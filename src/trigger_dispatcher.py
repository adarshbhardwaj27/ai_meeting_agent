"""
Priority-based async dispatcher for trigger processing.

Problem this solves
-------------------
The old design awaited _process_trigger() inline inside _main_loop(), which
means a slow LLM call (4–8 s) froze transcript processing entirely.  During
that window the bot could not detect a voice command like "Alex, summarize"
or any subsequent trigger.

New design
----------

    _main_loop()  →  dispatcher.dispatch()  [non-blocking, O(1)]
                           │
                    asyncio.PriorityQueue
                           │
                     Single worker task
                      ├─ dequeues highest-priority item
                      ├─ preempts active lower-priority LLM work on arrival
                      │  of a higher-priority trigger
                      ├─ wraps process_fn in asyncio.wait_for (hard LLM deadline)
                      └─ logs queue depth / preemptions / timeouts

Priority tiers
--------------
    0  EXPLICIT_COMMAND, VOICE_ADDRESS          ← user is waiting; never preempted
    1  DECISION_FORMING, CONFLICT_DETECTED      ← high-value proactive
    2  CONTRADICTION, FACTUAL_INCONSISTENCY,    ← medium-value proactive
       RISK_OVERLOOKED
    3  REPEATED_CONFUSION, CIRCULAR_DISCUSSION, ← low-value proactive
       MEMORY_GAP
    4  TOPIC_DRIFT, COLLABORATION_OPPORTUNITY   ← informational; dropped under load

Safety properties
-----------------
- Deduplication   : at most one queued item per TriggerType at any moment
                    (P0 is exempt — voice / commands must always reach the front)
- Backpressure    : tier ≥ 3 items are dropped when queue depth exceeds MAX_QUEUE_SIZE
- Preemption      : a tier-N trigger cancels an active tier-M task when N < M
                    (CancelledError propagates; _process_trigger's except Exception
                    does not catch it in Python 3.8+)
- LLM timeout     : per-tier hard deadline via asyncio.wait_for
- Observability   : structured log fields on every state transition
"""

import asyncio
import itertools
from typing import Callable, Coroutine, Any, Optional
import structlog

from src.context.models import Trigger, MeetingContext, TriggerType

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Priority tier mapping (lower number = higher priority)
# ---------------------------------------------------------------------------

TRIGGER_PRIORITY: dict[TriggerType, int] = {
    TriggerType.EXPLICIT_COMMAND:          0,
    TriggerType.VOICE_ADDRESS:             0,
    TriggerType.DECISION_FORMING:          1,
    TriggerType.CONFLICT_DETECTED:         1,
    TriggerType.CONTRADICTION:             2,
    TriggerType.FACTUAL_INCONSISTENCY:     2,
    TriggerType.RISK_OVERLOOKED:           2,
    TriggerType.REPEATED_CONFUSION:        3,
    TriggerType.CIRCULAR_DISCUSSION:       3,
    TriggerType.MEMORY_GAP:               3,
    TriggerType.TOPIC_DRIFT:              4,
    TriggerType.COLLABORATION_OPPORTUNITY: 4,
}

# Hard LLM generation deadline per priority tier (seconds).
# P0 is tightest — the user is literally waiting for a reply.
LLM_TIMEOUT_BY_TIER: dict[int, float] = {
    0: 10.0,
    1: 15.0,
    2: 15.0,
    3: 20.0,
    4: 20.0,
}

# Drop tier ≥ BACKPRESSURE_DROP_TIER when queue depth exceeds MAX_QUEUE_SIZE.
# This ensures voice / command triggers are never starved.
_MAX_QUEUE_SIZE    = 8
_BACKPRESSURE_TIER = 3   # drop tier 3 and 4 under load


# ---------------------------------------------------------------------------
# TriggerDispatcher
# ---------------------------------------------------------------------------

class TriggerDispatcher:
    """
    Priority-aware async dispatcher for trigger processing.

    The caller provides a *process_fn* coroutine factory — in production
    this is MeetingAssistant._process_trigger.  The dispatcher owns
    all scheduling concerns; process_fn owns all business logic.

    Thread-safety: designed for a single asyncio event loop.  All public
    methods must be called from that loop.
    """

    def __init__(
        self,
        process_fn: Callable[[Trigger, MeetingContext], Coroutine[Any, Any, None]],
    ) -> None:
        self._process_fn = process_fn

        # Priority queue entries: (priority, sequence, trigger, context)
        # Sequence (monotonic counter) breaks ties so same-priority items are FIFO.
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._seq = itertools.count()

        # Worker lifecycle
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None

        # In-flight tracking (used for preemption decisions)
        self._active_task: Optional[asyncio.Task] = None
        self._active_priority: int = 99  # sentinel: nothing in flight

        # Deduplication: trigger types currently sitting in the queue.
        # Does NOT include the item being actively processed.
        self._queued_types: set[TriggerType] = set()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background worker.  Call once after the event loop is running."""
        self._running = True
        self._worker_task = asyncio.create_task(
            self._worker_loop(), name="trigger_dispatcher_worker"
        )
        logger.info("trigger_dispatcher_started")

    async def stop(self) -> None:
        """
        Cancel in-flight work and shut down the worker cleanly.

        Does not drain the queue — any pending items are discarded.
        This matches the existing behaviour of MeetingAssistant.stop()
        which cancels background tasks without waiting for them.
        """
        self._running = False

        # Cancel the active LLM call first (fastest path to exit)
        if self._active_task and not self._active_task.done():
            self._active_task.cancel()
            try:
                await self._active_task
            except (asyncio.CancelledError, Exception):
                pass

        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        self._queued_types.clear()
        logger.info("trigger_dispatcher_stopped")

    async def drain(self) -> None:
        """
        Wait until the queue is empty and no item is actively processing.

        Useful in tests to avoid sleeping for a fixed duration.
        """
        await self._queue.join()
        # Give the active task one more round-trip to complete delivery
        if self._active_task and not self._active_task.done():
            try:
                await self._active_task
            except (asyncio.CancelledError, Exception):
                pass

    # ── Public API ───────────────────────────────────────────────────────────

    def dispatch(self, trigger: Trigger, context: MeetingContext) -> None:
        """
        Enqueue a trigger for priority-ordered processing.

        Non-blocking and O(1) — safe to call from _main_loop() on every
        polling cycle without stalling transcript ingestion.

        Decisions made here (in order):
          1. Backpressure  — drop tier ≥ 3 when queue is full
          2. Deduplication — skip if same type already queued (P1+ only)
          3. Preemption    — cancel active lower-priority task if needed
          4. Enqueue
        """
        if not self._running:
            return

        priority = TRIGGER_PRIORITY.get(trigger.type, 4)

        # ── 1. Backpressure ──────────────────────────────────────────────────
        if self._queue.qsize() >= _MAX_QUEUE_SIZE and priority >= _BACKPRESSURE_TIER:
            logger.info(
                "trigger_dropped_backpressure",
                trigger_type=trigger.type.value,
                priority=priority,
                queue_depth=self._queue.qsize(),
            )
            return

        # ── 2. Deduplication (P0 is exempt — commands must never be lost) ────
        if priority > 0 and trigger.type in self._queued_types:
            logger.debug(
                "trigger_deduplicated",
                trigger_type=trigger.type.value,
                priority=priority,
            )
            return

        # ── 3. Preemption ────────────────────────────────────────────────────
        # Cancel the active LLM task only when the new trigger is strictly
        # more important.  Preempting a P0 task never happens in practice
        # (nothing has lower priority than 0) but the guard is explicit.
        if (
            self._active_task
            and not self._active_task.done()
            and priority < self._active_priority
            and self._active_priority > 0  # never preempt another P0
        ):
            logger.info(
                "trigger_preempting_active_work",
                new_type=trigger.type.value,
                new_priority=priority,
                preempted_priority=self._active_priority,
            )
            self._active_task.cancel()

        # ── 4. Enqueue ───────────────────────────────────────────────────────
        self._queued_types.add(trigger.type)
        self._queue.put_nowait((priority, next(self._seq), trigger, context))

        logger.debug(
            "trigger_dispatched",
            trigger_type=trigger.type.value,
            priority=priority,
            queue_depth=self._queue.qsize(),
        )

    # ── Worker ───────────────────────────────────────────────────────────────

    async def _worker_loop(self) -> None:
        """
        Continuously dequeue the highest-priority item and process it.

        The outer loop never blocks on I/O itself — the 1-second
        asyncio.wait_for on queue.get() yields control regularly so
        other coroutines (including the main loop) can run.
        """
        while self._running:
            # Wait for the next item with a short timeout so we can check
            # self._running and exit cleanly when stop() is called.
            try:
                priority, _, trigger, context = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # Item is now "in-flight" — remove from dedup set
            self._queued_types.discard(trigger.type)
            self._active_priority = priority

            timeout = LLM_TIMEOUT_BY_TIER.get(priority, 20.0)

            self._active_task = asyncio.create_task(
                self._bounded_process(trigger, context, timeout),
                name=f"trigger_{trigger.type.value}",
            )

            try:
                await self._active_task
            except asyncio.CancelledError:
                # Preempted by a higher-priority trigger — this is expected
                # and intentional.  The active task was cancelled in dispatch().
                logger.info(
                    "trigger_preempted",
                    trigger_type=trigger.type.value,
                    priority=priority,
                )
            except Exception as e:
                # Should not reach here — _bounded_process swallows exceptions
                # internally, but guard defensively.
                logger.error(
                    "trigger_worker_unhandled_error",
                    trigger_type=trigger.type.value,
                    error=str(e),
                )
            finally:
                self._active_priority = 99
                self._active_task = None
                # Signal queue.join() (used by drain()) that this item is done
                self._queue.task_done()

    async def _bounded_process(
        self,
        trigger: Trigger,
        context: MeetingContext,
        timeout: float,
    ) -> None:
        """
        Call process_fn with a hard deadline.

        - asyncio.TimeoutError  → logged as a warning; no response is sent
        - asyncio.CancelledError → re-raised so preemption propagates correctly
        - Any other exception   → logged; worker continues
        """
        try:
            await asyncio.wait_for(
                self._process_fn(trigger, context),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "trigger_llm_timeout",
                trigger_type=trigger.type.value,
                timeout_s=timeout,
            )
        except asyncio.CancelledError:
            # Must re-raise — this is how _worker_loop detects preemption
            raise
        except Exception as e:
            logger.error(
                "trigger_process_error",
                trigger_type=trigger.type.value,
                error=str(e),
            )
