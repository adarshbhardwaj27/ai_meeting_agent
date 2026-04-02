"""
Real-Time Trigger Engine for the AI Meeting Assistant.

This engine continuously monitors the meeting context and transcript to detect
situations where the AI should send a chat message. It handles both:

1. Reactive triggers: Explicit commands like "@Alex summarize"
2. Proactive triggers: AI-initiated messages for conflicts, decisions, drift, etc.
"""

import asyncio
from collections import Counter
import re
from datetime import datetime, timedelta
from typing import List, Optional, Callable, Set
import structlog

from src.config import config, ConflictSensitivity
from src.context.models import (
    MeetingContext, TranscriptChunk, Trigger, TriggerType, CommandType,
    Concern, Decision, ConcernSeverity, DecisionStatus,
    SignalType, SignalCandidate,
)
from src.triggers.conversation_intelligence import ConversationIntelligence

logger = structlog.get_logger()


class TriggerEngine:
    """
    Continuously monitors for trigger conditions that warrant an AI response.
    
    The trigger engine runs on a tight loop (every ~500ms) to ensure
    real-time responsiveness during live conversations.
    """
    
    def __init__(
        self,
        on_trigger: Optional[Callable[[Trigger], None]] = None,
        check_interval_ms: int = None,
        enable_proactive: bool = None
    ):
        self.on_trigger = on_trigger
        self.check_interval_ms = check_interval_ms or config.trigger.check_interval_ms
        self.enable_proactive = enable_proactive if enable_proactive is not None else config.trigger.enable_proactive
        
        # State tracking
        self._running = False
        self._check_task: Optional[asyncio.Task] = None
        self._last_check_time: Optional[datetime] = None
        self._processed_chunk_ids: Set[str] = set()
        self._recent_triggers: List[Trigger] = []  # Prevent duplicate triggers
        
        # Trigger cooldowns (prevent spamming)
        self._last_trigger_by_type: dict[TriggerType, datetime] = {}
        self._cooldown_seconds = {
            TriggerType.EXPLICIT_COMMAND: 0,  # No cooldown for commands
            TriggerType.VOICE_ADDRESS: 0,     # No cooldown for voice-addressed questions
            TriggerType.DECISION_FORMING: 30,
            TriggerType.CONFLICT_DETECTED: 20,
            TriggerType.TOPIC_DRIFT: 60,
            TriggerType.REPEATED_CONFUSION: 45,
            TriggerType.COMPLEX_CONCEPT: 30,
            TriggerType.RISK_OVERLOOKED: 30,
            TriggerType.CIRCULAR_DISCUSSION: 60,
            # Intelligence-layer signal types
            TriggerType.CONTRADICTION: 15,
            TriggerType.MEMORY_GAP: 20,
            TriggerType.FACTUAL_INCONSISTENCY: 20,
            TriggerType.COLLABORATION_OPPORTUNITY: 30,
        }
        
        # Conversation intelligence (Tier-1 local scorer)
        self._intelligence = ConversationIntelligence()
        
        # Bot name for command detection
        self.bot_name = config.bot.name.lower()

        # Voice-mention detection state
        # When someone says "Alex, ..." we may need to wait for the next
        # chunk(s) to capture the full question before firing a trigger.
        self._voice_mention_patterns = self._build_voice_mention_patterns()
        self._voice_buffer_speaker: Optional[str] = None
        self._voice_buffer_text: str = ""
        self._voice_buffer_started: Optional[datetime] = None
        self._voice_buffer_last_chunk: Optional[datetime] = None  # when last content was appended
        self._voice_buffer_max_age = timedelta(seconds=15)  # max wait before discarding
        self._voice_settle_time  = timedelta(seconds=3)    # silence gap → speaker is done

        # Decision language patterns
        self._decision_patterns = [
            (r"let's (?:go with|use|do|stick with|try) (.+)", 0.8),
            (r"we(?:'ll| will| should) (?:go with|use|do) (.+)", 0.9),
            (r"(?:i think|i vote|i suggest) we (?:should )? (.+)", 0.6),
            (r"(?:agreed|ok|alright)[,.]? (?:we'll|let's) (.+)", 0.95),
            (r"(?:the decision is|it's decided|we've decided)[:\s]+ (.+)", 1.0),
            (r"(?:final answer|that's it)[,:]? (.+)", 0.9),
        ]
        
        # Concern/risk language patterns  
        self._concern_patterns = [
            r"(?:worried|concerned|afraid) about",
            r"(?:risk|issue|problem) with",
            r"what if .+ (?:fails|breaks|doesn't work)",
            r"(?:might|could|may) (?:cause|be|create) (?:problems|issues)",
        ]
        
        # Confusion patterns
        self._confusion_patterns = [
            r"(?:i don't|i do not) (?:understand|get|follow)",
            r"(?:what do you mean|can you explain|i(?:'m| am) confused)",
            r"(?:sorry|wait)[,]? (?:what|can you repeat)",
            r"(?:i(?:'m| am) lost|lost me there|a bit lost)",
            r"(?:i(?:'m| am) not following|not following)",
            r"(?:can we slow down|slow down(?: a bit)?|too fast)",
        ]

        self._semantic_stopwords = {
            "about", "after", "again", "against", "almost", "also", "although",
            "always", "among", "around", "because", "before", "being", "between",
            "both", "came", "come", "could", "each", "else", "ever", "from",
            "getting", "given", "going", "have", "into", "just", "kind", "like",
            "make", "maybe", "much", "must", "need", "only", "other", "really",
            "should", "since", "some", "still", "such", "than", "that", "their",
            "them", "then", "there", "these", "they", "thing", "this", "those",
            "through", "today", "topic", "very", "want", "were", "what", "when",
            "where", "which", "while", "with", "would", "yeah", "your", "lets",
            "weve", "were", "dont", "doesnt", "cant", "wont", "isnt", "arent",
            "team", "okay", "alright", "right", "need", "talk", "discuss"
        }
        
        # Command patterns (reactive triggers)
        # ORDER MATTERS: more specific commands must come before general ones
        # (e.g. ACTION_ITEMS before EXPLAIN, since "what are the action items"
        # would otherwise match EXPLAIN's "what are ..." pattern)
        from collections import OrderedDict
        self._command_patterns: dict[CommandType, list] = OrderedDict([
            (CommandType.SUMMARIZE, [
                r"summarize",
                r"summary",
                r"what (?:have we|did we) (?:discussed|talked about|covered)",
                r"where are we",
                r"catch me up",
            ]),
            (CommandType.ACTION_ITEMS, [
                r"action items",
                r"(?:what are the |list )tasks",
                r"to-?dos?",
                r"next steps",
            ]),
            (CommandType.RISKS, [
                r"(?:what are the |any )risks",
                r"concerns",
                r"(?:what could|what might) go wrong",
                r"issues",
            ]),
            (CommandType.DECISION, [
                r"(?:what|which) decision",
                r"what (?:are we|did we) (?:decide|choosing|leaning)",
                r"current (?:decision|direction)",
            ]),
            (CommandType.EXPLAIN, [
                r"explain",
                r"what (?:is|does) (.+)",
                r"define",
                r"meaning of",
            ]),
        ])
    
    async def start(self):
        """Start the trigger engine background processing."""
        self._running = True
        self._check_task = asyncio.create_task(self._check_loop())
        logger.info("trigger_engine_started")
    
    async def stop(self):
        """Stop the trigger engine."""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        logger.info("trigger_engine_stopped")
    
    async def check_for_triggers(
        self, 
        context: MeetingContext,
        new_chunks: Optional[List[TranscriptChunk]] = None,
        chat_text: Optional[str] = None
    ) -> List[Trigger]:
        """
        Check for trigger conditions given the current context.
        
        Args:
            context: Current meeting context
            new_chunks: New transcript chunks since last check (optional)
            chat_text: New chat message to check for commands (optional)
        
        Returns:
            List of triggered conditions
        """
        triggers = []
        
        # 1. Check for explicit commands in chat (highest priority)
        if chat_text:
            cmd_trigger = self._check_for_commands(chat_text)
            if cmd_trigger:
                triggers.append(cmd_trigger)
                # Return immediately for commands - they take priority
                return triggers
        
        # 1.5  Check for voice mentions in transcript ("Hey Alex, ...")
        #       Always run — even with no new chunks the settle-timer may flush
        #       the buffer from a previous cycle.
        voice_trigger = self._check_for_voice_mention(new_chunks or [], context)
        if voice_trigger:
            triggers.append(voice_trigger)
            return triggers  # Voice-addressed questions take priority

        # Skip proactive triggers if disabled
        if not self.enable_proactive:
            return triggers
        
        # Get recent transcript text for pattern matching
        recent_text = self._get_recent_text(context, new_chunks)
        if not recent_text:
            return triggers
        
        # 2. Check for decision + conflict (high priority)
        decision_trigger = self._check_for_decision_conflict(context, recent_text)
        if decision_trigger:
            triggers.append(decision_trigger)
        
        # 3. Check for overlooked risks
        risk_trigger = self._check_for_overlooked_risks(context, recent_text)
        if risk_trigger and not decision_trigger:  # Don't double-trigger
            triggers.append(risk_trigger)
        
        # 4. Check for topic drift
        drift_trigger = self._check_for_topic_drift(context, recent_text)
        if drift_trigger:
            triggers.append(drift_trigger)
        
        # 5. Check for repeated confusion
        confusion_trigger = self._check_for_repeated_confusion(context, recent_text)
        if confusion_trigger:
            triggers.append(confusion_trigger)
        
        # 6. Check for circular discussion
        circular_trigger = self._check_for_circular_discussion(context)
        if circular_trigger:
            triggers.append(circular_trigger)
        
        # 7. Intelligence-layer signals (Tier 1 — zero LLM cost)
        intelligence_signals = self._intelligence.score_signals(
            context=context, new_chunks=new_chunks
        )
        for signal in intelligence_signals:
            signal_trigger = self._promote_signal(signal)
            if signal_trigger:
                triggers.append(signal_trigger)

        # Filter out triggers on cooldown
        triggers = self._filter_cooldowns(triggers)
        
        # Update cooldown timestamps
        for trigger in triggers:
            self._last_trigger_by_type[trigger.type] = datetime.now()
            self._recent_triggers.append(trigger)
        
        return triggers
    
    # ── Signal promotion (Tier 1 → Trigger) ──────────────────────────

    _SIGNAL_TO_TRIGGER: dict[SignalType, TriggerType] = {
        SignalType.CONTRADICTION: TriggerType.CONTRADICTION,
        SignalType.MEMORY_GAP: TriggerType.MEMORY_GAP,
        SignalType.FACTUAL_INCONSISTENCY: TriggerType.FACTUAL_INCONSISTENCY,
        SignalType.COLLABORATION_OPPORTUNITY: TriggerType.COLLABORATION_OPPORTUNITY,
    }

    def _promote_signal(self, signal: SignalCandidate) -> Optional[Trigger]:
        """Convert a Tier-1 SignalCandidate into a Trigger if it qualifies."""
        trigger_type = self._SIGNAL_TO_TRIGGER.get(signal.signal_type)
        if trigger_type is None:
            return None

        trigger = Trigger.create_proactive(
            trigger_type=trigger_type,
            confidence=signal.confidence,
            reason=signal.reason,
        )
        trigger.relevant_transcript = signal.relevant_transcript

        # Attach evidence for the LLM prompt builder
        trigger.signal_evidence_a = signal.evidence_a
        trigger.signal_evidence_b = signal.evidence_b
        trigger.signal_speaker_a = signal.speaker_a
        trigger.signal_speaker_b = signal.speaker_b

        logger.info(
            "signal_promoted",
            signal_type=signal.signal_type.value,
            trigger_type=trigger_type.value,
            confidence=round(signal.confidence, 2),
        )
        return trigger

    # ── Voice-mention detection ────────────────────────────────────

    # Words that start a question or command — used by the question
    # heuristic when the bot's name appears mid/end-of-sentence.
    _QUESTION_STARTERS = frozenset({
        # Interrogative / wh-words
        'what', 'how', 'why', 'when', 'where', 'who', 'which',
        # Auxiliary verbs that open a yes/no question
        'can', 'could', 'should', 'would', 'will', 'do', 'does',
        'did', 'is', 'are', 'was', 'were', 'has', 'have',
        # Imperative / command verbs people use with bots
        'summarize', 'explain', 'help', 'tell', 'show', 'list',
        'find', 'check', 'give', 'describe', 'compare', 'analyze',
        'please',
    })

    _DIRECT_ADDRESS_CUE = re.compile(
        r"(?:^|\b)(?:"
        r"(?:i\s+(?:am|was)\s+asking)"
        r"|(?:i['’]?m\s+asking)"
        r"|asking"
        r"|ask"
        r"|(?:i\s+(?:want|wanted|need|needed|meant)\s+to\s+ask)"
        r"|(?:i\s+(?:am|was)\s+trying\s+to\s+ask)"
        r"|(?:let\s+me\s+ask)"
        r"|tell"
        r")\s*$",
        re.IGNORECASE,
    )

    def _build_voice_mention_patterns(self) -> List[re.Pattern]:
        """Build pattern to detect the bot's name at the START of speech.

        Allows zero or more filler/greeting words before the name, so
        "okay so alex what do you think?" is handled the same as
        "alex what do you think?".

        For name-at-END or name-in-MIDDLE detection, see
        ``_match_voice_mention`` which falls through to a second check
        with a question heuristic.
        """
        name = re.escape(self.bot_name)
        return [
            # Name at start, with zero or more filler/greeting/connector
            # words.  Real speech often chains fillers: "okay so alex",
            # "yeah but alex", "um well alex", etc.
            # Matches: "alex ...", "hey alex ...", "okay so alex ...",
            #          "yeah but alex, ...", "um well alex", "alex"
            re.compile(
                rf"^(?:(?:hey|hi|yo|ok|so|and|but|yeah|yea|like"
                rf"|well|um|uh|oh|right|okay|now)\s+)*{name}[,:\s]*(.*)",
                re.IGNORECASE,
            ),
        ]

    def _match_voice_mention(self, text: str) -> Optional[str]:
        """Detect the bot's name in *text* and extract the question part.

        Returns the question text (str) or ``None`` if no mention found.

        Three positions are handled:

        1. **START** — ``"hey alex what time is it?"``
           → returns ``"what time is it?"``  (most reliable, no heuristic)

        2. **END** — ``"what do you think alex?"``
           → returns ``"what do you think?"``  (requires question heuristic)

        3. **MIDDLE** — ``"can alex help with this?"``
           → returns ``"can help with this?"``  (requires question heuristic)

        For positions 2 and 3 a **question heuristic** is applied to
        avoid false positives like *"I talked to alex yesterday"*: the
        sentence must end with ``?`` or start with a question/command
        word (what, how, can, summarize, …).
        """
        text_stripped = text.strip()
        if not text_stripped:
            return None

        # ── 1. Name at START (most reliable) ────────────────────────
        for pattern in self._voice_mention_patterns:
            m = pattern.search(text_stripped)
            if m:
                return (m.group(1) or "").strip()

        # ── 2. Name elsewhere — require question heuristic ──────────
        name_re = re.compile(
            rf'\b{re.escape(self.bot_name)}\b', re.IGNORECASE
        )
        m = name_re.search(text_stripped)
        if not m:
            return None

        before = text_stripped[:m.start()].strip().rstrip(',: ')
        after  = text_stripped[m.end():].strip().lstrip(',: ')

        directed_payload = self._extract_direct_address_payload(before, after)
        if directed_payload:
            return directed_payload

        # Must look like a question / command to avoid 3rd-person refs
        if not self._looks_like_question(text_stripped):
            return None

        if not after or after.rstrip('?!. ') == '':
            # Name at END: "what do you think alex?"
            # Transfer any trailing punctuation back onto the question
            punct = after.strip() if after.strip() in ('?', '!', '.') else ''
            return (before + punct).strip()
        else:
            # Name in MIDDLE: "can alex help with this?"
            return f"{before} {after}".strip()

    def _extract_direct_address_payload(
        self,
        before_name_text: str,
        after_name_text: str,
    ) -> Optional[str]:
        """Extract a request from phrases like 'I am asking Alex ...'."""
        if not before_name_text or not after_name_text:
            return None

        if not self._DIRECT_ADDRESS_CUE.search(before_name_text.strip()):
            return None

        normalized_after = after_name_text.strip()
        normalized_after = re.sub(r"^(?:please\s+)", "", normalized_after, flags=re.IGNORECASE)
        normalized_after = re.sub(r"^(?:to\s+)", "", normalized_after, flags=re.IGNORECASE)
        if not normalized_after:
            return None

        if self._looks_like_question(normalized_after):
            return normalized_after

        normalized_lower = normalized_after.lower()
        for patterns in self._command_patterns.values():
            if any(re.search(pattern, normalized_lower) for pattern in patterns):
                return normalized_after

        return None

    @staticmethod
    def _looks_like_question(text: str) -> bool:
        """Return True if *text* looks like a question or command.

        Used as a guard for non-start name positions to reject
        3rd-person references ("I talked to alex yesterday").
        """
        text = text.strip()
        if not text:
            return False
        # Ends with a question mark → definitely a question
        if text[-1] == '?':
            return True
        # First word is a question / command word
        first_word = text.split()[0].lower().rstrip('.,!?')
        return first_word in TriggerEngine._QUESTION_STARTERS

    def _check_for_voice_mention(
        self,
        chunks: List[TranscriptChunk],
        context: MeetingContext,
    ) -> Optional[Trigger]:
        """
        Detect when someone says the bot's name out loud and collect
        everything they say afterwards.

        Simple rule: name at start of speech = talking to the bot.
        Buffer all subsequent speech (final + interim chunks keep the
        buffer alive).  Fire when:
          a) Punctuation at end (STT endpointed the sentence), OR
          b) 3 seconds of true silence (no chunks at all).
        Discard if 15 s pass with no meaningful content.
        """
        now = datetime.now()

        # ── 1. Expire stale buffer (max age exceeded) ──────────────
        if (
            self._voice_buffer_started
            and (now - self._voice_buffer_started) > self._voice_buffer_max_age
        ):
            logger.debug("voice_buffer_expired", text=self._voice_buffer_text[:60])
            self._clear_voice_buffer()

        # ── 2. Process incoming chunks FIRST ────────────────────────
        #    We must process chunks before checking the settle timer
        #    because interim chunks reset the settle clock.
        for chunk in chunks:
            text = chunk.text.strip()

            # Interim (non-final) chunks = "speaker is still talking".
            # Don't add their text (it'll be superseded by the final),
            # but reset the settle clock so we don't fire prematurely
            # during stutters, "ummm", or thinking pauses.
            if not chunk.is_final:
                if self._voice_buffer_started and text:
                    self._voice_buffer_last_chunk = now
                    logger.debug(
                        "voice_buffer_heartbeat",
                        interim_text=text[:40],
                        speaker=chunk.speaker,
                    )
                continue

            if not text:
                continue

            # ── Does this chunk contain the bot's name? ─────────────
            matched_name = False
            after_name_text = self._match_voice_mention(text)
            if after_name_text is not None:
                    matched_name = True
                    after_name = after_name_text

                    # Punctuation = STT already endpointed → fire now.
                    if after_name and after_name[-1] in "?.!":
                        self._clear_voice_buffer()
                        return self._create_voice_trigger(
                            speaker=chunk.speaker,
                            question_text=after_name,
                            context=context,
                        )

                    # ── If the buffer already has content, fire it
                    #    before starting the new question.  This
                    #    prevents the old question from being silently
                    #    dropped when someone says "alex" again.
                    pending_trigger = None
                    if (
                        self._voice_buffer_started
                        and self._voice_buffer_text.strip()
                    ):
                        logger.debug(
                            "voice_buffer_flushed_by_new_name",
                            old_text=self._voice_buffer_text[:60],
                            new_text=after_name[:40],
                        )
                        pending_trigger = self._create_voice_trigger(
                            speaker=self._voice_buffer_speaker or chunk.speaker,
                            question_text=self._voice_buffer_text.strip(),
                            context=context,
                        )

                    # Start the new buffer.
                    self._voice_buffer_speaker = chunk.speaker
                    self._voice_buffer_text = after_name
                    self._voice_buffer_started = now
                    self._voice_buffer_last_chunk = now

                    # Return the old trigger if there was one.
                    if pending_trigger:
                        return pending_trigger

            # ── Continuation of an already-buffered question ────────
            if not matched_name and self._voice_buffer_started:
                self._voice_buffer_text += " " + text
                self._voice_buffer_last_chunk = now
                content_added_this_cycle = True

                # Punctuation in the continuation → fire immediately.
                if self._voice_buffer_text.strip()[-1:] in ("?", ".", "!"):
                    trigger = self._create_voice_trigger(
                        speaker=self._voice_buffer_speaker or chunk.speaker,
                        question_text=self._voice_buffer_text.strip(),
                        context=context,
                    )
                    self._clear_voice_buffer()
                    return trigger

        # ── 3. Check settle timer AFTER processing chunks ───────────
        #    If no chunks arrived (or only interim), check if the speaker
        #    has been quiet long enough to flush the buffer.
        if (
            self._voice_buffer_started
            and self._voice_buffer_text.strip()
            and self._voice_buffer_last_chunk
            and (now - self._voice_buffer_last_chunk) >= self._voice_settle_time
        ):
            logger.debug(
                "voice_buffer_settled",
                text=self._voice_buffer_text[:60],
                settle_secs=round(
                    (now - self._voice_buffer_last_chunk).total_seconds(), 2
                ),
            )
            trigger = self._create_voice_trigger(
                speaker=self._voice_buffer_speaker,
                question_text=self._voice_buffer_text.strip(),
                context=context,
            )
            self._clear_voice_buffer()
            return trigger

        return None

    def _clear_voice_buffer(self):
        """Reset all voice-buffer state."""
        self._voice_buffer_speaker = None
        self._voice_buffer_text = ""
        self._voice_buffer_started = None
        self._voice_buffer_last_chunk = None

    def _create_voice_trigger(
        self,
        speaker: Optional[str],
        question_text: str,
        context: MeetingContext,
    ) -> Trigger:
        """
        Create a trigger for a voice-addressed question.

        First tries to match the question against known command patterns
        (summarize, risks, etc.).  Falls back to CUSTOM if none match.
        """
        question_lower = question_text.lower().strip()

        # Try to classify as a known command
        for cmd_type, patterns in self._command_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    args = None
                    if cmd_type == CommandType.EXPLAIN:
                        m = re.search(r"(?:what is|explain|define) (.+)", question_lower)
                        if m:
                            args = m.group(1).strip().rstrip("?.!")
                    trigger = Trigger.create_command(cmd_type, args)
                    trigger.type = TriggerType.VOICE_ADDRESS
                    trigger.reason = f"{speaker or 'Someone'} asked by voice: {cmd_type.value}"
                    trigger.relevant_transcript = question_text
                    logger.info(
                        "voice_command_detected",
                        speaker=speaker,
                        command=cmd_type.value,
                    )
                    return trigger

        # No known command matched — treat as a free-form question
        trigger = Trigger.create_command(CommandType.CUSTOM, question_text)
        trigger.type = TriggerType.VOICE_ADDRESS
        trigger.reason = f"{speaker or 'Someone'} asked by voice: \"{question_text}\""
        trigger.relevant_transcript = question_text
        logger.info("voice_question_detected", speaker=speaker, text=question_text[:80])
        return trigger

    def _get_recent_text(
        self, 
        context: MeetingContext, 
        new_chunks: Optional[List[TranscriptChunk]] = None
    ) -> str:
        """Get recent transcript text for pattern matching."""
        if new_chunks:
            return " ".join(c.text for c in new_chunks if c.is_final)
        
        # Get last ~2 minutes of transcript
        cutoff = datetime.now() - timedelta(minutes=2)
        recent = [c for c in context.recent_transcript if c.timestamp >= cutoff]
        return " ".join(c.text for c in recent if c.is_final)
    
    def _check_for_commands(self, chat_text: str) -> Optional[Trigger]:
        """Check if chat message contains a command for the bot."""
        text_lower = chat_text.lower()
        
        # Must mention the bot
        if f"@{self.bot_name}" not in text_lower:
            return None
        
        # Extract the command part (after @botname)
        command_part = text_lower.split(f"@{self.bot_name}", 1)[-1].strip()
        
        # Check each command type
        for cmd_type, patterns in self._command_patterns.items():
            for pattern in patterns:
                if re.search(pattern, command_part):
                    # Extract any arguments
                    args = None
                    if cmd_type == CommandType.EXPLAIN:
                        match = re.search(r"(?:what is|explain|define) (.+)", command_part)
                        if match:
                            args = match.group(1).strip()
                    
                    trigger = Trigger.create_command(cmd_type, args)
                    trigger.relevant_transcript = command_part
                    
                    logger.info(
                        "command_detected",
                        command=cmd_type.value,
                        args=args
                    )
                    return trigger
        
        # Unknown command - still respond
        trigger = Trigger.create_command(CommandType.CUSTOM, command_part)
        return trigger
    
    def _check_for_decision_conflict(
        self, 
        context: MeetingContext, 
        recent_text: str
    ) -> Optional[Trigger]:
        """Check if a decision is being made that conflicts with prior concerns."""
        text_lower = recent_text.lower()
        transcript_concerns = self._extract_recent_transcript_concerns(context)
        all_concerns = self._merge_concerns(
            context.unaddressed_concerns,
            transcript_concerns,
        )
        
        for pattern, confidence in self._decision_patterns:
            match = re.search(pattern, text_lower)
            if match:
                decision_text = match.group(1) if match.groups() else match.group(0)
                
                # Look for conflicting concerns
                conflicts = self._find_conflicting_concerns(
                    decision_text, 
                    all_concerns
                )
                
                if conflicts:
                    # High priority: decision conflicts with prior concern
                    trigger = Trigger.create_proactive(
                        TriggerType.CONFLICT_DETECTED,
                        confidence=min(0.95, confidence + 0.1),
                        reason=f"Decision about '{decision_text}' may conflict with earlier concern: '{conflicts[0].description}'"
                    )
                    trigger.conflicting_concern = conflicts[0]
                    trigger.relevant_transcript = recent_text[-500:]
                    
                    logger.info(
                        "conflict_detected",
                        decision=decision_text,
                        concern=conflicts[0].description
                    )
                    return trigger
                
                elif confidence >= config.trigger.decision_confidence_threshold:
                    # Decision forming without obvious conflict
                    trigger = Trigger.create_proactive(
                        TriggerType.DECISION_FORMING,
                        confidence=confidence,
                        reason=f"Team appears to be deciding: '{decision_text}'"
                    )
                    trigger.relevant_transcript = recent_text[-500:]
                    return trigger
        
        return None

    def _extract_recent_transcript_concerns(
        self,
        context: MeetingContext,
        minutes: int = 3,
    ) -> List[Concern]:
        """Extract concern-like statements from recent final transcript chunks."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        derived: List[Concern] = []

        for chunk in context.recent_transcript:
            if not chunk.is_final or chunk.timestamp < cutoff:
                continue

            text_lower = chunk.text.lower()
            if not any(re.search(pattern, text_lower) for pattern in self._concern_patterns):
                continue

            severity = ConcernSeverity.MEDIUM
            if any(term in text_lower for term in ("critical", "severe", "high risk", "major")):
                severity = ConcernSeverity.HIGH

            concern = Concern.create(
                description=chunk.text,
                raised_by=chunk.speaker,
                severity=severity,
            )
            concern.raised_at = chunk.timestamp
            derived.append(concern)

        return derived

    @staticmethod
    def _merge_concerns(*concern_groups: List[Concern]) -> List[Concern]:
        """Merge multiple concern lists while deduplicating by description."""
        merged: List[Concern] = []
        seen_descriptions: Set[str] = set()

        for group in concern_groups:
            for concern in group:
                key = concern.description.strip().lower()
                if key in seen_descriptions:
                    continue
                seen_descriptions.add(key)
                merged.append(concern)

        return merged
    
    def _find_conflicting_concerns(
        self, 
        decision_text: str, 
        concerns: List[Concern]
    ) -> List[Concern]:
        """Find concerns that might conflict with a decision."""
        conflicts = []
        decision_words = set(decision_text.lower().split())
        
        # Keywords that indicate potential conflict topics
        conflict_topics = {
            "kafka": ["latency", "complexity", "learning curve", "cost"],
            "microservices": ["complexity", "overhead", "debugging"],
            "cloud": ["cost", "vendor lock-in", "security"],
            "database": ["scaling", "migration", "consistency"],
            "api": ["versioning", "breaking changes", "security"],
        }
        
        for concern in concerns:
            concern_lower = concern.description.lower()
            concern_words = set(concern_lower.split())
            
            # Direct word overlap
            overlap = decision_words & concern_words
            if len(overlap) >= 1:
                conflicts.append(concern)
                continue
            
            # Topic-based conflict detection
            for topic, related_concerns in conflict_topics.items():
                if topic in decision_text.lower():
                    if any(rc in concern_lower for rc in related_concerns):
                        conflicts.append(concern)
                        break
        
        # Sort by severity (most severe first)
        conflicts.sort(key=lambda c: c.severity.value, reverse=True)
        return conflicts
    
    def _check_for_overlooked_risks(
        self, 
        context: MeetingContext, 
        recent_text: str
    ) -> Optional[Trigger]:
        """Check if the team is overlooking high-priority risks."""
        # Look for high-priority unaddressed concerns
        high_priority = context.high_priority_concerns
        if not high_priority:
            return None
        
        # Check if recent discussion mentions the risky topic without addressing concern
        for concern in high_priority:
            # Check how old the concern is
            age = datetime.now() - concern.raised_at
            if age < timedelta(minutes=3):
                continue  # Too recent, give them time
            
            # Check if the topic is being discussed
            concern_keywords = concern.description.lower().split()[:3]  # First few words
            if any(kw in recent_text.lower() for kw in concern_keywords):
                # Topic mentioned but concern not addressed
                trigger = Trigger.create_proactive(
                    TriggerType.RISK_OVERLOOKED,
                    confidence=0.7,
                    reason=f"Discussion continues but earlier {concern.severity.value} concern not addressed: '{concern.description}'"
                )
                trigger.conflicting_concern = concern
                trigger.relevant_transcript = recent_text[-500:]
                return trigger
        
        return None
    
    def _check_for_topic_drift(
        self, 
        context: MeetingContext, 
        recent_text: str
    ) -> Optional[Trigger]:
        """Check if the conversation has drifted from the active meeting focus."""
        recent_tokens = self._extract_keywords(recent_text)
        if len(recent_tokens) < config.trigger.topic_drift_min_tokens:
            return None

        anchor_terms = self._build_context_anchor_terms(context)
        if not anchor_terms:
            return None

        shared_terms = recent_tokens & anchor_terms
        similarity = len(shared_terms) / len(recent_tokens)
        novelty_ratio = 1.0 - similarity

        active_topic_terms = self._extract_keywords(
            context.current_topic.name if context.current_topic else ""
        )
        active_topic_overlap = len(recent_tokens & active_topic_terms)

        if not self._is_topic_drift(
            context=context,
            similarity=similarity,
            novelty_ratio=novelty_ratio,
            active_topic_overlap=active_topic_overlap,
            shared_terms_count=len(shared_terms),
        ):
            return None

        novel_terms = self._get_dominant_terms(recent_text, exclude_terms=anchor_terms)
        focus_label = self._describe_active_focus(context)
        drift_target = ", ".join(novel_terms[:3]) if novel_terms else "a different topic"
        confidence = min(0.82, 0.55 + (novelty_ratio * 0.25))

        trigger = Trigger.create_proactive(
            TriggerType.TOPIC_DRIFT,
            confidence=confidence,
            reason=f"Conversation has shifted away from {focus_label} toward {drift_target}"
        )
        trigger.relevant_transcript = recent_text[-300:]
        return trigger

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract normalized content words for lightweight topic comparison."""
        return {
            token for token in self._tokenize(text)
            if len(token) >= 3 and token not in self._semantic_stopwords
        }

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and lightly normalize text for semantic comparison."""
        raw_tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9'-]+", text.lower())
        return [self._normalize_token(token) for token in raw_tokens]

    def _normalize_token(self, token: str) -> str:
        """Normalize token variants to reduce superficial wording differences."""
        token = token.strip("'-")
        if len(token) > 4 and token.endswith("ies"):
            return token[:-3] + "y"
        if len(token) > 4 and token.endswith("ing"):
            return token[:-3]
        if len(token) > 3 and token.endswith("ed"):
            return token[:-2]
        if len(token) > 3 and token.endswith("es"):
            return token[:-2]
        if len(token) > 3 and token.endswith("s"):
            return token[:-1]
        return token

    def _build_context_anchor_terms(self, context: MeetingContext) -> Set[str]:
        """Build the current meeting focus from tracked state instead of canned phrases."""
        sources: List[str] = []

        if context.current_topic:
            sources.append(context.current_topic.name)

        sources.extend(topic.name for topic in context.topics_discussed[-3:])
        sources.extend(concern.description for concern in context.unaddressed_concerns[:5])
        sources.extend(decision.description for decision in context.decisions_pending[:5])
        sources.extend(decision.description for decision in context.decisions_confirmed[-3:])
        sources.extend(context.open_questions[-3:])
        sources.extend(context.action_items[-3:])

        anchors: Set[str] = set()
        for source in sources:
            anchors.update(self._extract_keywords(source))

        return anchors

    def _is_topic_drift(
        self,
        context: MeetingContext,
        similarity: float,
        novelty_ratio: float,
        active_topic_overlap: int,
        shared_terms_count: int,
    ) -> bool:
        """Decide whether low context alignment is strong enough to count as drift."""
        if similarity >= config.trigger.topic_drift_similarity_threshold:
            return False

        if novelty_ratio < 0.72:
            return False

        if shared_terms_count > 1:
            return False

        if context.current_topic and context.current_topic.duration < timedelta(minutes=1):
            return False

        if context.current_topic and active_topic_overlap > 0:
            return False

        return True

    def _get_dominant_terms(self, text: str, exclude_terms: Set[str]) -> List[str]:
        """Return the most salient non-anchor terms from the recent conversation."""
        counts = Counter(
            token for token in self._tokenize(text)
            if len(token) >= 3
            and token not in self._semantic_stopwords
            and token not in exclude_terms
        )
        return [term for term, _ in counts.most_common(5)]

    def _describe_active_focus(self, context: MeetingContext) -> str:
        """Produce a human-readable label for the current meeting focus."""
        if context.current_topic and context.current_topic.name:
            return f"'{context.current_topic.name}'"

        if context.decisions_pending:
            return f"pending decision '{context.decisions_pending[0].description}'"

        if context.unaddressed_concerns:
            return f"open concern '{context.unaddressed_concerns[0].description}'"

        return "the active meeting topic"
        return None
    
    def _check_for_repeated_confusion(
        self, 
        context: MeetingContext, 
        recent_text: str
    ) -> Optional[Trigger]:
        """Check if the same confusion is appearing multiple times."""
        window_text = context.get_recent_text(minutes=2) or recent_text
        text_lower = window_text.lower()
        
        # Count confusion indicators
        confusion_count = 0
        for pattern in self._confusion_patterns:
            matches = re.findall(pattern, text_lower)
            confusion_count += len(matches)
        
        # Threshold: 2+ confusion indicators in recent text
        if confusion_count >= 2:
            trigger = Trigger.create_proactive(
                TriggerType.REPEATED_CONFUSION,
                confidence=0.65,
                reason=f"Multiple signs of confusion detected ({confusion_count} indicators)"
            )
            trigger.relevant_transcript = window_text[-400:]
            return trigger
        
        return None
    
    def _check_for_circular_discussion(
        self, 
        context: MeetingContext
    ) -> Optional[Trigger]:
        """Check if the team is going in circles on a topic."""
        # Check if same decision has been discussed multiple times
        pending = context.decisions_pending
        if not pending:
            return None
        
        for decision in pending:
            # Decision discussed for a while without resolution
            age = datetime.now() - decision.proposed_at
            if age > timedelta(minutes=5) and decision.status == DecisionStatus.DISCUSSED:
                trigger = Trigger.create_proactive(
                    TriggerType.CIRCULAR_DISCUSSION,
                    confidence=0.6,
                    reason=f"Discussion about '{decision.description}' has been ongoing for {int(age.total_seconds()/60)} minutes without resolution"
                )
                trigger.conflicting_decision = decision
                return trigger
        
        return None
    
    def _filter_cooldowns(self, triggers: List[Trigger]) -> List[Trigger]:
        """Filter out triggers that are on cooldown."""
        filtered = []
        now = datetime.now()
        
        for trigger in triggers:
            cooldown = self._cooldown_seconds.get(trigger.type, 30)
            last_trigger = self._last_trigger_by_type.get(trigger.type)
            
            if last_trigger is None:
                filtered.append(trigger)
            elif (now - last_trigger).total_seconds() >= cooldown:
                filtered.append(trigger)
            else:
                logger.debug(
                    "trigger_on_cooldown",
                    type=trigger.type.value,
                    seconds_remaining=cooldown - (now - last_trigger).total_seconds()
                )
        
        return filtered
    
    async def _check_loop(self):
        """Background loop that periodically checks for triggers."""
        # This is called when integrated with the main orchestrator
        # For standalone use, call check_for_triggers directly
        while self._running:
            try:
                await asyncio.sleep(self.check_interval_ms / 1000.0)
                self._last_check_time = datetime.now()
            except asyncio.CancelledError:
                break
