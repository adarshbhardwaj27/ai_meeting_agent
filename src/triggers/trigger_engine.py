"""
Real-Time Trigger Engine for the AI Meeting Assistant.

This engine continuously monitors the meeting context and transcript to detect
situations where the AI should send a chat message. It handles both:

1. Reactive triggers: Explicit commands like "@Alex summarize"
2. Proactive triggers: AI-initiated messages for conflicts, decisions, drift, etc.
"""

import asyncio
import re
from datetime import datetime, timedelta
from typing import List, Optional, Callable, Set
import structlog

from src.config import config, ConflictSensitivity
from src.context.models import (
    MeetingContext, TranscriptChunk, Trigger, TriggerType, CommandType,
    Concern, Decision, ConcernSeverity, DecisionStatus
)

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
            TriggerType.DECISION_FORMING: 30,
            TriggerType.CONFLICT_DETECTED: 20,
            TriggerType.TOPIC_DRIFT: 60,
            TriggerType.REPEATED_CONFUSION: 45,
            TriggerType.COMPLEX_CONCEPT: 30,
            TriggerType.RISK_OVERLOOKED: 30,
            TriggerType.CIRCULAR_DISCUSSION: 60,
        }
        
        # Bot name for command detection
        self.bot_name = config.bot.name.lower()
        
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
            r"(?:what do you mean|can you explain|i'm confused)",
            r"(?:sorry|wait)[,]? (?:what|can you repeat)",
            r"(?:i'm lost|lost me there)",
        ]
        
        # Topic drift indicators
        self._drift_patterns = [
            r"(?:anyway|anyhow|back to|getting off track)",
            r"(?:that reminds me|speaking of which|on a different note)",
            r"(?:we're getting sidetracked|let's refocus)",
        ]
        
        # Command patterns (reactive triggers)
        self._command_patterns = {
            CommandType.SUMMARIZE: [
                r"summarize",
                r"summary",
                r"what (?:have we|did we) (?:discussed|talked about|covered)",
                r"where are we",
                r"catch me up",
            ],
            CommandType.RISKS: [
                r"(?:what are the |any )risks",
                r"concerns",
                r"(?:what could|what might) go wrong",
                r"issues",
            ],
            CommandType.EXPLAIN: [
                r"explain",
                r"what (?:is|does|are) (.+)",
                r"define",
                r"meaning of",
            ],
            CommandType.DECISION: [
                r"(?:what|which) decision",
                r"what (?:are we|did we) (?:decide|choosing|leaning)",
                r"current (?:decision|direction)",
            ],
            CommandType.ACTION_ITEMS: [
                r"action items",
                r"(?:what are the |list )tasks",
                r"to-?dos?",
                r"next steps",
            ],
        }
    
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
        
        # Filter out triggers on cooldown
        triggers = self._filter_cooldowns(triggers)
        
        # Update cooldown timestamps
        for trigger in triggers:
            self._last_trigger_by_type[trigger.type] = datetime.now()
            self._recent_triggers.append(trigger)
        
        return triggers
    
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
        
        for pattern, confidence in self._decision_patterns:
            match = re.search(pattern, text_lower)
            if match:
                decision_text = match.group(1) if match.groups() else match.group(0)
                
                # Look for conflicting concerns
                conflicts = self._find_conflicting_concerns(
                    decision_text, 
                    context.unaddressed_concerns
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
        """Check if the conversation has drifted from the main topic."""
        text_lower = recent_text.lower()
        
        # Look for explicit drift indicators
        for pattern in self._drift_patterns:
            if re.search(pattern, text_lower):
                trigger = Trigger.create_proactive(
                    TriggerType.TOPIC_DRIFT,
                    confidence=0.6,
                    reason="Conversation appears to be drifting from the main topic"
                )
                trigger.relevant_transcript = recent_text[-300:]
                return trigger
        
        return None
    
    def _check_for_repeated_confusion(
        self, 
        context: MeetingContext, 
        recent_text: str
    ) -> Optional[Trigger]:
        """Check if the same confusion is appearing multiple times."""
        text_lower = recent_text.lower()
        
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
            trigger.relevant_transcript = recent_text[-400:]
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
