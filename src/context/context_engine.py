"""
Live Context Engine for the Real-Time AI Meeting Assistant.

This engine maintains a continuous understanding of the meeting by:
1. Keeping a sliding window of recent transcript (~10 minutes)
2. Compressing older content into summaries
3. Tracking state: topics, concerns, decisions, questions
4. Updating context every few seconds
"""

import asyncio
from collections import deque
from datetime import datetime, timedelta
from typing import List, Optional, Callable, Deque
import re
import structlog

from src.config import config
from src.context.models import (
    MeetingContext, TranscriptChunk, Topic, Concern, Decision,
    ConcernSeverity, DecisionStatus, Event, EventType
)

logger = structlog.get_logger()


class ContextEngine:
    """
    Maintains live understanding of the meeting conversation.
    
    The context engine is the "memory" of the AI assistant, continuously
    processing transcript chunks and maintaining state about what's being
    discussed, what concerns have been raised, and what decisions are pending.
    """
    
    def __init__(
        self,
        meeting_id: str,
        platform: str,
        window_minutes: int = None,
        update_interval: float = None,
        on_context_updated: Optional[Callable[[MeetingContext], None]] = None
    ):
        self.window_minutes = window_minutes or config.context.window_minutes
        self.update_interval = update_interval or config.context.update_interval_seconds
        self.on_context_updated = on_context_updated
        
        # Initialize context
        self.context = MeetingContext(
            meeting_id=meeting_id,
            meeting_started_at=datetime.now(),
            platform=platform
        )
        
        # Processing buffers
        self._chunk_buffer: Deque[TranscriptChunk] = deque()
        self._pending_analysis = False
        
        # Background task handle
        self._update_task: Optional[asyncio.Task] = None
        self._running = False
        
        # State detection patterns
        self._concern_patterns = [
            r"(?:i'm |i am )?(?:worried|concerned|afraid) (?:about |that )?(.+)",
            r"(?:the |a )?(?:risk|problem|issue) (?:is |with )?(.+)",
            r"(?:what if|what about) (.+)\?",
            r"(?:we need to consider|don't forget about) (.+)",
            r"(.+) (?:might|could|may) (?:be a problem|cause issues|fail)",
        ]
        
        self._decision_patterns = [
            r"(?:let's |we should |we'll |we will )(?:go with|use|do|implement) (.+)",
            r"(?:i think we should|i vote for|i suggest we) (.+)",
            r"(?:the decision is|we've decided|it's decided)[:\s]+(.+)",
            r"(?:so we're going with|agreed[,\s]+we'll) (.+)",
        ]
        
        self._question_patterns = [
            r"(?:what|how|why|when|where|who|can|could|should|would) .+\?",
            r"(?:does anyone|has anyone|is there) .+\?",
        ]
    
    async def start(self):
        """Start the context engine background processing."""
        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("context_engine_started", meeting_id=self.context.meeting_id)
    
    async def stop(self):
        """Stop the context engine."""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        logger.info("context_engine_stopped", meeting_id=self.context.meeting_id)
    
    async def add_transcript_chunk(self, chunk: TranscriptChunk):
        """
        Add a new transcript chunk to be processed.
        This is called frequently as speech is transcribed.
        """
        self._chunk_buffer.append(chunk)
        
        # Add to recent transcript immediately
        self.context.recent_transcript.append(chunk)
        
        # Track participant
        if chunk.speaker:
            self.context.participants.add(chunk.speaker)
            self.context.active_speaker = chunk.speaker
        
        # Mark that we have pending analysis
        self._pending_analysis = True
        
        logger.debug(
            "transcript_chunk_added",
            text=chunk.text[:50],
            speaker=chunk.speaker
        )
    
    async def add_chat_message(self, sender: str, text: str):
        """Process an incoming chat message."""
        self.context.participants.add(sender)
        
        # Check if it's a command for our bot
        # (This is also handled by trigger engine, but we track it here too)
        bot_name = config.bot.name.lower()
        if f"@{bot_name}" in text.lower():
            logger.info("bot_mentioned_in_chat", sender=sender, text=text)
    
    async def _update_loop(self):
        """Background loop that periodically analyzes context."""
        while self._running:
            try:
                await asyncio.sleep(self.update_interval)
                
                if self._pending_analysis:
                    await self._analyze_context()
                    self._pending_analysis = False
                
                # Prune old transcript chunks
                await self._prune_sliding_window()
                
                # Update timestamp
                self.context.last_updated = datetime.now()
                
                # Notify listeners
                if self.on_context_updated:
                    self.on_context_updated(self.context)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("context_update_error", error=str(e))
    
    async def _analyze_context(self):
        """
        Analyze recent transcript for state changes.
        This runs every few seconds to keep context up-to-date.
        """
        # Get recent unprocessed text
        recent_text = self._get_recent_unprocessed_text()
        if not recent_text:
            return
        
        # Detect concerns
        await self._detect_concerns(recent_text)
        
        # Detect decisions
        await self._detect_decisions(recent_text)
        
        # Detect topic changes
        await self._detect_topic_changes(recent_text)
        
        # Detect open questions
        await self._detect_questions(recent_text)
        
        logger.debug(
            "context_analyzed",
            concerns=len(self.context.concerns_raised),
            pending_decisions=len(self.context.decisions_pending),
            confirmed_decisions=len(self.context.decisions_confirmed)
        )
    
    def _get_recent_unprocessed_text(self) -> str:
        """Get text from chunks that haven't been fully analyzed."""
        # For now, just get last few chunks
        # In production, track which chunks have been processed
        recent_chunks = list(self._chunk_buffer)[-20:]
        return " ".join(c.text for c in recent_chunks if c.is_final)
    
    async def _detect_concerns(self, text: str):
        """Detect concerns or risks mentioned in the text."""
        text_lower = text.lower()
        
        for pattern in self._concern_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                concern_text = match.group(1).strip() if match.groups() else match.group(0)
                
                # Avoid duplicates
                if any(concern_text in c.description.lower() 
                       for c in self.context.concerns_raised):
                    continue
                
                # Determine severity based on keywords
                severity = self._assess_concern_severity(concern_text)
                
                concern = Concern.create(
                    description=concern_text.capitalize(),
                    raised_by=self.context.active_speaker,
                    severity=severity
                )
                
                self.context.concerns_raised.append(concern)
                logger.info(
                    "concern_detected",
                    description=concern.description,
                    severity=severity.value
                )
    
    def _assess_concern_severity(self, text: str) -> ConcernSeverity:
        """Assess the severity of a concern based on language."""
        text_lower = text.lower()
        
        critical_words = ["critical", "blocker", "must", "cannot", "impossible", "fail"]
        high_words = ["significant", "major", "serious", "important", "risk"]
        medium_words = ["concern", "issue", "problem", "might", "could"]
        
        if any(word in text_lower for word in critical_words):
            return ConcernSeverity.CRITICAL
        elif any(word in text_lower for word in high_words):
            return ConcernSeverity.HIGH
        elif any(word in text_lower for word in medium_words):
            return ConcernSeverity.MEDIUM
        else:
            return ConcernSeverity.LOW
    
    async def _detect_decisions(self, text: str):
        """Detect decisions being made or proposed."""
        text_lower = text.lower()
        
        for pattern in self._decision_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                decision_text = match.group(1).strip() if match.groups() else match.group(0)
                
                # Check if this updates an existing decision
                existing = self._find_similar_decision(decision_text)
                if existing:
                    # Escalate status
                    if existing.status == DecisionStatus.PROPOSED:
                        existing.status = DecisionStatus.DISCUSSED
                    elif existing.status == DecisionStatus.DISCUSSED:
                        existing.status = DecisionStatus.TENTATIVE
                    continue
                
                # Create new decision
                decision = Decision.create(
                    description=decision_text.capitalize(),
                    proposed_by=self.context.active_speaker
                )
                
                # Check for related concerns
                related_concerns = self._find_related_concerns(decision_text)
                decision.related_concerns = [c.id for c in related_concerns]
                
                self.context.decisions_pending.append(decision)
                logger.info(
                    "decision_detected",
                    description=decision.description,
                    related_concerns=len(related_concerns)
                )
    
    def _find_similar_decision(self, text: str) -> Optional[Decision]:
        """Find an existing decision similar to the given text."""
        text_lower = text.lower()
        text_words = set(text_lower.split())
        
        for decision in self.context.decisions_pending:
            decision_words = set(decision.description.lower().split())
            # Check for significant word overlap
            overlap = len(text_words & decision_words)
            if overlap >= min(3, len(text_words)):
                return decision
        
        return None
    
    def _find_related_concerns(self, decision_text: str) -> List[Concern]:
        """Find concerns that might be related to a decision."""
        related = []
        decision_lower = decision_text.lower()
        decision_words = set(decision_lower.split())
        
        for concern in self.context.concerns_raised:
            concern_words = set(concern.description.lower().split())
            # Check for word overlap
            overlap = decision_words & concern_words
            if len(overlap) >= 1:
                related.append(concern)
        
        return related
    
    async def _detect_topic_changes(self, text: str):
        """Detect when the conversation topic changes."""
        # Simple heuristic: look for topic transition phrases
        topic_phrases = [
            r"(?:moving on to|let's talk about|next topic|now for) (.+)",
            r"(?:regarding|about|concerning) (.+)",
            r"(?:switching to|let's discuss) (.+)",
        ]
        
        text_lower = text.lower()
        for pattern in topic_phrases:
            match = re.search(pattern, text_lower)
            if match:
                new_topic_name = match.group(1).strip()
                
                # End current topic
                if self.context.current_topic:
                    self.context.current_topic.ended_at = datetime.now()
                    self.context.topics_discussed.append(self.context.current_topic)
                
                # Start new topic
                self.context.current_topic = Topic(
                    name=new_topic_name.capitalize(),
                    started_at=datetime.now()
                )
                
                logger.info("topic_changed", new_topic=new_topic_name)
                break
    
    async def _detect_questions(self, text: str):
        """Detect open questions in the conversation."""
        for pattern in self._question_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                question = match.group(0).strip()
                
                # Avoid duplicates
                if question.lower() not in [q.lower() for q in self.context.open_questions]:
                    self.context.open_questions.append(question)
                    
                    # Keep only recent questions (max 10)
                    if len(self.context.open_questions) > 10:
                        self.context.open_questions.pop(0)
    
    async def _prune_sliding_window(self):
        """Remove transcript chunks outside the sliding window."""
        cutoff = datetime.now() - timedelta(minutes=self.window_minutes)
        
        # Separate old chunks for compression
        old_chunks = [c for c in self.context.recent_transcript 
                      if c.timestamp < cutoff]
        
        # Keep recent chunks
        self.context.recent_transcript = [
            c for c in self.context.recent_transcript 
            if c.timestamp >= cutoff
        ]
        
        # Compress old chunks into summary (if significant)
        if old_chunks and len(old_chunks) >= 10:
            await self._compress_to_summary(old_chunks)
    
    async def _compress_to_summary(self, chunks: List[TranscriptChunk]):
        """
        Compress old transcript chunks into the historical summary.
        
        In production, this would use an LLM to create a concise summary.
        For the MVP, we use a simple text concatenation with truncation.
        """
        old_text = " ".join(c.text for c in chunks)
        
        # Simple compression: keep first and last parts
        if len(old_text) > 500:
            compressed = old_text[:250] + " ... " + old_text[-250:]
        else:
            compressed = old_text
        
        # Append to historical summary
        if self.context.historical_summary:
            self.context.historical_summary += f"\n\n[Earlier]: {compressed}"
        else:
            self.context.historical_summary = f"[Earlier discussion]: {compressed}"
        
        # Keep summary from growing too large
        max_summary_length = 2000
        if len(self.context.historical_summary) > max_summary_length:
            self.context.historical_summary = (
                self.context.historical_summary[-max_summary_length:]
            )
        
        logger.debug(
            "summary_compressed",
            chunks_compressed=len(chunks),
            summary_length=len(self.context.historical_summary)
        )
    
    def get_context_snapshot(self) -> MeetingContext:
        """Get a snapshot of the current context."""
        return self.context
    
    def get_formatted_context(self) -> str:
        """Get a formatted string representation of the context for LLM."""
        ctx = self.context
        
        parts = [
            f"Meeting Duration: {int(ctx.meeting_duration.total_seconds() / 60)} minutes",
            f"Participants: {', '.join(ctx.participants) or 'Unknown'}",
        ]
        
        if ctx.current_topic:
            parts.append(f"Current Topic: {ctx.current_topic.name}")
        
        if ctx.topics_discussed:
            parts.append(f"Topics Discussed: {', '.join(t.name for t in ctx.topics_discussed)}")
        
        if ctx.unaddressed_concerns:
            concerns_text = "; ".join(
                f"{c.description} ({c.severity.value})" 
                for c in ctx.unaddressed_concerns[:5]
            )
            parts.append(f"Unaddressed Concerns: {concerns_text}")
        
        if ctx.decisions_pending:
            decisions_text = "; ".join(
                f"{d.description} ({d.status.value})" 
                for d in ctx.decisions_pending[:5]
            )
            parts.append(f"Pending Decisions: {decisions_text}")
        
        if ctx.decisions_confirmed:
            confirmed_text = "; ".join(d.description for d in ctx.decisions_confirmed[:5])
            parts.append(f"Confirmed Decisions: {confirmed_text}")
        
        if ctx.historical_summary:
            parts.append(f"\nHistorical Summary:\n{ctx.historical_summary}")
        
        return "\n".join(parts)
