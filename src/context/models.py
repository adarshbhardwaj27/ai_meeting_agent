"""
Data models for the Real-Time AI Meeting Assistant.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Set, Dict, Any
import uuid


# =============================================================================
# Transcript Models
# =============================================================================

@dataclass
class TranscriptChunk:
    """A single chunk of transcribed speech."""
    text: str
    timestamp: datetime
    speaker: Optional[str] = None
    is_final: bool = True
    confidence: float = 1.0
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __post_init__(self):
        self.text = self.text.strip()


@dataclass
class TranscriptSegment:
    """A complete sentence or thought from the transcript."""
    text: str
    speaker: Optional[str]
    start_time: datetime
    end_time: datetime
    chunks: List[TranscriptChunk] = field(default_factory=list)
    
    @property
    def duration(self) -> timedelta:
        return self.end_time - self.start_time


# =============================================================================
# Context State Models
# =============================================================================

class ConcernSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DecisionStatus(Enum):
    PROPOSED = "proposed"
    DISCUSSED = "discussed"
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


@dataclass
class Concern:
    """A concern or risk raised during the meeting."""
    id: str
    description: str
    raised_by: Optional[str]
    raised_at: datetime
    severity: ConcernSeverity = ConcernSeverity.MEDIUM
    related_topics: List[str] = field(default_factory=list)
    addressed: bool = False
    addressed_at: Optional[datetime] = None
    
    @classmethod
    def create(cls, description: str, raised_by: Optional[str] = None, 
               severity: ConcernSeverity = ConcernSeverity.MEDIUM) -> "Concern":
        return cls(
            id=str(uuid.uuid4()),
            description=description,
            raised_by=raised_by,
            raised_at=datetime.now(),
            severity=severity
        )


@dataclass
class Decision:
    """A decision made or proposed during the meeting."""
    id: str
    description: str
    proposed_by: Optional[str]
    proposed_at: datetime
    status: DecisionStatus = DecisionStatus.PROPOSED
    supporters: List[str] = field(default_factory=list)
    objectors: List[str] = field(default_factory=list)
    related_concerns: List[str] = field(default_factory=list)  # Concern IDs
    confirmed_at: Optional[datetime] = None
    
    @classmethod
    def create(cls, description: str, proposed_by: Optional[str] = None) -> "Decision":
        return cls(
            id=str(uuid.uuid4()),
            description=description,
            proposed_by=proposed_by,
            proposed_at=datetime.now()
        )
    
    def confirm(self):
        self.status = DecisionStatus.CONFIRMED
        self.confirmed_at = datetime.now()


@dataclass
class Topic:
    """A topic being discussed in the meeting."""
    name: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    key_points: List[str] = field(default_factory=list)
    
    @property
    def is_active(self) -> bool:
        return self.ended_at is None
    
    @property
    def duration(self) -> timedelta:
        end = self.ended_at or datetime.now()
        return end - self.started_at


# =============================================================================
# Meeting Context (The Core State)
# =============================================================================

@dataclass
class MeetingContext:
    """
    The complete state of the meeting as understood by the AI.
    This is the primary data structure passed to the trigger engine and LLM.
    """
    # Meeting metadata
    meeting_id: str
    meeting_started_at: datetime
    platform: str  # "google_meet", "teams", "zoom"
    
    # Raw transcript - sliding window (configurable, default ~10 min)
    recent_transcript: List[TranscriptChunk] = field(default_factory=list)
    
    # Compressed summary of earlier discussion (beyond sliding window)
    historical_summary: str = ""
    
    # Tracked state
    current_topic: Optional[Topic] = None
    topics_discussed: List[Topic] = field(default_factory=list)
    concerns_raised: List[Concern] = field(default_factory=list)
    decisions_confirmed: List[Decision] = field(default_factory=list)
    decisions_pending: List[Decision] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    
    # Participants
    participants: Set[str] = field(default_factory=set)
    active_speaker: Optional[str] = None
    
    # Metadata
    last_updated: datetime = field(default_factory=datetime.now)
    tokens_in_window: int = 0
    
    @property
    def meeting_duration(self) -> timedelta:
        return datetime.now() - self.meeting_started_at
    
    @property
    def unaddressed_concerns(self) -> List[Concern]:
        return [c for c in self.concerns_raised if not c.addressed]
    
    @property
    def high_priority_concerns(self) -> List[Concern]:
        return [c for c in self.concerns_raised 
                if c.severity in (ConcernSeverity.HIGH, ConcernSeverity.CRITICAL)
                and not c.addressed]
    
    def get_recent_text(self, minutes: int = 3) -> str:
        """Get transcript text from the last N minutes."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        chunks = [
            c for c in self.recent_transcript
            if c.timestamp >= cutoff and c.is_final
        ]
        return " ".join(c.text for c in chunks)
    
    def get_context_for_llm(self) -> Dict[str, Any]:
        """Format context for LLM consumption."""
        return {
            "meeting_duration_minutes": int(self.meeting_duration.total_seconds() / 60),
            "current_topic": self.current_topic.name if self.current_topic else "Unknown",
            "topics_discussed": [t.name for t in self.topics_discussed],
            "unaddressed_concerns": [
                {"description": c.description, "raised_by": c.raised_by, "severity": c.severity.value}
                for c in self.unaddressed_concerns
            ],
            "pending_decisions": [
                {"description": d.description, "status": d.status.value}
                for d in self.decisions_pending
            ],
            "confirmed_decisions": [d.description for d in self.decisions_confirmed],
            "open_questions": self.open_questions[:5],
            "action_items": self.action_items[:5],
            "participants": list(self.participants),
            "historical_summary": self.historical_summary,
        }


# =============================================================================
# Trigger Models
# =============================================================================

class TriggerType(Enum):
    # Reactive triggers (user commands)
    EXPLICIT_COMMAND = "explicit_command"
    
    # Proactive triggers (AI-initiated)
    DECISION_FORMING = "decision_forming"
    CONFLICT_DETECTED = "conflict_detected"
    TOPIC_DRIFT = "topic_drift"
    REPEATED_CONFUSION = "repeated_confusion"
    COMPLEX_CONCEPT = "complex_concept"
    RISK_OVERLOOKED = "risk_overlooked"
    CIRCULAR_DISCUSSION = "circular_discussion"

    # Voice-addressed question (someone said the bot's name out loud)
    VOICE_ADDRESS = "voice_address"

    # Intelligence-layer signals (Tier 1 → Tier 2)
    CONTRADICTION = "contradiction"
    MEMORY_GAP = "memory_gap"
    FACTUAL_INCONSISTENCY = "factual_inconsistency"
    COLLABORATION_OPPORTUNITY = "collaboration_opportunity"


class CommandType(Enum):
    SUMMARIZE = "summarize"
    RISKS = "risks"
    EXPLAIN = "explain"
    DECISION = "decision"
    ACTION_ITEMS = "action_items"
    CUSTOM = "custom"


@dataclass
class Trigger:
    """A detected trigger that may result in an AI message."""
    id: str
    type: TriggerType
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    reason: str  # Human-readable explanation
    
    # For command triggers
    command_type: Optional[CommandType] = None
    command_args: Optional[str] = None
    
    # For conflict triggers
    conflicting_concern: Optional[Concern] = None
    conflicting_decision: Optional[Decision] = None
    
    # Related context
    relevant_transcript: str = ""

    # Signal evidence (populated by Tier-1 intelligence layer)
    signal_evidence_a: Optional[str] = None
    signal_evidence_b: Optional[str] = None
    signal_speaker_a: Optional[str] = None
    signal_speaker_b: Optional[str] = None
    
    @classmethod
    def create_command(cls, command_type: CommandType, 
                       args: Optional[str] = None) -> "Trigger":
        return cls(
            id=str(uuid.uuid4()),
            type=TriggerType.EXPLICIT_COMMAND,
            confidence=1.0,
            timestamp=datetime.now(),
            reason=f"User requested: {command_type.value}",
            command_type=command_type,
            command_args=args
        )
    
    @classmethod
    def create_proactive(cls, trigger_type: TriggerType, 
                         confidence: float, reason: str) -> "Trigger":
        return cls(
            id=str(uuid.uuid4()),
            type=trigger_type,
            confidence=confidence,
            timestamp=datetime.now(),
            reason=reason
        )


# =============================================================================
# Chat Models
# =============================================================================

class MessagePriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ChatMessage:
    """A message in the meeting chat."""
    id: str
    text: str
    sender: str
    timestamp: datetime
    is_from_bot: bool = False
    mentions_bot: bool = False
    
    @classmethod
    def from_meeting(cls, text: str, sender: str, 
                     bot_name: str = "Alex") -> "ChatMessage":
        return cls(
            id=str(uuid.uuid4()),
            text=text,
            sender=sender,
            timestamp=datetime.now(),
            is_from_bot=False,
            mentions_bot=f"@{bot_name}" in text.lower() or 
                        f"@{bot_name.lower()}" in text.lower()
        )


@dataclass
class AIResponse:
    """A response generated by the AI to be sent to chat."""
    id: str
    text: str
    trigger: Trigger
    priority: MessagePriority
    created_at: datetime
    sent: bool = False
    sent_at: Optional[datetime] = None
    
    @classmethod
    def create(cls, text: str, trigger: Trigger, 
               priority: MessagePriority = MessagePriority.MEDIUM) -> "AIResponse":
        return cls(
            id=str(uuid.uuid4()),
            text=text,
            trigger=trigger,
            priority=priority,
            created_at=datetime.now()
        )
    
    def mark_sent(self):
        self.sent = True
        self.sent_at = datetime.now()


# =============================================================================
# Event Models (for Event Bus)
# =============================================================================

class EventType(Enum):
    # Input events
    AUDIO_CHUNK = "audio_chunk"
    TRANSCRIPT_CHUNK = "transcript_chunk"
    CHAT_MESSAGE_RECEIVED = "chat_message_received"
    
    # Processing events
    CONTEXT_UPDATED = "context_updated"
    TRIGGER_FIRED = "trigger_fired"
    
    # Output events
    RESPONSE_READY = "response_ready"
    CHAT_MESSAGE_SENT = "chat_message_sent"
    
    # System events
    BOT_JOINED = "bot_joined"
    BOT_LEFT = "bot_left"
    ERROR = "error"


@dataclass
class Event:
    """Base event for the event bus."""
    id: str
    type: EventType
    timestamp: datetime
    payload: Any
    
    @classmethod
    def create(cls, event_type: EventType, payload: Any) -> "Event":
        return cls(
            id=str(uuid.uuid4()),
            type=event_type,
            timestamp=datetime.now(),
            payload=payload
        )


# =============================================================================
# Conversation Intelligence Models
# =============================================================================

class SignalType(Enum):
    """Types of conversational signals the intelligence layer can detect."""
    CONTRADICTION = "contradiction"           # X said A, now Y says not-A
    MEMORY_GAP = "memory_gap"                 # Team forgot / can't recall earlier point
    FACTUAL_INCONSISTENCY = "factual_inconsistency"  # Stated fact conflicts with prior data
    COLLABORATION_OPPORTUNITY = "collaboration_opportunity"  # Bot has useful context to add


@dataclass
class AssertionRecord:
    """
    A factual claim extracted from the transcript.

    The assertion ledger tracks what each participant has stated as fact
    so contradictions and memory gaps can be detected without an LLM call.
    """
    id: str
    text: str                       # The normalized claim
    speaker: Optional[str]
    timestamp: datetime
    keywords: Set[str] = field(default_factory=set)   # Content words for fast overlap
    negated: bool = False           # "we should NOT use X"
    superseded_by: Optional[str] = None  # ID of a later assertion that updates this one

    @classmethod
    def create(
        cls,
        text: str,
        speaker: Optional[str] = None,
        keywords: Optional[Set[str]] = None,
        negated: bool = False,
    ) -> "AssertionRecord":
        return cls(
            id=str(uuid.uuid4()),
            text=text,
            speaker=speaker,
            timestamp=datetime.now(),
            keywords=keywords or set(),
            negated=negated,
        )


@dataclass
class SignalCandidate:
    """
    A lightweight candidate produced by Tier-1 local scoring.

    Only candidates above the confidence threshold are escalated to the LLM
    (Tier 2) for response generation.
    """
    signal_type: SignalType
    confidence: float                     # 0.0 – 1.0
    reason: str                           # Human-readable explanation
    evidence_a: Optional[str] = None      # First piece of evidence (e.g. earlier claim)
    evidence_b: Optional[str] = None      # Second piece (e.g. current claim)
    speaker_a: Optional[str] = None
    speaker_b: Optional[str] = None
    related_assertion_ids: List[str] = field(default_factory=list)
    relevant_transcript: str = ""
