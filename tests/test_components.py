"""
Test suite for the Real-Time AI Meeting Assistant.

Includes:
- Unit tests for individual components
- Integration tests for the pipeline
- End-to-end simulation tests
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock, patch

# Import components
from src.context.models import (
    TranscriptChunk, MeetingContext, Concern, Decision,
    Trigger, TriggerType, CommandType, ConcernSeverity, DecisionStatus, Topic,
    SignalType, SignalCandidate, AssertionRecord, AIResponse, MessagePriority,
)
from src.context.context_engine import ContextEngine
from src.triggers.trigger_engine import TriggerEngine
from src.triggers.conversation_intelligence import ConversationIntelligence
from src.reasoning.llm_service import LLMService
from src.reasoning.speech_service import SpeechService
from src.chat.chat_manager import ChatManager
from src.config import LLMProvider


# =============================================================================
# Context Engine Tests
# =============================================================================

class TestContextEngine:
    """Tests for the ContextEngine."""
    
    @pytest.fixture
    def context_engine(self):
        """Create a context engine for testing."""
        engine = ContextEngine(
            meeting_id="test_meeting",
            platform="test"
        )
        return engine
    
    @pytest.mark.asyncio
    async def test_add_transcript_chunk(self, context_engine):
        """Test adding a transcript chunk."""
        chunk = TranscriptChunk(
            text="Hello, let's start the meeting",
            timestamp=datetime.now(),
            speaker="John"
        )
        
        await context_engine.add_transcript_chunk(chunk)
        
        assert len(context_engine.context.recent_transcript) == 1
        assert "John" in context_engine.context.participants
    
    @pytest.mark.asyncio
    async def test_detect_concerns(self, context_engine):
        """Test that concerns are detected from transcript."""
        # Start the engine briefly
        await context_engine.start()
        
        # Add transcript with concern language
        chunk = TranscriptChunk(
            text="I'm worried about the latency issues we might face",
            timestamp=datetime.now(),
            speaker="Sarah"
        )
        await context_engine.add_transcript_chunk(chunk)
        
        # Wait for analysis
        await asyncio.sleep(context_engine.update_interval + 0.5)
        
        await context_engine.stop()
        
        # Check that concern was detected
        assert len(context_engine.context.concerns_raised) > 0
        assert "latency" in context_engine.context.concerns_raised[0].description.lower()
    
    @pytest.mark.asyncio
    async def test_detect_decisions(self, context_engine):
        """Test that decisions are detected from transcript."""
        await context_engine.start()
        
        chunk = TranscriptChunk(
            text="Let's go with Kafka for the message queue",
            timestamp=datetime.now(),
            speaker="John"
        )
        await context_engine.add_transcript_chunk(chunk)
        
        await asyncio.sleep(context_engine.update_interval + 0.5)
        await context_engine.stop()
        
        assert len(context_engine.context.decisions_pending) > 0
    
    @pytest.mark.asyncio
    async def test_sliding_window(self, context_engine):
        """Test that old transcripts are pruned."""
        context_engine.window_minutes = 0.01  # Very short window for testing
        
        # Add old chunk
        old_chunk = TranscriptChunk(
            text="This is old",
            timestamp=datetime.now() - timedelta(minutes=1),
            speaker="John"
        )
        await context_engine.add_transcript_chunk(old_chunk)
        
        # Add recent chunk
        new_chunk = TranscriptChunk(
            text="This is new",
            timestamp=datetime.now(),
            speaker="Sarah"
        )
        await context_engine.add_transcript_chunk(new_chunk)
        
        # Manually trigger pruning
        await context_engine._prune_sliding_window()
        
        # Old chunk should be removed
        texts = [c.text for c in context_engine.context.recent_transcript]
        assert "This is new" in texts

    @pytest.mark.asyncio
    async def test_compress_to_summary_uses_llm_summarizer_when_threshold_met(self):
        """Older transcript batches should use the injected summarizer when large enough."""
        calls = []

        async def summarize(existing_summary, chunks):
            calls.append((existing_summary, [chunk.text for chunk in chunks]))
            return (
                "Summary: The team compared cloud provider options for the new system.\n"
                "Decisions: None\n"
                "Open Questions: Which provider to choose.\n"
                "Risks: Budget constraints.\n"
                "Action Items: None"
            )

        engine = ContextEngine(
            meeting_id="summary_test",
            platform="test",
            summary_update_threshold=20,
            context_summarizer=summarize,
        )

        chunks = [
            TranscriptChunk(
                text="We should compare AWS, Azure, and GCP for the new architecture.",
                timestamp=datetime.now() - timedelta(minutes=20),
                speaker="John",
            ),
            TranscriptChunk(
                text="Our budget is constrained, so cost matters a lot.",
                timestamp=datetime.now() - timedelta(minutes=19),
                speaker="Sarah",
            ),
            TranscriptChunk(
                text="We have not decided which provider to use yet.",
                timestamp=datetime.now() - timedelta(minutes=18),
                speaker="Mike",
            ),
            TranscriptChunk(
                text="We need to keep the summary clean for later prompts.",
                timestamp=datetime.now() - timedelta(minutes=17),
                speaker="John",
            ),
        ]

        await engine._compress_to_summary(chunks)

        assert calls
        assert engine.context.historical_summary.startswith("Summary:")
        assert "Budget constraints" in engine.context.historical_summary
    
    def test_get_recent_text(self, context_engine):
        """Test getting recent transcript text."""
        # Add some chunks
        for i in range(5):
            chunk = TranscriptChunk(
                text=f"Message {i}",
                timestamp=datetime.now() - timedelta(minutes=i),
                speaker="John"
            )
            context_engine.context.recent_transcript.append(chunk)
        
        # Get last 2 minutes
        recent = context_engine.context.get_recent_text(minutes=2)
        
        assert "Message 0" in recent
        assert "Message 1" in recent

    @pytest.mark.asyncio
    async def test_interim_chunks_not_persisted(self, context_engine):
        """Interim transcript chunks should not pollute persistent context."""
        interim = TranscriptChunk(
            text="Hello the",
            timestamp=datetime.now(),
            speaker="John",
            is_final=False,
        )
        final = TranscriptChunk(
            text="Hello there",
            timestamp=datetime.now(),
            speaker="John",
            is_final=True,
        )

        await context_engine.add_transcript_chunk(interim)
        await context_engine.add_transcript_chunk(final)

        assert len(context_engine.context.recent_transcript) == 1
        assert context_engine.context.recent_transcript[0].text == "Hello there"

    @pytest.mark.asyncio
    async def test_duplicate_final_chunks_ignored(self, context_engine):
        """Duplicate final transcript chunks should only be stored once."""
        now = datetime.now()
        chunk1 = TranscriptChunk(
            text="Let's review the proposal",
            timestamp=now,
            speaker="Sarah",
        )
        chunk2 = TranscriptChunk(
            text="Let's review the proposal",
            timestamp=now + timedelta(seconds=1),
            speaker="Sarah",
        )

        await context_engine.add_transcript_chunk(chunk1)
        await context_engine.add_transcript_chunk(chunk2)

        assert len(context_engine.context.recent_transcript) == 1

    @pytest.mark.asyncio
    async def test_detect_action_items(self, context_engine):
        """Action items should be extracted into structured context."""
        context_engine.update_interval = 0.1
        await context_engine.start()

        try:
            chunk = TranscriptChunk(
                text="John will send the design doc by Friday.",
                timestamp=datetime.now(),
                speaker="Sarah",
            )
            await context_engine.add_transcript_chunk(chunk)

            await asyncio.sleep(context_engine.update_interval + 0.3)

            assert any(
                "john: send the design doc by friday" in item.lower()
                for item in context_engine.context.action_items
            )
        finally:
            await context_engine.stop()

    def test_get_context_snapshot_returns_copy(self, context_engine):
        """Context snapshots should not expose the live mutable object."""
        context_engine.context.open_questions.append("What is the rollout plan?")
        context_engine.context.action_items.append("Sarah: confirm rollout date")

        snapshot = context_engine.get_context_snapshot()
        snapshot.open_questions.append("Mutated in test")
        snapshot.action_items.append("Injected into snapshot")

        assert "Mutated in test" not in context_engine.context.open_questions
        assert "Injected into snapshot" not in context_engine.context.action_items


# =============================================================================
# Trigger Engine Tests
# =============================================================================

class TestTriggerEngine:
    """Tests for the TriggerEngine."""
    
    @pytest.fixture
    def trigger_engine(self):
        """Create a trigger engine for testing."""
        return TriggerEngine()
    
    @pytest.fixture
    def mock_context(self):
        """Create a mock meeting context."""
        context = MeetingContext(
            meeting_id="test",
            meeting_started_at=datetime.now() - timedelta(minutes=10),
            platform="test"
        )
        
        # Add a concern
        context.concerns_raised.append(
            Concern.create(
                description="Latency issues with current architecture",
                raised_by="Sarah",
                severity=ConcernSeverity.HIGH
            )
        )
        
        return context
    
    @pytest.mark.asyncio
    async def test_detect_command(self, trigger_engine):
        """Test detection of explicit commands."""
        triggers = await trigger_engine.check_for_triggers(
            context=MeetingContext(
                meeting_id="test",
                meeting_started_at=datetime.now(),
                platform="test"
            ),
            chat_text="@Alex summarize what we discussed"
        )
        
        assert len(triggers) == 1
        assert triggers[0].type == TriggerType.EXPLICIT_COMMAND
        assert triggers[0].command_type == CommandType.SUMMARIZE
    
    @pytest.mark.asyncio
    async def test_detect_risks_command(self, trigger_engine):
        """Test detection of risks command."""
        triggers = await trigger_engine.check_for_triggers(
            context=MeetingContext(
                meeting_id="test",
                meeting_started_at=datetime.now(),
                platform="test"
            ),
            chat_text="@Alex what are the risks here?"
        )
        
        assert len(triggers) == 1
        assert triggers[0].command_type == CommandType.RISKS
    
    @pytest.mark.asyncio
    async def test_detect_decision_conflict(self, trigger_engine, mock_context):
        """Test detection of decision that conflicts with concern."""
        # Add transcript with decision that conflicts with latency concern
        chunk = TranscriptChunk(
            text="Let's go with Kafka for latency optimization",
            timestamp=datetime.now(),
            speaker="John"
        )
        mock_context.recent_transcript.append(chunk)
        
        triggers = await trigger_engine.check_for_triggers(
            context=mock_context,
            new_chunks=[chunk]
        )
        
        # Should detect either conflict or decision forming
        assert len(triggers) >= 1
        trigger_types = [t.type for t in triggers]
        assert TriggerType.CONFLICT_DETECTED in trigger_types or TriggerType.DECISION_FORMING in trigger_types
    
    @pytest.mark.asyncio
    async def test_cooldown(self, trigger_engine, mock_context):
        """Test that triggers respect cooldown."""
        chunk = TranscriptChunk(
            text="Let's go with the first option",
            timestamp=datetime.now(),
            speaker="John"
        )
        mock_context.recent_transcript.append(chunk)
        
        # First trigger should fire
        triggers1 = await trigger_engine.check_for_triggers(
            context=mock_context,
            new_chunks=[chunk]
        )
        
        # Second immediate trigger should be blocked by cooldown
        triggers2 = await trigger_engine.check_for_triggers(
            context=mock_context,
            new_chunks=[chunk]
        )
        
        # Commands have no cooldown, proactive triggers do
        proactive_triggers = [t for t in triggers2 if t.type != TriggerType.EXPLICIT_COMMAND]
        assert len(proactive_triggers) == 0 or len(triggers2) <= len(triggers1)
    
    @pytest.mark.asyncio
    async def test_no_trigger_for_normal_speech(self, trigger_engine, mock_context):
        """Test that normal speech doesn't trigger unnecessarily."""
        chunk = TranscriptChunk(
            text="The weather is nice today",
            timestamp=datetime.now(),
            speaker="John"
        )
        mock_context.recent_transcript.append(chunk)
        
        triggers = await trigger_engine.check_for_triggers(
            context=mock_context,
            new_chunks=[chunk]
        )
        
        # Should not trigger for random conversation
        assert len(triggers) == 0

    @pytest.mark.asyncio
    async def test_detect_topic_drift_from_context_shift(self, trigger_engine, mock_context):
        """Test that topic drift is detected from context misalignment, not cue phrases."""
        mock_context.current_topic = Topic(
            name="Real-time notification architecture",
            started_at=datetime.now() - timedelta(minutes=4)
        )

        chunk = TranscriptChunk(
            text=(
                "We should compare hotel options, travel approvals, venue budget, "
                "and the offsite catering plan before Friday."
            ),
            timestamp=datetime.now(),
            speaker="John"
        )
        mock_context.recent_transcript.append(chunk)

        triggers = await trigger_engine.check_for_triggers(
            context=mock_context,
            new_chunks=[chunk]
        )

        assert any(trigger.type == TriggerType.TOPIC_DRIFT for trigger in triggers)

    @pytest.mark.asyncio
    async def test_no_topic_drift_when_conversation_matches_active_focus(self, trigger_engine, mock_context):
        """Test that active-topic discussion stays below the drift threshold."""
        mock_context.current_topic = Topic(
            name="Real-time notification architecture",
            started_at=datetime.now() - timedelta(minutes=4)
        )

        chunk = TranscriptChunk(
            text=(
                "We still need the notification architecture to handle latency, "
                "scaling, and delivery guarantees for the real-time pipeline."
            ),
            timestamp=datetime.now(),
            speaker="Sarah"
        )
        mock_context.recent_transcript.append(chunk)

        triggers = await trigger_engine.check_for_triggers(
            context=mock_context,
            new_chunks=[chunk]
        )

        assert all(trigger.type != TriggerType.TOPIC_DRIFT for trigger in triggers)


# =============================================================================
# Model Tests
# =============================================================================

class TestModels:
    """Tests for data models."""
    
    def test_concern_creation(self):
        """Test creating a concern."""
        concern = Concern.create(
            description="Database might not scale",
            raised_by="Mike",
            severity=ConcernSeverity.HIGH
        )
        
        assert concern.description == "Database might not scale"
        assert concern.raised_by == "Mike"
        assert concern.severity == ConcernSeverity.HIGH
        assert concern.addressed is False
    
    def test_decision_creation(self):
        """Test creating a decision."""
        decision = Decision.create(
            description="Use PostgreSQL for the database",
            proposed_by="Sarah"
        )
        
        assert decision.status == DecisionStatus.PROPOSED
        assert decision.proposed_by == "Sarah"
    
    def test_decision_confirmation(self):
        """Test confirming a decision."""
        decision = Decision.create(description="Use Redis")
        decision.confirm()
        
        assert decision.status == DecisionStatus.CONFIRMED
        assert decision.confirmed_at is not None
    
    def test_trigger_command_creation(self):
        """Test creating a command trigger."""
        trigger = Trigger.create_command(
            CommandType.SUMMARIZE,
            args="last 5 minutes"
        )
        
        assert trigger.type == TriggerType.EXPLICIT_COMMAND
        assert trigger.command_type == CommandType.SUMMARIZE
        assert trigger.confidence == 1.0
    
    def test_trigger_proactive_creation(self):
        """Test creating a proactive trigger."""
        trigger = Trigger.create_proactive(
            TriggerType.CONFLICT_DETECTED,
            confidence=0.85,
            reason="Decision conflicts with earlier concern"
        )
        
        assert trigger.type == TriggerType.CONFLICT_DETECTED
        assert trigger.confidence == 0.85
    
    def test_meeting_context_properties(self):
        """Test MeetingContext computed properties."""
        context = MeetingContext(
            meeting_id="test",
            meeting_started_at=datetime.now() - timedelta(minutes=30),
            platform="test"
        )
        
        # Add concerns
        context.concerns_raised.append(
            Concern.create("Issue 1", severity=ConcernSeverity.LOW)
        )
        context.concerns_raised.append(
            Concern.create("Issue 2", severity=ConcernSeverity.HIGH)
        )
        addressed = Concern.create("Issue 3")
        addressed.addressed = True
        context.concerns_raised.append(addressed)
        
        assert len(context.unaddressed_concerns) == 2
        assert len(context.high_priority_concerns) == 1
        assert context.meeting_duration.total_seconds() >= 30 * 60


# =============================================================================
# Conversation Intelligence Tests
# =============================================================================

class TestConversationIntelligence:
    """Tests for the Tier-1 local signal scorer."""

    @pytest.fixture
    def intel(self):
        return ConversationIntelligence()

    @pytest.fixture
    def base_context(self):
        return MeetingContext(
            meeting_id="intel_test",
            meeting_started_at=datetime.now() - timedelta(minutes=10),
            platform="test",
        )

    # ── Contradiction detection ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_detect_contradiction_across_speakers(self, intel, base_context):
        """When speaker B negates what speaker A asserted, fire a contradiction."""
        earlier = [
            TranscriptChunk(
                text="Kafka can easily meet our 100ms SLA.",
                timestamp=datetime.now() - timedelta(minutes=3),
                speaker="Mike",
            )
        ]
        # Feed earlier statement into the ledger
        intel.score_signals(context=base_context, new_chunks=earlier)

        later = [
            TranscriptChunk(
                text="Kafka cannot meet our 100ms SLA, the batching adds too much latency.",
                timestamp=datetime.now(),
                speaker="Sarah",
            )
        ]
        signals = intel.score_signals(context=base_context, new_chunks=later)

        contradictions = [s for s in signals if s.signal_type == SignalType.CONTRADICTION]
        assert len(contradictions) >= 1
        assert contradictions[0].confidence >= 0.55
        assert contradictions[0].speaker_a == "Mike"
        assert contradictions[0].speaker_b == "Sarah"

    @pytest.mark.asyncio
    async def test_no_contradiction_for_same_speaker_self_correction(self, intel, base_context):
        """Same speaker updating their own view should NOT fire a contradiction."""
        earlier = [
            TranscriptChunk(
                text="I think we should use Redis for caching.",
                timestamp=datetime.now() - timedelta(minutes=2),
                speaker="John",
            )
        ]
        intel.score_signals(context=base_context, new_chunks=earlier)

        later = [
            TranscriptChunk(
                text="Actually we should not use Redis for caching, Memcached is better.",
                timestamp=datetime.now(),
                speaker="John",
            )
        ]
        signals = intel.score_signals(context=base_context, new_chunks=later)

        contradictions = [s for s in signals if s.signal_type == SignalType.CONTRADICTION]
        assert len(contradictions) == 0

    # ── Memory gap detection ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_detect_memory_gap_with_matching_context(self, intel, base_context):
        """When someone asks 'does anyone remember X' and the bot has X, fire memory gap."""
        # Seed the assertion ledger with an earlier statement
        earlier = [
            TranscriptChunk(
                text="Our database migration deadline is March 15th.",
                timestamp=datetime.now() - timedelta(minutes=5),
                speaker="Sarah",
            )
        ]
        intel.score_signals(context=base_context, new_chunks=earlier)

        later = [
            TranscriptChunk(
                text="Does anyone remember when the database migration deadline was?",
                timestamp=datetime.now(),
                speaker="John",
            )
        ]
        signals = intel.score_signals(context=base_context, new_chunks=later)

        memory_gaps = [s for s in signals if s.signal_type == SignalType.MEMORY_GAP]
        assert len(memory_gaps) >= 1
        assert "database" in memory_gaps[0].evidence_a.lower() or "migration" in memory_gaps[0].evidence_a.lower()

    @pytest.mark.asyncio
    async def test_detect_memory_gap_with_colloquial_recall_language(self, intel, base_context):
        """Colloquial recall phrases like 'I'm blanking on' should fire memory gap."""
        earlier = [
            TranscriptChunk(
                text="The deployment window is Thursday 2am to 6am.",
                timestamp=datetime.now() - timedelta(minutes=5),
                speaker="Sarah",
            )
        ]
        intel.score_signals(context=base_context, new_chunks=earlier)

        later = [
            TranscriptChunk(
                text="I am blanking on when the deployment window is.",
                timestamp=datetime.now(),
                speaker="John",
            )
        ]
        signals = intel.score_signals(context=base_context, new_chunks=later)

        memory_gaps = [s for s in signals if s.signal_type == SignalType.MEMORY_GAP]
        assert len(memory_gaps) >= 1

    @pytest.mark.asyncio
    async def test_no_memory_gap_without_recall_language(self, intel, base_context):
        """Normal speech without recall phrases should not fire memory gap."""
        earlier = [
            TranscriptChunk(
                text="The sprint ends on Friday.",
                timestamp=datetime.now() - timedelta(minutes=3),
                speaker="Mike",
            )
        ]
        intel.score_signals(context=base_context, new_chunks=earlier)

        later = [
            TranscriptChunk(
                text="Let's plan our tasks for the rest of the sprint.",
                timestamp=datetime.now(),
                speaker="Sarah",
            )
        ]
        signals = intel.score_signals(context=base_context, new_chunks=later)

        memory_gaps = [s for s in signals if s.signal_type == SignalType.MEMORY_GAP]
        assert len(memory_gaps) == 0

    # ── Collaboration opportunity ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_detect_collaboration_when_concern_overlaps_speech(self, intel, base_context):
        """When current speech overlaps an old unaddressed concern, fire collaboration."""
        # Add an old concern to context
        concern = Concern.create(
            description="Latency issues with Kafka message batching",
            raised_by="Sarah",
            severity=ConcernSeverity.HIGH,
        )
        concern.raised_at = datetime.now() - timedelta(minutes=5)
        base_context.concerns_raised.append(concern)

        chunks = [
            TranscriptChunk(
                text="I think Kafka message batching is the right approach for throughput.",
                timestamp=datetime.now(),
                speaker="Mike",
            )
        ]
        signals = intel.score_signals(context=base_context, new_chunks=chunks)

        collabs = [s for s in signals if s.signal_type == SignalType.COLLABORATION_OPPORTUNITY]
        assert len(collabs) >= 1
        assert "latency" in collabs[0].reason.lower() or "kafka" in collabs[0].reason.lower()

    @pytest.mark.asyncio
    async def test_no_collaboration_for_very_recent_concern(self, intel, base_context):
        """A concern raised < 2 min ago should NOT fire collaboration (still fresh)."""
        concern = Concern.create(
            description="Database scaling problems",
            raised_by="John",
            severity=ConcernSeverity.MEDIUM,
        )
        concern.raised_at = datetime.now() - timedelta(seconds=30)
        base_context.concerns_raised.append(concern)

        chunks = [
            TranscriptChunk(
                text="We need to think about database scaling right now.",
                timestamp=datetime.now(),
                speaker="Sarah",
            )
        ]
        signals = intel.score_signals(context=base_context, new_chunks=chunks)

        collabs = [s for s in signals if s.signal_type == SignalType.COLLABORATION_OPPORTUNITY]
        assert len(collabs) == 0

    # ── End-to-end: intelligence signals through TriggerEngine ───────

    @pytest.mark.asyncio
    async def test_contradiction_promoted_to_trigger(self):
        """A contradiction signal should be promoted to a CONTRADICTION trigger."""
        trigger_engine = TriggerEngine()

        context = MeetingContext(
            meeting_id="promo_test",
            meeting_started_at=datetime.now() - timedelta(minutes=10),
            platform="test",
        )

        # Feed earlier claim
        earlier_chunk = TranscriptChunk(
            text="Our API supports up to 500 requests per second.",
            timestamp=datetime.now() - timedelta(minutes=3),
            speaker="Mike",
        )
        context.recent_transcript.append(earlier_chunk)
        await trigger_engine.check_for_triggers(
            context=context, new_chunks=[earlier_chunk]
        )

        # Feed contradicting claim from different speaker
        later_chunk = TranscriptChunk(
            text="Our API does not support more than 200 requests per second.",
            timestamp=datetime.now(),
            speaker="Sarah",
        )
        context.recent_transcript.append(later_chunk)
        triggers = await trigger_engine.check_for_triggers(
            context=context, new_chunks=[later_chunk]
        )

        contradiction_triggers = [t for t in triggers if t.type == TriggerType.CONTRADICTION]
        assert len(contradiction_triggers) >= 1
        assert contradiction_triggers[0].signal_evidence_a is not None

    @pytest.mark.asyncio
    async def test_memory_gap_promoted_to_trigger(self):
        """A memory-gap signal should be promoted to a MEMORY_GAP trigger."""
        trigger_engine = TriggerEngine()

        context = MeetingContext(
            meeting_id="memory_test",
            meeting_started_at=datetime.now() - timedelta(minutes=10),
            platform="test",
        )

        earlier_chunk = TranscriptChunk(
            text="The deployment window is Thursday 2am to 6am.",
            timestamp=datetime.now() - timedelta(minutes=5),
            speaker="Sarah",
        )
        context.recent_transcript.append(earlier_chunk)
        await trigger_engine.check_for_triggers(
            context=context, new_chunks=[earlier_chunk]
        )

        later_chunk = TranscriptChunk(
            text="Does anyone remember when the deployment window is?",
            timestamp=datetime.now(),
            speaker="John",
        )
        context.recent_transcript.append(later_chunk)
        triggers = await trigger_engine.check_for_triggers(
            context=context, new_chunks=[later_chunk]
        )

        memory_triggers = [t for t in triggers if t.type == TriggerType.MEMORY_GAP]
        assert len(memory_triggers) >= 1


# =============================================================================
# Voice-Address Tests
# =============================================================================

class TestVoiceAddress:
    """Tests for voice-mention detection (someone says the bot's name out loud)."""

    @pytest.fixture
    def trigger_engine(self):
        return TriggerEngine()

    @pytest.fixture
    def base_context(self):
        return MeetingContext(
            meeting_id="voice_test",
            meeting_started_at=datetime.now() - timedelta(minutes=10),
            platform="test",
        )

    @pytest.mark.asyncio
    async def test_voice_mention_with_full_question(self, trigger_engine, base_context):
        """'Alex, can you summarize?' in a single transcript chunk fires VOICE_ADDRESS."""
        chunk = TranscriptChunk(
            text="Alex, can you summarize what we discussed so far?",
            timestamp=datetime.now(),
            speaker="Sarah",
        )
        base_context.recent_transcript.append(chunk)

        triggers = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[chunk]
        )

        assert len(triggers) == 1
        assert triggers[0].type == TriggerType.VOICE_ADDRESS
        assert triggers[0].command_type == CommandType.SUMMARIZE

    @pytest.mark.asyncio
    async def test_voice_mention_hey_prefix(self, trigger_engine, base_context):
        """'Hey Alex, what are the risks?' detects both name and known command."""
        chunk = TranscriptChunk(
            text="Hey Alex, what are the risks here?",
            timestamp=datetime.now(),
            speaker="John",
        )
        base_context.recent_transcript.append(chunk)

        triggers = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[chunk]
        )

        assert len(triggers) == 1
        assert triggers[0].type == TriggerType.VOICE_ADDRESS
        assert triggers[0].command_type == CommandType.RISKS

    @pytest.mark.asyncio
    async def test_voice_mention_freeform_question(self, trigger_engine, base_context):
        """A question that doesn't match a known command gets CUSTOM type."""
        chunk = TranscriptChunk(
            text="Alex, do you think we should postpone the migration?",
            timestamp=datetime.now(),
            speaker="Mike",
        )
        base_context.recent_transcript.append(chunk)

        triggers = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[chunk]
        )

        assert len(triggers) == 1
        assert triggers[0].type == TriggerType.VOICE_ADDRESS
        assert triggers[0].command_type == CommandType.CUSTOM
        assert "postpone" in triggers[0].relevant_transcript.lower()

    @pytest.mark.asyncio
    async def test_voice_mention_multi_chunk_buffering(self, trigger_engine, base_context):
        """Name in chunk 1, question in chunk 2 — should still fire."""
        chunk1 = TranscriptChunk(
            text="Alex,",
            timestamp=datetime.now(),
            speaker="Sarah",
        )
        base_context.recent_transcript.append(chunk1)

        # First call: starts buffering
        triggers1 = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[chunk1]
        )
        assert len(triggers1) == 0  # Not enough text yet

        chunk2 = TranscriptChunk(
            text="what are the action items from today?",
            timestamp=datetime.now(),
            speaker="Sarah",
        )
        base_context.recent_transcript.append(chunk2)

        # Second call: buffer completes
        triggers2 = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[chunk2]
        )

        assert len(triggers2) == 1
        assert triggers2[0].type == TriggerType.VOICE_ADDRESS
        assert triggers2[0].command_type == CommandType.ACTION_ITEMS

    @pytest.mark.asyncio
    async def test_voice_mention_name_alone_then_question(self, trigger_engine, base_context):
        """'Alex' as standalone chunk, then question in next final chunk.

        Real STT often finalizes the name as its own utterance:
          🎤 'alex'
          💬 'do you know'  (interim)
          💬 'do you know who is the prime minister of india'  (interim)
          🎤 'do you know who is the prime minister of india'  (final)
        """
        chunk1 = TranscriptChunk(
            text="alex",
            timestamp=datetime.now(),
            speaker="Adarsh",
            is_final=True,
        )
        base_context.recent_transcript.append(chunk1)

        triggers1 = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[chunk1]
        )
        assert len(triggers1) == 0  # Just the name — buffer started

        # Interim chunks keep the buffer alive
        interim = TranscriptChunk(
            text="do you know who is the",
            timestamp=datetime.now(),
            speaker="Adarsh",
            is_final=False,
        )
        await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[interim]
        )

        # Final chunk with the full question
        chunk2 = TranscriptChunk(
            text="do you know who is the prime minister of india",
            timestamp=datetime.now(),
            speaker="Adarsh",
            is_final=True,
        )
        base_context.recent_transcript.append(chunk2)

        # No punctuation, so won't fire yet — need settle time
        triggers2 = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[chunk2]
        )
        assert len(triggers2) == 0

        # Simulate silence (settle time elapsed — 1.5s threshold)
        trigger_engine._voice_buffer_last_chunk = datetime.now() - timedelta(seconds=2)

        triggers3 = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[]
        )
        assert len(triggers3) == 1
        assert triggers3[0].type == TriggerType.VOICE_ADDRESS
        assert "prime minister" in triggers3[0].relevant_transcript.lower()

    @pytest.mark.asyncio
    async def test_voice_mention_trailing_incomplete_word(self, trigger_engine, base_context):
        """'Alex do you currently' should NOT fire — it's incomplete.

        The full question 'alex do you currently speak' arrives as:
          chunk 1 (final): 'alex do you currently'
          chunk 2 (final): 'speak'
        Neither has punctuation, so the bot uses *time-based settling*:
        it waits until the speaker goes quiet for ~1.5 s, then fires.
        """
        chunk1 = TranscriptChunk(
            text="alex do you currently",
            timestamp=datetime.now(),
            speaker="Sarthak",
        )
        base_context.recent_transcript.append(chunk1)

        triggers1 = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[chunk1]
        )
        # Should NOT fire yet — just started buffering
        assert len(triggers1) == 0

        chunk2 = TranscriptChunk(
            text="speak",
            timestamp=datetime.now(),
            speaker="Sarthak",
        )
        base_context.recent_transcript.append(chunk2)

        triggers2 = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[chunk2]
        )
        # Still no fire — no punctuation and settle time hasn't elapsed
        assert len(triggers2) == 0

        # Simulate 2 seconds of silence (settle time = 1.5s)
        trigger_engine._voice_buffer_last_chunk = datetime.now() - timedelta(seconds=2)

        triggers3 = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[]
        )
        assert len(triggers3) == 1
        assert triggers3[0].type == TriggerType.VOICE_ADDRESS
        assert "speak" in triggers3[0].relevant_transcript.lower()

    @pytest.mark.asyncio
    async def test_interim_chunks_keep_buffer_alive(self, trigger_engine, base_context):
        """Interim (non-final) chunks act as 'still speaking' heartbeat.

        When someone says 'Alex, do you... ummm... speak?':
          chunk 1 (final):   'alex do you'     → starts buffer
          chunk 2 (interim): 'ummm'            → heartbeat, no text added
          chunk 3 (final):   'speak?'          → punctuation → fire

        Without the interim heartbeat, the settle timer would fire after
        chunk 1 with an incomplete question.
        """
        chunk1 = TranscriptChunk(
            text="alex do you",
            timestamp=datetime.now(),
            speaker="Sarthak",
            is_final=True,
        )
        base_context.recent_transcript.append(chunk1)

        triggers1 = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[chunk1]
        )
        assert len(triggers1) == 0  # buffering started

        # Simulate settle time passing, BUT an interim chunk arrives (filler sound)
        trigger_engine._voice_buffer_last_chunk = datetime.now() - timedelta(seconds=2)

        interim_chunk = TranscriptChunk(
            text="um",
            timestamp=datetime.now(),
            speaker="Sarthak",
            is_final=False,  # ← interim = still speaking
        )

        # The interim chunk should reset the settle clock → no fire
        triggers2 = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[interim_chunk]
        )
        assert len(triggers2) == 0  # heartbeat prevented premature fire

        # Now the final chunk with the complete question arrives
        chunk3 = TranscriptChunk(
            text="currently speak?",
            timestamp=datetime.now(),
            speaker="Sarthak",
            is_final=True,
        )
        base_context.recent_transcript.append(chunk3)

        triggers3 = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[chunk3]
        )
        assert len(triggers3) == 1
        assert triggers3[0].type == TriggerType.VOICE_ADDRESS
        assert "speak" in triggers3[0].relevant_transcript.lower()

    @pytest.mark.asyncio
    async def test_no_trigger_when_name_mentioned_in_passing(self, trigger_engine, base_context):
        """Mentioning the name without a question/request should NOT trigger."""
        chunk = TranscriptChunk(
            text="I think Alex would agree with that approach.",
            timestamp=datetime.now(),
            speaker="John",
        )
        base_context.recent_transcript.append(chunk)

        triggers = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[chunk]
        )

        voice_triggers = [t for t in triggers if t.type == TriggerType.VOICE_ADDRESS]
        assert len(voice_triggers) == 0

    @pytest.mark.asyncio
    async def test_voice_start_third_person_reference_does_not_trigger(self, trigger_engine, base_context):
        """'alex mentioned the deadline...' should not be treated as a direct address."""
        chunk = TranscriptChunk(
            text="alex mentioned the deadline is next friday",
            timestamp=datetime.now(),
            speaker="Mike",
        )

        triggers = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[chunk]
        )

        assert [t for t in triggers if t.type == TriggerType.VOICE_ADDRESS] == []

    @pytest.mark.asyncio
    async def test_voice_name_from_reference_does_not_trigger(self, trigger_engine, base_context):
        """'yeah alex from the product team ...' should not be treated as a direct address."""
        chunk = TranscriptChunk(
            text="yeah alex from the product team said he will handle the integration",
            timestamp=datetime.now(),
            speaker="Sarah",
        )

        triggers = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[chunk]
        )

        assert [t for t in triggers if t.type == TriggerType.VOICE_ADDRESS] == []

    @pytest.mark.asyncio
    async def test_voice_takes_priority_over_proactive(self, trigger_engine, base_context):
        """Voice-addressed question should return immediately, skipping proactive checks."""
        # Add context that would normally trigger proactive signals
        concern = Concern.create(
            description="Database scaling problems",
            raised_by="John",
            severity=ConcernSeverity.HIGH,
        )
        concern.raised_at = datetime.now() - timedelta(minutes=5)
        base_context.concerns_raised.append(concern)

        chunk = TranscriptChunk(
            text="Alex, summarize the database scaling discussion.",
            timestamp=datetime.now(),
            speaker="Sarah",
        )
        base_context.recent_transcript.append(chunk)

        triggers = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[chunk]
        )

        # Should get exactly 1 trigger (voice), not additional proactive ones
        assert len(triggers) == 1
        assert triggers[0].type == TriggerType.VOICE_ADDRESS

    @pytest.mark.asyncio
    async def test_voice_mention_so_prefix(self, trigger_engine, base_context):
        """'so alex which should we use' — common connector prefix must match."""
        chunk = TranscriptChunk(
            text="so alex which should we use?",
            timestamp=datetime.now(),
            speaker="Adarsh Bhardwaj",
        )
        base_context.recent_transcript.append(chunk)

        triggers = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[chunk]
        )

        assert len(triggers) == 1
        assert triggers[0].type == TriggerType.VOICE_ADDRESS
        assert "which should we use" in triggers[0].relevant_transcript

    @pytest.mark.asyncio
    async def test_voice_multiple_fillers(self, trigger_engine, base_context):
        """'okay so alex what do you think' — multiple fillers before name
        should match the start pattern (regex allows * not ?)."""
        chunk = TranscriptChunk(
            text="okay so alex what do you think about this?",
            timestamp=datetime.now(),
            speaker="Adarsh",
        )
        triggers = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[chunk]
        )
        assert len(triggers) == 1
        assert triggers[0].type == TriggerType.VOICE_ADDRESS
        assert "what do you think" in triggers[0].relevant_transcript

    @pytest.mark.asyncio
    async def test_voice_name_at_end_question(self, trigger_engine, base_context):
        """'what do you think alex?' — name at END of question fires."""
        chunk = TranscriptChunk(
            text="what do you think alex?",
            timestamp=datetime.now(),
            speaker="Mike",
        )
        triggers = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[chunk]
        )
        assert len(triggers) == 1
        assert triggers[0].type == TriggerType.VOICE_ADDRESS
        assert "what do you think" in triggers[0].relevant_transcript

    @pytest.mark.asyncio
    async def test_voice_name_at_end_command_after_settle(self, trigger_engine, base_context):
        """'summarize alex' should trigger after settle and classify as summarize."""
        chunk = TranscriptChunk(
            text="summarize alex",
            timestamp=datetime.now(),
            speaker="Mike",
        )
        base_context.recent_transcript.append(chunk)

        triggers = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[chunk]
        )
        assert len(triggers) == 0

        trigger_engine._voice_buffer_last_chunk = datetime.now() - timedelta(seconds=2)
        triggers = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[]
        )

        assert len(triggers) == 1
        assert triggers[0].type == TriggerType.VOICE_ADDRESS
        assert triggers[0].command_type == CommandType.SUMMARIZE
        assert triggers[0].relevant_transcript.lower() == "summarize"

    @pytest.mark.asyncio
    async def test_voice_name_in_middle_question(self, trigger_engine, base_context):
        """'can alex help with the migration?' — name in MIDDLE fires."""
        chunk = TranscriptChunk(
            text="can alex help with the migration?",
            timestamp=datetime.now(),
            speaker="Sarah",
        )
        triggers = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[chunk]
        )
        assert len(triggers) == 1
        assert triggers[0].type == TriggerType.VOICE_ADDRESS
        assert "migration" in triggers[0].relevant_transcript

    @pytest.mark.asyncio
    async def test_voice_name_repeated_in_request_keeps_command(self, trigger_engine, base_context):
        """'alex can you alex summarize' should still resolve to summarize."""
        chunk = TranscriptChunk(
            text="alex can you alex summarize",
            timestamp=datetime.now(),
            speaker="Sarah",
        )
        base_context.recent_transcript.append(chunk)

        triggers = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[chunk]
        )
        assert len(triggers) == 0

        trigger_engine._voice_buffer_last_chunk = datetime.now() - timedelta(seconds=2)
        triggers = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[]
        )

        assert len(triggers) == 1
        assert triggers[0].type == TriggerType.VOICE_ADDRESS
        assert triggers[0].command_type == CommandType.SUMMARIZE
        assert "summarize" in triggers[0].relevant_transcript.lower()

    @pytest.mark.asyncio
    async def test_voice_name_with_filler_then_command_after_settle(self, trigger_engine, base_context):
        """'so alex summarize' should trigger after settle and classify as summarize."""
        chunk = TranscriptChunk(
            text="so alex summarize",
            timestamp=datetime.now(),
            speaker="Adarsh Bhardwaj",
        )
        base_context.recent_transcript.append(chunk)

        triggers = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[chunk]
        )
        assert len(triggers) == 0

        trigger_engine._voice_buffer_last_chunk = datetime.now() - timedelta(seconds=2)
        triggers = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[]
        )

        assert len(triggers) == 1
        assert triggers[0].type == TriggerType.VOICE_ADDRESS
        assert triggers[0].command_type == CommandType.SUMMARIZE
        assert triggers[0].relevant_transcript.lower() == "summarize"

    @pytest.mark.asyncio
    async def test_voice_clause_reset_then_name_triggers_followup(self, trigger_engine, base_context):
        """A clause reset like 'oh wait alex how are you' should trigger the follow-up request."""
        chunk = TranscriptChunk(
            text="so harsh can you circle back to me tomorrow oh wait alex how are you",
            timestamp=datetime.now(),
            speaker="Adarsh Bhardwaj",
        )
        base_context.recent_transcript.append(chunk)

        triggers = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[chunk]
        )
        assert len(triggers) == 0

        trigger_engine._voice_buffer_last_chunk = datetime.now() - timedelta(seconds=2)
        triggers = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[]
        )

        assert len(triggers) == 1
        assert triggers[0].type == TriggerType.VOICE_ADDRESS
        assert triggers[0].command_type == CommandType.CUSTOM
        assert triggers[0].relevant_transcript.lower() == "how are you"

    @pytest.mark.asyncio
    async def test_voice_embedded_asking_statement_triggers(self, trigger_engine, base_context):
        """Embedded 'asking Alex' phrasing should count as direct voice requests."""
        chunk = TranscriptChunk(
            text="no no i am asking alex what things you can help me with",
            timestamp=datetime.now(),
            speaker="Adarsh Bhardwaj",
        )
        triggers = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[chunk]
        )
        assert len(triggers) == 0

        trigger_engine._voice_buffer_last_chunk = datetime.now() - timedelta(seconds=4)
        triggers = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[]
        )

        assert len(triggers) == 1
        assert triggers[0].type == TriggerType.VOICE_ADDRESS
        assert "what things you can help me with" in triggers[0].relevant_transcript.lower()

    @pytest.mark.asyncio
    async def test_voice_embedded_request_keeps_command_classification(self, trigger_engine, base_context):
        """Embedded direct-address requests should still map to known commands."""
        chunk = TranscriptChunk(
            text="i want to ask alex to summarize the risks here",
            timestamp=datetime.now(),
            speaker="John",
        )
        triggers = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[chunk]
        )
        assert len(triggers) == 0

        trigger_engine._voice_buffer_last_chunk = datetime.now() - timedelta(seconds=4)
        triggers = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[]
        )

        assert len(triggers) == 1
        assert triggers[0].type == TriggerType.VOICE_ADDRESS
        assert triggers[0].command_type == CommandType.SUMMARIZE

    @pytest.mark.asyncio
    async def test_voice_past_reference_with_ask_does_not_trigger(self, trigger_engine, base_context):
        """Past-tense references should stay non-triggering even if they include ask/Alex."""
        chunk = TranscriptChunk(
            text="i asked alex yesterday about the deployment timeline",
            timestamp=datetime.now(),
            speaker="John",
        )
        triggers = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[chunk]
        )
        voice_triggers = [t for t in triggers if t.type == TriggerType.VOICE_ADDRESS]
        assert len(voice_triggers) == 0

    @pytest.mark.asyncio
    async def test_voice_3rd_person_no_trigger(self, trigger_engine, base_context):
        """'I talked to alex yesterday' — 3rd person statement must NOT trigger."""
        chunk = TranscriptChunk(
            text="I talked to alex yesterday about the deployment",
            timestamp=datetime.now(),
            speaker="John",
        )
        triggers = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[chunk]
        )
        voice_triggers = [t for t in triggers if t.type == TriggerType.VOICE_ADDRESS]
        assert len(voice_triggers) == 0

    @pytest.mark.asyncio
    async def test_voice_name_at_end_no_question(self, trigger_engine, base_context):
        """'great work alex' — name at end of non-question must NOT trigger."""
        chunk = TranscriptChunk(
            text="great work alex",
            timestamp=datetime.now(),
            speaker="Sarah",
        )
        triggers = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[chunk]
        )
        voice_triggers = [t for t in triggers if t.type == TriggerType.VOICE_ADDRESS]
        assert len(voice_triggers) == 0

    @pytest.mark.asyncio
    async def test_voice_buffer_overwrite_fires_old(self, trigger_engine, base_context):
        """When a new 'alex' question arrives while buffer has content,
        the OLD question should fire first — never silently dropped."""
        # First question (no punctuation → goes into buffer)
        chunk1 = TranscriptChunk(
            text="alex our website is a streaming service",
            timestamp=datetime.now(),
            speaker="Adarsh Bhardwaj",
        )
        await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[chunk1]
        )

        # Continuation (appended to buffer)
        chunk2 = TranscriptChunk(
            text="what will be the cost of using azure",
            timestamp=datetime.now(),
            speaker="Adarsh Bhardwaj",
        )
        await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[chunk2]
        )

        # NEW "alex" question arrives before settle timer — old buffer should fire
        chunk3 = TranscriptChunk(
            text="alex what is the date today",
            timestamp=datetime.now(),
            speaker="Adarsh Bhardwaj",
        )
        triggers = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[chunk3]
        )

        # The OLD buffered question should fire (not be dropped)
        assert len(triggers) == 1
        assert triggers[0].type == TriggerType.VOICE_ADDRESS
        assert "streaming service" in triggers[0].relevant_transcript
        assert "cost" in triggers[0].relevant_transcript

        # The NEW question is now in the buffer — let settle timer fire it
        import asyncio
        await asyncio.sleep(2.0)
        triggers2 = await trigger_engine.check_for_triggers(
            context=base_context, new_chunks=[]
        )
        assert len(triggers2) == 1
        assert "date today" in triggers2[0].relevant_transcript

    @pytest.mark.asyncio
    async def test_decision_conflict_detected_from_recent_transcript_before_context_analysis(self):
        """An immediate decision after a concern should still fire conflict detection."""
        trigger_engine = TriggerEngine()

        context = MeetingContext(
            meeting_id="conflict_recent_transcript",
            meeting_started_at=datetime.now() - timedelta(minutes=10),
            platform="test",
        )

        concern_chunk = TranscriptChunk(
            text="I am worried about the latency issues with Kafka message batching.",
            timestamp=datetime.now() - timedelta(seconds=2),
            speaker="Sarah",
        )
        context.recent_transcript.append(concern_chunk)
        await trigger_engine.check_for_triggers(context=context, new_chunks=[concern_chunk])

        decision_chunk = TranscriptChunk(
            text="We should use Kafka for the event pipeline.",
            timestamp=datetime.now(),
            speaker="Mike",
        )
        context.recent_transcript.append(decision_chunk)
        triggers = await trigger_engine.check_for_triggers(
            context=context,
            new_chunks=[decision_chunk],
        )

        assert any(t.type == TriggerType.CONFLICT_DETECTED for t in triggers)

    @pytest.mark.asyncio
    async def test_voice_buffer_suppresses_proactive_decision_trigger(self, trigger_engine, base_context):
        """While collecting a voice-addressed request, proactive triggers should be suppressed."""
        chunk = TranscriptChunk(
            text="alex we should use azure",
            timestamp=datetime.now(),
            speaker="Adarsh Bhardwaj",
        )
        base_context.recent_transcript.append(chunk)

        triggers = await trigger_engine.check_for_triggers(
            context=base_context,
            new_chunks=[chunk],
        )

        assert triggers == []

    @pytest.mark.asyncio
    async def test_proactive_trigger_not_repeated_without_new_final_chunks(self):
        """Periodic empty polls should not re-fire proactive triggers from stale transcript."""
        trigger_engine = TriggerEngine()
        context = MeetingContext(
            meeting_id="decision_polling",
            meeting_started_at=datetime.now() - timedelta(minutes=10),
            platform="test",
        )

        chunk = TranscriptChunk(
            text="we should use PostgreSQL for the main database",
            timestamp=datetime.now(),
            speaker="Mike",
        )
        context.recent_transcript.append(chunk)

        first_triggers = await trigger_engine.check_for_triggers(
            context=context,
            new_chunks=[chunk],
        )
        assert any(t.type == TriggerType.DECISION_FORMING for t in first_triggers)

        trigger_engine._last_trigger_by_type[TriggerType.DECISION_FORMING] = datetime.now() - timedelta(seconds=31)

        second_triggers = await trigger_engine.check_for_triggers(
            context=context,
            new_chunks=[],
        )
        assert second_triggers == []

    @pytest.mark.asyncio
    async def test_duplicate_proactive_trigger_suppressed_on_same_text(self):
        """Identical decision text should not keep re-triggering after cooldown expires."""
        trigger_engine = TriggerEngine()
        context = MeetingContext(
            meeting_id="decision_duplicate",
            meeting_started_at=datetime.now() - timedelta(minutes=10),
            platform="test",
        )

        chunk1 = TranscriptChunk(
            text="we should use PostgreSQL for the main database",
            timestamp=datetime.now(),
            speaker="Mike",
        )
        context.recent_transcript.append(chunk1)
        first_triggers = await trigger_engine.check_for_triggers(
            context=context,
            new_chunks=[chunk1],
        )
        assert any(t.type == TriggerType.DECISION_FORMING for t in first_triggers)

        trigger_engine._last_trigger_by_type[TriggerType.DECISION_FORMING] = datetime.now() - timedelta(seconds=31)

        chunk2 = TranscriptChunk(
            text="we should use PostgreSQL for the main database",
            timestamp=datetime.now() + timedelta(seconds=1),
            speaker="Mike",
        )
        context.recent_transcript.append(chunk2)
        second_triggers = await trigger_engine.check_for_triggers(
            context=context,
            new_chunks=[chunk2],
        )
        assert [t for t in second_triggers if t.type == TriggerType.DECISION_FORMING] == []

    @pytest.mark.asyncio
    async def test_repeated_confusion_detected_across_recent_turns(self):
        """Softer confusion across nearby turns should trigger repeated confusion."""
        trigger_engine = TriggerEngine()

        context = MeetingContext(
            meeting_id="soft_confusion_recent_turns",
            meeting_started_at=datetime.now() - timedelta(minutes=10),
            platform="test",
        )

        first_chunk = TranscriptChunk(
            text="i am a bit lost on the rollback plan",
            timestamp=datetime.now() - timedelta(seconds=3),
            speaker="John",
        )
        context.recent_transcript.append(first_chunk)
        await trigger_engine.check_for_triggers(context=context, new_chunks=[first_chunk])

        second_chunk = TranscriptChunk(
            text="can we slow down because i am not following the rollback plan",
            timestamp=datetime.now(),
            speaker="John",
        )
        context.recent_transcript.append(second_chunk)
        triggers = await trigger_engine.check_for_triggers(
            context=context,
            new_chunks=[second_chunk],
        )

        assert any(t.type == TriggerType.REPEATED_CONFUSION for t in triggers)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full pipeline."""
    
    @pytest.mark.asyncio
    async def test_transcript_to_trigger_pipeline(self):
        """Test the full pipeline from transcript to trigger detection."""
        # Setup
        context_engine = ContextEngine(
            meeting_id="integration_test",
            platform="test"
        )
        trigger_engine = TriggerEngine()
        
        await context_engine.start()
        await trigger_engine.start()
        
        try:
            # Simulate conversation
            chunks = [
                TranscriptChunk(
                    text="I'm concerned about the latency in our system",
                    timestamp=datetime.now(),
                    speaker="Sarah"
                ),
                TranscriptChunk(
                    text="Let's use a caching layer",
                    timestamp=datetime.now(),
                    speaker="John"
                ),
            ]
            
            for chunk in chunks:
                await context_engine.add_transcript_chunk(chunk)
            
            # Wait for processing
            await asyncio.sleep(context_engine.update_interval + 1)
            
            # Verify concerns detected
            assert len(context_engine.context.concerns_raised) > 0
            
            # Check for triggers
            triggers = await trigger_engine.check_for_triggers(
                context=context_engine.get_context_snapshot()
            )
            
            # Pipeline should work without errors
            assert isinstance(triggers, list)
            
        finally:
            await context_engine.stop()
            await trigger_engine.stop()


class TestChatManager:
    """Tests for chat delivery behavior."""

    @pytest.mark.asyncio
    async def test_voice_address_bypasses_proactive_rate_limit(self):
        """Voice-addressed answers should be treated as reactive, not proactive."""
        bot = Mock()
        bot.send_chat = AsyncMock(return_value=True)

        manager = ChatManager(bot)
        manager._last_proactive_time = datetime.now()

        voice_trigger = Trigger.create_command(CommandType.CUSTOM, "help with this")
        voice_trigger.type = TriggerType.VOICE_ADDRESS
        voice_response = AIResponse.create(
            text="Here is the answer.",
            trigger=voice_trigger,
            priority=MessagePriority.HIGH,
        )

        sent = await manager.send(voice_response)

        assert sent is True
        bot.send_chat.assert_awaited_once()

        bot.send_chat.reset_mock()

        proactive_trigger = Trigger.create_proactive(
            TriggerType.DECISION_FORMING,
            confidence=0.8,
            reason="Decision is forming",
        )
        proactive_response = AIResponse.create(
            text="Potential proactive note.",
            trigger=proactive_trigger,
        )

        sent_proactive = await manager.send(proactive_response)

        assert sent_proactive is False
        bot.send_chat.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_voice_address_prefers_speech_on_supported_bot(self):
        """Direct voice-addressed replies should use speech when the bot supports it."""
        bot = Mock()
        bot.supports_speech_output = True
        bot.send_speech = AsyncMock(return_value=True)
        bot.send_chat = AsyncMock(return_value=True)

        manager = ChatManager(bot)

        voice_trigger = Trigger.create_command(CommandType.CUSTOM, "help with this")
        voice_trigger.type = TriggerType.VOICE_ADDRESS
        voice_response = AIResponse.create(
            text="Here is the answer.",
            trigger=voice_trigger,
            priority=MessagePriority.HIGH,
        )

        sent = await manager.send(voice_response)

        assert sent is True
        bot.send_speech.assert_awaited_once_with("Here is the answer.")
        bot.send_chat.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_voice_address_shortens_spoken_payload(self):
        """Spoken delivery should trim long answers to keep TTS payloads small."""
        bot = Mock()
        bot.supports_speech_output = True
        bot.send_speech = AsyncMock(return_value=True)
        bot.send_chat = AsyncMock(return_value=True)

        manager = ChatManager(bot)

        voice_trigger = Trigger.create_command(CommandType.CUSTOM, "help with this")
        voice_trigger.type = TriggerType.VOICE_ADDRESS
        voice_response = AIResponse.create(
            text=(
                "Here is the quick answer. "
                "The main risk is timeline slippage because integration and QA are both under-resourced this week. "
                "I can summarize action items if useful."
            ),
            trigger=voice_trigger,
            priority=MessagePriority.HIGH,
        )

        sent = await manager.send(voice_response)

        assert sent is True
        spoken_text = bot.send_speech.await_args.args[0]
        assert len(spoken_text) <= 160
        assert "I can summarize action items if useful." not in spoken_text
        bot.send_chat.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_voice_address_falls_back_to_chat_when_speech_fails(self):
        """Speech delivery failures should fall back to the existing chat path."""
        bot = Mock()
        bot.supports_speech_output = True
        bot.send_speech = AsyncMock(return_value=False)
        bot.send_chat = AsyncMock(return_value=True)

        manager = ChatManager(bot)

        voice_trigger = Trigger.create_command(CommandType.CUSTOM, "help with this")
        voice_trigger.type = TriggerType.VOICE_ADDRESS
        voice_response = AIResponse.create(
            text="Here is the answer.",
            trigger=voice_trigger,
            priority=MessagePriority.HIGH,
        )

        sent = await manager.send(voice_response)

        assert sent is True
        bot.send_speech.assert_awaited_once_with("Here is the answer.")
        bot.send_chat.assert_awaited_once_with(f"{manager.bot_label}: Here is the answer.")


class TestLLMServiceFallback:
    """Tests for deterministic local fallbacks when the LLM is unavailable."""

    @pytest.fixture
    def fallback_context(self):
        context = MeetingContext(
            meeting_id="fallback_test",
            meeting_started_at=datetime.now() - timedelta(minutes=12),
            platform="test",
        )
        context.current_topic = Topic(
            name="Architecture Review",
            started_at=datetime.now() - timedelta(minutes=5),
        )
        context.participants.update({"Sarah", "John"})
        context.concerns_raised.append(
            Concern.create(
                description="Latency risk in the current design",
                raised_by="Sarah",
                severity=ConcernSeverity.HIGH,
            )
        )
        context.decisions_pending.append(
            Decision.create(
                description="Use a caching layer for reads",
                proposed_by="John",
            )
        )
        context.action_items.append("John: send the design doc by Friday")
        context.open_questions.append("What is the rollout plan?")
        context.recent_transcript.extend([
            TranscriptChunk(
                text="We should review the architecture tradeoffs.",
                timestamp=datetime.now() - timedelta(minutes=1),
                speaker="Sarah",
            ),
            TranscriptChunk(
                text="John will send the design doc by Friday.",
                timestamp=datetime.now() - timedelta(seconds=30),
                speaker="Sarah",
            ),
        ])
        return context

    @pytest.mark.asyncio
    async def test_summary_command_uses_local_fallback(self, fallback_context):
        """Summaries should still work from local context when the LLM is down."""
        service = LLMService()
        service._client = None

        trigger = Trigger.create_command(CommandType.SUMMARIZE)
        response = await service.generate_response(
            trigger=trigger,
            context=fallback_context,
            recent_transcript=fallback_context.get_recent_text(minutes=5),
        )

        assert response is not None
        assert "Topic:" in response.text or "Decisions:" in response.text

    @pytest.mark.asyncio
    async def test_action_items_command_uses_local_fallback(self, fallback_context):
        """Action items should come from structured context without the LLM."""
        service = LLMService()
        service._client = None

        trigger = Trigger.create_command(CommandType.ACTION_ITEMS)
        response = await service.generate_response(
            trigger=trigger,
            context=fallback_context,
            recent_transcript=fallback_context.get_recent_text(minutes=5),
        )

        assert response is not None
        assert "Action:" in response.text
        assert "design doc" in response.text.lower()

    @pytest.mark.asyncio
    async def test_proactive_trigger_skips_without_llm(self, fallback_context):
        """Proactive messages should stay silent instead of sending generic fallback text."""
        service = LLMService()
        service._client = None

        trigger = Trigger.create_proactive(
            TriggerType.CONTRADICTION,
            confidence=0.8,
            reason="Statements conflict",
        )

        response = await service.generate_response(
            trigger=trigger,
            context=fallback_context,
            recent_transcript=fallback_context.get_recent_text(minutes=5),
        )

        assert response is None

    @pytest.mark.asyncio
    async def test_voice_summary_uses_fast_path_without_llm(self, fallback_context):
        """Voice summarize requests should skip the LLM for lower latency."""
        service = LLMService()
        service._client = object()
        service._call_llm = AsyncMock(side_effect=AssertionError("LLM should not be called"))

        trigger = Trigger.create_command(CommandType.SUMMARIZE)
        trigger.type = TriggerType.VOICE_ADDRESS
        trigger.relevant_transcript = "summarize"

        response = await service.generate_response(
            trigger=trigger,
            context=fallback_context,
            recent_transcript=fallback_context.get_recent_text(minutes=5),
        )

        assert response is not None
        assert "discussion is about" in response.text.lower() or "likely direction" in response.text.lower()
        assert "topic:" not in response.text.lower()

    @pytest.mark.asyncio
    async def test_voice_summary_avoids_noisy_transcript_dump(self, fallback_context):
        """Voice summaries should normalize noisy topic fragments instead of parroting them back."""
        service = LLMService()
        service._client = object()
        service._call_llm = AsyncMock(side_effect=AssertionError("LLM should not be called"))

        fallback_context.current_topic = Topic(
            name=(
                "Which cloud architecture which cloud provider we should use "
                "which provider aws azure which one we should use or gcp even"
            ),
            started_at=datetime.now() - timedelta(minutes=4),
        )
        fallback_context.concerns_raised.clear()
        fallback_context.decisions_pending.clear()
        fallback_context.recent_transcript = [
            TranscriptChunk(
                text="I was confused about which cloud provider we should use.",
                timestamp=datetime.now() - timedelta(seconds=45),
                speaker="Adarsh Bhardwaj",
            ),
            TranscriptChunk(
                text="AWS, Azure, and GCP are all options and our budget is constrained.",
                timestamp=datetime.now() - timedelta(seconds=15),
                speaker="Adarsh Bhardwaj",
            ),
        ]

        trigger = Trigger.create_command(CommandType.SUMMARIZE)
        trigger.type = TriggerType.VOICE_ADDRESS
        trigger.relevant_transcript = "can you summarize till now"

        response = await service.generate_response(
            trigger=trigger,
            context=fallback_context,
            recent_transcript=fallback_context.get_recent_text(minutes=5),
        )

        assert response is not None
        assert "which cloud architecture which cloud provider" not in response.text.lower()
        assert "cloud provider" in response.text.lower()
        assert "budget" in response.text.lower() or "no final choice" in response.text.lower()

    @pytest.mark.asyncio
    async def test_voice_smalltalk_uses_fast_path_without_llm(self, fallback_context):
        """Common voice small-talk should respond locally without waiting for the LLM."""
        service = LLMService()
        service._client = object()
        service._call_llm = AsyncMock(side_effect=AssertionError("LLM should not be called"))

        trigger = Trigger.create_command(CommandType.CUSTOM, "how are you")
        trigger.type = TriggerType.VOICE_ADDRESS
        trigger.relevant_transcript = "how are you"

        response = await service.generate_response(
            trigger=trigger,
            context=fallback_context,
            recent_transcript=fallback_context.get_recent_text(minutes=5),
        )

        assert response is not None
        assert "doing well" in response.text.lower()

    @pytest.mark.asyncio
    async def test_voice_identity_question_uses_speaker_name_fast_path(self, fallback_context):
        """Speaker-grounded identity questions should answer with the speaker name."""
        service = LLMService()
        service._client = object()
        service._call_llm = AsyncMock(side_effect=AssertionError("LLM should not be called"))

        trigger = Trigger.create_command(CommandType.EXPLAIN, "my name")
        trigger.type = TriggerType.VOICE_ADDRESS
        trigger.speaker = "Adarsh Bhardwaj"
        trigger.relevant_transcript = "what is my name"

        response = await service.generate_response(
            trigger=trigger,
            context=fallback_context,
            recent_transcript=fallback_context.get_recent_text(minutes=5),
        )

        assert response is not None
        assert response.text == "Your name is Adarsh Bhardwaj."

    @pytest.mark.asyncio
    async def test_context_summary_is_sanitized_into_fixed_sections(self):
        """Historical context summarization should normalize the LLM output into stable sections."""
        service = LLMService()
        service._client = object()
        service._call_llm = AsyncMock(
            return_value=(
                "**Decisions:** None\n"
                "Summary: The team discussed cloud provider tradeoffs.\n"
                "Risks: Budget constraints and migration complexity.\n"
                "Action Items: John to share cost estimates."
            )
        )

        chunks = [
            TranscriptChunk(
                text="We should compare cloud providers for the migration.",
                timestamp=datetime.now() - timedelta(minutes=12),
                speaker="John",
            ),
            TranscriptChunk(
                text="Budget is a major concern for this rollout.",
                timestamp=datetime.now() - timedelta(minutes=11),
                speaker="Sarah",
            ),
        ]

        summary = await service.summarize_context("", chunks)

        assert summary is not None
        assert summary.startswith("Summary: The team discussed cloud provider tradeoffs.")
        assert "Decisions: None" in summary
        assert "Open Questions: None" in summary
        assert "Risks: Budget constraints and migration complexity." in summary
        assert "Action Items: John to share cost estimates." in summary

    @pytest.mark.asyncio
    async def test_context_summary_passes_model_override_when_configured(self):
        """Background context cleanup should use its dedicated model when configured."""
        service = LLMService()
        service._client = object()
        service._summary_model = "gpt-4o-mini"
        service._call_llm = AsyncMock(
            return_value=(
                "Summary: Team reviewed cloud costs.\n"
                "Decisions: None\n"
                "Open Questions: None\n"
                "Risks: Budget pressure.\n"
                "Action Items: None"
            )
        )

        chunks = [
            TranscriptChunk(
                text="We need a cheaper model for background summarization.",
                timestamp=datetime.now() - timedelta(minutes=5),
                speaker="Adarsh",
            )
        ]

        await service.summarize_context("", chunks)

        assert service._call_llm.await_args.kwargs["model_override"] == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_call_llm_uses_model_override_for_openai(self):
        """Model overrides should be honored at the client call site."""
        create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )
        )
        service = LLMService()
        service.provider = LLMProvider.OPENAI
        service._model = "primary-model"
        service._client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )

        result = await service._call_llm(
            "system",
            "user",
            model_override="summary-model",
        )

        assert result == "ok"
        assert create.await_args.kwargs["model"] == "summary-model"


class TestSpeechService:
    """Speech synthesis provider behavior."""

    def test_speech_service_unavailable_without_elevenlabs_key(self, monkeypatch):
        import src.reasoning.speech_service as speech_module

        monkeypatch.setattr(speech_module.config.speech, "enable_voice_replies", True)
        monkeypatch.setattr(speech_module.config.speech, "elevenlabs_api_key", None)

        service = SpeechService()

        assert service.available is False

    @pytest.mark.asyncio
    async def test_speech_service_calls_elevenlabs_convert_api(self, monkeypatch):
        import src.reasoning.speech_service as speech_module

        calls = []

        class DummyResponse:
            def __init__(self):
                self.status_code = 200
                self.content = b"fake-mp3"
                self.text = "ok"

        class DummyAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def post(self, url, params=None, headers=None, json=None):
                calls.append(
                    {
                        "url": url,
                        "params": params,
                        "headers": headers,
                        "json": json,
                    }
                )
                return DummyResponse()

        monkeypatch.setattr(speech_module.config.speech, "enable_voice_replies", True)
        monkeypatch.setattr(speech_module.config.speech, "elevenlabs_api_key", "test-key")
        monkeypatch.setattr(speech_module.config.speech, "elevenlabs_model_id", "eleven_flash_v2_5")
        monkeypatch.setattr(speech_module.config.speech, "elevenlabs_voice_id", "voice-123")
        monkeypatch.setattr(speech_module.config.speech, "elevenlabs_output_format", "mp3_44100_128")
        monkeypatch.setattr(speech_module.config.speech, "elevenlabs_latency_mode", 3)
        monkeypatch.setattr(speech_module.httpx, "AsyncClient", DummyAsyncClient)

        service = SpeechService()
        audio = await service.synthesize_mp3("hello world")

        assert audio == b"fake-mp3"
        assert len(calls) == 1
        assert calls[0]["url"].endswith("/text-to-speech/voice-123")
        assert calls[0]["params"] == {
            "output_format": "mp3_44100_128",
            "optimize_streaming_latency": 3,
        }
        assert calls[0]["headers"]["xi-api-key"] == "test-key"
        assert calls[0]["json"] == {
            "text": "hello world",
            "model_id": "eleven_flash_v2_5",
        }


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
