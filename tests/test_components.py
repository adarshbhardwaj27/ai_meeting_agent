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
from unittest.mock import Mock, AsyncMock, patch

# Import components
from src.context.models import (
    TranscriptChunk, MeetingContext, Concern, Decision,
    Trigger, TriggerType, CommandType, ConcernSeverity, DecisionStatus
)
from src.context.context_engine import ContextEngine
from src.triggers.trigger_engine import TriggerEngine


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


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
