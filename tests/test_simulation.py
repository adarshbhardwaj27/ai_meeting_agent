"""
End-to-end simulation test for the Real-Time AI Meeting Assistant.

This script simulates a realistic meeting conversation and shows
how the AI assistant would respond in real-time.

Run with: python -m tests.test_simulation
"""

import asyncio
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.context import ContextEngine, TranscriptChunk, TriggerType
from src.triggers import TriggerEngine
from src.reasoning import LLMService
from src.config import config


# Simulated meeting conversation
MEETING_SCRIPT = [
    {
        "speaker": "John (PM)",
        "text": "Alright team, let's kick off this architecture discussion. We need to design the real-time notification system.",
        "delay": 2
    },
    {
        "speaker": "Sarah (Engineer)",
        "text": "Before we dive in, I want to flag that latency is a major concern. Our SLA requires sub-100ms delivery.",
        "delay": 3
    },
    {
        "speaker": "Mike (Engineer)",
        "text": "Good point. We also need to consider scale - we're expecting 10x growth next quarter.",
        "delay": 2
    },
    {
        "speaker": "John (PM)",
        "text": "Right, so we need something that handles both. What are our options?",
        "delay": 2
    },
    {
        "speaker": "Mike (Engineer)", 
        "text": "I've been looking at Kafka. It's great for handling high throughput.",
        "delay": 2
    },
    {
        "speaker": "Sarah (Engineer)",
        "text": "But Kafka adds latency. There's network hops, batching delays. I'm not sure it fits our SLA.",
        "delay": 3
    },
    {
        "speaker": "Mike (Engineer)",
        "text": "We could tune the batching settings to minimize delay.",
        "delay": 2
    },
    {
        "speaker": "John (PM)",
        "text": "Let's just go with Kafka - it's industry standard and our infra team knows it.",
        "delay": 3  # This should trigger conflict detection!
    },
    {
        "speaker": "Sarah (Engineer)",
        "text": "Wait, are we sure? What about the latency concerns I raised?",
        "delay": 3
    },
    {
        "speaker": "John (PM)",
        "text": "Good question. Mike, can Kafka meet the 100ms SLA?",
        "delay": 2
    },
    {
        "speaker": "Mike (Engineer)",
        "text": "Honestly, I'm not sure. We'd need to benchmark it.",
        "delay": 2
    },
    {
        "speaker": "John (PM)",
        "text": "Okay, let's table the Kafka decision for now. What other options?",
        "delay": 3
    },
    {
        "speaker": "Sarah (Engineer)",
        "text": "What about Redis Streams? It's designed for real-time with much lower latency.",
        "delay": 2
    },
    {
        "speaker": "Mike (Engineer)",
        "text": "Redis could work, but I'm worried about durability. What if we lose messages?",
        "delay": 3  # New concern!
    },
    {
        "speaker": "John (PM)",
        "text": "Good point. So we need: low latency, high scale, and durability. Anything else?",
        "delay": 2
    },
    {
        "speaker": "Sarah (Engineer)",
        "text": "Cost is a factor too. Managed Kafka can get expensive.",
        "delay": 2
    },
    {
        "speaker": "command",
        "text": "@Alex summarize where we are",  # Explicit command
        "delay": 3
    },
    {
        "speaker": "John (PM)",
        "text": "Okay, I think we should schedule a follow-up with benchmarks. Let's go with Redis for MVP.",
        "delay": 3  # Decision!
    },
    {
        "speaker": "command",
        "text": "@Alex what are the risks with this decision?",
        "delay": 3
    },
]


async def run_simulation():
    """Run the meeting simulation."""
    print("\n" + "="*70)
    print("🎬 REAL-TIME AI MEETING ASSISTANT - SIMULATION")
    print("="*70)
    print(f"\n📋 Bot Name: {config.bot.label}")
    print("📋 This simulation demonstrates real-time chat interventions")
    print("="*70 + "\n")
    
    # Initialize components
    context_engine = ContextEngine(
        meeting_id=f"sim_{datetime.now().strftime('%H%M%S')}",
        platform="simulation"
    )
    
    trigger_engine = TriggerEngine()
    llm_service = LLMService()
    
    # Start engines
    await context_engine.start()
    await trigger_engine.start()
    
    # Track stats
    messages_sent = 0
    triggers_fired = 0
    
    try:
        for entry in MEETING_SCRIPT:
            speaker = entry["speaker"]
            text = entry["text"]
            delay = entry.get("delay", 2)
            
            # Check if this is a command
            is_command = speaker == "command" or "@Alex" in text.lower()
            
            if is_command:
                # Commands come from chat
                print(f"\n💬 [CHAT] {text}")
                
                # Check for command trigger
                triggers = await trigger_engine.check_for_triggers(
                    context=context_engine.get_context_snapshot(),
                    chat_text=text
                )
            else:
                # Regular speech
                print(f"\n🎤 {speaker}: \"{text}\"")
                
                # Add to transcript
                chunk = TranscriptChunk(
                    text=text,
                    timestamp=datetime.now(),
                    speaker=speaker.split()[0]  # First name only
                )
                await context_engine.add_transcript_chunk(chunk)
                
                # Small delay for context processing
                await asyncio.sleep(0.3)
                
                # Check for proactive triggers
                triggers = await trigger_engine.check_for_triggers(
                    context=context_engine.get_context_snapshot(),
                    new_chunks=[chunk]
                )
            
            # Process any triggers
            for trigger in triggers:
                triggers_fired += 1
                
                # Get context
                context = context_engine.get_context_snapshot()
                recent_text = context.get_recent_text(minutes=5)
                
                # Generate response
                response = await llm_service.generate_response(
                    trigger=trigger,
                    context=context,
                    recent_transcript=recent_text
                )
                
                if response:
                    messages_sent += 1
                    
                    # Display with visual distinction
                    trigger_label = trigger.type.value.replace("_", " ").title()
                    print(f"\n{'─'*50}")
                    print(f"🤖 {config.bot.label}:")
                    print(f"   {response.text}")
                    print(f"   [Trigger: {trigger_label}]")
                    print(f"{'─'*50}")
            
            # Simulate real-time delay
            await asyncio.sleep(delay)
        
        # Show final stats
        print("\n" + "="*70)
        print("📊 SIMULATION COMPLETE")
        print("="*70)
        print(f"   Total conversation turns: {len(MEETING_SCRIPT)}")
        print(f"   Triggers detected: {triggers_fired}")
        print(f"   AI messages sent: {messages_sent}")
        print(f"   Concerns tracked: {len(context_engine.context.concerns_raised)}")
        print(f"   Decisions tracked: {len(context_engine.context.decisions_pending)}")
        print("="*70 + "\n")
        
        # Show tracked state
        print("📋 TRACKED CONCERNS:")
        for concern in context_engine.context.concerns_raised:
            status = "✅" if concern.addressed else "⚠️"
            print(f"   {status} {concern.description} ({concern.severity.value})")
        
        print("\n📋 TRACKED DECISIONS:")
        for decision in context_engine.context.decisions_pending:
            print(f"   📌 {decision.description} ({decision.status.value})")
        
    finally:
        await context_engine.stop()
        await trigger_engine.stop()


async def run_quick_demo():
    """Run a quick demo without LLM calls."""
    print("\n" + "="*70)
    print("🎬 QUICK DEMO (No LLM required)")
    print("="*70 + "\n")
    
    context_engine = ContextEngine(
        meeting_id="quick_demo",
        platform="demo"
    )
    
    trigger_engine = TriggerEngine()
    
    await context_engine.start()
    
    # Quick conversation
    messages = [
        ("John", "I'm worried about the database performance"),
        ("Sarah", "Let's go with PostgreSQL for the database"),
    ]
    
    for speaker, text in messages:
        chunk = TranscriptChunk(text=text, timestamp=datetime.now(), speaker=speaker)
        await context_engine.add_transcript_chunk(chunk)
        print(f"🎤 {speaker}: \"{text}\"")
        await asyncio.sleep(0.5)
    
    # Wait for processing
    await asyncio.sleep(context_engine.update_interval + 0.5)
    
    # Check triggers
    triggers = await trigger_engine.check_for_triggers(
        context=context_engine.get_context_snapshot()
    )
    
    print(f"\n📊 Concerns detected: {len(context_engine.context.concerns_raised)}")
    print(f"📊 Decisions detected: {len(context_engine.context.decisions_pending)}")
    print(f"📊 Triggers fired: {len(triggers)}")
    
    for trigger in triggers:
        print(f"   ⚡ {trigger.type.value}: {trigger.reason}")
    
    await context_engine.stop()
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run quick demo without LLM")
    args = parser.parse_args()
    
    if args.quick:
        asyncio.run(run_quick_demo())
    else:
        asyncio.run(run_simulation())
