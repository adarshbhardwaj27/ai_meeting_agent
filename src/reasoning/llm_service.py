"""
LLM Reasoning Service for the AI Meeting Assistant.

This service generates short, actionable chat messages based on:
- The trigger that fired
- Recent transcript
- Compressed context

All outputs are constrained to be 1-3 lines, neutral, and easy to ignore.
"""

import asyncio
from typing import Optional, Dict, Any
from datetime import datetime
import structlog

from src.config import config, LLMProvider
from src.context.models import (
    MeetingContext, Trigger, TriggerType, CommandType, AIResponse, MessagePriority
)

logger = structlog.get_logger()


# System prompts for different trigger types
SYSTEM_PROMPTS = {
    TriggerType.EXPLICIT_COMMAND: """You are {bot_name}, a silent AI assistant in a live meeting chat.
A participant asked you a direct question. Respond helpfully but VERY briefly.

RULES:
- Maximum 2-3 short lines
- Be direct and factual
- No fluff or preamble
- No "I think" or "I suggest"
- Use the meeting context provided
- If you don't have enough info, say so briefly""",

    TriggerType.CONFLICT_DETECTED: """You are {bot_name}, a silent AI assistant observing a live meeting.
You noticed a potential conflict between what's being discussed now and an earlier concern.

RULES:
- One line pointing out the potential conflict
- Be neutral, not alarming
- Frame as observation, not warning
- Make it easy to ignore if not relevant
- No "I noticed" or "I observed" - just state it""",

    TriggerType.DECISION_FORMING: """You are {bot_name}, a silent AI assistant observing a live meeting.
The team appears to be making a decision. If there's useful context to add, do so briefly.

RULES:
- One line max unless there's critical context
- Only speak if you have genuinely useful input
- If the decision seems fine, say nothing (return SKIP)
- No "I think" or "consider" - just factual observation""",

    TriggerType.RISK_OVERLOOKED: """You are {bot_name}, a silent AI assistant observing a live meeting.
An important risk mentioned earlier seems to be getting overlooked.

RULES:
- One line gentle reminder
- Reference who raised it if known
- Frame as "earlier mentioned" not "you forgot"
- Be easy to dismiss if they've already considered it""",

    TriggerType.TOPIC_DRIFT: """You are {bot_name}, a silent AI assistant observing a live meeting.
The conversation seems to have drifted from the main topic.

RULES:
- Only mention if drift seems unintentional
- One line noting the original topic
- Very gentle - they may have drifted intentionally
- Return SKIP if the tangent seems valuable""",

    TriggerType.REPEATED_CONFUSION: """You are {bot_name}, a silent AI assistant observing a live meeting.
There seems to be repeated confusion about something.

RULES:
- Brief clarification if you can provide one
- Point out the confusion pattern if you can't clarify
- One line max
- If confusion is complex, suggest they pause to align""",

    TriggerType.CIRCULAR_DISCUSSION: """You are {bot_name}, a silent AI assistant observing a live meeting.
The team seems to be going in circles on a topic.

RULES:
- Gently note that this has been discussed for a while
- Optionally summarize the main positions in one line
- Suggest a decision approach only if asked
- Very light touch - they may be working through it""",

    TriggerType.CONTRADICTION: """You are {bot_name}, a silent AI assistant observing a live meeting.
You noticed that a current statement appears to contradict something said earlier.

RULES:
- One line quoting both statements neutrally
- Frame as "Earlier [speaker] mentioned X, just now [speaker] said Y — worth clarifying?"
- Be genuinely helpful, not gotcha
- If the contradiction is trivial or self-correction, return SKIP
- Never take sides""",

    TriggerType.MEMORY_GAP: """You are {bot_name}, a silent AI assistant observing a live meeting.
Someone is trying to recall something that was discussed earlier, and you have the answer in your context.

RULES:
- One line with the specific fact they're trying to remember
- Attribute it: "Earlier [speaker] mentioned …"
- Be a helpful note-taker, not a know-it-all
- If you're not confident in the match, return SKIP""",

    TriggerType.FACTUAL_INCONSISTENCY: """You are {bot_name}, a silent AI assistant observing a live meeting.
Two statements contain different numbers or facts about the same topic.

RULES:
- One line noting both figures neutrally
- Frame as "Just checking — earlier the figure was X, now it's Y"
- Do NOT say which is correct
- If the difference is insignificant or clearly a rounding, return SKIP""",

    TriggerType.COLLABORATION_OPPORTUNITY: """You are {bot_name}, a silent AI assistant observing a live meeting.
The current discussion is directly related to an earlier concern, decision, or question that hasn't been addressed yet.

RULES:
- One line connecting the current discussion to the earlier point
- Frame as "This may relate to [speaker]'s earlier point about …"
- Only speak if the connection is genuinely useful
- If it's obvious they already know, return SKIP""",

    TriggerType.VOICE_ADDRESS: """You are {bot_name}, an AI assistant in a live meeting.
A participant just said your name out loud and asked you something directly.
This is a voice question — they are speaking to YOU, not typing in chat.

RULES:
- Respond as if someone at the table asked you a question
- Be concise (2-3 lines max) but genuinely helpful
- Use the full meeting context to give the best answer
- Address the speaker by name if known
- Be natural and conversational — this is a spoken interaction, your reply appears in chat
- If they asked a known command (summarize, risks, etc.) follow that command's format""",
}

COMMAND_PROMPTS = {
    CommandType.SUMMARIZE: """Provide a 2-3 line summary of what has been discussed so far.
Include: main topics, key concerns raised, and any decisions (tentative or confirmed).
Be factual and brief.""",

    CommandType.RISKS: """List the main risks/concerns that have been raised in 2-3 lines.
Note which ones are still open vs addressed.
Prioritize high-severity concerns.""",

    CommandType.EXPLAIN: """Explain the requested term or concept in 1-2 lines.
Use simple language appropriate for the meeting context.
If you don't know, say so briefly.""",

    CommandType.DECISION: """In 1-2 lines, state what decision the team seems to be leaning toward.
Note if it's tentative or confirmed.
Mention any open concerns about this decision.""",

    CommandType.ACTION_ITEMS: """List any action items or next steps mentioned in 2-3 lines.
Include who was assigned if mentioned.
Be brief - just the key tasks.""",

    CommandType.CUSTOM: """The user asked a custom question. Answer helpfully in 2-3 lines max.
Use the meeting context to inform your answer.
If the question is unclear, ask for clarification briefly.""",
}


class LLMService:
    """
    Service for generating AI chat messages using LLM.
    
    Supports OpenAI, Anthropic, and Azure OpenAI backends.
    All responses are constrained to be brief and meeting-appropriate.
    """
    
    def __init__(self):
        self.provider = config.llm.provider
        self.bot_name = config.bot.name
        self.max_chars = config.chat.max_chars
        self.max_lines = config.chat.max_lines
        
        # Initialize the appropriate client
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the LLM client based on configured provider."""
        if self.provider == LLMProvider.OPENAI:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=config.llm.openai_api_key)
                self._model = config.llm.openai_model
                logger.info("llm_initialized", provider="openai", model=self._model)
            except ImportError:
                logger.warning("openai_not_installed")
                
        elif self.provider == LLMProvider.ANTHROPIC:
            try:
                from anthropic import AsyncAnthropic
                self._client = AsyncAnthropic(api_key=config.llm.anthropic_api_key)
                self._model = config.llm.anthropic_model
                logger.info("llm_initialized", provider="anthropic", model=self._model)
            except ImportError:
                logger.warning("anthropic_not_installed")
                
        elif self.provider == LLMProvider.AZURE_OPENAI:
            try:
                from openai import AsyncAzureOpenAI
                self._client = AsyncAzureOpenAI(
                    api_key=config.llm.azure_openai_api_key,
                    azure_endpoint=config.llm.azure_openai_endpoint,
                    api_version="2024-02-15-preview"
                )
                self._model = config.llm.azure_openai_deployment
                logger.info("llm_initialized", provider="azure_openai", model=self._model)
            except ImportError:
                logger.warning("openai_not_installed")
    
    async def generate_response(
        self,
        trigger: Trigger,
        context: MeetingContext,
        recent_transcript: str
    ) -> Optional[AIResponse]:
        """
        Generate a chat response for the given trigger and context.
        
        Args:
            trigger: The trigger that fired
            context: Current meeting context
            recent_transcript: Recent transcript text (last 2-3 minutes)
        
        Returns:
            AIResponse if a message should be sent, None if AI decides to skip
        """
        # Build the prompt
        system_prompt = self._build_system_prompt(trigger)
        user_prompt = self._build_user_prompt(trigger, context, recent_transcript)
        
        # Call LLM
        response_text = await self._call_llm(system_prompt, user_prompt)
        
        if response_text is None:
            return None
        
        # Check for SKIP response
        if response_text.strip().upper() == "SKIP" or not response_text.strip():
            logger.info("llm_chose_to_skip", trigger=trigger.type.value)
            return None
        
        # Enforce constraints
        response_text = self._enforce_constraints(response_text)
        
        # Determine priority
        priority = self._determine_priority(trigger)
        
        # Create response
        response = AIResponse.create(
            text=response_text,
            trigger=trigger,
            priority=priority
        )
        
        logger.info(
            "response_generated",
            trigger=trigger.type.value,
            length=len(response_text),
            priority=priority.value
        )
        
        return response
    
    def _build_system_prompt(self, trigger: Trigger) -> str:
        """Build the system prompt for the trigger type."""
        base_prompt = SYSTEM_PROMPTS.get(
            trigger.type, 
            SYSTEM_PROMPTS[TriggerType.EXPLICIT_COMMAND]
        )
        
        return base_prompt.format(bot_name=self.bot_name)
    
    def _build_user_prompt(
        self, 
        trigger: Trigger, 
        context: MeetingContext,
        recent_transcript: str
    ) -> str:
        """Build the user prompt with context and trigger info."""
        parts = []
        
        # Meeting context
        parts.append("=== MEETING CONTEXT ===")
        parts.append(f"Duration: {int(context.meeting_duration.total_seconds() / 60)} minutes")
        parts.append(f"Participants: {', '.join(context.participants) or 'Unknown'}")
        
        if context.current_topic:
            parts.append(f"Current Topic: {context.current_topic.name}")
        
        if context.unaddressed_concerns:
            concerns = [f"- {c.description} ({c.severity.value})" 
                       for c in context.unaddressed_concerns[:3]]
            parts.append(f"Open Concerns:\n" + "\n".join(concerns))
        
        if context.decisions_pending:
            decisions = [f"- {d.description} ({d.status.value})" 
                        for d in context.decisions_pending[:3]]
            parts.append(f"Pending Decisions:\n" + "\n".join(decisions))
        
        if context.historical_summary:
            parts.append(f"\nHistorical Summary:\n{context.historical_summary[:500]}")
        
        # Recent transcript
        parts.append("\n=== RECENT CONVERSATION (last 2-3 min) ===")
        parts.append(recent_transcript[-1500:])  # Limit transcript length
        
        # Trigger information
        parts.append("\n=== TRIGGER ===")
        parts.append(f"Type: {trigger.type.value}")
        parts.append(f"Reason: {trigger.reason}")
        
        if trigger.conflicting_concern:
            parts.append(f"Related Concern: {trigger.conflicting_concern.description}")
        
        if trigger.conflicting_decision:
            parts.append(f"Related Decision: {trigger.conflicting_decision.description}")

        # Signal evidence (from Tier-1 intelligence layer)
        if trigger.signal_evidence_a or trigger.signal_evidence_b:
            parts.append("\n=== SIGNAL EVIDENCE ===")
            if trigger.signal_evidence_a:
                speaker_a = trigger.signal_speaker_a or "Someone"
                parts.append(f"Earlier ({speaker_a}): \"{trigger.signal_evidence_a}\"")
            if trigger.signal_evidence_b:
                speaker_b = trigger.signal_speaker_b or "Someone"
                parts.append(f"Now ({speaker_b}): \"{trigger.signal_evidence_b}\"")
        
        # Command-specific instructions
        if trigger.type in (TriggerType.EXPLICIT_COMMAND, TriggerType.VOICE_ADDRESS) and trigger.command_type:
            cmd_prompt = COMMAND_PROMPTS.get(
                trigger.command_type, 
                COMMAND_PROMPTS[CommandType.CUSTOM]
            )
            parts.append(f"\n=== TASK ===\n{cmd_prompt}")
            
            if trigger.command_args:
                parts.append(f"User specified: {trigger.command_args}")
        else:
            parts.append("\n=== TASK ===")
            parts.append("Generate a brief chat message (1-3 lines max) based on the trigger.")
            parts.append("If no message is needed, respond with just: SKIP")
        
        return "\n".join(parts)
    
    async def _call_llm(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Call the LLM and return the response text."""
        if self._client is None:
            logger.error("llm_client_not_initialized")
            return self._fallback_response()
        
        try:
            if self.provider in (LLMProvider.OPENAI, LLMProvider.AZURE_OPENAI):
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=150,  # Keep responses short
                    temperature=0.7,
                )
                return response.choices[0].message.content
                
            elif self.provider == LLMProvider.ANTHROPIC:
                response = await self._client.messages.create(
                    model=self._model,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    max_tokens=150,
                )
                return response.content[0].text
                
        except Exception as e:
            logger.error("llm_call_failed", error=str(e))
            return None
    
    def _fallback_response(self) -> str:
        """Generate a fallback response when LLM is unavailable."""
        return "I'm having trouble processing right now. Please try again."
    
    def _enforce_constraints(self, text: str) -> str:
        """Enforce message length and formatting constraints."""
        # Remove any markdown formatting
        text = text.replace("**", "").replace("*", "")
        text = text.replace("```", "").replace("`", "")
        
        # Remove any bullet points
        lines = text.strip().split("\n")
        lines = [line.lstrip("•-*").strip() for line in lines]
        
        # Limit number of lines
        if len(lines) > self.max_lines:
            lines = lines[:self.max_lines]
        
        text = " | ".join(line for line in lines if line)
        
        # Limit total characters
        if len(text) > self.max_chars:
            text = text[:self.max_chars - 3] + "..."
        
        return text.strip()
    
    def _determine_priority(self, trigger: Trigger) -> MessagePriority:
        """Determine the message priority based on trigger type."""
        priority_map = {
            TriggerType.EXPLICIT_COMMAND: MessagePriority.HIGH,
            TriggerType.VOICE_ADDRESS: MessagePriority.HIGH,
            TriggerType.CONFLICT_DETECTED: MessagePriority.HIGH,
            TriggerType.CONTRADICTION: MessagePriority.HIGH,
            TriggerType.FACTUAL_INCONSISTENCY: MessagePriority.HIGH,
            TriggerType.MEMORY_GAP: MessagePriority.MEDIUM,
            TriggerType.RISK_OVERLOOKED: MessagePriority.MEDIUM,
            TriggerType.DECISION_FORMING: MessagePriority.MEDIUM,
            TriggerType.CIRCULAR_DISCUSSION: MessagePriority.MEDIUM,
            TriggerType.COLLABORATION_OPPORTUNITY: MessagePriority.MEDIUM,
            TriggerType.REPEATED_CONFUSION: MessagePriority.LOW,
            TriggerType.TOPIC_DRIFT: MessagePriority.LOW,
            TriggerType.COMPLEX_CONCEPT: MessagePriority.LOW,
        }
        return priority_map.get(trigger.type, MessagePriority.MEDIUM)
    
    async def test_connection(self) -> bool:
        """Test if the LLM connection is working."""
        try:
            response = await self._call_llm(
                "You are a test assistant.",
                "Respond with just: OK"
            )
            return response is not None and "OK" in response.upper()
        except Exception as e:
            logger.error("llm_connection_test_failed", error=str(e))
            return False
