"""
LLM Reasoning Service for the AI Meeting Assistant.

This service generates short, actionable chat messages based on:
- The trigger that fired
- Recent transcript
- Compressed context

All outputs are constrained to be 1-3 lines, neutral, and easy to ignore.
"""

import asyncio
from dataclasses import dataclass
import re
from typing import Optional, Dict, Any, List
from datetime import datetime
import structlog

from src.config import config, LLMProvider
from src.context.models import (
    MeetingContext, Trigger, TriggerType, CommandType, AIResponse, MessagePriority,
    TranscriptChunk,
)

logger = structlog.get_logger()


@dataclass(frozen=True)
class TriggerAdjudication:
    """Single-call LLM decision about whether and how a proactive trigger should respond."""

    action: str
    final_trigger_type: TriggerType
    reason: str
    response_text: str


_ADJUDICATED_PROACTIVE_TRIGGER_TYPES = frozenset({
    TriggerType.DECISION_FORMING,
    TriggerType.CONFLICT_DETECTED,
    TriggerType.TOPIC_DRIFT,
    TriggerType.REPEATED_CONFUSION,
    TriggerType.CONTRADICTION,
    TriggerType.FACTUAL_INCONSISTENCY,
    TriggerType.COLLABORATION_OPPORTUNITY,
})

_ADJUDICATION_OUTPUT_TRIGGER_TYPES = tuple(
    trigger_type
    for trigger_type in TriggerType
    if trigger_type not in {TriggerType.EXPLICIT_COMMAND, TriggerType.VOICE_ADDRESS}
)


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
- One line max
- Tie the current proposal to the earlier concern or tradeoff
- Prefer a corrective nudge: what should be checked, confirmed, or decided next
- Be neutral, not alarming
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
- Otherwise ask the single most useful alignment question or propose a short recap
- Do not merely say people are confused
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
- One line max
- Prefer a resolution-oriented message: identify the mismatch and ask for the exact confirmation or update needed
- If context clearly indicates the current or more reliable answer, state it briefly
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
- One line max
- Prefer resolving the mismatch: cite both figures and ask for the source-of-truth confirmation
- If context clearly identifies the confirmed value or source, state it briefly
- Do not merely point out that the numbers differ
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
- Keep it to 1-2 short spoken sentences, ideally under 160 characters
- Use the full meeting context to give the best answer
- Address the speaker by name if known
- Be natural and conversational — this is a spoken interaction and your reply may be spoken aloud
- Do not ask a follow-up question unless it is absolutely necessary
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
        self._summary_model = (config.llm.context_summary_model or "").strip() or None
        
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
        fast_response_text = self._build_fast_voice_response(
            trigger=trigger,
            context=context,
            recent_transcript=recent_transcript,
        )
        if fast_response_text is not None:
            response_text = fast_response_text
        else:
            adjudication = await self._adjudicate_trigger(
                trigger=trigger,
                context=context,
                recent_transcript=recent_transcript,
            )

            if adjudication is not None:
                trigger.adjudication_action = adjudication.action
                trigger.adjudication_reason = adjudication.reason or None

                logger.info(
                    "trigger_adjudicated",
                    proposed_trigger=(trigger.proposed_type or trigger.type).value,
                    final_trigger=adjudication.final_trigger_type.value,
                    action=adjudication.action,
                )

                if adjudication.action == "skip":
                    logger.info(
                        "llm_chose_to_skip",
                        trigger=trigger.type.value,
                        adjudication_reason=trigger.adjudication_reason,
                    )
                    return None

                self._apply_adjudicated_trigger_type(trigger, adjudication.final_trigger_type)
                response_text = adjudication.response_text
            else:
                # Build the prompt
                system_prompt = self._build_system_prompt(trigger)
                user_prompt = self._build_user_prompt(trigger, context, recent_transcript)

                # Call LLM
                response_text = await self._call_llm(
                    system_prompt,
                    user_prompt,
                    trigger=trigger,
                )

            if response_text is None:
                response_text = self._build_local_fallback_response(
                    trigger=trigger,
                    context=context,
                    recent_transcript=recent_transcript,
                )
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

    async def _adjudicate_trigger(
        self,
        trigger: Trigger,
        context: MeetingContext,
        recent_transcript: str,
    ) -> Optional[TriggerAdjudication]:
        """Use one LLM call to skip, confirm/retype, and generate the final response."""
        if not self._should_adjudicate_trigger(trigger):
            return None

        system_prompt = self._build_trigger_adjudication_system_prompt()
        user_prompt = self._build_trigger_adjudication_user_prompt(
            trigger=trigger,
            context=context,
            recent_transcript=recent_transcript,
        )
        response_text = await self._call_llm(
            system_prompt,
            user_prompt,
            trigger=trigger,
            max_tokens_override=220,
            temperature=0.2,
        )
        return self._parse_trigger_adjudication(response_text, trigger.type)

    async def summarize_context(
        self,
        existing_summary: str,
        chunks: List[TranscriptChunk],
    ) -> Optional[str]:
        """Compress older transcript into a persistent structured meeting memory."""
        if self._client is None or not chunks:
            return None

        transcript_lines = self._prepare_summary_transcript_lines(chunks)
        if not transcript_lines:
            return None

        system_prompt = (
            "You compress meeting history into a clean persistent memory for later LLM use.\n"
            "Rules:\n"
            "- Remove filler, greetings, wake words, duplicate transcript fragments, and ASR noise\n"
            "- Keep only durable facts: topic, options compared, decisions, open questions, risks, and action items\n"
            "- Preserve uncertainty; if no decision is made, say that explicitly\n"
            "- Do not invent details not present in the transcript or prior memory\n"
            "- Output plain text with exactly these five lines:\n"
            "Summary: ...\n"
            "Decisions: ...\n"
            "Open Questions: ...\n"
            "Risks: ...\n"
            "Action Items: ...\n"
            "- Use 'None' if a section has nothing meaningful\n"
            "- Keep the full output under 900 characters"
        )

        joined_transcript = "\n".join(transcript_lines)
        user_prompt = (
            "Existing persistent meeting memory:\n"
            f"{(existing_summary or 'None')[:1200]}\n\n"
            "New older transcript to merge:\n"
            f"{joined_transcript}\n\n"
            "Rewrite the persistent meeting memory now."
        )

        response_text = await self._call_llm(
            system_prompt,
            user_prompt,
            max_tokens_override=260,
            temperature=0.2,
            model_override=self._summary_model,
        )
        return self._sanitize_context_summary(response_text)

    async def audit_summary_for_factual_errors(
        self,
        existing_summary: str,
        chunks: List[TranscriptChunk],
    ) -> Optional[Trigger]:
        """Use the summary-time LLM pass to catch factual issues worth surfacing."""
        if self._client is None or not chunks:
            return None

        transcript_lines = self._prepare_summary_transcript_lines(chunks)
        if not transcript_lines:
            return None

        system_prompt = (
            "You audit meeting memory for material factual inconsistencies worth surfacing as a proactive trigger.\n"
            "Flag issues like:\n"
            "- obvious arithmetic or factual errors in a statement\n"
            "- materially conflicting figures or facts about the same topic\n"
            "Do not flag speculation, opinion differences, tentative estimates, or clean self-corrections.\n"
            "Output plain text with exactly these six lines:\n"
            "Trigger: yes|no\n"
            "Confidence: 0.00 to 1.00\n"
            "Reason: ...\n"
            "Evidence A: ...\n"
            "Evidence B: ...\n"
            "Relevant Transcript: ...\n"
            "If there is no worthwhile inconsistency, output Trigger: no and use None for the other fields."
        )

        joined_transcript = "\n".join(transcript_lines)
        user_prompt = (
            "Existing persistent meeting memory:\n"
            f"{(existing_summary or 'None')[:1200]}\n\n"
            "New older transcript under review:\n"
            f"{joined_transcript}\n\n"
            "Should a factual_inconsistency trigger be raised now?"
        )

        response_text = await self._call_llm(
            system_prompt,
            user_prompt,
            max_tokens_override=180,
            temperature=0.1,
            model_override=self._summary_model,
        )
        return self._parse_summary_factual_audit(response_text)

    def _build_fast_voice_response(
        self,
        trigger: Trigger,
        context: MeetingContext,
        recent_transcript: str,
    ) -> Optional[str]:
        """Return low-latency local responses for common spoken interactions."""
        if trigger.type != TriggerType.VOICE_ADDRESS:
            return None

        speaker_name = (trigger.speaker or "").strip()

        if trigger.command_type in {
            CommandType.SUMMARIZE,
            CommandType.RISKS,
            CommandType.DECISION,
            CommandType.ACTION_ITEMS,
        }:
            logger.info(
                "voice_fast_path_used",
                command=(trigger.command_type.value if trigger.command_type else "custom"),
            )
            if trigger.command_type == CommandType.SUMMARIZE:
                return self._fallback_spoken_summary(context, recent_transcript)
            return self._build_local_fallback_response(
                trigger=trigger,
                context=context,
                recent_transcript=recent_transcript,
            )

        request_text = (trigger.relevant_transcript or trigger.command_args or "").lower()
        normalized_text = re.sub(r"[^a-z0-9\s?]", " ", request_text)
        normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

        if speaker_name and normalized_text in {"what is my name", "what s my name", "who am i"}:
            logger.info("voice_smalltalk_fast_path", kind="identity")
            return f"Your name is {speaker_name}."

        if "how are you" in normalized_text:
            logger.info("voice_smalltalk_fast_path", kind="how_are_you")
            return "I'm doing well. How can I help?"

        if "are you there" in normalized_text or "can you hear me" in normalized_text:
            logger.info("voice_smalltalk_fast_path", kind="presence_check")
            return "Yes, I'm here and listening."

        return None

    def _should_adjudicate_trigger(self, trigger: Trigger) -> bool:
        """Only adjudicate the proactive trigger classes that are currently noisy."""
        if self._client is None:
            return False

        return trigger.type in _ADJUDICATED_PROACTIVE_TRIGGER_TYPES

    @staticmethod
    def _apply_adjudicated_trigger_type(trigger: Trigger, final_trigger_type: TriggerType) -> None:
        """Persist the LLM-confirmed final trigger type on the live trigger object."""
        if final_trigger_type != trigger.type:
            trigger.proposed_type = trigger.proposed_type or trigger.type
            trigger.type = final_trigger_type
            trigger.adjudication_action = "retyped"
        else:
            trigger.adjudication_action = "confirmed"

    def _build_trigger_adjudication_system_prompt(self) -> str:
        """Prompt for single-call proactive trigger adjudication and response generation."""
        allowed_types = ", ".join(trigger_type.value for trigger_type in _ADJUDICATION_OUTPUT_TRIGGER_TYPES)
        return (
            "You review a candidate proactive trigger for a live meeting assistant.\n"
            "Decide whether the assistant should speak, and if so which proactive trigger type fits best.\n"
            "Allowed final trigger types: "
            f"{allowed_types}.\n"
            "Never output explicit_command or voice_address.\n"
            "If the moment is weak, redundant, already being handled by the team, or not worth interrupting, choose skip.\n"
            "If the evidence is concrete and directly useful, prefer a short response over skipping.\n"
            "Examples of concrete value: a real conflict with an earlier concern, repeated confusion that needs clarification, or a clear numeric mismatch.\n"
            "If you respond, the response must be 1-3 short lines, neutral, factual, and easy to ignore.\n"
            "For contradiction, factual_inconsistency, repeated_confusion, and conflict_detected, do not merely describe the problem back to the room.\n"
            "Prefer a corrective next step: clarify the exact point, ask for the source of truth, confirm the current decision, or provide the missing fact if the context already makes it clear.\n"
            "Bad: John and Sarah have conflicting statements about the deployment deadline.\n"
            "Better: Earlier John said Tuesday and Sarah said Thursday. Can we confirm the current deployment deadline and update the ticket?\n"
            "Bad: There seems to be a discrepancy in the API capacity statements.\n"
            "Better: Earlier we used 500 rps, now 200 rps was mentioned. Can we confirm the source-of-truth limit before sizing the rollout?\n"
            "Output exactly this format:\n"
            "Action: respond|skip\n"
            "Final Trigger: <one allowed trigger type>\n"
            "Reason: <one short sentence>\n"
            "Response:\n"
            "<assistant message, or NONE if Action: skip>"
        )

    @staticmethod
    def _build_trigger_response_goal(trigger_type: TriggerType) -> str:
        """Describe what a useful response should accomplish for this trigger."""
        goals = {
            TriggerType.CONFLICT_DETECTED: (
                "Move the discussion forward by connecting the current proposal to the earlier concern "
                "and naming the specific check, tradeoff, or decision the team should resolve now."
            ),
            TriggerType.DECISION_FORMING: (
                "Help the team choose by naming the key tradeoff, decision frame, or next decision step that will unblock them."
            ),
            TriggerType.TOPIC_DRIFT: (
                "Only redirect if it adds value; otherwise skip. If you respond, gently point back to the active topic or blocker."
            ),
            TriggerType.REPEATED_CONFUSION: (
                "Clarify the point directly if the answer is available. Otherwise ask the crispest possible alignment question. "
                "Do not just say the team is confused."
            ),
            TriggerType.CONTRADICTION: (
                "Resolve the mismatch. If the current or authoritative answer is evident from context, state it briefly. "
                "Otherwise ask for the precise confirmation or update needed. Do not just say the statements conflict."
            ),
            TriggerType.FACTUAL_INCONSISTENCY: (
                "Resolve the numeric or factual mismatch. If a source-of-truth value is clear from context, state it. "
                "Otherwise ask for source-of-truth confirmation. Do not just point out the inconsistency."
            ),
            TriggerType.COLLABORATION_OPPORTUNITY: (
                "Connect the current thread to the overlooked concern, question, or dependency and suggest the practical follow-up."
            ),
            TriggerType.RISK_OVERLOOKED: (
                "Re-surface the missing risk in a way that prompts a concrete check, mitigation, or owner discussion."
            ),
            TriggerType.CIRCULAR_DISCUSSION: (
                "Summarize the stuck choice or suggest a lightweight decision frame only if it helps break the loop."
            ),
            TriggerType.MEMORY_GAP: (
                "Answer the recall gap directly with the specific fact the team is trying to remember."
            ),
        }
        return goals.get(
            trigger_type,
            "Add the most useful missing context in the least disruptive way.",
        )

    def _build_trigger_adjudication_user_prompt(
        self,
        trigger: Trigger,
        context: MeetingContext,
        recent_transcript: str,
    ) -> str:
        """Build the adjudication payload with trigger evidence and current meeting state."""
        parts = []
        parts.append("=== CANDIDATE TRIGGER ===")
        parts.append(f"Proposed Trigger: {trigger.type.value}")
        parts.append(f"Confidence: {trigger.confidence:.2f}")
        parts.append(f"Reason: {trigger.reason}")
        parts.append("\n=== RESPONSE GOAL ===")
        parts.append(self._build_trigger_response_goal(trigger.type))

        if trigger.signal_evidence_a or trigger.signal_evidence_b:
            parts.append("\n=== SIGNAL EVIDENCE ===")
            if trigger.signal_evidence_a:
                speaker_a = trigger.signal_speaker_a or "Someone"
                parts.append(f"Earlier ({speaker_a}): {trigger.signal_evidence_a}")
            if trigger.signal_evidence_b:
                speaker_b = trigger.signal_speaker_b or "Someone"
                parts.append(f"Now ({speaker_b}): {trigger.signal_evidence_b}")

        if trigger.relevant_transcript:
            parts.append("\n=== RELEVANT TRANSCRIPT ===")
            parts.append(trigger.relevant_transcript[-600:])

        parts.append("\n=== MEETING CONTEXT ===")
        parts.append(f"Duration: {int(context.meeting_duration.total_seconds() / 60)} minutes")
        parts.append(f"Participants: {', '.join(context.participants) or 'Unknown'}")
        if context.current_topic:
            parts.append(f"Current Topic: {context.current_topic.name}")
        if context.unaddressed_concerns:
            parts.append(
                "Open Concerns:\n" + "\n".join(
                    f"- {concern.description} ({concern.severity.value})"
                    for concern in context.unaddressed_concerns[:3]
                )
            )
        if context.decisions_pending:
            parts.append(
                "Pending Decisions:\n" + "\n".join(
                    f"- {decision.description} ({decision.status.value})"
                    for decision in context.decisions_pending[:3]
                )
            )
        if context.open_questions:
            parts.append("Open Questions:\n" + "\n".join(f"- {question}" for question in context.open_questions[:3]))
        if context.action_items:
            parts.append("Action Items:\n" + "\n".join(f"- {item}" for item in context.action_items[:3]))
        if context.historical_summary:
            parts.append(f"Historical Summary:\n{context.historical_summary[:500]}")

        parts.append("\n=== RECENT CONVERSATION ===")
        parts.append(recent_transcript[-1800:] or "None")
        return "\n".join(parts)
    
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

        if context.open_questions:
            questions = [f"- {question}" for question in context.open_questions[:3]]
            parts.append(f"Open Questions:\n" + "\n".join(questions))

        if context.action_items:
            action_items = [f"- {item}" for item in context.action_items[:3]]
            parts.append(f"Action Items:\n" + "\n".join(action_items))
        
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
            parts.append(self._build_trigger_response_goal(trigger.type))
            parts.append("If no message is needed, respond with just: SKIP")
        
        return "\n".join(parts)
    
    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        trigger: Optional[Trigger] = None,
        max_tokens_override: Optional[int] = None,
        temperature: float = 0.7,
        model_override: Optional[str] = None,
    ) -> Optional[str]:
        """Call the LLM and return the response text."""
        if self._client is None:
            logger.warning("llm_client_not_initialized")
            return None

        max_tokens = max_tokens_override or (
            80 if trigger and trigger.type == TriggerType.VOICE_ADDRESS else 150
        )
        model_name = model_override or self._model

        # Tighter deadline for voice/commands (user is waiting); generous for
        # proactive calls.  This is defense-in-depth on top of the dispatcher's
        # outer per-tier deadline — it guarantees a single provider call can
        # never hang indefinitely regardless of call site.
        llm_timeout = (
            8.0 if trigger and trigger.type == TriggerType.VOICE_ADDRESS else 25.0
        )

        try:
            if self.provider in (LLMProvider.OPENAI, LLMProvider.AZURE_OPENAI):
                response = await asyncio.wait_for(
                    self._client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    ),
                    timeout=llm_timeout,
                )
                return response.choices[0].message.content

            elif self.provider == LLMProvider.ANTHROPIC:
                response = await asyncio.wait_for(
                    self._client.messages.create(
                        model=model_name,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    ),
                    timeout=llm_timeout,
                )
                return response.content[0].text

        except asyncio.TimeoutError:
            logger.warning(
                "llm_call_timeout",
                provider=self.provider.value,
                timeout_s=llm_timeout,
                trigger_type=trigger.type.value if trigger else None,
            )
            return None
        except Exception as e:
            logger.error("llm_call_failed", error=str(e))
            return None

    def _build_local_fallback_response(
        self,
        trigger: Trigger,
        context: MeetingContext,
        recent_transcript: str,
    ) -> Optional[str]:
        """Build a deterministic fallback when the LLM is unavailable.

        For proactive triggers, silence is safer than a generic low-confidence
        reply. For direct user requests, respond from local structured context
        whenever possible.
        """
        if trigger.type not in (TriggerType.EXPLICIT_COMMAND, TriggerType.VOICE_ADDRESS):
            logger.info("llm_unavailable_skipping_proactive", trigger=trigger.type.value)
            return None

        command_type = trigger.command_type or CommandType.CUSTOM

        if command_type == CommandType.SUMMARIZE:
            return self._fallback_summary(context, recent_transcript)
        if command_type == CommandType.RISKS:
            return self._fallback_risks(context)
        if command_type == CommandType.DECISION:
            return self._fallback_decision(context)
        if command_type == CommandType.ACTION_ITEMS:
            return self._fallback_action_items(context)
        if command_type in (CommandType.EXPLAIN, CommandType.CUSTOM):
            request_text = (trigger.command_args or trigger.relevant_transcript or "that").strip()
            if len(request_text) > 80:
                request_text = request_text[:77] + "..."
            return (
                f"I heard the question about '{request_text}', but detailed Q&A isn't available right now. "
                f"I can still help with summaries, risks, decisions, and action items."
            )

        return "I don't have a reliable local answer for that yet."

    def _fallback_summary(self, context: MeetingContext, recent_transcript: str) -> str:
        """Build a concise summary from local structured context."""
        parts = []
        topic = self._select_summary_topic(context, recent_transcript)

        if topic:
            parts.append(f"Topic: {topic}")

        decisions = context.decisions_confirmed[-1:] + context.decisions_pending[:1]
        if decisions:
            decision_text = "; ".join(d.description for d in decisions[:2])
            parts.append(f"Decisions: {decision_text}")

        if context.unaddressed_concerns:
            concerns = "; ".join(c.description for c in context.unaddressed_concerns[:2])
            parts.append(f"Concerns: {concerns}")

        if context.action_items:
            parts.append(f"Next: {context.action_items[0]}")

        if not parts and recent_transcript.strip():
            discussion_hint = self._infer_topic_from_text(recent_transcript)
            if discussion_hint:
                parts.append(f"Recent discussion: {discussion_hint}")
            else:
                snippet = recent_transcript.strip()
                if len(snippet) > 180:
                    snippet = snippet[-180:]
                parts.append(f"Recent discussion: {snippet}")

        return "\n".join(parts[:3]) or "No strong meeting context captured yet."

    def _fallback_spoken_summary(self, context: MeetingContext, recent_transcript: str) -> str:
        """Build a short, spoken-friendly summary for voice requests."""
        topic = self._select_summary_topic(context, recent_transcript)
        concern = self._select_primary_concern(context, recent_transcript)
        decision = self._select_primary_decision(context)
        no_decision = self._infer_no_decision_clause(recent_transcript) if decision is None else None

        sentences = []
        if topic:
            sentences.append(f"So far, the discussion is about {topic}.")

        detail_clauses = []
        if decision:
            detail_clauses.append(f"a likely direction is {decision}")
        elif no_decision:
            detail_clauses.append(no_decision)

        if concern:
            detail_clauses.append(f"the main concern is {concern}")
        elif not detail_clauses and context.action_items:
            detail_clauses.append(f"the next step is {self._to_sentence_fragment(context.action_items[0])}")

        if detail_clauses:
            sentence = self._join_spoken_clauses(detail_clauses)
            if sentence:
                sentences.append(sentence)

        if not sentences:
            discussion_hint = self._infer_topic_from_text(recent_transcript)
            if discussion_hint:
                return f"So far, the discussion is about {discussion_hint}."
            return "I do not have a clean summary yet."

        return " ".join(sentences[:2])

    def _select_summary_topic(self, context: MeetingContext, recent_transcript: str) -> Optional[str]:
        """Pick the cleanest topic label available for summaries."""
        if context.current_topic and context.current_topic.name:
            cleaned_topic = self._clean_topic_candidate(context.current_topic.name)
            if cleaned_topic:
                return cleaned_topic

        return self._infer_topic_from_text(recent_transcript)

    def _clean_topic_candidate(self, topic: str) -> Optional[str]:
        """Normalize a topic and reject noisy transcript-like labels."""
        cleaned = " ".join((topic or "").split()).strip(" ,.!?;:")
        if not cleaned:
            return None

        lowered = cleaned.lower()
        if self._looks_like_noisy_topic(lowered):
            return self._infer_topic_from_text(cleaned)

        return lowered

    def _looks_like_noisy_topic(self, text: str) -> bool:
        """Heuristic for transcript fragments that should not be spoken as a topic."""
        tokens = [re.sub(r"[^a-z0-9]+", "", token) for token in text.split()]
        tokens = [token for token in tokens if token]
        if not tokens:
            return True

        question_like = {"which", "what", "how", "who", "why", "should", "could"}
        unique_ratio = len(set(tokens)) / len(tokens)
        return (
            len(tokens) > 8
            or len(text) > 80
            or sum(token in question_like for token in tokens) >= 2
            or unique_ratio < 0.75
            or self.bot_name.lower() in tokens
        )

    def _infer_topic_from_text(self, text: str) -> Optional[str]:
        """Infer a concise topic phrase from noisy transcript text."""
        normalized = self._normalize_summary_text(text)
        if not normalized:
            return None

        if ("cloud" in normalized or "provider" in normalized) and any(
            provider in normalized for provider in ("aws", "azure", "gcp", "google cloud")
        ):
            if any(word in normalized for word in ("budget", "constrain", "cost")):
                return "cloud provider options under a tight budget"
            return "the best cloud provider and architecture to use"

        if "architecture" in normalized and "cloud" in normalized:
            return "cloud architecture choices"

        if any(word in normalized for word in ("budget", "cost", "pricing")):
            return "budget constraints"

        keywords = self._extract_summary_keywords(normalized)
        if not keywords:
            return None
        if len(keywords) == 1:
            return keywords[0]
        if len(keywords) == 2:
            return f"{keywords[0]} and {keywords[1]}"
        return f"{keywords[0]}, {keywords[1]}, and {keywords[2]}"

    def _normalize_summary_text(self, text: str) -> str:
        """Normalize free-form transcript text for summary heuristics."""
        normalized = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
        normalized = re.sub(r"\b(?:alex|so|um|uh|hmm|like|just|really|today|okay|ok)\b", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _extract_summary_keywords(self, text: str) -> list[str]:
        """Extract a few meaningful keywords from transcript text."""
        stopwords = {
            "about", "after", "again", "also", "been", "being", "between", "build", "building",
            "can", "could", "corporate", "discuss", "discussing", "from", "have", "help", "into",
            "it", "its", "just", "make", "more", "only", "people", "question", "right", "should",
            "something", "talk", "that", "them", "they", "this", "till", "use", "used", "using",
            "want", "what", "which", "with", "would", "your", "you", "we", "will", "till", "now",
        }
        keywords = []
        seen = set()
        for token in text.split():
            if len(token) < 4 or token in stopwords or token in seen:
                continue
            seen.add(token)
            keywords.append(token)
            if len(keywords) == 3:
                break
        return keywords

    def _select_primary_concern(self, context: MeetingContext, recent_transcript: str) -> Optional[str]:
        """Get the most important concern for a summary."""
        if context.high_priority_concerns:
            return self._to_sentence_fragment(context.high_priority_concerns[0].description)
        if context.unaddressed_concerns:
            return self._to_sentence_fragment(context.unaddressed_concerns[0].description)

        normalized = self._normalize_summary_text(recent_transcript)
        if any(word in normalized for word in ("budget", "constrain", "cost")):
            return "budget constraints"
        if any(word in normalized for word in ("latency", "risk", "issue", "problem")):
            for keyword in ("latency", "risk", "issue", "problem"):
                if keyword in normalized:
                    return f"{keyword} concerns"
        return None

    def _select_primary_decision(self, context: MeetingContext) -> Optional[str]:
        """Get the clearest decision phrase for a spoken summary."""
        if context.decisions_confirmed:
            return self._decision_phrase(context.decisions_confirmed[-1].description)
        if context.decisions_pending:
            return self._decision_phrase(context.decisions_pending[-1].description)
        return None

    def _decision_phrase(self, description: str) -> str:
        """Convert a decision description into a sentence-friendly phrase."""
        text = self._to_sentence_fragment(description)
        lowered = text.lower()

        for prefix in ("we should ", "we will ", "we'll ", "let's ", "going with ", "go with "):
            if lowered.startswith(prefix):
                return f"to {lowered[len(prefix):].strip()}"

        first_word = lowered.split()[0] if lowered.split() else ""
        if first_word in {"use", "choose", "adopt", "implement", "build", "move"}:
            return f"to {lowered}"

        return lowered

    def _infer_no_decision_clause(self, recent_transcript: str) -> Optional[str]:
        """Infer when the discussion is still open and comparing options."""
        normalized = self._normalize_summary_text(recent_transcript)
        providers = []
        for label, pattern in (
            ("AWS", r"\baws\b"),
            ("Azure", r"\bazure\b"),
            ("GCP", r"\bgcp\b|\bgoogle cloud\b"),
        ):
            if re.search(pattern, normalized):
                providers.append(label)

        if len(providers) >= 2:
            if len(providers) == 2:
                return f"no final choice is locked yet between {providers[0]} and {providers[1]}"
            return f"no final choice is locked yet between {providers[0]}, {providers[1]}, and {providers[2]}"

        if any(phrase in normalized for phrase in ("which one", "what do you think", "confused about", "should use", "should we use")):
            return "no final decision is locked yet"

        return None

    def _to_sentence_fragment(self, text: str) -> str:
        """Normalize text fragments before inserting them into spoken sentences."""
        cleaned = " ".join((text or "").split()).strip(" ,.!?;:")
        if not cleaned:
            return ""
        return cleaned[0].lower() + cleaned[1:] if len(cleaned) > 1 else cleaned.lower()

    def _join_spoken_clauses(self, clauses: list[str]) -> Optional[str]:
        """Join 1-2 spoken clauses into a short sentence."""
        filtered = [clause.strip().rstrip(". ") for clause in clauses if clause.strip()]
        if not filtered:
            return None
        if len(filtered) == 1:
            sentence = filtered[0]
        else:
            sentence = f"{filtered[0]}, and {filtered[1]}"
        return sentence[0].upper() + sentence[1:] + "."

    def _fallback_risks(self, context: MeetingContext) -> str:
        concerns = context.high_priority_concerns or context.unaddressed_concerns
        if not concerns:
            return "No major open risks captured yet."
        lines = [f"Risk: {concern.description}" for concern in concerns[:3]]
        return "\n".join(lines)

    def _fallback_decision(self, context: MeetingContext) -> str:
        if context.decisions_confirmed:
            decision = context.decisions_confirmed[-1]
            return f"Confirmed decision: {decision.description}"
        if context.decisions_pending:
            decision = context.decisions_pending[-1]
            return f"Current direction: {decision.description} ({decision.status.value})"
        return "No clear decision captured yet."

    def _fallback_action_items(self, context: MeetingContext) -> str:
        if not context.action_items:
            return "No action items captured yet."
        return "\n".join(f"Action: {item}" for item in context.action_items[:3])

    def _sanitize_context_summary(self, text: Optional[str]) -> Optional[str]:
        """Normalize LLM-compressed historical summary into a fixed plain-text schema."""
        if not text or not text.strip():
            return None

        cleaned = text.replace("**", "").replace("`", "").strip()
        cleaned_lines = [line.strip().lstrip("-*• ").strip() for line in cleaned.splitlines() if line.strip()]

        prefixes = {
            "summary": "Summary",
            "decisions": "Decisions",
            "open questions": "Open Questions",
            "risks": "Risks",
            "action items": "Action Items",
        }
        sections: Dict[str, str] = {}

        for line in cleaned_lines:
            if ":" not in line:
                continue
            raw_key, raw_value = line.split(":", 1)
            normalized_key = raw_key.strip().lower()
            if normalized_key in prefixes and normalized_key not in sections:
                sections[normalized_key] = " ".join(raw_value.split()).strip() or "None"

        if not sections:
            sections["summary"] = " ".join(cleaned.split())[:300] or "None"

        ordered_lines = []
        for key in ("summary", "decisions", "open questions", "risks", "action items"):
            ordered_lines.append(f"{prefixes[key]}: {sections.get(key, 'None')}")

        normalized = "\n".join(ordered_lines)
        if len(normalized) > 1000:
            normalized = normalized[:997].rstrip() + "..."
        return normalized

    @staticmethod
    def _prepare_summary_transcript_lines(chunks: List[TranscriptChunk]) -> List[str]:
        """Normalize transcript chunks for summary-time LLM calls."""
        transcript_lines = []
        for chunk in chunks:
            text = " ".join((chunk.text or "").split()).strip()
            if not text:
                continue
            speaker = chunk.speaker or "Unknown"
            transcript_lines.append(f"[{speaker}] {text}")

        if len(transcript_lines) > 16:
            transcript_lines = (
                transcript_lines[:6]
                + [f"[{len(transcript_lines) - 12} intermediate utterances omitted]"]
                + transcript_lines[-6:]
            )

        return transcript_lines

    def _parse_summary_factual_audit(self, text: Optional[str]) -> Optional[Trigger]:
        """Parse summary-audit output into a factual inconsistency trigger."""
        if not text or not text.strip():
            return None

        fields: Dict[str, str] = {}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            fields[key.strip().lower()] = value.strip()

        if fields.get("trigger", "no").lower() not in {"yes", "true"}:
            return None

        confidence = 0.78
        try:
            confidence = float(fields.get("confidence", confidence))
        except (TypeError, ValueError):
            confidence = 0.78
        confidence = max(0.0, min(confidence, 1.0))

        reason = fields.get("reason") or "LLM summary audit found a likely factual inconsistency."
        evidence_a = fields.get("evidence a")
        evidence_b = fields.get("evidence b")
        relevant_transcript = fields.get("relevant transcript")

        trigger = Trigger.create_proactive(
            trigger_type=TriggerType.FACTUAL_INCONSISTENCY,
            confidence=confidence,
            reason=reason,
        )
        trigger.signal_evidence_a = None if evidence_a in {None, "", "None"} else evidence_a
        trigger.signal_evidence_b = None if evidence_b in {None, "", "None"} else evidence_b
        trigger.relevant_transcript = (
            "" if relevant_transcript in {None, "", "None"} else relevant_transcript
        )
        return trigger

    def _parse_trigger_adjudication(
        self,
        text: Optional[str],
        proposed_trigger_type: TriggerType,
    ) -> Optional[TriggerAdjudication]:
        """Parse the combined adjudication-and-response LLM output."""
        if not text or not text.strip():
            return None

        cleaned = text.strip()
        if "Action:" not in cleaned and "Final Trigger:" not in cleaned:
            return TriggerAdjudication(
                action="respond",
                final_trigger_type=proposed_trigger_type,
                reason="LLM returned an unstructured answer; treated as confirmed.",
                response_text=cleaned,
            )

        fields: Dict[str, str] = {}
        response_lines: List[str] = []
        in_response_block = False

        for raw_line in cleaned.splitlines():
            stripped = raw_line.strip()
            if not stripped and not in_response_block:
                continue

            if in_response_block:
                response_lines.append(raw_line.rstrip())
                continue

            if ":" not in stripped:
                continue

            key, value = stripped.split(":", 1)
            normalized_key = key.strip().lower()
            if normalized_key == "response":
                in_response_block = True
                if value.strip():
                    response_lines.append(value.strip())
            else:
                fields[normalized_key] = value.strip()

        action = fields.get("action", "respond").strip().lower()
        final_trigger_type = self._coerce_adjudicated_trigger_type(
            fields.get("final trigger"),
            proposed_trigger_type,
        )
        reason = fields.get("reason") or ""
        response_text = "\n".join(line for line in response_lines if line.strip()).strip()

        if action in {"skip", "none", "no"}:
            return TriggerAdjudication(
                action="skip",
                final_trigger_type=final_trigger_type,
                reason=reason or "LLM chose to skip the proactive interruption.",
                response_text="",
            )

        if not response_text or response_text.upper() == "NONE":
            return TriggerAdjudication(
                action="skip",
                final_trigger_type=final_trigger_type,
                reason=reason or "LLM declined to provide a response.",
                response_text="",
            )

        return TriggerAdjudication(
            action="respond",
            final_trigger_type=final_trigger_type,
            reason=reason,
            response_text=response_text,
        )

    @staticmethod
    def _coerce_adjudicated_trigger_type(
        raw_value: Optional[str],
        fallback: TriggerType,
    ) -> TriggerType:
        """Map a parsed trigger label back to a supported proactive trigger type."""
        if not raw_value:
            return fallback

        normalized = raw_value.strip().lower().replace("-", "_").replace(" ", "_")
        try:
            trigger_type = TriggerType(normalized)
        except ValueError:
            return fallback

        if trigger_type not in _ADJUDICATION_OUTPUT_TRIGGER_TYPES:
            return fallback

        return trigger_type
    
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
