"""Ultra-fast local trigger scoring for proactive interventions.

This module provides a tiny, dependency-free scoring model that can run on
every trigger cycle in a few milliseconds. It is intentionally designed as a
bootstrap model:

- deterministic feature extraction from recent meeting turns
- linear scoring with calibrated weights
- no network calls or heavyweight ML dependencies

The interface is built so an offline-trained model can replace the hand-tuned
weights later without changing TriggerEngine's call pattern.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set

import structlog

from src.context.models import MeetingContext, TranscriptChunk, Trigger, TriggerType

logger = structlog.get_logger()


@dataclass(frozen=True)
class LinearTriggerProfile:
    """Simple linear classifier profile for a trigger type."""

    trigger_type: TriggerType
    bias: float
    threshold: float
    weights: Dict[str, float]


class FastTriggerModel:
    """Fast local scorer that supplements heuristic triggers.

    The first shipped profile focuses on REPEATED_CONFUSION because it is a
    common product gap: natural multi-speaker uncertainty often does not match
    a single keyword regex, but still shows a clear conversational pattern.
    """

    _RECENT_WINDOW = timedelta(minutes=2)
    _MAX_TURNS = 8

    _STOPWORDS: Set[str] = {
        "a", "an", "and", "are", "as", "at", "be", "because", "but", "by",
        "can", "could", "did", "do", "does", "for", "from", "had", "has",
        "have", "how", "i", "if", "in", "is", "it", "its", "just", "let",
        "lets", "like", "may", "maybe", "might", "my", "not", "of", "on",
        "or", "our", "same", "she", "so", "that", "the", "their", "them",
        "then", "there", "they", "this", "to", "too", "us", "was", "we",
        "what", "when", "where", "which", "who", "why", "with", "would",
        "yeah", "you", "your",
    }

    _QUESTION_STARTERS: Set[str] = {
        "what", "why", "how", "when", "where", "who", "which",
        "can", "could", "should", "would", "will", "did", "does", "do",
        "is", "are", "was", "were",
    }

    _STRONG_CONFUSION_PATTERNS: Sequence[re.Pattern[str]] = (
        re.compile(r"\bi(?:'m| am) confused\b", re.IGNORECASE),
        re.compile(r"\bwhat do you mean\b", re.IGNORECASE),
        re.compile(r"\bcan you explain\b", re.IGNORECASE),
        re.compile(r"\bi(?:'m| am) not following\b", re.IGNORECASE),
        re.compile(r"\bi(?:'m| am) lost\b", re.IGNORECASE),
        re.compile(r"\blost me there\b", re.IGNORECASE),
        re.compile(r"\bcan we slow down\b", re.IGNORECASE),
        re.compile(r"\btoo fast\b", re.IGNORECASE),
    )

    _SOFT_UNCERTAINTY_PATTERNS: Sequence[re.Pattern[str]] = (
        re.compile(r"\bweird\b", re.IGNORECASE),
        re.compile(r"\bstrange\b", re.IGNORECASE),
        re.compile(r"\bodd\b", re.IGNORECASE),
        re.compile(r"\bnot making sense\b", re.IGNORECASE),
        re.compile(r"\bnothing is making sense\b", re.IGNORECASE),
        re.compile(r"\bmissing something\b", re.IGNORECASE),
        re.compile(r"\bsame here\b", re.IGNORECASE),
        re.compile(r"\bno idea\b", re.IGNORECASE),
        re.compile(r"\bcan'?t figure (?:it )?out\b", re.IGNORECASE),
        re.compile(r"\bcan'?t tell\b", re.IGNORECASE),
    )

    _ISSUE_PATTERNS: Sequence[re.Pattern[str]] = (
        re.compile(r"\b500\b"),
        re.compile(r"\berror\b", re.IGNORECASE),
        re.compile(r"\bissue\b", re.IGNORECASE),
        re.compile(r"\bproblem\b", re.IGNORECASE),
        re.compile(r"\bfail(?:ing|s|ed)?\b", re.IGNORECASE),
        re.compile(r"\bempty\b", re.IGNORECASE),
        re.compile(r"\bmiddleware\b", re.IGNORECASE),
        re.compile(r"\bcontroller\b", re.IGNORECASE),
        re.compile(r"\bdeploy(?:ment)?\b", re.IGNORECASE),
        re.compile(r"\benv\b", re.IGNORECASE),
        re.compile(r"\bconfig\b", re.IGNORECASE),
        re.compile(r"\blog(?:ger|s)?\b", re.IGNORECASE),
        re.compile(r"\bdebug\b", re.IGNORECASE),
    )

    _RESOLUTION_PATTERNS: Sequence[re.Pattern[str]] = (
        re.compile(r"\bfixed\b", re.IGNORECASE),
        re.compile(r"\bsolved\b", re.IGNORECASE),
        re.compile(r"\bresolved\b", re.IGNORECASE),
        re.compile(r"\bworks now\b", re.IGNORECASE),
        re.compile(r"\bgot it\b", re.IGNORECASE),
        re.compile(r"\bthat makes sense\b", re.IGNORECASE),
        re.compile(r"\bfound it\b", re.IGNORECASE),
        re.compile(r"\bnevermind\b", re.IGNORECASE),
    )

    _CONFUSION_PROFILE = LinearTriggerProfile(
        trigger_type=TriggerType.REPEATED_CONFUSION,
        bias=-2.9,
        threshold=0.62,
        weights={
            "strong_confusion_ratio": 2.0,
            "soft_uncertainty_ratio": 1.3,
            "question_turn_ratio": 0.6,
            "speaker_switch_ratio": 0.5,
            "uncertain_speaker_ratio": 1.3,
            "topic_overlap_ratio": 0.8,
            "issue_stall_ratio": 0.8,
            "turn_count_ratio": 0.3,
            "resolution_ratio": -1.2,
        },
    )

    def predict_triggers(
        self,
        context: MeetingContext,
        new_chunks: Optional[Sequence[TranscriptChunk]] = None,
    ) -> List[Trigger]:
        """Predict proactive triggers from recent conversational dynamics."""
        turns = self._recent_turns(context)
        if len(turns) < 2:
            return []

        confusion = self._predict_repeated_confusion(context, turns)
        return [confusion] if confusion else []

    def _predict_repeated_confusion(
        self,
        context: MeetingContext,
        turns: Sequence[TranscriptChunk],
    ) -> Optional[Trigger]:
        features = self._build_confusion_features(context, turns)

        # Keep a hard floor so the model doesn't trigger on ordinary technical discussion.
        uncertainty_signal = features["strong_confusion_hits"] + features["soft_uncertainty_hits"]
        if uncertainty_signal < 2 and features["uncertain_speakers"] < 2:
            return None

        if not (
            features["strong_confusion_hits"] >= 2
            or features["uncertain_speakers"] >= 2
            or (
                features["strong_confusion_hits"] >= 1
                and features["soft_uncertainty_hits"] >= 2
            )
        ):
            return None

        probability = self._score(features, self._CONFUSION_PROFILE)
        if probability < self._CONFUSION_PROFILE.threshold:
            return None

        trigger = Trigger.create_proactive(
            TriggerType.REPEATED_CONFUSION,
            confidence=round(probability, 3),
            reason=self._build_confusion_reason(features),
        )
        trigger.relevant_transcript = " ".join(turn.text for turn in turns)[-500:]

        logger.debug(
            "fast_trigger_model_fired",
            trigger_type=trigger.type.value,
            confidence=trigger.confidence,
            strong_confusion_hits=features["strong_confusion_hits"],
            soft_uncertainty_hits=features["soft_uncertainty_hits"],
            uncertain_speakers=features["uncertain_speakers"],
        )
        return trigger

    def _build_confusion_features(
        self,
        context: MeetingContext,
        turns: Sequence[TranscriptChunk],
    ) -> Dict[str, float]:
        combined_text = " ".join(turn.text for turn in turns)
        strong_confusion_hits = self._count_pattern_hits(combined_text, self._STRONG_CONFUSION_PATTERNS)
        soft_uncertainty_hits = self._count_pattern_hits(combined_text, self._SOFT_UNCERTAINTY_PATTERNS)
        issue_stall_hits = self._count_pattern_hits(combined_text, self._ISSUE_PATTERNS)
        resolution_hits = self._count_pattern_hits(combined_text, self._RESOLUTION_PATTERNS)

        turn_count = len(turns)
        distinct_speakers = {turn.speaker or "Unknown" for turn in turns}
        uncertain_speakers = {
            turn.speaker or "Unknown"
            for turn in turns
            if self._turn_has_uncertainty(turn.text)
        }
        question_turns = sum(1 for turn in turns if self._looks_like_question(turn.text))
        speaker_switches = sum(
            1
            for previous, current in zip(turns, turns[1:])
            if (previous.speaker or "") != (current.speaker or "")
        )
        topic_overlap_ratio = self._average_topic_overlap(turns)

        return {
            "strong_confusion_hits": float(strong_confusion_hits),
            "soft_uncertainty_hits": float(soft_uncertainty_hits),
            "uncertain_speakers": float(len(uncertain_speakers)),
            "strong_confusion_ratio": min(1.0, strong_confusion_hits / 2.0),
            "soft_uncertainty_ratio": min(1.0, soft_uncertainty_hits / 3.0),
            "question_turn_ratio": question_turns / max(1, turn_count),
            "speaker_switch_ratio": speaker_switches / max(1, turn_count - 1),
            "uncertain_speaker_ratio": len(uncertain_speakers) / max(1, len(distinct_speakers)),
            "topic_overlap_ratio": topic_overlap_ratio,
            "issue_stall_ratio": min(1.0, issue_stall_hits / 4.0),
            "turn_count_ratio": min(1.0, turn_count / 6.0),
            "resolution_ratio": min(1.0, resolution_hits / 2.0),
            "open_question_pressure": min(1.0, len(context.open_questions) / 4.0),
        }

    @classmethod
    def _build_confusion_reason(cls, features: Dict[str, float]) -> str:
        reasons = []
        if features["uncertain_speakers"] >= 2:
            reasons.append(f"{int(features['uncertain_speakers'])} speakers showed uncertainty")
        if features["strong_confusion_hits"] >= 1:
            reasons.append(f"{int(features['strong_confusion_hits'])} strong confusion cues")
        if features["soft_uncertainty_hits"] >= 2:
            reasons.append(f"{int(features['soft_uncertainty_hits'])} softer uncertainty cues")
        if features["topic_overlap_ratio"] >= 0.3:
            reasons.append("same unresolved topic repeated across turns")
        if not reasons:
            reasons.append("conversation shows accumulating confusion")
        return "Fast trigger model detected likely repeated confusion: " + ", ".join(reasons[:3])

    def _recent_turns(self, context: MeetingContext) -> List[TranscriptChunk]:
        cutoff = datetime.now() - self._RECENT_WINDOW
        turns = [
            chunk
            for chunk in context.recent_transcript
            if chunk.is_final and chunk.text.strip() and chunk.timestamp >= cutoff
        ]
        return turns[-self._MAX_TURNS:]

    @classmethod
    def _count_pattern_hits(
        cls,
        text: str,
        patterns: Sequence[re.Pattern[str]],
    ) -> int:
        return sum(len(pattern.findall(text)) for pattern in patterns)

    @classmethod
    def _turn_has_uncertainty(cls, text: str) -> bool:
        return bool(
            cls._count_pattern_hits(text, cls._STRONG_CONFUSION_PATTERNS)
            or cls._count_pattern_hits(text, cls._SOFT_UNCERTAINTY_PATTERNS)
        )

    @classmethod
    def _looks_like_question(cls, text: str) -> bool:
        normalized = text.strip()
        if not normalized:
            return False
        if normalized.endswith("?"):
            return True
        first_word = normalized.split()[0].lower().rstrip(".,!?:;")
        return first_word in cls._QUESTION_STARTERS

    @classmethod
    def _average_topic_overlap(cls, turns: Sequence[TranscriptChunk]) -> float:
        keyword_turns = [cls._content_keywords(turn.text) for turn in turns]
        overlaps: List[float] = []
        for previous, current in zip(keyword_turns, keyword_turns[1:]):
            if not previous or not current:
                continue
            overlaps.append(len(previous & current) / max(1, len(current)))
        if not overlaps:
            return 0.0
        return sum(overlaps) / len(overlaps)

    @classmethod
    def _content_keywords(cls, text: str) -> Set[str]:
        raw_tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9'-]+", text.lower())
        normalized = [cls._normalize_token(token) for token in raw_tokens]
        counts = Counter(
            token for token in normalized if len(token) >= 3 and token not in cls._STOPWORDS
        )
        return {token for token, count in counts.items() if count >= 1}

    @staticmethod
    def _normalize_token(token: str) -> str:
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

    @staticmethod
    def _score(features: Dict[str, float], profile: LinearTriggerProfile) -> float:
        total = profile.bias
        for feature_name, weight in profile.weights.items():
            total += features.get(feature_name, 0.0) * weight
        return 1.0 / (1.0 + math.exp(-total))