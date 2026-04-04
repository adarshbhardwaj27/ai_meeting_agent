"""
Conversation Intelligence — Tier-1 Local Signal Scorer.

This module is the "brain" of the meeting assistant.  It runs every trigger
cycle (~500 ms) with ZERO LLM calls, producing lightweight SignalCandidates
that the TriggerEngine can promote to full Trigger objects when they cross
a confidence threshold.

Architecture (two-tier):
  Tier 1  (this file)  — cheap, local, every cycle
      • Assertion ledger: tracks factual claims per speaker
      • Contradiction scorer: detects when a new statement negates an earlier one
      • Memory-gap detector: notices when the team can't recall something the
        bot has in its context window
      • Collaboration-opportunity scorer: spots moments where the bot's context
        (concerns, decisions, questions) would genuinely help the conversation
      • Factual-inconsistency detector: catches when two speakers state
        conflicting numbers / names / facts

  Tier 2  (LLMService)  — expensive, only when Tier 1 fires
      • Receives a SignalCandidate with pre-packaged evidence
      • Generates a 1-3 line chat message (or SKIP)

Why this matters:
  A naïve approach would send every utterance to the LLM asking "should I
  speak?".  At 2 utterances/second × $0.01/call that's $72/hour.  Tier 1
  does the same job for $0 by operating on normalised keyword sets, an
  assertion ledger, and simple vector-free similarity scoring.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import structlog

from src.config import config
from src.context.models import (
    AssertionRecord,
    MeetingContext,
    SignalCandidate,
    SignalType,
    TranscriptChunk,
)

logger = structlog.get_logger()

# ── Tunables (exposed via config in a later step) ───────────────────

_CONTRADICTION_CONFIDENCE_FLOOR = 0.55
_MEMORY_GAP_CONFIDENCE_FLOOR = 0.50
_FACTUAL_CONFIDENCE_FLOOR = 0.55
_COLLABORATION_CONFIDENCE_FLOOR = 0.45
_ASSERTION_MAX_AGE_MINUTES = 30
_MIN_KEYWORD_OVERLAP_FOR_RELATION = 2
_COLLABORATION_MIN_CONCERN_AGE = timedelta(minutes=4)
_NEGATION_WORDS = frozenset({
    "not", "no", "never", "don't", "dont", "doesn't", "doesnt",
    "shouldn't", "shouldnt", "won't", "wont", "can't", "cant",
    "cannot", "isn't", "isnt", "aren't", "arent", "wasn't", "wasnt",
    "weren't", "werent", "wouldn't", "wouldnt", "neither", "nor",
    "against", "disagree", "reject", "wrong", "incorrect", "false",
    "impossible", "unlike", "opposite",
})
_MEMORY_PHRASES = re.compile(
    r"(?:(?:do(?:es)? anyone|anybody|who) (?:remember|recall|know))"
    r"|(?:what (?:did|was|were) (?:we|they|he|she) (?:say|decide|agree|mention))"
    r"|(?:i (?:can'?t|cannot|don'?t) (?:remember|recall))"
    r"|(?:i(?:'m| am)? blanking on)"
    r"|(?:i(?:'m| am)? drawing a blank)"
    r"|(?:it'?s slipping my mind)"
    r"|(?:wasn'?t there (?:something|a point|a concern))"
    r"|(?:remind me (?:what|who|when|why))"
    r"|(?:did (?:we|someone|anybody) (?:already|ever) (?:discuss|mention|cover|address))",
    re.IGNORECASE,
)
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "it", "its", "this", "that", "these", "those", "i", "we", "you",
    "he", "she", "they", "me", "us", "him", "her", "them",
    "my", "our", "your", "his", "their", "mine", "ours", "yours",
    "so", "if", "then", "than", "as", "just", "about", "also",
    "very", "really", "actually", "basically", "like", "right",
    "okay", "yeah", "yes", "well", "um", "uh", "let", "lets",
    "think", "know", "said", "say", "go", "going", "get",
})
# Patterns that indicate a quantitative claim (numbers, percentages, durations)
_NUMERIC_CLAIM = re.compile(
    r"\b\d+(?:\.\d+)?(?:\s*(?:%|percent|ms|millisecond|second|minute|hour"
    r"|day|week|month|gb|mb|kb|tb|k|m|x|times|users?|requests?|calls?))\b",
    re.IGNORECASE,
)
_GENERIC_RELATION_TERMS = frozenset({
    "action", "actions", "add", "approach", "approaches", "change", "changes",
    "consider", "considering", "contract", "contracts", "decide", "decision",
    "decisions", "define", "doc", "docs", "document", "documents", "flow",
    "item", "items", "long", "need", "needs", "option", "options", "owner",
    "owners", "plan", "plans", "question", "questions", "scope", "short",
    "step", "steps", "support", "supports", "term", "terms", "test", "tests",
    "today", "update", "updates", "work", "works",
})
_CONTRADICTION_EXPLORATORY_PHRASES = re.compile(
    r"\b(?:if we|can we|should we|could we|would we|option(?:s)?|alternatively|"
    r"short-term|long-term|leaning|prefer(?:s)?|suggest(?:s|ed)?|spike|toggle|"
    r"unclear|who should|need to decide|need to consider|action items?|next steps?|"
    r"rollback|hotfix|want(?:s)?|requested|asking for|asked for|require(?:s|d)?)\b",
    re.IGNORECASE,
)
_CONTRADICTION_ADDITIVE_PHRASES = re.compile(
    r"^(?:also|and|plus)\b|\b(?:also|in addition|we also need|we also have|"
    r"need to consider|must also|that could explain|which could explain|"
    r"root cause)\b",
    re.IGNORECASE,
)
_RETROSPECTIVE_ASSUMPTION_PHRASES = re.compile(
    r"\b(?:assumed|thought|expected|planned|intended|didn't realize|did not realize)\b",
    re.IGNORECASE,
)
_COLLABORATION_ALREADY_COVERED_PHRASES = re.compile(
    r"\b(?:let'?s|we(?:'ll| will)|i(?:'ll| will)|action items?|next steps?|"
    r"can we|pick|choose|decide|owner|assign|schedule|define|update|patch|"
    r"hotfix|rollback|acceptable|works?|manual sign-off|approval sla|"
    r"short-term|long-term|combined approach)\b",
    re.IGNORECASE,
)


class ConversationIntelligence:
    """
    Zero-cost Tier-1 signal scorer that runs every trigger cycle.

    Public API used by TriggerEngine:
        score_signals(context, new_chunks) -> List[SignalCandidate]
    """

    def __init__(self) -> None:
        # ── Assertion ledger ──────────────────────────────────────────
        self._assertions: List[AssertionRecord] = []
        self._assertions_by_speaker: Dict[str, List[AssertionRecord]] = defaultdict(list)
        self._processed_chunk_ids: Set[str] = set()

    # ══════════════════════════════════════════════════════════════════
    #  Public entry point
    # ══════════════════════════════════════════════════════════════════

    def score_signals(
        self,
        context: MeetingContext,
        new_chunks: Optional[List[TranscriptChunk]] = None,
    ) -> List[SignalCandidate]:
        """
        Run all Tier-1 detectors and return any signals above threshold.

        This is called every trigger cycle (~500 ms).  It must be FAST
        and must NEVER call an LLM.
        """
        if not new_chunks:
            return []

        # Only process chunks we haven't seen
        fresh = [c for c in new_chunks if c.chunk_id not in self._processed_chunk_ids]
        if not fresh:
            return []
        for c in fresh:
            self._processed_chunk_ids.add(c.chunk_id)

        # Extract assertions from new speech
        new_assertions = self._extract_assertions(fresh)

        signals: List[SignalCandidate] = []

        # ── Detector 1: Contradictions ────────────────────────────────
        signals.extend(self._detect_contradictions(new_assertions))

        # ── Detector 2: Memory gaps ───────────────────────────────────
        signals.extend(self._detect_memory_gaps(fresh, context))

        # ── Detector 3: Factual inconsistencies (numeric claims) ──────
        signals.extend(self._detect_factual_inconsistencies(new_assertions))

        # ── Detector 4: Collaboration opportunities ───────────────────
        signals.extend(self._detect_collaboration_opportunities(fresh, context))

        # Store new assertions in the ledger
        for a in new_assertions:
            self._assertions.append(a)
            if a.speaker:
                self._assertions_by_speaker[a.speaker].append(a)

        # Prune stale assertions
        self._prune_old_assertions()

        return self._dedupe_signals(signals)

    # ══════════════════════════════════════════════════════════════════
    #  Assertion extraction
    # ══════════════════════════════════════════════════════════════════

    def _extract_assertions(
        self, chunks: List[TranscriptChunk]
    ) -> List[AssertionRecord]:
        """
        Extract factual claims from transcript chunks.

        A "claim" is any declarative sentence that contains at least two
        content keywords.  We track whether it is negated so we can
        detect contradictions later.
        """
        assertions: List[AssertionRecord] = []
        for chunk in chunks:
            if not chunk.is_final:
                continue
            sentences = self._split_sentences(chunk.text)
            for sentence in sentences:
                keywords = self._content_keywords(sentence)
                if len(keywords) < 2:
                    continue
                negated = self._is_negated(sentence)
                assertions.append(
                    AssertionRecord.create(
                        text=sentence.strip(),
                        speaker=chunk.speaker,
                        keywords=keywords,
                        negated=negated,
                    )
                )
        return assertions

    # ══════════════════════════════════════════════════════════════════
    #  Detector 1: Contradictions
    # ══════════════════════════════════════════════════════════════════

    def _detect_contradictions(
        self, new_assertions: List[AssertionRecord]
    ) -> List[SignalCandidate]:
        """
        Check if any new assertion contradicts an earlier one.

        Contradiction ≈ high keyword overlap + opposite polarity (negation
        flip) or directly opposing quantitative claims.
        """
        signals: List[SignalCandidate] = []
        for new in new_assertions:
            for old in self._assertions:
                if old.superseded_by:
                    continue
                # Same speaker updating their own view isn't a contradiction
                # worth surfacing (people self-correct all the time)
                if old.speaker and old.speaker == new.speaker:
                    continue

                overlap = old.keywords & new.keywords
                if len(overlap) < _MIN_KEYWORD_OVERLAP_FOR_RELATION:
                    continue

                substantive_overlap = self._substantive_overlap(overlap)
                if self._should_skip_contradiction(old.text, new.text, substantive_overlap):
                    continue

                # Polarity flip (one negated, one not)
                if old.negated != new.negated:
                    confidence = self._contradiction_confidence(old, new, overlap)
                    if confidence >= _CONTRADICTION_CONFIDENCE_FLOOR:
                        signals.append(
                            SignalCandidate(
                                signal_type=SignalType.CONTRADICTION,
                                confidence=confidence,
                                reason=(
                                    f"{new.speaker or 'Someone'} said \"{new.text}\" "
                                    f"which appears to contradict "
                                    f"{old.speaker or 'an earlier statement'}: \"{old.text}\""
                                ),
                                evidence_a=old.text,
                                evidence_b=new.text,
                                speaker_a=old.speaker,
                                speaker_b=new.speaker,
                                related_assertion_ids=[old.id, new.id],
                                relevant_transcript=new.text,
                            )
                        )
        return signals

    def _contradiction_confidence(
        self,
        old: AssertionRecord,
        new: AssertionRecord,
        overlap: Set[str],
    ) -> float:
        """Score how likely two assertions are a real contradiction."""
        # Base: keyword overlap ratio (Jaccard on content words)
        union = old.keywords | new.keywords
        jaccard = len(overlap) / len(union) if union else 0.0
        score = 0.40 + (jaccard * 0.35)

        # Boost if different speakers (cross-speaker contradiction is more valuable)
        if old.speaker and new.speaker and old.speaker != new.speaker:
            score += 0.10

        # Boost if the overlapping keywords include a "topic noun" (> 4 chars)
        topic_nouns = {w for w in overlap if len(w) > 4}
        if topic_nouns:
            score += 0.05 * min(len(topic_nouns), 3)

        return min(score, 0.95)

    # ══════════════════════════════════════════════════════════════════
    #  Detector 2: Memory gaps
    # ══════════════════════════════════════════════════════════════════

    def _detect_memory_gaps(
        self,
        chunks: List[TranscriptChunk],
        context: MeetingContext,
    ) -> List[SignalCandidate]:
        """
        Detect when someone is trying to recall something the bot already
        has in its context window.
        """
        signals: List[SignalCandidate] = []
        combined = " ".join(c.text for c in chunks if c.is_final)
        if not _MEMORY_PHRASES.search(combined):
            return signals

        # They're trying to remember something — can we help?
        query_keywords = self._content_keywords(combined)
        if len(query_keywords) < 2:
            return signals

        # Search assertion ledger for matching earlier claims
        best_match: Optional[AssertionRecord] = None
        best_overlap = 0
        for assertion in reversed(self._assertions):
            overlap = len(assertion.keywords & query_keywords)
            if overlap >= _MIN_KEYWORD_OVERLAP_FOR_RELATION and overlap > best_overlap:
                best_match = assertion
                best_overlap = overlap

        # Also check meeting concerns, decisions, and questions
        context_matches: List[str] = []
        for concern in context.unaddressed_concerns:
            concern_kw = self._content_keywords(concern.description)
            if len(concern_kw & query_keywords) >= 2:
                label = f"{concern.raised_by or 'Someone'} raised: \"{concern.description}\""
                context_matches.append(label)

        for decision in context.decisions_pending + context.decisions_confirmed:
            dec_kw = self._content_keywords(decision.description)
            if len(dec_kw & query_keywords) >= 2:
                context_matches.append(
                    f"Decision: \"{decision.description}\" ({decision.status.value})"
                )

        if best_match or context_matches:
            evidence = best_match.text if best_match else context_matches[0]
            confidence = min(
                0.90,
                _MEMORY_GAP_CONFIDENCE_FLOOR
                + (best_overlap * 0.08)
                + (0.10 if context_matches else 0.0),
            )
            signals.append(
                SignalCandidate(
                    signal_type=SignalType.MEMORY_GAP,
                    confidence=confidence,
                    reason=(
                        f"The team is trying to recall something — the bot has "
                        f"relevant context: \"{evidence}\""
                    ),
                    evidence_a=evidence,
                    evidence_b=combined[:200],
                    speaker_a=best_match.speaker if best_match else None,
                    speaker_b=chunks[0].speaker if chunks else None,
                    related_assertion_ids=[best_match.id] if best_match else [],
                    relevant_transcript=combined[:300],
                )
            )
        return signals

    # ══════════════════════════════════════════════════════════════════
    #  Detector 3: Factual inconsistencies (numeric / named claims)
    # ══════════════════════════════════════════════════════════════════

    def _detect_factual_inconsistencies(
        self, new_assertions: List[AssertionRecord]
    ) -> List[SignalCandidate]:
        """
        Detect when a quantitative claim conflicts with a prior one.

        Example: "Our SLA is 100ms" vs later "We promised 50ms SLA".
        Both are about "SLA" + a number, but the numbers differ.
        """
        signals: List[SignalCandidate] = []
        for new in new_assertions:
            new_numbers = self._extract_numbers(new.text)
            if not new_numbers:
                continue
            for old in self._assertions:
                if old.superseded_by:
                    continue
                overlap = old.keywords & new.keywords
                if len(overlap) < _MIN_KEYWORD_OVERLAP_FOR_RELATION:
                    continue
                old_numbers = self._extract_numbers(old.text)
                if not old_numbers:
                    continue
                # Numbers present in both and they differ
                if new_numbers != old_numbers:
                    confidence = min(
                        0.90,
                        _FACTUAL_CONFIDENCE_FLOOR
                        + (len(overlap) * 0.07)
                        + 0.10,  # numeric mismatch bonus
                    )
                    signals.append(
                        SignalCandidate(
                            signal_type=SignalType.FACTUAL_INCONSISTENCY,
                            confidence=confidence,
                            reason=(
                                f"Numeric mismatch: \"{new.text}\" vs earlier "
                                f"\"{old.text}\" — overlapping topic "
                                f"({', '.join(sorted(overlap))})"
                            ),
                            evidence_a=old.text,
                            evidence_b=new.text,
                            speaker_a=old.speaker,
                            speaker_b=new.speaker,
                            related_assertion_ids=[old.id, new.id],
                            relevant_transcript=new.text,
                        )
                    )
        return signals

    # ══════════════════════════════════════════════════════════════════
    #  Detector 4: Collaboration opportunities
    # ══════════════════════════════════════════════════════════════════

    def _detect_collaboration_opportunities(
        self,
        chunks: List[TranscriptChunk],
        context: MeetingContext,
    ) -> List[SignalCandidate]:
        """
        Detect when the bot's tracked context (concerns, decisions, questions)
        is highly relevant to the current conversation but hasn't been
        surfaced yet.

        This replaces the old hardcoded "risk_overlooked" and
        "decision_forming" regex approaches with a single unified scorer.
        """
        signals: List[SignalCandidate] = []
        combined = " ".join(c.text for c in chunks if c.is_final)
        recent_kw = self._content_keywords(combined)
        if len(recent_kw) < 3:
            return signals

        if self._discussion_already_has_next_steps(combined):
            return signals

        # Check unaddressed concerns that overlap with current speech
        for concern in context.unaddressed_concerns:
            age = datetime.now() - concern.raised_at
            if age < _COLLABORATION_MIN_CONCERN_AGE:
                continue  # Too fresh — they still have it in mind
            concern_kw = self._content_keywords(concern.description)
            overlap = recent_kw & concern_kw
            if len(overlap) >= _MIN_KEYWORD_OVERLAP_FOR_RELATION:
                confidence = min(
                    0.85,
                    _COLLABORATION_CONFIDENCE_FLOOR
                    + (len(overlap) * 0.08)
                    + (0.10 if concern.severity.value in ("high", "critical") else 0.0),
                )
                signals.append(
                    SignalCandidate(
                        signal_type=SignalType.COLLABORATION_OPPORTUNITY,
                        confidence=confidence,
                        reason=(
                            f"Discussion touches on "
                            f"\"{concern.description}\" "
                            f"({concern.severity.value} concern raised by "
                            f"{concern.raised_by or 'someone'} "
                            f"{int(age.total_seconds() / 60)} min ago) "
                            f"— not yet addressed"
                        ),
                        evidence_a=concern.description,
                        evidence_b=combined[:200],
                        speaker_a=concern.raised_by,
                        speaker_b=chunks[-1].speaker if chunks else None,
                        relevant_transcript=combined[:300],
                    )
                )

        # Check open questions that now have relevant context
        for question in context.open_questions[-5:]:
            q_kw = self._content_keywords(question)
            overlap = recent_kw & q_kw
            if len(overlap) >= _MIN_KEYWORD_OVERLAP_FOR_RELATION:
                confidence = min(
                    0.75,
                    _COLLABORATION_CONFIDENCE_FLOOR + (len(overlap) * 0.08),
                )
                signals.append(
                    SignalCandidate(
                        signal_type=SignalType.COLLABORATION_OPPORTUNITY,
                        confidence=confidence,
                        reason=(
                            f"Current conversation may answer the earlier open "
                            f"question: \"{question}\""
                        ),
                        evidence_a=question,
                        evidence_b=combined[:200],
                        relevant_transcript=combined[:300],
                    )
                )

        return signals

    @staticmethod
    def _substantive_overlap(overlap: Set[str]) -> Set[str]:
        """Remove generic planning words from overlap before contradiction scoring."""
        return {token for token in overlap if token not in _GENERIC_RELATION_TERMS}

    def _should_skip_contradiction(
        self,
        old_text: str,
        new_text: str,
        substantive_overlap: Set[str],
    ) -> bool:
        """Filter low-value contradiction pairs that are really planning or diagnosis."""
        if not substantive_overlap:
            return True

        if _RETROSPECTIVE_ASSUMPTION_PHRASES.search(old_text) or _RETROSPECTIVE_ASSUMPTION_PHRASES.search(new_text):
            return True

        if _CONTRADICTION_EXPLORATORY_PHRASES.search(old_text) or _CONTRADICTION_EXPLORATORY_PHRASES.search(new_text):
            return True

        if _CONTRADICTION_ADDITIVE_PHRASES.search(new_text):
            return True

        return False

    @staticmethod
    def _discussion_already_has_next_steps(text: str) -> bool:
        """Skip collaboration prompts when the team is already assigning actions or converging."""
        return bool(_COLLABORATION_ALREADY_COVERED_PHRASES.search(text))

    def _dedupe_signals(self, signals: List[SignalCandidate]) -> List[SignalCandidate]:
        """Keep the highest-confidence signal per type and transcript span."""
        deduped: Dict[Tuple[SignalType, str], SignalCandidate] = {}

        for signal in signals:
            signature_text = signal.relevant_transcript or signal.evidence_b or signal.reason
            normalized_signature = " ".join(self._content_keywords(signature_text or ""))[:160]
            key = (signal.signal_type, normalized_signature)
            existing = deduped.get(key)
            if existing is None or signal.confidence > existing.confidence:
                deduped[key] = signal

        return list(deduped.values())

    # ══════════════════════════════════════════════════════════════════
    #  Text utilities (shared across detectors)
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentence-like segments."""
        # Split on sentence-ending punctuation, or on common speech connectors
        parts = re.split(r'(?<=[.!?])\s+|(?:,\s*(?:but|however|although|yet))', text)
        return [p.strip() for p in parts if p and len(p.strip()) > 10]

    @staticmethod
    def _content_keywords(text: str) -> Set[str]:
        """Extract content words (non-stopwords, length >= 3)."""
        raw = re.findall(r"[a-zA-Z][a-zA-Z0-9'-]+", text.lower())
        return {
            w.strip("'-")
            for w in raw
            if len(w) >= 3 and w.strip("'-") not in _STOPWORDS
        }

    @staticmethod
    def _is_negated(sentence: str) -> bool:
        """Check if a sentence contains negation."""
        words = set(sentence.lower().split())
        return bool(words & _NEGATION_WORDS)

    @staticmethod
    def _extract_numbers(text: str) -> Set[str]:
        """Extract numeric claims (e.g. '100ms', '50%', '10x')."""
        return set(_NUMERIC_CLAIM.findall(text.lower()))

    def _prune_old_assertions(self) -> None:
        """Remove assertions older than the max age."""
        cutoff = datetime.now() - timedelta(minutes=_ASSERTION_MAX_AGE_MINUTES)
        self._assertions = [a for a in self._assertions if a.timestamp >= cutoff]
        for speaker in self._assertions_by_speaker:
            self._assertions_by_speaker[speaker] = [
                a for a in self._assertions_by_speaker[speaker]
                if a.timestamp >= cutoff
            ]
