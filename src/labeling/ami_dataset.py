"""AMI corpus helpers for browsing and labeling meeting windows.

The AMI annotations are stored as NITE XML layers. This module reconstructs
human-readable turns from the XML files so they can be labeled in a small UI
without requiring the original NXT tooling.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence
import xml.etree.ElementTree as ET


_NITE_NS = {"nite": "http://nite.sourceforge.net/"}
_WORD_INDEX_RE = re.compile(r"(\d+)$")
_HREF_RE = re.compile(r"#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?")


def _tag_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _word_index(word_id: str) -> int:
    match = _WORD_INDEX_RE.search(word_id)
    if not match:
        raise ValueError(f"Unable to parse word index from '{word_id}'")
    return int(match.group(1))


def _parse_href_range(href: str) -> tuple[str, str, str]:
    match = _HREF_RE.search(href)
    if not match:
        raise ValueError(f"Unsupported href format: {href}")
    file_name = href.split("#", 1)[0]
    start_id = match.group(1)
    end_id = match.group(2) or start_id
    return file_name, start_id, end_id


def _pointer_target_id(pointer_href: str) -> str:
    match = _HREF_RE.search(pointer_href)
    if not match:
        raise ValueError(f"Unsupported pointer href format: {pointer_href}")
    return match.group(1)


@dataclass
class AmiToken:
    token_id: str
    text: str
    start_time: float
    end_time: float
    is_punctuation: bool = False


@dataclass
class RangeAnnotation:
    source_id: str
    start_index: int
    end_index: int
    label: str
    addressee: Optional[str] = None


@dataclass
class TopicAnnotation:
    label: str
    start_index: int
    end_index: int


@dataclass
class AmiTurn:
    meeting_id: str
    speaker: str
    start_time: float
    end_time: float
    text: str
    dialogue_acts: List[str] = field(default_factory=list)
    adjacency_types: List[str] = field(default_factory=list)
    addressee: Optional[str] = None
    topic: Optional[str] = None
    act_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker": self.speaker,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "text": self.text,
            "dialogue_acts": self.dialogue_acts,
            "adjacency_types": self.adjacency_types,
            "addressee": self.addressee,
            "topic": self.topic,
        }


@dataclass
class AmiWindow:
    meeting_id: str
    example_id: str
    window_index: int
    start_time: float
    end_time: float
    topic: Optional[str]
    transcript_window: List[AmiTurn]
    weak_signals: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meeting_id": self.meeting_id,
            "example_id": self.example_id,
            "window_index": self.window_index,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "topic": self.topic,
            "transcript_window": [turn.to_dict() for turn in self.transcript_window],
            "weak_signals": self.weak_signals,
        }


class AmiDataset:
    """Load AMI XML annotations and expose meeting windows for labeling."""

    _MERGE_SAME_SPEAKER_GAP_SECONDS = 1.0

    def __init__(self, dataset_root: str | Path):
        self.dataset_root = Path(dataset_root)
        self.words_dir = self.dataset_root / "words"
        self.segments_dir = self.dataset_root / "segments"
        self.dialogue_acts_dir = self.dataset_root / "dialogueActs"
        self.topics_dir = self.dataset_root / "topics"
        self.ontologies_dir = self.dataset_root / "ontologies"
        self._da_type_map = self._load_ontology_map(self.ontologies_dir / "da-types.xml", "da-type")
        self._ap_type_map = self._load_ontology_map(self.ontologies_dir / "ap-types.xml", "ap-type")
        self._meeting_turn_cache: Dict[str, List[AmiTurn]] = {}
        self._meeting_window_cache: Dict[tuple[str, int, int], List[AmiWindow]] = {}

    def list_meetings(self) -> List[str]:
        meetings = {path.name.split(".", 1)[0] for path in self.segments_dir.glob("*.segments.xml")}
        return sorted(meetings)

    def build_windows(self, meeting_id: str, window_size: int = 6, step: int = 3) -> List[AmiWindow]:
        cache_key = (meeting_id, window_size, step)
        if cache_key in self._meeting_window_cache:
            return self._meeting_window_cache[cache_key]

        turns = self.load_meeting_turns(meeting_id)
        windows: List[AmiWindow] = []
        if not turns:
            self._meeting_window_cache[cache_key] = windows
            return windows

        start_index = 0
        window_index = 0
        while start_index < len(turns):
            end_index = min(len(turns), start_index + window_size)
            end_index = self._extend_window_end_for_continuation(turns, end_index)
            chunk = turns[start_index:end_index]
            if len(chunk) < 3 and windows:
                break
            if not chunk:
                break

            dominant_topic = self._dominant_topic(chunk)
            weak_signals = self._build_weak_signals(chunk)
            windows.append(
                AmiWindow(
                    meeting_id=meeting_id,
                    example_id=f"{meeting_id}_{window_index:04d}",
                    window_index=window_index,
                    start_time=chunk[0].start_time,
                    end_time=chunk[-1].end_time,
                    topic=dominant_topic,
                    transcript_window=chunk,
                    weak_signals=weak_signals,
                )
            )
            window_index += 1
            start_index += max(1, step)

        self._meeting_window_cache[cache_key] = windows
        return windows

    @classmethod
    def _extend_window_end_for_continuation(cls, turns: Sequence[AmiTurn], end_index: int) -> int:
        if end_index <= 0 or end_index >= len(turns):
            return end_index

        last_turn = turns[end_index - 1]
        if not cls._looks_truncated(last_turn.text):
            return end_index

        extended_end = end_index
        lookahead_end = min(len(turns), end_index + 3)
        for candidate_index in range(end_index, lookahead_end):
            candidate = turns[candidate_index]
            if (
                candidate.speaker == last_turn.speaker
                and candidate.start_time - last_turn.end_time <= cls._MERGE_SAME_SPEAKER_GAP_SECONDS
            ):
                extended_end = candidate_index + 1

        return extended_end

    @staticmethod
    def _looks_truncated(text: str) -> bool:
        stripped = text.strip()
        if not stripped or stripped.endswith((".", "?", "!")):
            return False

        words = stripped.split()
        if len(words) < 3:
            return False

        return len(words[-1].strip("'\"")) <= 2

    def load_meeting_turns(self, meeting_id: str) -> List[AmiTurn]:
        if meeting_id in self._meeting_turn_cache:
            return self._meeting_turn_cache[meeting_id]

        topic_ranges = self._load_topic_ranges(meeting_id)
        adjacency_map = self._load_adjacency_pairs(meeting_id)
        turns: List[AmiTurn] = []

        for segment_path in sorted(self.segments_dir.glob(f"{meeting_id}.*.segments.xml")):
            file_parts = segment_path.name.split(".")
            if len(file_parts) < 3:
                continue
            speaker = file_parts[1]
            words_path = self.words_dir / f"{meeting_id}.{speaker}.words.xml"
            if not words_path.exists():
                continue

            words_in_order, word_lookup = self._load_words(words_path)
            word_positions = {token.token_id: index for index, token in enumerate(words_in_order)}
            dialogue_annotations = self._load_dialogue_annotations(meeting_id, speaker, word_positions)
            topic_annotations = topic_ranges.get(f"{meeting_id}.{speaker}.words.xml", [])
            turns.extend(
                self._build_turns_from_segments(
                    meeting_id=meeting_id,
                    speaker=speaker,
                    segment_path=segment_path,
                    word_lookup=word_lookup,
                    word_positions=word_positions,
                    dialogue_annotations=dialogue_annotations,
                    topic_annotations=topic_annotations,
                    adjacency_map=adjacency_map,
                )
            )

        merged_turns = self._merge_adjacent_turns(sorted(turns, key=lambda turn: (turn.start_time, turn.end_time, turn.speaker)))
        self._meeting_turn_cache[meeting_id] = merged_turns
        return merged_turns

    def _load_words(self, words_path: Path) -> tuple[List[AmiToken], Dict[str, AmiToken]]:
        tree = ET.parse(words_path)
        root = tree.getroot()
        tokens: List[AmiToken] = []
        for element in root:
            tag = _tag_name(element)
            token_id = element.attrib.get("{http://nite.sourceforge.net/}id") or element.attrib.get("nite:id") or element.attrib.get("id")
            if not token_id:
                continue

            if tag == "w":
                text = (element.text or "").strip()
                token = AmiToken(
                    token_id=token_id,
                    text=text,
                    start_time=float(element.attrib.get("starttime", 0.0)),
                    end_time=float(element.attrib.get("endtime", 0.0)),
                    is_punctuation=element.attrib.get("punc") == "true",
                )
            elif tag == "vocalsound":
                sound_type = element.attrib.get("type", "sound")
                token = AmiToken(
                    token_id=token_id,
                    text=f"[{sound_type}]",
                    start_time=float(element.attrib.get("starttime", 0.0)),
                    end_time=float(element.attrib.get("endtime", 0.0)),
                )
            else:
                continue
            tokens.append(token)

        return tokens, {token.token_id: token for token in tokens}

    def _load_dialogue_annotations(
        self,
        meeting_id: str,
        speaker: str,
        word_positions: Dict[str, int],
    ) -> List[RangeAnnotation]:
        dialog_path = self.dialogue_acts_dir / f"{meeting_id}.{speaker}.dialog-act.xml"
        if not dialog_path.exists():
            return []

        tree = ET.parse(dialog_path)
        root = tree.getroot()
        annotations: List[RangeAnnotation] = []

        for dact in root:
            if _tag_name(dact) != "dact":
                continue
            act_id = dact.attrib.get("{http://nite.sourceforge.net/}id") or dact.attrib.get("nite:id") or dact.attrib.get("id")
            if not act_id:
                continue

            label = "Unknown"
            for pointer in dact.findall("nite:pointer", _NITE_NS):
                if pointer.attrib.get("role") == "da-aspect":
                    label = self._da_type_map.get(_pointer_target_id(pointer.attrib["href"]), label)
                    break

            addressee = dact.attrib.get("addressee")
            for child in dact.findall("nite:child", _NITE_NS):
                file_name, start_id, end_id = _parse_href_range(child.attrib["href"])
                if not file_name.endswith(f"{meeting_id}.{speaker}.words.xml"):
                    continue
                if start_id not in word_positions or end_id not in word_positions:
                    continue
                start_index = min(word_positions[start_id], word_positions[end_id])
                end_index = max(word_positions[start_id], word_positions[end_id])
                annotations.append(
                    RangeAnnotation(
                        source_id=act_id,
                        start_index=start_index,
                        end_index=end_index,
                        label=label,
                        addressee=addressee,
                    )
                )

        return annotations

    def _load_topic_ranges(self, meeting_id: str) -> Dict[str, List[TopicAnnotation]]:
        topic_path = self.topics_dir / f"{meeting_id}.topic.xml"
        if not topic_path.exists():
            return {}

        tree = ET.parse(topic_path)
        root = tree.getroot()
        topic_ranges: Dict[str, List[TopicAnnotation]] = defaultdict(list)

        for topic in root:
            if _tag_name(topic) != "topic":
                continue

            label = (topic.attrib.get("other_description") or "").strip()
            if not label:
                for pointer in topic.findall("nite:pointer", _NITE_NS):
                    if pointer.attrib.get("role") == "scenario_topic_type":
                        label = self._normalize_topic_pointer(pointer.attrib["href"])
                        break

            for child in topic.findall("nite:child", _NITE_NS):
                file_name, start_id, end_id = _parse_href_range(child.attrib["href"])
                topic_ranges[file_name].append(
                    TopicAnnotation(
                        label=label or "Unlabeled topic",
                        start_index=min(_word_index(start_id), _word_index(end_id)),
                        end_index=max(_word_index(start_id), _word_index(end_id)),
                    )
                )

        return topic_ranges

    def _load_adjacency_pairs(self, meeting_id: str) -> Dict[str, List[str]]:
        pair_path = self.dialogue_acts_dir / f"{meeting_id}.adjacency-pairs.xml"
        if not pair_path.exists():
            return {}

        tree = ET.parse(pair_path)
        root = tree.getroot()
        adjacency_map: Dict[str, set[str]] = defaultdict(set)

        for pair in root:
            if _tag_name(pair) != "adjacency-pair":
                continue

            pair_type = "Linked"
            linked_ids: List[str] = []
            for pointer in pair.findall("nite:pointer", _NITE_NS):
                role = pointer.attrib.get("role")
                target_id = _pointer_target_id(pointer.attrib["href"])
                if role == "type":
                    pair_type = self._ap_type_map.get(target_id, pair_type)
                elif role in {"source", "target"}:
                    linked_ids.append(target_id)

            for act_id in linked_ids:
                adjacency_map[act_id].add(pair_type)

        return {act_id: sorted(values) for act_id, values in adjacency_map.items()}

    def _build_turns_from_segments(
        self,
        *,
        meeting_id: str,
        speaker: str,
        segment_path: Path,
        word_lookup: Dict[str, AmiToken],
        word_positions: Dict[str, int],
        dialogue_annotations: Sequence[RangeAnnotation],
        topic_annotations: Sequence[TopicAnnotation],
        adjacency_map: Dict[str, List[str]],
    ) -> List[AmiTurn]:
        tree = ET.parse(segment_path)
        root = tree.getroot()
        turns: List[AmiTurn] = []

        for segment in root:
            if _tag_name(segment) != "segment":
                continue

            segment_word_ids: List[str] = []
            for child in segment.findall("nite:child", _NITE_NS):
                _, start_id, end_id = _parse_href_range(child.attrib["href"])
                if start_id not in word_positions or end_id not in word_positions:
                    continue
                start_index = min(word_positions[start_id], word_positions[end_id])
                end_index = max(word_positions[start_id], word_positions[end_id])
                for token_id, index in word_positions.items():
                    if start_index <= index <= end_index:
                        segment_word_ids.append(token_id)

            if not segment_word_ids:
                continue

            segment_word_ids = sorted(segment_word_ids, key=lambda token_id: word_positions[token_id])
            tokens = [word_lookup[token_id] for token_id in segment_word_ids if token_id in word_lookup]
            text = self._render_text(tokens)
            if not text:
                continue

            segment_start = min(word_positions[segment_word_ids[0]], word_positions[segment_word_ids[-1]])
            segment_end = max(word_positions[segment_word_ids[0]], word_positions[segment_word_ids[-1]])
            overlapping_annotations = [
                annotation
                for annotation in dialogue_annotations
                if not (annotation.end_index < segment_start or annotation.start_index > segment_end)
            ]
            act_ids = [annotation.source_id for annotation in overlapping_annotations]
            dialogue_acts = sorted({annotation.label for annotation in overlapping_annotations})
            addressees = sorted({annotation.addressee for annotation in overlapping_annotations if annotation.addressee})
            adjacency_types = sorted(
                {
                    pair_type
                    for act_id in act_ids
                    for pair_type in adjacency_map.get(act_id, [])
                }
            )
            topic = self._pick_topic(segment_start, segment_end, topic_annotations)

            turns.append(
                AmiTurn(
                    meeting_id=meeting_id,
                    speaker=speaker,
                    start_time=float(segment.attrib.get("transcriber_start", 0.0)),
                    end_time=float(segment.attrib.get("transcriber_end", 0.0)),
                    text=text,
                    dialogue_acts=dialogue_acts,
                    adjacency_types=adjacency_types,
                    addressee=", ".join(addressees) if addressees else None,
                    topic=topic,
                    act_ids=act_ids,
                )
            )

        return turns

    @staticmethod
    def _render_text(tokens: Sequence[AmiToken]) -> str:
        parts: List[str] = []
        for token in tokens:
            text = token.text.strip()
            if not text:
                continue

            if not parts:
                parts.append(text)
                continue

            if token.is_punctuation or text in {".", ",", "?", "!", ";", ":", ")", "]"}:
                parts[-1] += text
            elif parts[-1].endswith(("(", "[", "{", "/")):
                parts[-1] += text
            else:
                parts.append(text)

        return " ".join(parts).strip()

    @staticmethod
    def _pick_topic(segment_start: int, segment_end: int, topic_annotations: Sequence[TopicAnnotation]) -> Optional[str]:
        best_label: Optional[str] = None
        best_overlap = -1
        for annotation in topic_annotations:
            overlap = min(segment_end, annotation.end_index) - max(segment_start, annotation.start_index) + 1
            if overlap > best_overlap and overlap > 0:
                best_overlap = overlap
                best_label = annotation.label
        return best_label

    @staticmethod
    def _merge_adjacent_turns(turns: Sequence[AmiTurn]) -> List[AmiTurn]:
        if not turns:
            return []

        merged: List[AmiTurn] = []
        for turn in turns:
            if (
                merged
                and merged[-1].speaker == turn.speaker
                and turn.start_time - merged[-1].end_time <= AmiDataset._MERGE_SAME_SPEAKER_GAP_SECONDS
            ):
                previous = merged[-1]
                merged[-1] = AmiTurn(
                    meeting_id=previous.meeting_id,
                    speaker=previous.speaker,
                    start_time=previous.start_time,
                    end_time=turn.end_time,
                    text=(previous.text + " " + turn.text).strip(),
                    dialogue_acts=sorted(set(previous.dialogue_acts + turn.dialogue_acts)),
                    adjacency_types=sorted(set(previous.adjacency_types + turn.adjacency_types)),
                    addressee=previous.addressee or turn.addressee,
                    topic=previous.topic or turn.topic,
                    act_ids=sorted(set(previous.act_ids + turn.act_ids)),
                )
            else:
                merged.append(turn)
        return merged

    @staticmethod
    def _dominant_topic(turns: Sequence[AmiTurn]) -> Optional[str]:
        topics = [turn.topic for turn in turns if turn.topic]
        if not topics:
            return None
        return Counter(topics).most_common(1)[0][0]

    @staticmethod
    def _build_weak_signals(turns: Sequence[AmiTurn]) -> Dict[str, Any]:
        dialogue_act_counts = Counter(act for turn in turns for act in turn.dialogue_acts)
        adjacency_counts = Counter(kind for turn in turns for kind in turn.adjacency_types)
        speakers = sorted({turn.speaker for turn in turns})
        understanding_related = sum(
            dialogue_act_counts.get(name, 0)
            for name in ("Comment-About-Understanding", "Elicit-Comment-Understanding", "Elicit-Inform")
        )
        return {
            "speaker_count": len(speakers),
            "speakers": speakers,
            "dialogue_act_counts": dict(dialogue_act_counts.most_common(6)),
            "adjacency_counts": dict(adjacency_counts.most_common(6)),
            "understanding_signal_count": understanding_related,
        }

    @staticmethod
    def _load_ontology_map(path: Path, element_name: str) -> Dict[str, str]:
        if not path.exists():
            return {}

        tree = ET.parse(path)
        root = tree.getroot()
        mapping: Dict[str, str] = {}
        for element in root.iter():
            if _tag_name(element) != element_name:
                continue
            element_id = element.attrib.get("{http://nite.sourceforge.net/}id") or element.attrib.get("nite:id") or element.attrib.get("id")
            if not element_id:
                continue
            if element_id.endswith("root"):
                continue
            mapping[element_id] = element.attrib.get("gloss") or element.attrib.get("name") or element_id
        return mapping

    @staticmethod
    def _normalize_topic_pointer(pointer_href: str) -> str:
        pointer_id = _pointer_target_id(pointer_href)
        return pointer_id.replace(".", " ")