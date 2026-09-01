from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

from src.streaming_ml_repair.sequence_head.lstm_head import (
    PAD_TOKEN_ID, MISSING_TOKEN_ID,
)

NORMAL = "NORMAL"
RECOVERED_ML = "RECOVERED_ML"
RECOVERED_ML_SEQ = "RECOVERED_ML_SEQ"
UNRECOVERED_ML = "UNRECOVERED_ML"

@dataclass
class PrefixEntry:
    activity_id: int
    provenance: str
    confidence: float

class PerCasePrefixBuffer:

    def __init__(self, window_size: int):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = window_size
        self._buffers: dict = defaultdict(lambda: deque(maxlen=window_size))

    def append(self, case_id, activity_id: int, provenance: str, confidence: float) -> None:
        self._buffers[case_id].append(PrefixEntry(
            activity_id=activity_id, provenance=provenance, confidence=confidence,
        ))

    def get_prefix_ids(self, case_id) -> list:
        entries = list(self._buffers.get(case_id, ()))
        pad_count = self.window_size - len(entries)
        ids = [PAD_TOKEN_ID] * pad_count + [e.activity_id for e in entries]
        return ids

    def snapshot_prefix_ids(self, case_id) -> tuple:
        return tuple(self.get_prefix_ids(case_id))

    def get_prefix_entries(self, case_id) -> list:
        return list(self._buffers.get(case_id, ()))

    def composition(self, case_id) -> dict:
        comp = {NORMAL: 0, RECOVERED_ML: 0, RECOVERED_ML_SEQ: 0, UNRECOVERED_ML: 0}
        for e in self._buffers.get(case_id, ()):
            if e.provenance in comp:
                comp[e.provenance] += 1
        return comp

    def contamination_fraction(self, case_id) -> float:
        entries = list(self._buffers.get(case_id, ()))
        if not entries:
            return 0.0
        non_normal = sum(1 for e in entries if e.provenance != NORMAL)
        return non_normal / len(entries)
