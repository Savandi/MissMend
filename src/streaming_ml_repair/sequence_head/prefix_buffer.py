"""Per-case prefix buffer for the sequence rescue head.

Maintains a rolling window of the most recent N events per case, each tagged
with provenance (NORMAL, RECOVERED_ML, RECOVERED_ML_SEQ, UNRECOVERED_ML).
The provenance tag is what enables the autoregressive prefix-contamination
analysis: at rescue invocation we know how much of the prefix is observed
vs repaired, and we can stratify the rescue F1 by contamination fraction.

The buffer is maintained eagerly for every arriving event, regardless of
whether the rescue branch fires, because we don't know in advance which
events will need rescue.

This module is provider-agnostic: it just stores activity-id tokens. Whether
the head also wants z latents (z_latent input mode) is decided in the
pipeline integration; the activity-id buffer is always maintained because
the autoregressive analysis applies regardless of input mode.
"""
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
    """One entry in a per-case prefix buffer."""
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
        """Record an event in its case prefix."""
        self._buffers[case_id].append(PrefixEntry(
            activity_id=activity_id, provenance=provenance, confidence=confidence,
        ))

    def get_prefix_ids(self, case_id) -> list:
        """Return the activity-id list for this case's prefix, left-padded with
        PAD_TOKEN_ID to exactly window_size positions. Used as direct input to
        the activity-id LSTM head."""
        entries = list(self._buffers.get(case_id, ()))
        pad_count = self.window_size - len(entries)
        ids = [PAD_TOKEN_ID] * pad_count + [e.activity_id for e in entries]
        return ids

    def snapshot_prefix_ids(self, case_id) -> tuple:
        """Return an IMMUTABLE value-copy snapshot of the prefix activity-id
        sequence (padded). The returned tuple is decoupled from the live buffer
        so subsequent calls to append() cannot mutate the snapshot. This is
        the binding used by the rescue head to ensure the training pair
        (prefix_t, a_t) is built from the prefix BEFORE a_t is admitted to the
        buffer."""
        return tuple(self.get_prefix_ids(case_id))

    def get_prefix_entries(self, case_id) -> list:
        """Return the raw PrefixEntry list (no padding) for this case's prefix.
        Used by the contamination analysis."""
        return list(self._buffers.get(case_id, ()))

    def composition(self, case_id) -> dict:
        """Return a dict counting provenance kinds in this case's prefix.
        Used to log autoregressive contamination at rescue invocation."""
        comp = {NORMAL: 0, RECOVERED_ML: 0, RECOVERED_ML_SEQ: 0, UNRECOVERED_ML: 0}
        for e in self._buffers.get(case_id, ()):
            if e.provenance in comp:
                comp[e.provenance] += 1
        return comp

    def contamination_fraction(self, case_id) -> float:
        """Fraction of the case's prefix that is NOT directly observed
        (i.e. repaired by ML head, repaired by SEQ head, or unrecovered).
        Returns 0.0 if the prefix is empty."""
        entries = list(self._buffers.get(case_id, ()))
        if not entries:
            return 0.0
        non_normal = sum(1 for e in entries if e.provenance != NORMAL)
        return non_normal / len(entries)
