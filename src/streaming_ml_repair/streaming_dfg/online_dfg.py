from __future__ import annotations
from collections import OrderedDict, defaultdict
from typing import Hashable, Optional, Set, Tuple

class OnlineDFG:

    def __init__(
        self,
        max_edges: int = 50000,
        decay_every: int = 5000,
        decay_factor: float = 0.9,
        min_count: float = 0.1,
        parallel_min_transitions: float = 5.0,
        parallel_min_ratio: float = 0.30,
        redetect_every: int = 1000,
    ):
        if max_edges < 1:
            raise ValueError(f"max_edges must be >= 1, got {max_edges}")
        if not (0.0 < decay_factor <= 1.0):
            raise ValueError(f"decay_factor must be in (0, 1], got {decay_factor}")
        if not (0.0 < parallel_min_ratio <= 1.0):
            raise ValueError(f"parallel_min_ratio must be in (0, 1], got {parallel_min_ratio}")

        self.max_edges = int(max_edges)
        self.decay_every = int(decay_every)
        self.decay_factor = float(decay_factor)
        self.min_count = float(min_count)
        self.parallel_min_transitions = float(parallel_min_transitions)
        self.parallel_min_ratio = float(parallel_min_ratio)
        self.redetect_every = int(redetect_every)

        self._edges: "OrderedDict[Tuple[Hashable, Hashable], float]" = OrderedDict()
        self._parallel_pairs: Set[Tuple[Hashable, Hashable]] = set()
        self._parallel_siblings: dict = defaultdict(set)
        self._n_added: int = 0
        self._n_added_since_redetect: int = 0

    def add(self, prev_activity: Hashable, current_activity: Hashable) -> None:
        if prev_activity is None or current_activity is None:
            return
        key = (prev_activity, current_activity)
        if key in self._edges:
            self._edges[key] += 1.0
            self._edges.move_to_end(key)
        else:
            self._edges[key] = 1.0
            if len(self._edges) > self.max_edges:
                self._edges.popitem(last=False)
        self._n_added += 1
        self._n_added_since_redetect += 1
        if self.decay_every > 0 and self._n_added % self.decay_every == 0:
            self._decay()
        if self._n_added_since_redetect >= self.redetect_every:
            self.redetect_parallel_pairs()

    def redetect_parallel_pairs(self) -> None:
        pairs: Set[Tuple[Hashable, Hashable]] = set()
        siblings: dict = defaultdict(set)
        edges = dict(self._edges)
        seen_unordered: Set[frozenset] = set()
        for (a, b), c_ab in edges.items():
            if a == b:
                continue
            unordered = frozenset((a, b))
            if unordered in seen_unordered:
                continue
            seen_unordered.add(unordered)
            c_ba = edges.get((b, a), 0.0)
            min_c = min(c_ab, c_ba)
            max_c = max(c_ab, c_ba)
            if (min_c >= self.parallel_min_transitions
                    and max_c > 0.0
                    and (min_c / max_c) >= self.parallel_min_ratio):
                pairs.add((a, b))
                pairs.add((b, a))
                siblings[a].add(b)
                siblings[b].add(a)
        self._parallel_pairs = pairs
        self._parallel_siblings = siblings
        self._n_added_since_redetect = 0

    def parallel_features_for(self, prev_activity: Hashable) -> Tuple[float, float, float]:
        if prev_activity is None:
            return 0.0, 0.0, 0.0
        siblings = self._parallel_siblings.get(prev_activity, set())
        is_parallel = 1.0 if siblings else 0.0
        n_siblings = float(len(siblings))
        if not siblings:
            return is_parallel, n_siblings, 0.0
        total_out = 0.0
        parallel_out = 0.0
        for (a, b), c in self._edges.items():
            if a == prev_activity:
                total_out += c
                if b in siblings:
                    parallel_out += c
        density = (parallel_out / total_out) if total_out > 0.0 else 0.0
        return is_parallel, n_siblings, float(density)

    def n_edges(self) -> int:
        return len(self._edges)

    def n_parallel_pairs(self) -> int:
        return len(self._parallel_pairs) // 2

    def _decay(self) -> None:
        to_remove = []
        for key in list(self._edges.keys()):
            self._edges[key] *= self.decay_factor
            if self._edges[key] < self.min_count:
                to_remove.append(key)
        for key in to_remove:
            del self._edges[key]
