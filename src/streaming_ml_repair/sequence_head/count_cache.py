from __future__ import annotations
from collections import OrderedDict, defaultdict
from typing import Hashable, Optional, Tuple

class CountCache:

    def __init__(
        self,
        order: int = 3,
        max_prefixes: int = 50000,
        decay_every: int = 5000,
        decay_factor: float = 0.9,
        min_count: float = 0.1,
    ):
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        if not (0.0 < decay_factor <= 1.0):
            raise ValueError(f"decay_factor must be in (0, 1], got {decay_factor}")
        self.order = int(order)
        self.max_prefixes = int(max_prefixes)
        self.decay_every = int(decay_every)
        self.decay_factor = float(decay_factor)
        self.min_count = float(min_count)

        self._table: "OrderedDict[Tuple[Hashable, ...], dict]" = OrderedDict()
        self._n_added: int = 0

    def add(self, prefix: Tuple[Hashable, ...], next_activity: Hashable) -> None:
        key = tuple(prefix)
        if key in self._table:
            entry = self._table[key]
            entry[next_activity] = entry.get(next_activity, 0.0) + 1.0
            self._table.move_to_end(key)
        else:
            self._table[key] = {next_activity: 1.0}
            if len(self._table) > self.max_prefixes:
                self._table.popitem(last=False)
        self._n_added += 1
        if self.decay_every > 0 and self._n_added % self.decay_every == 0:
            self._decay()

    def predict(
        self, prefix: Tuple[Hashable, ...]
    ) -> Tuple[Optional[Hashable], float, float]:
        key = tuple(prefix)
        entry = self._table.get(key)
        if not entry:
            return None, 0.0, 0.0
        self._table.move_to_end(key)
        top_activity, top_count = max(entry.items(), key=lambda kv: kv[1])
        support = sum(entry.values())
        if support <= 0.0:
            return None, 0.0, 0.0
        return top_activity, float(top_count / support), float(support)

    def __len__(self) -> int:
        return len(self._table)

    def total_observations(self) -> float:
        return float(sum(sum(e.values()) for e in self._table.values()))

    def _decay(self) -> None:
        to_remove = []
        for key, entry in self._table.items():
            for act in list(entry.keys()):
                entry[act] *= self.decay_factor
                if entry[act] < self.min_count:
                    del entry[act]
            if not entry:
                to_remove.append(key)
        for key in to_remove:
            del self._table[key]
