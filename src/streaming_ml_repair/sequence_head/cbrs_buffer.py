from __future__ import annotations
import random
from collections import defaultdict
from typing import Any

class CBRSBuffer:

    def __init__(self, capacity: int, random_seed: int | None = None):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._rng = random.Random(random_seed)
        self._items: list = []
        self._seen: dict = defaultdict(int)

    def __len__(self) -> int:
        return len(self._items)

    def class_counts(self) -> dict:
        counts: dict = defaultdict(int)
        for _, c in self._items:
            counts[c] += 1
        return counts

    def _largest_classes(self) -> list:
        counts = self.class_counts()
        if not counts:
            return []
        max_c = max(counts.values())
        return [c for c, v in counts.items() if v == max_c]

    def _replace_random_of_class(self, class_label) -> int:
        candidates = [i for i, (_, c) in enumerate(self._items) if c == class_label]
        return self._rng.choice(candidates)

    def add(self, sample: Any, class_label: Any) -> None:
        self._seen[class_label] += 1
        if len(self._items) < self.capacity:
            self._items.append((sample, class_label))
            return

        largest = self._largest_classes()
        is_arriving_one_of_largest = class_label in largest

        if is_arriving_one_of_largest:
            m_c = sum(1 for _, c in self._items if c == class_label)
            n_c = self._seen[class_label]
            if n_c > 0 and self._rng.random() < (m_c / n_c):
                idx = self._replace_random_of_class(class_label)
                self._items[idx] = (sample, class_label)
            return

        chosen_majority_class = self._rng.choice(largest)
        idx = self._replace_random_of_class(chosen_majority_class)
        self._items[idx] = (sample, class_label)

    def sample_batch(self, batch_size: int) -> list:
        if len(self._items) <= batch_size:
            return list(self._items)
        return self._rng.sample(self._items, batch_size)
