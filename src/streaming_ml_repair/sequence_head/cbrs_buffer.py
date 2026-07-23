"""Class-balanced reservoir sampling buffer for online sequence-head training.

Follows Chrysakis & Moens (2020) "Online Continual Learning from Imbalanced
Data" (ICML 2020). Vanilla reservoir sampling for a fixed-size buffer K under
a Zipfian class stream concentrates the buffer on majority classes and
starves the tail. CBRS guarantees that each class receives an approximately
equal share of the buffer over a long enough stream by tracking per-class
seen counts and admitting new examples in a way that protects under-
represented classes.

The variant used here is the *full* CBRS algorithm:

1. While buffer not full: append every arriving example.
2. Once full, for each new (target_class) example:
     a. Identify the current largest class in the buffer.
     b. If target_class is one of the largest classes:
          With probability (b_size_of_class / count_seen_for_class), replace
          a uniformly random sample of the current largest class.
        Else: discard the new example.
     c. If target_class is NOT one of the largest classes:
          Replace a uniformly random sample of the current largest class.

The "largest class in buffer" tie-break protects the tail: whenever a tail
class appears, it always displaces a majority class sample, so tail classes
accumulate. Over a long Zipfian stream, the buffer composition converges
to uniform across the classes that have appeared at least once.
"""
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
        """Current count of each class inside the buffer."""
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
        """Replace a uniformly random buffer slot whose class == class_label,
        returning that slot's index. Caller assigns the new item."""
        candidates = [i for i, (_, c) in enumerate(self._items) if c == class_label]
        return self._rng.choice(candidates)

    def add(self, sample: Any, class_label: Any) -> None:
        """Admit an arriving (sample, class) example into the buffer per CBRS."""
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
        """Return up to batch_size items drawn uniformly without replacement
        from the buffer. If the buffer has fewer than batch_size items, return
        them all."""
        if len(self._items) <= batch_size:
            return list(self._items)
        return self._rng.sample(self._items, batch_size)
