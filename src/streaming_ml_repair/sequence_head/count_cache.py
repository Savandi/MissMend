"""Bounded-order n-gram count cache for prefix-deterministic streams.

This module provides ``CountCache``: a streaming, non-parametric memoriser that
records, for every observed length-k activity prefix, the empirical distribution
over the following activity. It is a complementary recovery signal to the
SDAE/BFR feature-view cluster matcher and the LSTM rescue head.

Motivation
~~~~~~~~~~
On prefix-deterministic streams (ViennaLine71, CybersecIoT, Chess Piece
Production) the next activity is often a (near-)deterministic function of the
last 2-3 activities. The cluster matcher conditions on the joint
multi-perspective representation, but its commit decisions can be wrong with
high confidence when the IoT/data signal is misleading and the prefix evidence
is much sharper. Likewise, the LSTM rescue head spends ~1M parameters trying to
learn what is essentially a deterministic lookup, and class-balanced replay
mildly blurs the deterministic mappings.

A count table memorises a deterministic mapping exactly, in O(1) per update,
and provides a directly interpretable confidence (the fraction of the dominant
successor among observed transitions for that prefix). It does not generalise
across prefixes, which is the right inductive bias for low-entropy streams.

Streaming compliance
~~~~~~~~~~~~~~~~~~~~
The cache uses bounded memory through proportional decay every ``decay_every``
labelled events: every stored count is multiplied by ``decay_factor`` in
(0, 1]. This preserves relative frequencies while shrinking the effective
sample size so the cache adapts to drift and old prefixes do not dominate
indefinitely. Memory is also bounded by an LRU eviction once the number of
distinct prefixes exceeds ``max_prefixes``.

All operations on a single event are O(k) for the prefix tuple hash and O(1)
for the update. No backpropagation, no gradient state, no GPU residence.

API
~~~
``CountCache.add(prefix, next_activity)`` records a labelled transition.

``CountCache.predict(prefix)`` returns ``(top_activity, dominance, support)``
where ``dominance`` is the fraction of observations of ``prefix`` followed by
``top_activity`` (in [0, 1]) and ``support`` is the total number of
observations of ``prefix`` (a positive integer if the prefix has been seen,
else 0). If the prefix has not been observed at all, returns
``(None, 0.0, 0)``.

The decision policies (override cluster commits, emit on abstention, etc.) live
in the pipeline orchestrator, not in this module. This module is a pure cache.
"""
from __future__ import annotations
from collections import OrderedDict, defaultdict
from typing import Hashable, Optional, Tuple

class CountCache:
    """Bounded-memory n-gram count table with proportional decay.

    Args:
        order: prefix length k. Each cache key is the most recent k activities
            of a case. Default 3.
        max_prefixes: maximum number of distinct prefixes stored. Older
            prefixes are evicted in LRU order when the cap is exceeded. Default
            50000.
        decay_every: trigger a proportional decay sweep every this many added
            transitions. 0 disables decay. Default 5000.
        decay_factor: multiplier applied to every stored count during a decay
            sweep, in (0, 1]. Default 0.9.
        min_count: counts below this threshold after decay are pruned.
            Default 0.1.
    """

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
        """Record a labelled (prefix, next_activity) transition.

        The prefix is automatically truncated/padded by the caller; this method
        assumes ``prefix`` is already of length ``self.order`` (or shorter for
        the start of a case, in which case the caller must left-pad). The cache
        keys on whatever tuple it receives.
        """
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
        """Return ``(top_activity, dominance, support)`` for the prefix.

        ``dominance`` is the fraction of observations following ``prefix`` that
        matched ``top_activity``; ``support`` is the total weighted observation
        count (a float because of decay). Returns ``(None, 0.0, 0.0)`` for an
        unseen prefix.
        """
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
        """Apply proportional decay; prune sub-threshold entries.

        Every stored count is multiplied by ``decay_factor``. Entries whose
        every count falls below ``min_count`` are removed. The relative
        successor proportions per prefix are preserved exactly; only the
        effective sample size shrinks.
        """
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
