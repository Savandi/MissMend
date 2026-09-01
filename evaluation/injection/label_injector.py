import random
import copy
from contextlib import contextmanager

class MissingLabelInjector:

    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.injected_indices = set()
        self.ground_truth = {}

    def inject_random(self, events, rate=0.1):
        labelled_indices = [
            i for i, e in enumerate(events)
            if e.concept_name and str(e.concept_name).strip()
        ]
        n_inject = int(len(labelled_indices) * rate)
        selected = self.rng.sample(labelled_indices, min(n_inject, len(labelled_indices)))

        for idx in selected:
            self.ground_truth[idx] = events[idx].concept_name
            events[idx].concept_name = None
            self.injected_indices.add(idx)

        return events

    @contextmanager
    def inject_in_place(self, events, rate=0.1):
        labelled_indices = [
            i for i, e in enumerate(events)
            if e.concept_name and str(e.concept_name).strip()
        ]
        n_inject = int(len(labelled_indices) * rate)
        selected = self.rng.sample(labelled_indices, min(n_inject, len(labelled_indices)))

        for idx in selected:
            self.ground_truth[idx] = events[idx].concept_name
            events[idx].concept_name = None
            self.injected_indices.add(idx)

        try:
            yield events
        finally:
            for idx, original_label in self.ground_truth.items():
                events[idx].concept_name = original_label

    def inject_burst(self, events, rate=0.1, burst_size=5):
        labelled_indices = [
            i for i, e in enumerate(events)
            if e.concept_name and str(e.concept_name).strip()
        ]
        n_inject = int(len(labelled_indices) * rate)
        injected = 0

        while injected < n_inject and labelled_indices:
            start_pos = self.rng.randint(0, len(labelled_indices) - 1)
            for j in range(burst_size):
                if start_pos + j < len(labelled_indices) and injected < n_inject:
                    idx = labelled_indices[start_pos + j]
                    if idx not in self.injected_indices:
                        self.ground_truth[idx] = events[idx].concept_name
                        events[idx].concept_name = None
                        self.injected_indices.add(idx)
                        injected += 1

        return events

    def inject_activity_specific(self, events, target_activities, rate=0.5):
        for i, e in enumerate(events):
            if (e.concept_name and str(e.concept_name) in target_activities
                    and self.rng.random() < rate and i not in self.injected_indices):
                self.ground_truth[i] = e.concept_name
                e.concept_name = None
                self.injected_indices.add(i)

        return events

    def get_ground_truth(self):
        return dict(self.ground_truth)

    def reset(self):
        self.injected_indices = set()
        self.ground_truth = {}
