import numpy as np
from collections import defaultdict

class Activity2Vec:

    def __init__(self, embedding_dim=8, context_window=3, learning_rate=0.01, negative_samples=5, seed=42):
        self.embedding_dim = embedding_dim
        self.context_window = context_window
        self.lr = learning_rate
        self.negative_samples = negative_samples
        self.rng = np.random.default_rng(seed)
        self.vocab = {}
        self.embeddings = {}
        self.context_embeddings = {}
        self.activity_counts = defaultdict(int)
        self.total_count = 0
        self.case_sequences = defaultdict(list)

    def _get_or_create(self, activity):
        if activity not in self.vocab:
            idx = len(self.vocab)
            self.vocab[activity] = idx
            self.embeddings[activity] = self.rng.normal(0, 0.1, self.embedding_dim).astype(np.float32)
            self.context_embeddings[activity] = self.rng.normal(0, 0.1, self.embedding_dim).astype(np.float32)
        return self.embeddings[activity]

    def update_sequence(self, case_id, activity):
        self._get_or_create(activity)
        self.case_sequences[case_id].append(activity)
        self.activity_counts[activity] += 1
        self.total_count += 1

        seq = self.case_sequences[case_id]
        if len(seq) < 2:
            return

        pos = len(seq) - 1
        target = seq[pos]
        start = max(0, pos - self.context_window)

        for i in range(start, pos):
            context = seq[i]
            self._train_pair(target, context)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def _train_pair(self, target, context):
        t_emb = self.embeddings[target]
        c_emb = self.context_embeddings[context]

        score = np.dot(t_emb, c_emb)
        grad = (self._sigmoid(score) - 1.0) * self.lr
        self.embeddings[target] -= grad * c_emb
        self.context_embeddings[context] -= grad * t_emb

        for _ in range(self.negative_samples):
            neg = self._sample_negative(target)
            if neg is None:
                continue
            n_emb = self.context_embeddings[neg]
            score = np.dot(t_emb, n_emb)
            grad = self._sigmoid(score) * self.lr
            self.embeddings[target] -= grad * n_emb
            self.context_embeddings[neg] -= grad * t_emb

    def _sample_negative(self, exclude):
        if len(self.vocab) < 2:
            return None
        activities = list(self.vocab.keys())
        freqs = np.array([self.activity_counts[a] ** 0.75 for a in activities])
        freqs /= freqs.sum()
        for _ in range(10):
            idx = self.rng.choice(len(activities), p=freqs)
            if activities[idx] != exclude:
                return activities[idx]
        return None

    def get(self, activity):
        if activity in self.embeddings:
            return self.embeddings[activity].tolist()
        return self._get_or_create(activity).tolist()

    def batch_train(self, sequences, epochs=5):
        for activity_seq in sequences:
            for act in activity_seq:
                self._get_or_create(act)

        for _ in range(epochs):
            for seq in sequences:
                for pos in range(1, len(seq)):
                    target = seq[pos]
                    start = max(0, pos - self.context_window)
                    for i in range(start, pos):
                        self._train_pair(target, seq[i])
