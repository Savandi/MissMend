import numpy as np
from collections import defaultdict

class PPMIActivityEmbeddings:

    def __init__(self, embedding_dim=32, context_window=3):
        self.embedding_dim = embedding_dim
        self.context_window = context_window
        self.cooccurrence = defaultdict(lambda: defaultdict(float))
        self.activity_freq = defaultdict(float)
        self.total_pairs = 0.0
        self.vocab = {}
        self.embeddings = {}
        self.case_sequences = defaultdict(list)
        self._needs_rebuild = True

    def update_sequence(self, case_id, activity):
        self.case_sequences[case_id].append(activity)
        seq = self.case_sequences[case_id]
        pos = len(seq) - 1

        if activity not in self.vocab:
            self.vocab[activity] = len(self.vocab)

        start = max(0, pos - self.context_window)
        for i in range(start, pos):
            context = seq[i]
            self.cooccurrence[activity][context] += 1.0
            self.cooccurrence[context][activity] += 1.0
            self.activity_freq[activity] += 1.0
            self.activity_freq[context] += 1.0
            self.total_pairs += 2.0

        self._needs_rebuild = True

    def _build_ppmi_matrix(self):
        activities = sorted(self.vocab.keys())
        n = len(activities)
        if n == 0:
            return np.zeros((0, 0)), activities

        act_to_idx = {a: i for i, a in enumerate(activities)}
        matrix = np.zeros((n, n))

        for a in activities:
            for b in self.cooccurrence[a]:
                if b in act_to_idx:
                    cooc = self.cooccurrence[a][b]
                    freq_a = self.activity_freq[a]
                    freq_b = self.activity_freq[b]
                    if freq_a > 0 and freq_b > 0 and self.total_pairs > 0:
                        pmi = np.log(max(cooc * self.total_pairs / (freq_a * freq_b), 1e-10))
                        matrix[act_to_idx[a], act_to_idx[b]] = max(pmi, 0.0)

        return matrix, activities

    def rebuild_embeddings(self):
        matrix, activities = self._build_ppmi_matrix()
        if len(activities) == 0:
            return

        dim = min(self.embedding_dim, len(activities) - 1)
        if dim <= 0:
            for act in activities:
                self.embeddings[act] = np.zeros(self.embedding_dim).tolist()
            return

        try:
            U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
            emb_matrix = U[:, :dim] * np.sqrt(S[:dim])
        except np.linalg.LinAlgError:
            emb_matrix = np.random.normal(0, 0.1, (len(activities), dim))

        for i, act in enumerate(activities):
            emb = np.zeros(self.embedding_dim)
            emb[:dim] = emb_matrix[i]
            self.embeddings[act] = emb.tolist()

        self._needs_rebuild = False

    def get(self, activity):
        if self._needs_rebuild and self.total_pairs > 100:
            self.rebuild_embeddings()
        if activity in self.embeddings:
            return self.embeddings[activity]
        return [0.0] * self.embedding_dim
