import numpy as np

class CandidateCluster:

    def __init__(self, cluster_id, latent_dim, epsilon=1e-4):
        self.cluster_id = cluster_id
        self.latent_dim = latent_dim
        self.epsilon = epsilon
        self.n = 0
        self.sum_vec = np.zeros(latent_dim)
        self.sum_sq_mat = np.zeros((latent_dim, latent_dim))
        self.first_seen = None
        self.last_seen = None
        self.members = []

    @property
    def centroid(self):
        if self.n == 0:
            return np.zeros(self.latent_dim)
        return self.sum_vec / self.n

    @property
    def sample_covariance(self):
        if self.n < 2:
            return np.eye(self.latent_dim) * self.epsilon
        mean = self.centroid
        cov = (self.sum_sq_mat / self.n) - np.outer(mean, mean)
        return cov

    def _ledoit_wolf_shrinkage(self, S):
        d = self.latent_dim
        mu = np.trace(S) / d
        target = mu * np.eye(d)
        if not self.members or len(self.members) < 2:
            return target, 1.0
        X = np.asarray(self.members, dtype=np.float64)
        n = X.shape[0]
        mean = np.mean(X, axis=0)
        Xc = X - mean
        num = 0.0
        for i in range(n):
            xi = Xc[i]
            outer_i = np.outer(xi, xi)
            diff = outer_i - S
            num += np.sum(diff * diff)
        num /= (n * n)
        denom = np.sum((S - target) ** 2)
        if denom <= 0:
            return target, 1.0
        lam = max(0.0, min(1.0, num / denom))
        shrunk = (1.0 - lam) * S + lam * target
        return shrunk, lam

    @property
    def covariance(self):
        if self.n < 2:
            return np.eye(self.latent_dim) * max(self.epsilon, 1e-3)
        S = self.sample_covariance
        shrunk, _ = self._ledoit_wolf_shrinkage(S)
        shrunk += np.eye(self.latent_dim) * self.epsilon
        return shrunk

    @property
    def inv_covariance(self):
        try:
            return np.linalg.inv(self.covariance)
        except np.linalg.LinAlgError:
            return np.eye(self.latent_dim) / max(self.epsilon, 1e-3)

    def update(self, z, event_idx, store_member=True):
        if self.first_seen is None:
            self.first_seen = event_idx
        self.last_seen = event_idx
        self.n += 1
        self.sum_vec += z
        self.sum_sq_mat += np.outer(z, z)
        if store_member:
            self.members.append(np.asarray(z, dtype=np.float64))

    def mahalanobis_distance(self, z):
        if self.n < 2:
            return float('inf')
        diff = z - self.centroid
        try:
            val = diff @ self.inv_covariance @ diff
            if not np.isfinite(val) or val < 0:
                return float('inf')
            return float(np.sqrt(max(val, 0.0)))
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            return float('inf')

class CandidateActivityBFR:

    def __init__(self, latent_dim, n_new=10, persist_time=5,
                 separation_threshold=3.0, match_threshold=2.0,
                 min_members_for_match=3, fuzzifier=2.0, epsilon=1e-4,
                 max_members_stored=200):
        self.latent_dim = latent_dim
        self.n_new = n_new
        self.persist_time = persist_time
        self.separation_threshold = separation_threshold
        self.match_threshold = match_threshold
        self.min_members_for_match = min_members_for_match
        self.fuzzifier = fuzzifier
        self.epsilon = epsilon
        self.max_members_stored = max_members_stored
        self.candidates = {}
        self.next_id = 0
        self.event_count = 0

    def add(self, z):
        self.event_count += 1
        z = np.asarray(z, dtype=np.float64)
        best_id = self._find_best_candidate(z)
        if best_id is not None:
            cand = self.candidates[best_id]
            store = len(cand.members) < self.max_members_stored
            cand.update(z, self.event_count, store_member=store)
        else:
            new_id = self.next_id
            self.next_id += 1
            cand = CandidateCluster(new_id, self.latent_dim, self.epsilon)
            cand.update(z, self.event_count, store_member=True)
            self.candidates[new_id] = cand

    def _find_best_candidate(self, z):
        eligible = [(cid, cand) for cid, cand in self.candidates.items()
                    if cand.n >= self.min_members_for_match]
        if not eligible:
            return None

        best_id = None
        best_dist = float('inf')
        for cid, cand in eligible:
            d = cand.mahalanobis_distance(z)
            if d < best_dist:
                best_dist = d
                best_id = cid

        if best_id is not None and best_dist <= self.match_threshold:
            return best_id
        return None

    def _separation_to_reference(self, candidate, reference_clusters):
        min_dist = float('inf')
        for ref in reference_clusters.values():
            if ref.n < 2:
                continue
            d = candidate.mahalanobis_distance(ref.centroid)
            if d < min_dist:
                min_dist = d
        return min_dist

    def check_promotion(self, reference_clusters):
        promoted = []
        for cid in list(self.candidates.keys()):
            cand = self.candidates[cid]
            if cand.n < self.n_new:
                continue
            if cand.first_seen is None or cand.last_seen is None:
                continue
            if (cand.last_seen - cand.first_seen) < self.persist_time:
                continue
            separation = self._separation_to_reference(cand, reference_clusters)
            if separation <= self.separation_threshold:
                continue

            promoted.append({
                'cluster_id': cid,
                'centroid': cand.centroid.copy(),
                'members': [m.copy() for m in cand.members],
                'size': cand.n,
                'first_seen': cand.first_seen,
                'last_seen': cand.last_seen,
                'separation': separation,
            })
            del self.candidates[cid]

        return promoted

    def summary(self):
        return {
            'num_candidates': len(self.candidates),
            'event_count': self.event_count,
            'candidates': [
                {'id': c.cluster_id, 'n': c.n,
                 'age': (c.last_seen - c.first_seen) if c.first_seen is not None else 0}
                for c in self.candidates.values()
            ],
        }
