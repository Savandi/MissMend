import numpy as np
from scipy.spatial.distance import mahalanobis

class BFRCluster:

    def __init__(self, cluster_id, label, latent_dim, epsilon=1e-4):
        self.cluster_id = cluster_id
        self.label = label
        self.latent_dim = latent_dim
        self.epsilon = epsilon
        self.n = 0
        self.sum_vec = np.zeros(latent_dim)
        self.sum_sq_mat = np.zeros((latent_dim, latent_dim))

    @property
    def centroid(self):
        if self.n == 0:
            return np.zeros(self.latent_dim)
        return self.sum_vec / self.n

    @property
    def covariance(self):
        if self.n < 2:
            return np.eye(self.latent_dim) * self.epsilon
        mean = self.centroid
        cov = (self.sum_sq_mat / self.n) - np.outer(mean, mean)
        cov += np.eye(self.latent_dim) * self.epsilon
        return cov

    @property
    def inv_covariance(self):
        try:
            return np.linalg.inv(self.covariance)
        except np.linalg.LinAlgError:
            return np.eye(self.latent_dim) / self.epsilon

    def update(self, z):
        self.n += 1
        self.sum_vec += z
        self.sum_sq_mat += np.outer(z, z)

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

class StreamingBFR:

    def __init__(self, latent_dim, n_min, n_reliable, alpha=0.5,
                 fuzzifier=2.0, lambda_exp=1.0, epsilon=1e-4,
                 confidence_mode='entropy'):
        self.latent_dim = latent_dim
        self.n_min = n_min
        self.n_reliable = n_reliable
        self.alpha = alpha
        self.fuzzifier = fuzzifier
        self.lambda_exp = lambda_exp
        self.epsilon = epsilon
        self.confidence_mode = confidence_mode
        self.clusters = {}
        self.in_grace_period = True

    def add_cluster(self, label):
        cluster_id = len(self.clusters)
        self.clusters[label] = BFRCluster(cluster_id, label, self.latent_dim, self.epsilon)
        return self.clusters[label]

    def update_cluster(self, label, z):
        if label not in self.clusters:
            self.add_cluster(label)
        self.clusters[label].update(z)
        self._check_grace_period()

    def merge_cluster(self, source_label, target_label):
        if source_label not in self.clusters or target_label not in self.clusters:
            return False
        if source_label == target_label:
            return False
        src = self.clusters[source_label]
        tgt = self.clusters[target_label]
        tgt.n += src.n
        tgt.sum_vec += src.sum_vec
        tgt.sum_sq_mat += src.sum_sq_mat
        del self.clusters[source_label]
        return True

    def find_merge_candidate(self, labelled_label, merge_threshold=2.0,
                              synthetic_prefix='UNDISCOVERED_'):
        if labelled_label not in self.clusters:
            return None
        labelled_cluster = self.clusters[labelled_label]
        if labelled_cluster.n < self.n_min:
            return None

        best_candidate = None
        best_distance = float('inf')

        for label, cluster in self.clusters.items():
            if not str(label).startswith(synthetic_prefix):
                continue
            if cluster.n < self.n_min:
                continue
            d_forward = labelled_cluster.mahalanobis_distance(cluster.centroid)
            d_backward = cluster.mahalanobis_distance(labelled_cluster.centroid)
            symmetric_d = (d_forward + d_backward) / 2.0
            if symmetric_d < best_distance and symmetric_d < merge_threshold:
                best_distance = symmetric_d
                best_candidate = label

        return best_candidate

    def _check_grace_period(self):
        if not self.in_grace_period:
            return
        if all(c.n >= self.n_min for c in self.clusters.values()):
            self.in_grace_period = False

    def _compute_distances(self, z):
        distances = {}
        for label, cluster in self.clusters.items():
            if cluster.n >= self.n_min:
                distances[label] = cluster.mahalanobis_distance(z)
        return distances

    def _compute_fuzzy_memberships(self, distances):
        if not distances:
            return {}
        labels = list(distances.keys())
        dists = np.array([distances[l] for l in labels])

        zero_mask = dists == 0.0
        if np.any(zero_mask):
            memberships = {l: 0.0 for l in labels}
            zero_idx = np.where(zero_mask)[0][0]
            memberships[labels[zero_idx]] = 1.0
            return memberships

        dists = np.maximum(dists, 1e-10)
        inf_mask = ~np.isfinite(dists)
        dists[inf_mask] = 1e10
        exp = 2.0 / (self.fuzzifier - 1.0)
        memberships = {}
        with np.errstate(divide='ignore', invalid='ignore'):
            for i, label in enumerate(labels):
                ratios = (dists[i] / dists) ** exp
                ratios = np.where(np.isfinite(ratios), ratios, 1e10)
                total = np.sum(ratios)
                memberships[label] = 1.0 / total if total > 0 else 0.0
        return memberships

    def _compute_entropy(self, memberships):
        if not memberships:
            return 0.0
        values = np.array(list(memberships.values()))
        values = values[values > 1e-15]
        if len(values) <= 1:
            return 0.0
        return -np.sum(values * np.log(values))

    def _max_entropy(self):
        k = sum(1 for c in self.clusters.values() if c.n >= self.n_min)
        if k <= 1:
            return 1.0
        return np.log(k)

    def _reliability_weight(self, label):
        if label not in self.clusters:
            return 0.0
        n = self.clusters[label].n
        if n < self.n_min:
            return 0.0
        return min(1.0, n / self.n_reliable)

    def compute_confidence(self, z):
        distances = self._compute_distances(z)
        if not distances:
            return None, None, 0.0

        memberships = self._compute_fuzzy_memberships(distances)
        if not memberships:
            return None, None, 0.0

        best_label = max(memberships, key=memberships.get)
        u_star = memberships[best_label]
        sorted_memberships = sorted(memberships.values(), reverse=True)
        u_second = sorted_memberships[1] if len(sorted_memberships) > 1 else 0.0
        w = self._reliability_weight(best_label)

        if self.confidence_mode == 'entropy':
            entropy = self._compute_entropy(memberships)
            h_max = self._max_entropy()
            entropy_term = (1.0 - entropy / h_max) if h_max > 0 else 1.0
            conf = u_star * (entropy_term ** self.lambda_exp) * w
        elif self.confidence_mode == 'margin':
            margin = u_star - u_second
            conf = margin * w
        elif self.confidence_mode == 'log_combined':
            entropy = self._compute_entropy(memberships)
            log_conf = np.log(max(u_star, 1e-10)) - self.lambda_exp * entropy + np.log(max(w, 1e-10))
            conf = 1.0 / (1.0 + np.exp(-log_conf))
        else:
            conf = u_star * w

        return best_label, memberships, conf

    def match(self, z):
        best_label, memberships, conf = self.compute_confidence(z)
        if best_label is None:
            return None, 0.0, 'UNRECOVERED_ML'
        if conf >= self.alpha:
            return best_label, conf, 'RECOVERED_ML'
        return None, conf, 'UNRECOVERED_ML'

    def argmax_label(self, z):
        """Return the cluster matcher's argmax label irrespective of the alpha
        gate. Returns None only when no clusters meet n_min. Used by the
        sequence-head agreement gate, where the cluster's best guess is needed
        even on events the cluster matcher itself abstained on."""
        best_label, _, _ = self.compute_confidence(z)
        return best_label

    def selective_reset(self, per_cluster_errors_pre, per_cluster_errors_post, delta=2.0):
        cleared = []
        for label in list(self.clusters.keys()):
            if label in per_cluster_errors_pre and label in per_cluster_errors_post:
                pre = per_cluster_errors_pre[label]
                post = per_cluster_errors_post[label]
                if pre > 0 and post / pre > delta:
                    self.clusters[label] = BFRCluster(
                        self.clusters[label].cluster_id,
                        label, self.latent_dim, self.epsilon
                    )
                    cleared.append(label)
        if cleared:
            self.in_grace_period = True
        return cleared

    def calibrate(self, labelled_data):
        if len(labelled_data) < 10:
            return

        best_alpha = self.alpha
        best_lambda = self.lambda_exp
        best_f1 = 0.0

        for alpha_candidate in np.arange(0.1, 0.9, 0.1):
            for lambda_candidate in np.arange(0.5, 3.0, 0.5):
                self.alpha = alpha_candidate
                self.lambda_exp = lambda_candidate
                correct = 0
                total = 0
                for z, true_label in labelled_data:
                    pred_label, conf, flag = self.match(z)
                    if flag == 'RECOVERED_ML':
                        total += 1
                        if pred_label == true_label:
                            correct += 1
                precision = correct / total if total > 0 else 0
                recall = correct / len(labelled_data) if labelled_data else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                if f1 > best_f1:
                    best_f1 = f1
                    best_alpha = alpha_candidate
                    best_lambda = lambda_candidate

        self.alpha = best_alpha
        self.lambda_exp = best_lambda
