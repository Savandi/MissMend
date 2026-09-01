import numpy as np
import torch
import math
import random
from collections import defaultdict
from river.drift import ADWIN

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
from src.streaming_ml_repair.clustering.streaming_fuzzy_bfr import StreamingBFR
from src.streaming_ml_repair.sdae.streaming_sparse_denoising_autoencoder import SparseDenoisingAutoencoder
from src.streaming_ml_repair.sdae.cluster_aware_loss import ClusterAwareLoss
from src.streaming_ml_repair.feature_vector.iot_feature_builder import IoTFeatureBuilder
from src.streaming_ml_repair.feature_vector.online_activity_embeddings import DynamicActivityEmbeddings
from src.streaming_ml_repair.feature_vector.online_onehot_encoder import OnlineOneHotEncoder
from src.streaming_ml_repair.feature_vector.data_payload_builder import DataPayloadBuilder
from src.streaming_ml_repair.undiscovered.candidate_activity_bfr import CandidateActivityBFR
from src.streaming_ml_repair.sdae.masked_pretraining import MaskedActivityPretrainer
from src.streaming_ml_repair.sequence_head import SequenceRescueHead
from src.streaming_ml_repair.sequence_head.prefix_buffer import (
    PerCasePrefixBuffer, PAD_TOKEN_ID, MISSING_TOKEN_ID,
)
from src.streaming_ml_repair.sequence_head.cbrs_buffer import CBRSBuffer
from src.streaming_ml_repair.sequence_head.count_cache import CountCache
from src.streaming_ml_repair.streaming_dfg.online_dfg import OnlineDFG
from src.streaming_ml_repair.calibration import TemperatureScaler, PlattScaler

class StreamingMLRepairPipeline:

    def __init__(self, config):
        self.config = config
        self.sensor_vocabulary = config.get('sensor_vocabulary', [])
        self.latent_dim = config.get('latent_dim', 32)
        self.hidden_dims = config.get('hidden_dims', [128, 64])
        self.window_size = config.get('window_size', 10)
        self.n_min = config.get('n_min', 20)
        self.n_reliable = config.get('n_reliable', 50)
        self.alpha = config.get('alpha', 0.5)
        self.fuzzifier = config.get('fuzzifier', 2.0)
        self.lambda_exp = config.get('lambda_exp', 1.0)
        self.sparsity_lambda = config.get('sparsity_lambda', 0.01)
        self.noise_std = config.get('noise_std', 0.1)
        self.warmup_events = config.get('warmup_events', 500)
        self.retrain_interval = config.get('retrain_interval', 1000)
        self.delta = config.get('delta', 2.0)
        self.embedding_dim = config.get('embedding_dim', 8)
        self.max_lifecycle_categories = config.get('max_lifecycle_categories', 20)
        self.training_epochs = config.get('training_epochs', 100)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.auto_calibrate = config.get('auto_calibrate', True)

        self.iot_builder = IoTFeatureBuilder(self.sensor_vocabulary)
        self.activity_embeddings = DynamicActivityEmbeddings(self.embedding_dim, seed=42)
        self.lifecycle_encoder = OnlineOneHotEncoder(self.max_lifecycle_categories)
        self.endpoint_encoder = OnlineOneHotEncoder(config.get('max_endpoints', 100))
        self.activity_uuid_encoder = OnlineOneHotEncoder(config.get('max_uuids', 100))
        self.data_payload_builder = DataPayloadBuilder(
            max_keys=config.get('max_data_keys', 50),
            hash_dim=config.get('data_hash_dim', 8)
        )
        self.data_attribute_keys = config.get('data_attribute_keys', ['concept:endpoint', 'cpee:activity_uuid'])

        self.input_dim = None
        self.sdae = None
        self.bfr = None
        self.pretrainer = None
        self.cf_feature_count = self.window_size * (self.embedding_dim + 1)
        self.adwin_global = ADWIN(delta=config.get('adwin_delta', 0.002))
        self.adwin_per_cluster = defaultdict(lambda: ADWIN(delta=config.get('adwin_delta', 0.002)))
        self.undiscovered_buffer = CandidateActivityBFR(
            latent_dim=config.get('latent_dim', 32),
            n_new=config.get('undiscovered_n_new', 10),
            persist_time=config.get('undiscovered_persist_time', 5),
            separation_threshold=config.get('undiscovered_separation', 3.0),
            match_threshold=config.get('candidate_match_threshold', 2.0),
            min_members_for_match=config.get('candidate_min_members', 3),
            fuzzifier=config.get('fuzzifier', 2.0),
            epsilon=config.get('epsilon', 1e-4),
            max_members_stored=config.get('candidate_max_members_stored', 200),
        )

        self.case_windows = defaultdict(list)
        self.case_activity_counts = defaultdict(lambda: defaultdict(int))
        self.case_first_ts = {}
        self.case_event_totals = defaultdict(int)
        self.enable_case_features = config.get('enable_case_features', False)
        self.dfg = defaultdict(lambda: defaultdict(int))
        self.outgoing_dfg = defaultdict(lambda: defaultdict(int))
        self.activity_counts = defaultdict(int)
        self.warmup_buffer = []
        self.event_count = 0
        self.is_warmed_up = False
        self.recent_labelled = []
        self.recent_labelled_per_cluster = defaultdict(list)
        self.max_recent_per_cluster = config.get('max_recent_per_cluster', 200)
        self.per_cluster_errors_pre = defaultdict(list)
        self.drift_detected = False

        self.results = []
        self.reconstruction_errors = []
        self.reconciliation_log = []
        self.cluster_drift_count = defaultdict(int)
        self.severe_drift_count = defaultdict(int)
        self.gradual_drift_count = 0
        self.warmup_activities_seen = set()
        self.warmup_last_new_activity_at = 0
        self.warmup_stability_window = config.get('warmup_stability_window', 500)
        self.warmup_min_events = config.get('warmup_min_events', 200)
        self.warmup_min_activities = config.get('warmup_min_activities', 5)
        self.warmup_sufficient_activities = config.get(
            'warmup_sufficient_activities', 20
        )
        self.warmup_max_events = config.get(
            'warmup_max_events', 2 * self.warmup_events
        )

        self.enable_cluster_calibration = config.get('enable_cluster_calibration', False)
        self.enable_sequence_head = config.get('enable_sequence_head', False)
        self.sequence_input_mode = config.get('sequence_input_mode', 'activity_id')
        self.alpha_seq = config.get('alpha_seq', self.alpha)
        self.cbrs_capacity = config.get('cbrs_capacity', 500)
        self.cbrs_batch_size = config.get('cbrs_batch_size', 16)
        self.sequence_hidden_dim = config.get('sequence_hidden_dim', 64)
        self.sequence_embedding_dim = config.get('sequence_embedding_dim', 64)
        self.sequence_dropout = config.get('sequence_dropout', 0.1)
        self.sequence_lr = config.get('sequence_lr', 1e-3)
        self.sequence_warmup_epochs = config.get('sequence_warmup_epochs', 5)
        self.sequence_num_layers = config.get('sequence_num_layers', 1)
        self.calibration_holdout_fraction = config.get('calibration_holdout_fraction', 0.10)
        self.enable_agreement_gating = config.get('enable_agreement_gating', False)
        self.alpha_seq_high = config.get('alpha_seq_high', 0.70)
        self.agreement_topk = config.get('agreement_topk', 3)
        self.agreement_high_conf_override = config.get(
            'agreement_high_conf_override', 1.01
        )
        self.agreement_gated_blocks = 0
        self.agreement_high_conf_bypasses = 0
        self.retro_alpha_cluster = config.get(
            'retro_alpha_cluster', config.get('alpha_cluster', self.alpha)
        )
        self.retro_alpha_seq = config.get(
            'retro_alpha_seq', config.get('alpha_seq', self.alpha)
        )

        self.sequence_head = None
        self.sequence_optimizer = None
        self.prefix_buffer = PerCasePrefixBuffer(window_size=self.window_size)
        self.replay_buffer = None
        self.temp_scaler = TemperatureScaler()
        self.platt_scaler = PlattScaler()

        self.enable_count_cache = bool(config.get('enable_count_cache', False))
        self.count_cache = None
        if self.enable_count_cache:
            self.count_cache = CountCache(
                order=int(config.get('count_cache_order', 3)),
                max_prefixes=int(config.get('count_cache_max_prefixes', 50000)),
                decay_every=int(config.get('count_cache_decay_every', 5000)),
                decay_factor=float(config.get('count_cache_decay_factor', 0.9)),
                min_count=float(config.get('count_cache_min_count', 0.1)),
            )
        self.count_cache_override_dominance = float(
            config.get('count_cache_override_dominance', 0.90))
        self.count_cache_override_support = float(
            config.get('count_cache_override_support', 20.0))
        self.count_cache_emit_dominance = float(
            config.get('count_cache_emit_dominance', 0.95))
        self.count_cache_emit_support = float(
            config.get('count_cache_emit_support', 50.0))
        self.count_cache_overrides = 0
        self.count_cache_direct_emits = 0

        self.enable_online_dfg = bool(config.get('enable_online_dfg', False))
        self.online_dfg = None
        if self.enable_online_dfg:
            self.online_dfg = OnlineDFG(
                max_edges=int(config.get('dfg_max_edges', 50000)),
                decay_every=int(config.get('dfg_decay_every', 5000)),
                decay_factor=float(config.get('dfg_decay_factor', 0.9)),
                min_count=float(config.get('dfg_min_count', 0.1)),
                parallel_min_transitions=float(config.get('dfg_parallel_min_transitions', 5.0)),
                parallel_min_ratio=float(config.get('dfg_parallel_min_ratio', 0.30)),
                redetect_every=int(config.get('dfg_redetect_every', 1000)),
            )
        self.holdout_calibration_pairs: list = []
        self.seq_activity_to_id: dict = {}
        self.seq_id_to_activity: dict = {}
        self._seq_vocab_frozen: bool = False
        self.seq_rescue_invocations = 0
        self.seq_rescue_commits = 0
        self.contamination_log = []

        self.enable_retroactive_warmup_recovery = config.get(
            'enable_retroactive_warmup_recovery', True
        )
        self.retro_events = 0
        self.retro_tp_cluster = 0
        self.retro_tp_rescue = 0
        self.retro_abs = 0

        self.alpha_cluster = config.get('alpha_cluster', self.alpha)
        self.class_prior_beta = config.get('class_prior_beta', 0.0)
        self.prior_adjustments = 0
        self.margin_threshold = config.get('margin_threshold', 0.0)
        self.margin_abstentions = 0
        self.arbitration = config.get('arbitration', False)
        self.arbitration_margin = config.get('arbitration_margin', 0.05)
        self.arbitration_invocations = 0
        self.arbitration_overrides = 0

    def _should_end_warmup(self):
        if self.event_count < self.warmup_min_events:
            return False

        current_label = None
        if self.warmup_buffer:
            _, label, _ = self.warmup_buffer[-1]
            current_label = label
        if current_label and current_label not in self.warmup_activities_seen:
            self.warmup_activities_seen.add(current_label)
            self.warmup_last_new_activity_at = self.event_count

        if self.event_count >= self.warmup_max_events:
            return True

        soft_target_reached = self.event_count >= self.warmup_events
        n_activities = len(self.warmup_activities_seen)

        if soft_target_reached and n_activities >= self.warmup_sufficient_activities:
            return True

        events_since_new = self.event_count - self.warmup_last_new_activity_at
        enough_activities = n_activities >= self.warmup_min_activities
        stable_window = events_since_new >= self.warmup_stability_window
        if enough_activities and stable_window and soft_target_reached:
            return True

        return False

    def _build_feature_vector(self, event):
        cf = self._build_cf_features(event)
        data = self._build_data_features(event)
        iot = self.iot_builder.build(event.sensor_readings, event.case_id, self.window_size)
        if self.config.get('iot_presence_only', False):
            iot = (iot != 0.0).astype(iot.dtype)
        elif self.config.get('disable_iot_features', False):
            iot = np.zeros_like(iot)
        return np.concatenate([cf, data, iot]).astype(np.float32)

    def _build_cf_features(self, event):
        features = []
        window = self.case_windows.get(event.case_id, [])
        cf_mode = self.config.get('cf_feature_mode', 'prev_only')

        for i in range(self.window_size):
            if i < len(window):
                prev = window[-(i + 1)]
                prev_activity = prev.get('concept_name', '')
                emb = self.activity_embeddings.get(prev_activity) if prev_activity else [0.0] * self.embedding_dim
                features.extend(emb)
            else:
                features.extend([0.0] * self.embedding_dim)

        current_activity = str(event.concept_name) if event.concept_name else ''
        case_seq = self.case_windows.get(event.case_id, [])
        seq_len = len(case_seq)
        prev_activity = case_seq[-1].get('concept_name', '') if seq_len >= 1 else ''

        include_current = cf_mode in ('hybrid', 'current_only')
        include_prev = cf_mode in ('hybrid', 'prev_only')

        if include_current:
            incoming = self.dfg.get(current_activity, {})
            features.append(float(sum(incoming.values())))
            incoming_values = list(incoming.values())
            if incoming_values:
                total = sum(incoming_values)
                entropy = -sum((c / total) * math.log(c / total) for c in incoming_values if c > 0)
                features.append(entropy)
            else:
                features.append(0.0)

        if include_prev:
            outgoing = self.outgoing_dfg.get(prev_activity, {}) if prev_activity else {}
            features.append(float(sum(outgoing.values())))
            outgoing_values = list(outgoing.values())
            if outgoing_values:
                total = sum(outgoing_values)
                entropy = -sum((c / total) * math.log(c / total) for c in outgoing_values if c > 0)
                features.append(entropy)
            else:
                features.append(0.0)

        features.append(float(seq_len))

        if include_current:
            if current_activity:
                occurrence = self.case_activity_counts[event.case_id].get(current_activity, 0)
                features.append(float(occurrence))
            else:
                features.append(0.0)

        if include_prev:
            if prev_activity:
                prev_occurrence = self.case_activity_counts[event.case_id].get(prev_activity, 0)
                features.append(float(prev_occurrence))
            else:
                features.append(0.0)

        if include_current:
            if seq_len >= 2:
                prev_act = case_seq[-1].get('concept_name', '')
                features.append(1.0 if prev_act == current_activity else 0.0)
            else:
                features.append(0.0)

        if include_prev:
            if seq_len >= 2:
                prev_prev_activity = case_seq[-2].get('concept_name', '')
                features.append(1.0 if prev_prev_activity and prev_prev_activity == prev_activity else 0.0)
            else:
                features.append(0.0)

        if seq_len >= 1:
            prev_ts = case_seq[-1].get('timestamp')
            curr_ts = event.timestamp
            if prev_ts and curr_ts:
                try:
                    prev_naive = prev_ts.replace(tzinfo=None) if prev_ts.tzinfo else prev_ts
                    curr_naive = curr_ts.replace(tzinfo=None) if curr_ts.tzinfo else curr_ts
                    delta = (curr_naive - prev_naive).total_seconds()
                    features.append(min(delta / 3600.0, 24.0))
                except Exception:
                    features.append(0.0)
            else:
                features.append(0.0)
        else:
            features.append(0.0)

        if self.enable_case_features:
            case_id = event.case_id
            total_events = self.case_event_totals.get(case_id, 0)
            distinct_activities = len(self.case_activity_counts.get(case_id, {}))
            repetition_ratio = (total_events / max(distinct_activities, 1)
                                if distinct_activities > 0 else 0.0)
            first_ts = self.case_first_ts.get(case_id)
            curr_ts = event.timestamp
            case_age_hours = 0.0
            if first_ts is not None and curr_ts is not None:
                try:
                    first_naive = first_ts.replace(tzinfo=None) if first_ts.tzinfo else first_ts
                    curr_naive = curr_ts.replace(tzinfo=None) if curr_ts.tzinfo else curr_ts
                    delta_sec = max((curr_naive - first_naive).total_seconds(), 0.0)
                    case_age_hours = math.log1p(delta_sec / 3600.0)
                except Exception:
                    case_age_hours = 0.0
            features.append(math.log1p(float(total_events)))
            features.append(math.log1p(float(distinct_activities)))
            features.append(min(repetition_ratio, 10.0))
            features.append(case_age_hours)
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

        if self.enable_online_dfg and self.online_dfg is not None and prev_activity:
            is_parallel, n_siblings, density = self.online_dfg.parallel_features_for(prev_activity)
            features.append(float(is_parallel))
            features.append(float(n_siblings))
            features.append(float(density))
        else:
            features.extend([0.0, 0.0, 0.0])

        return np.array(features, dtype=np.float32)

    def _hash_to_vector(self, value, dim=16):
        import hashlib
        if not value:
            return [0.0] * dim
        h = hashlib.md5(str(value).encode()).hexdigest()
        vec = []
        for i in range(dim):
            byte_val = int(h[i * 2:(i * 2) + 2], 16)
            vec.append((byte_val / 255.0) * 2 - 1)
        return vec

    def _build_data_features(self, event):
        features = []
        lifecycle_onehot = self.lifecycle_encoder.encode(event.lifecycle or '')
        features.extend(lifecycle_onehot)

        if self.data_attribute_keys:
            for attr_key in self.data_attribute_keys:
                value = event.attributes.get(attr_key, '')
                if isinstance(value, (int, float)):
                    features.append(float(value))
                elif isinstance(value, str):
                    features.extend(self._hash_to_vector(value, dim=16))
                else:
                    features.extend([0.0] * 16)

        if event.timestamp:
            features.append(event.timestamp.hour / 24.0)
            features.append(event.timestamp.minute / 60.0)
            features.append(event.timestamp.weekday() / 7.0)
        else:
            features.extend([0.0, 0.0, 0.0])

        return np.array(features, dtype=np.float32)

    def _update_context(self, event, confidence=1.0):
        activity = str(event.concept_name) if event.concept_name else ''

        entry = {
            'concept_name': activity,
            'timestamp': event.timestamp,
            'confidence': confidence,
        }
        self.case_windows[event.case_id].append(entry)
        if len(self.case_windows[event.case_id]) > self.window_size:
            self.case_windows[event.case_id] = self.case_windows[event.case_id][-self.window_size:]

        if event.case_id not in self.case_first_ts and event.timestamp is not None:
            self.case_first_ts[event.case_id] = event.timestamp
        self.case_event_totals[event.case_id] += 1

        if activity:
            self.activity_embeddings.get(activity)
            self.activity_counts[activity] += 1
            self.case_activity_counts[event.case_id][activity] += 1

            window = self.case_windows[event.case_id]
            if len(window) >= 2:
                prev_activity = window[-2].get('concept_name', '')
                if prev_activity:
                    self.dfg[activity][prev_activity] += 1
                    self.outgoing_dfg[prev_activity][activity] += 1

    def _init_models(self, input_dim):
        self.input_dim = input_dim
        self.sdae = SparseDenoisingAutoencoder(
            input_dim=input_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
            sparsity_lambda=self.sparsity_lambda,
            noise_std=self.noise_std
        )
        self.bfr = StreamingBFR(
            latent_dim=self.latent_dim,
            n_min=self.n_min,
            n_reliable=self.n_reliable,
            alpha=self.alpha,
            fuzzifier=self.fuzzifier,
            lambda_exp=self.lambda_exp,
            confidence_mode=self.config.get('confidence_mode', 'entropy')
        )

    def _train_sdae(self, feature_vectors, labels=None):
        self.sdae.train()
        params = list(self.sdae.parameters())

        if labels is not None and self.pretrainer is None:
            self.pretrainer = MaskedActivityPretrainer(self.latent_dim, mask_ratio=0.15)
            self.pretrainer.fit_labels(labels)

        if self.pretrainer and self.pretrainer.classifier is not None:
            params += list(self.pretrainer.classifier.parameters())

        optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        dataset = torch.tensor(np.array(feature_vectors), dtype=torch.float32)
        batch_size = min(64, len(dataset))
        cluster_loss_fn = ClusterAwareLoss(separation_weight=1.0, compactness_weight=0.5)

        labels_encoded = None
        if labels is not None and self.pretrainer:
            labels_encoded = self.pretrainer.encode_labels(labels)

        for epoch in range(self.training_epochs):
            indices = torch.randperm(len(dataset))
            for start in range(0, len(dataset), batch_size):
                batch_idx = indices[start:start + batch_size]
                batch = dataset[batch_idx]
                optimizer.zero_grad()

                masked_batch, mask = self.pretrainer.mask_features(batch, self.cf_feature_count) if self.pretrainer else (batch, None)
                x_hat, z = self.sdae(masked_batch)
                recon_loss = self.sdae.loss(batch, x_hat, z)

                loss = recon_loss

                if labels is not None:
                    batch_labels = [labels[i] for i in batch_idx.tolist()]
                    cluster_loss = cluster_loss_fn.compute(z, batch_labels)
                    loss = loss + cluster_loss

                if labels_encoded is not None and self.pretrainer:
                    batch_label_idx = labels_encoded[batch_idx]
                    masked_loss = self.pretrainer.compute_loss(z, batch_label_idx)
                    loss = loss + masked_loss

                loss.backward()
                optimizer.step()

        self.sdae.eval()

    def _encode(self, feature_vector):
        self.sdae.eval()
        with torch.no_grad():
            x = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
            x_hat, z = self.sdae(x)
            recon_error = torch.nn.functional.mse_loss(x_hat, x).item()
        return z.squeeze(0).numpy(), recon_error

    def _standardise_features(self, feature_vectors):
        fv_array = np.array(feature_vectors)
        self._feature_means = np.mean(fv_array, axis=0)
        self._feature_stds = np.std(fv_array, axis=0)
        self._feature_stds = np.maximum(self._feature_stds, 1e-8)
        return (fv_array - self._feature_means) / self._feature_stds

    def _standardise_single(self, feature_vector):
        if not hasattr(self, '_feature_means'):
            return feature_vector
        return (feature_vector - self._feature_means) / self._feature_stds

    def _discover_data_attributes(self):
        from collections import Counter
        key_counts = Counter()
        skip = {'data', 'raw', 'cpee:description'}
        for _, _, event in self.warmup_buffer:
            for k, v in event.attributes.items():
                if k not in skip and isinstance(v, str) and len(v) > 0:
                    key_counts[k] += 1
        self.data_attribute_keys = [k for k, _ in key_counts.most_common(self.max_data_attributes)]
        if not self.data_attribute_keys:
            self.data_attribute_keys = []

    def _warmup(self):
        labelled = [(fv, label) for fv, label, _ in self.warmup_buffer if label]
        if len(labelled) < self.n_min:
            return

        feature_vectors = [fv for fv, _ in labelled]
        labels = [label for _, label in labelled]
        standardised = self._standardise_features(feature_vectors)
        feature_vectors_std = [standardised[i] for i in range(len(standardised))]

        need_calibration = self.enable_cluster_calibration or self.enable_sequence_head
        holdout_idxs: set = set()
        if need_calibration:
            rng = random.Random(42)
            n_total = len(labelled)
            n_holdout = max(1, int(n_total * self.calibration_holdout_fraction))
            all_idxs = list(range(n_total))
            rng.shuffle(all_idxs)
            holdout_idxs = set(all_idxs[:n_holdout])
        train_idxs = [i for i in range(len(labelled)) if i not in holdout_idxs]

        sdae_feature_vectors = [feature_vectors_std[i] for i in train_idxs]
        sdae_labels = [labels[i] for i in train_idxs]
        self._init_models(len(feature_vectors_std[0]))
        self._train_sdae(sdae_feature_vectors, sdae_labels)

        for i in train_idxs:
            z, _ = self._encode(feature_vectors_std[i])
            self.bfr.update_cluster(labels[i], z)

        if self.auto_calibrate:
            calibration_data = []
            for i in train_idxs:
                z, _ = self._encode(feature_vectors_std[i])
                calibration_data.append((z, labels[i]))
            self.bfr.calibrate(calibration_data)

        if self.enable_cluster_calibration and holdout_idxs:
            plant_confs = []
            plant_correct = []
            for i in holdout_idxs:
                z, _ = self._encode(feature_vectors_std[i])
                best_label, _, cluster_conf = self.bfr.compute_confidence(z)
                if best_label is None or cluster_conf is None:
                    continue
                plant_confs.append(float(cluster_conf))
                plant_correct.append(1 if best_label == labels[i] else 0)
            if plant_confs:
                self.platt_scaler.fit(plant_confs, plant_correct)
                self.holdout_calibration_pairs = list(zip(plant_confs, plant_correct))

        if self.enable_sequence_head:
            self._warmup_sequence_head(labelled, feature_vectors_std, holdout_idxs, train_idxs)
            if not self.enable_retroactive_warmup_recovery:
                self._seed_prefix_buffer_from_warmup()

        self.is_warmed_up = True

    def _seed_prefix_buffer_from_warmup(self):
        if not self.enable_sequence_head:
            return
        for _, label, event in self.warmup_buffer:
            if label:
                act_id = self._seq_activity_id(label)
                provenance = 'NORMAL'
                confidence = 1.0
            else:
                act_id = MISSING_TOKEN_ID
                provenance = 'UNRECOVERED_ML'
                confidence = 0.0
            self.prefix_buffer.append(
                case_id=event.case_id,
                activity_id=act_id,
                provenance=provenance,
                confidence=confidence,
            )

    def _seq_activity_id(self, label):
        if not label:
            return MISSING_TOKEN_ID
        s = str(label)
        if s not in self.seq_activity_to_id:
            if self._seq_vocab_frozen:
                return MISSING_TOKEN_ID
            new_id = len(self.seq_activity_to_id) + 2
            self.seq_activity_to_id[s] = new_id
            self.seq_id_to_activity[new_id] = s
        return self.seq_activity_to_id[s]

    def _warmup_sequence_head(self, labelled, feature_vectors_std,
                              holdout_idxs, train_idxs):
        for _, label, _ in self.warmup_buffer:
            if label:
                self._seq_activity_id(label)
        vocab_size = len(self.seq_activity_to_id) + 2

        if vocab_size < 4:
            return

        from collections import defaultdict
        case_history: dict = defaultdict(list)
        train_examples = []
        labelled_position = 0
        for fv, label, event in self.warmup_buffer:
            case_id = event.case_id
            prefix = case_history[case_id][-self.window_size:]
            pad_count = self.window_size - len(prefix)
            prefix_padded = [PAD_TOKEN_ID] * pad_count + prefix
            target_id = self._seq_activity_id(label) if label else MISSING_TOKEN_ID
            if label:
                train_examples.append((prefix_padded, target_id, labelled_position))
                labelled_position += 1
            case_history[case_id].append(target_id if label else MISSING_TOKEN_ID)

        if len(train_examples) < 10:
            return

        holdout_idxs = list(holdout_idxs)
        train_idxs = list(train_idxs)

        self.sequence_head = SequenceRescueHead(
            activity_vocab_size=vocab_size,
            latent_dim=self.latent_dim if self.sequence_input_mode == 'z_latent' else None,
            embedding_dim=self.sequence_embedding_dim,
            hidden_dim=self.sequence_hidden_dim,
            dropout=self.sequence_dropout,
            input_mode=self.sequence_input_mode,
            num_layers=self.sequence_num_layers,
        )
        self.sequence_optimizer = torch.optim.AdamW(
            self.sequence_head.parameters(), lr=self.sequence_lr
        )
        self.replay_buffer = CBRSBuffer(capacity=self.cbrs_capacity, random_seed=42)

        criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
        train_prefixes = torch.tensor(
            [train_examples[i][0] for i in train_idxs], dtype=torch.long
        )
        train_targets = torch.tensor(
            [train_examples[i][1] for i in train_idxs], dtype=torch.long
        )

        if self.sequence_input_mode == 'z_latent':
            raise NotImplementedError(
                "z_latent input mode requires cached per-event z history "
                "during warmup; not yet implemented. Use activity_id mode for now."
            )

        self.sequence_head.train()
        for epoch in range(self.sequence_warmup_epochs):
            self.sequence_optimizer.zero_grad()
            logits = self.sequence_head(train_prefixes)
            loss = criterion(logits, train_targets)
            loss.backward()
            self.sequence_optimizer.step()

        for i in train_idxs:
            prefix_ids, target_id, _ = train_examples[i]
            self.replay_buffer.add(sample=(prefix_ids, target_id), class_label=target_id)

        if holdout_idxs:
            self.sequence_head.eval()
            with torch.no_grad():
                holdout_prefixes = torch.tensor(
                    [train_examples[i][0] for i in holdout_idxs], dtype=torch.long
                )
                holdout_targets = torch.tensor(
                    [train_examples[i][1] for i in holdout_idxs], dtype=torch.long
                )
                holdout_logits = self.sequence_head(holdout_prefixes).cpu().numpy()
                self.temp_scaler.fit(holdout_logits, holdout_targets.cpu().numpy())

        self._seq_vocab_frozen = True

    def process_event(self, event):
        self.event_count += 1
        has_label = event.concept_name and str(event.concept_name).strip()
        label = str(event.concept_name) if has_label else None

        feature_vector = self._build_feature_vector(event)

        if not self.is_warmed_up:
            self._update_context(event)
            self.warmup_buffer.append((feature_vector, label, event))
            result = {
                'event': event,
                'original_label': label,
                'recovered_label': label,
                'confidence': 1.0 if has_label else 0.0,
                'provenance': 'NORMAL' if has_label else 'UNRECOVERED_ML',
            }
            self.results.append(result)
            if self._should_end_warmup():
                self._warmup()
                if self.enable_retroactive_warmup_recovery and self.is_warmed_up:
                    self._retroactive_warmup_recovery()
                    result = self.results[-1]
            return result

        feature_vector_std = self._standardise_single(feature_vector)
        z, recon_error = self._encode(feature_vector_std)
        self.reconstruction_errors.append(recon_error)

        prefix_t = self.prefix_buffer.snapshot_prefix_ids(event.case_id) \
                   if (self.enable_sequence_head or self.enable_count_cache) else None

        if has_label:
            self.bfr.update_cluster(label, z)
            self._update_context(event)
            self.recent_labelled.append((feature_vector_std, label))
            if len(self.recent_labelled) > self.retrain_interval:
                self.recent_labelled = self.recent_labelled[-self.retrain_interval:]
            self.recent_labelled_per_cluster[label].append(feature_vector_std)
            if len(self.recent_labelled_per_cluster[label]) > self.max_recent_per_cluster:
                self.recent_labelled_per_cluster[label] = self.recent_labelled_per_cluster[label][-self.max_recent_per_cluster:]
            self.per_cluster_errors_pre[label].append(recon_error)
            if len(self.per_cluster_errors_pre[label]) > 100:
                self.per_cluster_errors_pre[label] = self.per_cluster_errors_pre[label][-100:]
            self._try_reconcile_synthetic_cluster(label)
            if self.enable_sequence_head and self.sequence_head is not None:
                self._sequence_head_online_update_snapshot(prefix_t, label)
            if self.enable_count_cache and self.count_cache is not None and prefix_t is not None:
                self._count_cache_add(prefix_t, label)
            if self.enable_online_dfg and self.online_dfg is not None:
                window = self.case_windows.get(event.case_id, [])
                if window:
                    prev_label = window[-1].get('concept_name')
                    if prev_label:
                        self.online_dfg.add(prev_label, label)
            self._update_prefix_buffer(event, label, 'NORMAL', 1.0)
            result = {
                'event': event,
                'original_label': label,
                'recovered_label': label,
                'confidence': 1.0,
                'provenance': 'NORMAL',
            }
        else:
            best_label, memberships, conf_raw = self.bfr.compute_confidence(z)
            cluster_conf = float(conf_raw) if conf_raw is not None else 0.0
            cluster_conf = self._prior_adjust(best_label, cluster_conf)
            if self.enable_cluster_calibration and self.platt_scaler.is_fitted \
                    and cluster_conf is not None:
                cluster_conf = float(self.platt_scaler.calibrate(float(cluster_conf)))
            cluster_margin = 0.0
            if memberships:
                vals = sorted(memberships.values(), reverse=True)
                u_star = vals[0]
                u_second = vals[1] if len(vals) > 1 else 0.0
                cluster_margin = u_star - u_second
            recovered_label = None
            confidence = cluster_conf
            flag = 'UNRECOVERED_ML'
            if best_label is not None and cluster_conf is not None \
                    and cluster_conf >= self.alpha_cluster:
                recovered_label = best_label
                confidence = cluster_conf
                flag = 'RECOVERED_ML'
            if (self.margin_threshold > 0.0
                    and flag == 'RECOVERED_ML'
                    and cluster_margin < self.margin_threshold):
                flag = 'UNRECOVERED_ML'
                recovered_label = None
                self.margin_abstentions += 1
            if (self.enable_count_cache and self.count_cache is not None
                    and flag == 'RECOVERED_ML' and prefix_t is not None):
                cache_label, cache_dom, cache_sup = self._count_cache_predict(prefix_t)
                if (cache_label is not None
                        and cache_label != recovered_label
                        and cache_dom >= self.count_cache_override_dominance
                        and cache_sup >= self.count_cache_override_support):
                    flag = 'UNRECOVERED_ML'
                    recovered_label = None
                    self.count_cache_overrides += 1

            rescued = False

            if (self.arbitration
                    and flag == 'RECOVERED_ML'
                    and self.enable_sequence_head
                    and self.sequence_head is not None
                    and (cluster_margin < self.arbitration_margin
                         or cluster_conf < self.alpha_cluster + self.arbitration_margin)):
                seq_label, seq_conf = self._sequence_head_predict_snapshot(prefix_t)
                self.arbitration_invocations += 1
                if (seq_label is not None
                        and seq_conf >= self.alpha_seq
                        and seq_label != recovered_label
                        and seq_conf > cluster_conf + self.arbitration_margin):
                    recovered_label = seq_label
                    confidence = seq_conf
                    flag = 'RECOVERED_ML_SEQ'
                    rescued = True
                    self.arbitration_overrides += 1

            if flag == 'UNRECOVERED_ML':
                self.undiscovered_buffer.add(z)
                promotions = self.undiscovered_buffer.check_promotion(self.bfr.clusters)
                for promo in promotions:
                    new_label = f"UNDISCOVERED_{len(self.bfr.clusters)}"
                    for member_z in promo['members']:
                        self.bfr.update_cluster(new_label, member_z)
                    self.adwin_per_cluster[new_label] = ADWIN(
                        delta=self.config.get('adwin_delta', 0.002))

                if self.enable_sequence_head and self.sequence_head is not None:
                    seq_label, seq_conf = self._sequence_head_predict_snapshot(prefix_t)
                    self.seq_rescue_invocations += 1
                    self.contamination_log.append({
                        'event_idx': self.event_count,
                        'composition': self.prefix_buffer.composition(event.case_id),
                        'contamination_fraction': self.prefix_buffer.contamination_fraction(event.case_id),
                        'seq_conf': float(seq_conf),
                    })
                    commit_rescue = seq_label is not None and seq_conf >= self.alpha_seq
                    if commit_rescue:
                        commit_rescue = self._rescue_gate_allows(
                            seq_label, memberships, seq_conf=seq_conf
                        )
                    if commit_rescue and self.enable_agreement_gating == 'tiered':
                        cluster_argmax = self.bfr.argmax_label(z)
                        agrees = cluster_argmax is not None and cluster_argmax == seq_label
                        if not agrees and seq_conf < self.alpha_seq_high:
                            commit_rescue = False
                    if commit_rescue:
                        recovered_label = seq_label
                        confidence = seq_conf
                        flag = 'RECOVERED_ML_SEQ'
                        rescued = True
                        self.seq_rescue_commits += 1

            if (flag == 'UNRECOVERED_ML'
                    and self.enable_count_cache
                    and self.count_cache is not None
                    and prefix_t is not None):
                cache_label, cache_dom, cache_sup = self._count_cache_predict(prefix_t)
                if (cache_label is not None
                        and cache_dom >= self.count_cache_emit_dominance
                        and cache_sup >= self.count_cache_emit_support):
                    recovered_label = cache_label
                    confidence = cache_dom
                    flag = 'RECOVERED_ML'
                    rescued = True
                    self.count_cache_direct_emits += 1

            self._update_context(event, confidence if confidence is not None else 0.0)
            self._update_prefix_buffer(
                event, recovered_label if rescued or flag == 'RECOVERED_ML' else None,
                flag, float(confidence) if confidence is not None else 0.0
            )
            result = {
                'event': event,
                'original_label': None,
                'recovered_label': recovered_label,
                'confidence': float(confidence) if confidence is not None else 0.0,
                'provenance': flag,
                'z': z,
            }

        self._check_drift(recon_error, label=label if has_label else None)
        self.results.append(result)
        return result

    def _rescue_gate_allows(self, seq_label, memberships, seq_conf=None):
        mode = self.enable_agreement_gating
        if not mode:
            return True
        if seq_conf is not None and seq_conf >= self.agreement_high_conf_override:
            self.agreement_high_conf_bypasses += 1
            return True
        if not memberships:
            return True
        if mode == 'tiered':
            return True
        if mode == 'topk':
            k = self.agreement_topk
            top_k_labels = {l for l, _ in sorted(
                memberships.items(), key=lambda kv: -kv[1])[:k]}
            if seq_label in top_k_labels:
                return True
            self.agreement_gated_blocks += 1
            return False
        cluster_argmax = max(memberships.items(), key=lambda kv: kv[1])[0]
        if seq_label == cluster_argmax:
            return True
        self.agreement_gated_blocks += 1
        return False

    def _prior_adjust(self, label, confidence):
        if self.class_prior_beta <= 0.0 or not label or confidence is None:
            return confidence
        if not self.activity_counts:
            return confidence
        total = sum(self.activity_counts.values())
        if total <= 0:
            return confidence
        p_class = self.activity_counts.get(label, 1) / total
        p_uniform = 1.0 / max(1, len(self.activity_counts))
        if p_class <= 0:
            return confidence
        factor = (p_uniform / p_class) ** self.class_prior_beta
        self.prior_adjustments += 1
        return min(1.0, confidence * factor)

    def _retroactive_warmup_recovery(self):
        n = len(self.warmup_buffer)
        for i, (fv, lbl, ev) in enumerate(self.warmup_buffer):
            if lbl is not None:
                if self.enable_sequence_head:
                    act_id = self._seq_activity_id(lbl)
                    self.prefix_buffer.append(
                        case_id=ev.case_id,
                        activity_id=act_id,
                        provenance='NORMAL',
                        confidence=1.0,
                    )
                continue
            if i >= len(self.results):
                continue

            self.retro_events += 1
            try:
                fv_std = self._standardise_single(fv)
                z, _ = self._encode(fv_std)
            except Exception:
                continue

            best_label, memberships, conf_raw = self.bfr.compute_confidence(z)
            cluster_conf = float(conf_raw) if conf_raw is not None else 0.0
            cluster_conf = self._prior_adjust(best_label, cluster_conf)
            if self.enable_cluster_calibration and self.platt_scaler.is_fitted \
                    and cluster_conf is not None:
                cluster_conf = float(self.platt_scaler.calibrate(float(cluster_conf)))
            cluster_margin = 0.0
            if memberships:
                vals = sorted(memberships.values(), reverse=True)
                cluster_margin = vals[0] - (vals[1] if len(vals) > 1 else 0.0)

            recovered_label = None
            confidence = cluster_conf
            flag = 'UNRECOVERED_ML'
            if best_label is not None and cluster_conf is not None \
                    and cluster_conf >= self.retro_alpha_cluster:
                recovered_label = best_label
                confidence = cluster_conf
                flag = 'RECOVERED_ML'
            if (self.margin_threshold > 0.0
                    and flag == 'RECOVERED_ML'
                    and cluster_margin < self.margin_threshold):
                flag = 'UNRECOVERED_ML'
                recovered_label = None
                self.margin_abstentions += 1
            if (self.arbitration
                    and flag == 'RECOVERED_ML'
                    and self.enable_sequence_head
                    and self.sequence_head is not None
                    and (cluster_margin < self.arbitration_margin
                         or cluster_conf < self.retro_alpha_cluster + self.arbitration_margin)):
                prefix_t = self.prefix_buffer.snapshot_prefix_ids(ev.case_id)
                seq_label, seq_conf = self._sequence_head_predict_snapshot(prefix_t)
                self.arbitration_invocations += 1
                if (seq_label is not None
                        and seq_conf >= self.retro_alpha_seq
                        and seq_label != recovered_label
                        and seq_conf > cluster_conf + self.arbitration_margin):
                    recovered_label = seq_label
                    confidence = float(seq_conf)
                    flag = 'RECOVERED_ML_SEQ'
                    self.arbitration_overrides += 1

            if flag == 'UNRECOVERED_ML' \
                    and self.enable_sequence_head and self.sequence_head is not None:
                prefix_t = self.prefix_buffer.snapshot_prefix_ids(ev.case_id)
                seq_label, seq_conf = self._sequence_head_predict_snapshot(prefix_t)
                if seq_label is not None and seq_conf >= self.retro_alpha_seq:
                    commit_rescue = self._rescue_gate_allows(
                        seq_label, memberships, seq_conf=seq_conf
                    )
                    if commit_rescue and self.enable_agreement_gating == 'tiered':
                        cluster_argmax = self.bfr.argmax_label(z)
                        agrees = cluster_argmax is not None and cluster_argmax == seq_label
                        if not agrees and seq_conf < self.alpha_seq_high:
                            commit_rescue = False
                    if commit_rescue:
                        recovered_label = seq_label
                        confidence = float(seq_conf)
                        flag = 'RECOVERED_ML_SEQ'

            self.results[i] = {
                'event': ev,
                'original_label': None,
                'recovered_label': recovered_label,
                'confidence': float(confidence) if confidence is not None else 0.0,
                'provenance': flag,
            }
            self._update_prefix_buffer(
                ev,
                recovered_label if flag in ('RECOVERED_ML', 'RECOVERED_ML_SEQ') else None,
                flag,
                float(confidence) if confidence is not None else 0.0,
            )

            if flag == 'RECOVERED_ML':
                self.retro_tp_cluster += 1
            elif flag == 'RECOVERED_ML_SEQ':
                self.retro_tp_rescue += 1
            else:
                self.retro_abs += 1

    def _try_reconcile_synthetic_cluster(self, labelled_label):
        if not self.config.get('enable_synthetic_reconciliation', True):
            return
        if str(labelled_label).startswith('UNDISCOVERED_'):
            return
        merge_threshold = self.config.get('synthetic_merge_threshold', 2.0)
        candidate = self.bfr.find_merge_candidate(labelled_label, merge_threshold)
        if candidate is None:
            return
        merged = self.bfr.merge_cluster(source_label=candidate, target_label=labelled_label)
        if not merged:
            return
        if candidate in self.adwin_per_cluster:
            del self.adwin_per_cluster[candidate]
        if candidate in self.per_cluster_errors_pre:
            del self.per_cluster_errors_pre[candidate]
        if candidate in self.recent_labelled_per_cluster:
            del self.recent_labelled_per_cluster[candidate]
        self._relabel_past_results(from_label=candidate, to_label=labelled_label)
        self.reconciliation_log.append({
            'synthetic_label': candidate,
            'real_label': labelled_label,
            'event_index': len(self.results),
        })

    def _relabel_past_results(self, from_label, to_label):
        for r in self.results:
            if r.get('recovered_label') == from_label and r.get('provenance') == 'RECOVERED_ML':
                r['recovered_label'] = to_label
                r['reconciled_from'] = from_label

    def _check_drift(self, recon_error, label=None):
        self.adwin_global.update(recon_error)

        if label:
            self.adwin_per_cluster[label].update(recon_error)
            if self.adwin_per_cluster[label].drift_detected:
                errors = self.per_cluster_errors_pre.get(label, [])
                if len(errors) >= 20:
                    mid = len(errors) // 2
                    pre_mean = np.mean(errors[:mid])
                    post_mean = np.mean(errors[mid:])
                    severe = pre_mean > 0 and post_mean / pre_mean > self.delta
                    if severe:
                        self.severe_drift_count[label] += 1
                        self._decay_cluster(label)
                    self._handle_gradual_drift()
                    return

        if self.adwin_global.drift_detected:
            self._handle_gradual_drift()
            if self.enable_sequence_head and self.sequence_head is not None:
                self._reset_sequence_head_on_drift()
            if self.enable_online_dfg and self.online_dfg is not None:
                self.online_dfg.redetect_parallel_pairs()

    def _reset_sequence_head_on_drift(self):
        if self.sequence_head is None:
            return
        for module in self.sequence_head.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        self.sequence_optimizer = torch.optim.AdamW(
            self.sequence_head.parameters(), lr=self.sequence_lr
        )
        self.temp_scaler = TemperatureScaler()
        self.platt_scaler = PlattScaler()

    def _decay_cluster(self, label):
        decay_factor = self.config.get('drift_decay_factor', 0.5)
        if label not in self.bfr.clusters:
            return
        cluster = self.bfr.clusters[label]
        cluster.n *= decay_factor
        cluster.sum_vec *= decay_factor
        cluster.sum_sq_mat *= decay_factor
        self.cluster_drift_count[label] += 1
        if label in self.adwin_per_cluster:
            self.adwin_per_cluster[label] = ADWIN(delta=self.config.get('adwin_delta', 0.002))
        if label in self.per_cluster_errors_pre:
            self.per_cluster_errors_pre[label] = []

    def _handle_gradual_drift(self):
        if len(self.recent_labelled) < 50:
            return
        self.gradual_drift_count += 1
        fvs = [fv for fv, _ in self.recent_labelled[-200:]]
        labels = [l for _, l in self.recent_labelled[-200:]]
        self._train_sdae(fvs, labels)

    def process_stream(self, event_generator):
        for event in event_generator:
            yield self.process_event(event)

    def get_summary(self):
        total = len(self.results)
        normal = sum(1 for r in self.results if r['provenance'] == 'NORMAL')
        recovered = sum(1 for r in self.results if r['provenance'] == 'RECOVERED_ML')
        recovered_seq = sum(1 for r in self.results if r['provenance'] == 'RECOVERED_ML_SEQ')
        unrecovered = sum(1 for r in self.results if r['provenance'] == 'UNRECOVERED_ML')
        avg_conf = np.mean([r['confidence'] for r in self.results
                            if r['provenance'] in ('RECOVERED_ML', 'RECOVERED_ML_SEQ')]) \
                   if (recovered + recovered_seq) > 0 else 0.0

        return {
            'total_events': total,
            'normal': normal,
            'recovered': recovered,
            'recovered_seq': recovered_seq,
            'unrecovered': unrecovered,
            'recovery_rate': (recovered + recovered_seq) / (recovered + recovered_seq + unrecovered)
                              if (recovered + recovered_seq + unrecovered) > 0 else 0,
            'avg_recovery_confidence': avg_conf,
            'num_clusters': len(self.bfr.clusters) if self.bfr else 0,
            'reconciliations': len(self.reconciliation_log),
            'reconciliation_log': self.reconciliation_log,
            'cluster_decay_events': dict(self.cluster_drift_count),
            'severe_drift_events': dict(self.severe_drift_count),
            'gradual_drift_events': self.gradual_drift_count,
            'seq_rescue_invocations': self.seq_rescue_invocations,
            'seq_rescue_commits': self.seq_rescue_commits,
            'seq_head_enabled': self.enable_sequence_head and self.sequence_head is not None,
            'retro_enabled': self.enable_retroactive_warmup_recovery,
            'retro_events': self.retro_events,
            'retro_tp_cluster': self.retro_tp_cluster,
            'retro_tp_rescue': self.retro_tp_rescue,
            'retro_abstained': self.retro_abs,
        }

    def _update_prefix_buffer(self, event, label_or_none, provenance, confidence):
        if not self.enable_sequence_head:
            return
        if label_or_none and provenance in ('NORMAL', 'RECOVERED_ML', 'RECOVERED_ML_SEQ'):
            act_id = self._seq_activity_id(label_or_none)
        else:
            act_id = MISSING_TOKEN_ID
        self.prefix_buffer.append(
            case_id=event.case_id,
            activity_id=act_id,
            provenance=provenance,
            confidence=float(confidence),
        )

    def _count_cache_key(self, prefix_t):
        if self.count_cache is None or prefix_t is None:
            return None
        order = self.count_cache.order
        return tuple(prefix_t[-order:])

    def _count_cache_add(self, prefix_t, label):
        key = self._count_cache_key(prefix_t)
        if key is None:
            return
        act_id = self._seq_activity_id(label)
        if act_id in (PAD_TOKEN_ID, MISSING_TOKEN_ID):
            return
        self.count_cache.add(key, act_id)

    def _count_cache_predict(self, prefix_t):
        key = self._count_cache_key(prefix_t)
        if key is None:
            return None, 0.0, 0.0
        top_id, dominance, support = self.count_cache.predict(key)
        if top_id is None or top_id in (PAD_TOKEN_ID, MISSING_TOKEN_ID):
            return None, 0.0, 0.0
        label = self.seq_id_to_activity.get(int(top_id))
        return label, float(dominance), float(support)

    def _sequence_head_predict_snapshot(self, prefix_t):
        if self.sequence_head is None or prefix_t is None:
            return None, 0.0
        x = torch.tensor([list(prefix_t)], dtype=torch.long)
        self.sequence_head.eval()
        with torch.no_grad():
            logits = self.sequence_head(x)
        if self.temp_scaler.is_fitted:
            probs = self.temp_scaler.calibrate(logits.numpy())[0]
        else:
            scaled = logits.numpy()[0]
            scaled = scaled - scaled.max()
            ex = np.exp(scaled)
            probs = ex / ex.sum()
        probs_clipped = probs.copy()
        probs_clipped[PAD_TOKEN_ID] = 0.0
        probs_clipped[MISSING_TOKEN_ID] = 0.0
        pred_id = int(np.argmax(probs_clipped))
        pred_conf = float(probs_clipped[pred_id])
        pred_label = self.seq_id_to_activity.get(pred_id)
        return pred_label, pred_conf

    def _sequence_head_online_update_snapshot(self, prefix_t, label):
        if self.sequence_head is None or self.replay_buffer is None or prefix_t is None:
            return
        prefix_ids = list(prefix_t)
        target_id = self._seq_activity_id(label)
        self.replay_buffer.add(sample=(prefix_ids, target_id), class_label=target_id)
        batch = self.replay_buffer.sample_batch(self.cbrs_batch_size)
        if len(batch) < 2:
            return
        prefixes = torch.tensor([s[0] for s, _ in batch], dtype=torch.long)
        targets = torch.tensor([s[1] for s, _ in batch], dtype=torch.long)
        self.sequence_head.train()
        self.sequence_optimizer.zero_grad()
        logits = self.sequence_head(prefixes)
        loss = torch.nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.sequence_head.parameters(), 1.0)
        self.sequence_optimizer.step()
