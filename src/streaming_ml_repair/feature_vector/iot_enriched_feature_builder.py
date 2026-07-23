import numpy as np
from src.streaming_ml_repair.feature_vector.iot_feature_builder import IoTFeatureBuilder

class ExtendedFeatureVectorBuilder:

    def __init__(self, sensor_vocabulary, data_columns=None, excluded_columns=None,
                 use_control_flow=True, sensor_window_size=10):
        self.iot_builder = IoTFeatureBuilder(sensor_vocabulary)
        self.data_columns = data_columns or []
        self.excluded_columns = set(excluded_columns or [])
        self.use_control_flow = use_control_flow
        self.sensor_window_size = sensor_window_size
        self.online_onehot_encoders = {}
        self.activity_embeddings = None
        self.running_means = {}
        self.running_vars = {}
        self.running_count = 0

    @staticmethod
    def _is_null(value):
        if value is None or value == -1:
            return True
        if isinstance(value, str):
            return value.strip().lower() in {"", "nan", "null", "none", "na"}
        return False

    def _build_data_features(self, event_dict):
        features = []
        for col in self.data_columns:
            if col in self.excluded_columns:
                continue
            value = event_dict.get(col, None)
            if col in self.online_onehot_encoders:
                features.extend(self.online_onehot_encoders[col].encode(value))
            else:
                if self._is_null(value):
                    features.append(0.0)
                else:
                    try:
                        features.append(float(value))
                    except (TypeError, ValueError):
                        features.append(0.0)
        return features

    def _build_cf_features(self, control_flow_features, cf_feature_names):
        features = []
        if not self.use_control_flow or control_flow_features is None:
            return features
        for cf_name in cf_feature_names:
            cf_value = control_flow_features.get(cf_name, None)
            if cf_name.startswith("prev_"):
                if (self.activity_embeddings and
                        hasattr(self.activity_embeddings, 'embeddings') and
                        len(self.activity_embeddings.embeddings) > 0 and
                        cf_value is not None):
                    emb = self.activity_embeddings.get(cf_value)
                    if emb is None:
                        emb = [0.0] * self.activity_embeddings.embedding_dim
                else:
                    dim = getattr(self.activity_embeddings, 'embedding_dim', 8) if self.activity_embeddings else 8
                    emb = [0.0] * dim
                features.extend(emb)
            else:
                if self._is_null(cf_value):
                    features.append(0.0)
                else:
                    try:
                        features.append(float(cf_value))
                    except (TypeError, ValueError):
                        features.append(0.0)
        return features

    def build(self, event_dict, control_flow_features, cf_feature_names,
              sensor_readings, case_id, confidence_weight=1.0):
        data_feats = self._build_data_features(event_dict)
        cf_feats = self._build_cf_features(control_flow_features, cf_feature_names)
        cf_feats = [f * confidence_weight for f in cf_feats]
        iot_feats = self.iot_builder.build(sensor_readings, case_id, self.sensor_window_size)
        combined = np.array(data_feats + cf_feats, dtype=np.float32)
        return np.concatenate([combined, iot_feats])

    def standardise(self, feature_vector):
        self.running_count += 1
        if self.running_count == 1:
            self.running_means = feature_vector.copy()
            self.running_vars = np.zeros_like(feature_vector)
            return np.zeros_like(feature_vector)
        old_mean = self.running_means.copy()
        self.running_means += (feature_vector - self.running_means) / self.running_count
        self.running_vars += (feature_vector - old_mean) * (feature_vector - self.running_means)
        std = np.sqrt(self.running_vars / (self.running_count - 1))
        std = np.maximum(std, 1e-8)
        return (feature_vector - self.running_means) / std

    def build_and_standardise(self, event_dict, control_flow_features, cf_feature_names,
                              sensor_readings, case_id, confidence_weight=1.0):
        raw = self.build(event_dict, control_flow_features, cf_feature_names,
                         sensor_readings, case_id, confidence_weight)
        return self.standardise(raw)
