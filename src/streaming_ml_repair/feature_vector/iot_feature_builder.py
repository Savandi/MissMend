import numpy as np
from collections import defaultdict

class IoTFeatureBuilder:

    def __init__(self, sensor_vocabulary):
        self.sensor_vocab = sorted(sensor_vocabulary)
        self.sensor_to_idx = {s: i for i, s in enumerate(self.sensor_vocab)}
        self.num_sensors = len(self.sensor_vocab)
        self.case_sensor_history = defaultdict(lambda: defaultdict(list))

    @property
    def feature_dim(self):
        return self.num_sensors * 6

    def _get_temporal_stats(self, history):
        if len(history) == 0:
            return 0.0, 0.0, 0.0, 0.0
        values = np.array(history, dtype=np.float64)
        mean = np.mean(values)
        std = np.std(values) if len(values) > 1 else 0.0
        if len(values) >= 2:
            x = np.arange(len(values), dtype=np.float64)
            slope = np.polyfit(x, values, 1)[0] if len(values) >= 2 else 0.0
            delta = values[-1] - values[-2]
        else:
            slope = 0.0
            delta = 0.0
        return mean, std, slope, delta

    def build(self, sensor_readings, case_id, window_size=10):
        values = np.zeros(self.num_sensors, dtype=np.float32)
        presence = np.zeros(self.num_sensors, dtype=np.float32)
        means = np.zeros(self.num_sensors, dtype=np.float32)
        stds = np.zeros(self.num_sensors, dtype=np.float32)
        slopes = np.zeros(self.num_sensors, dtype=np.float32)
        deltas = np.zeros(self.num_sensors, dtype=np.float32)

        for sensor_key, reading in sensor_readings.items():
            if sensor_key not in self.sensor_to_idx:
                continue
            idx = self.sensor_to_idx[sensor_key]

            try:
                numeric_val = float(reading)
            except (TypeError, ValueError):
                numeric_val = 0.0

            values[idx] = numeric_val
            presence[idx] = 1.0

            history = self.case_sensor_history[case_id][sensor_key]
            history.append(numeric_val)
            if len(history) > window_size:
                self.case_sensor_history[case_id][sensor_key] = history[-window_size:]
                history = self.case_sensor_history[case_id][sensor_key]

            means[idx], stds[idx], slopes[idx], deltas[idx] = self._get_temporal_stats(history)

        return np.concatenate([values, presence, means, stds, slopes, deltas])

    @property
    def feature_dim(self):
        return self.num_sensors * 6

    def get_feature_names(self):
        names = []
        for prefix in ['val', 'pres', 'mean', 'std', 'slope', 'delta']:
            for sensor in self.sensor_vocab:
                names.append(f"iot_{prefix}_{sensor}")
        return names

    def evict_case(self, case_id):
        if case_id in self.case_sensor_history:
            del self.case_sensor_history[case_id]
