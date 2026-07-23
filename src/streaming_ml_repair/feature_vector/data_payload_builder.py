import numpy as np
import hashlib

class DataPayloadBuilder:

    def __init__(self, max_keys=50, hash_dim=8):
        self.max_keys = max_keys
        self.hash_dim = hash_dim
        self.key_vocab = {}
        self.next_idx = 0

    @property
    def feature_dim(self):
        return self.max_keys * (3 + self.hash_dim) + self.hash_dim + 1

    def _register_key(self, key_name):
        if key_name not in self.key_vocab and self.next_idx < self.max_keys:
            self.key_vocab[key_name] = self.next_idx
            self.next_idx += 1

    def _hash_value(self, value):
        h = hashlib.md5(str(value).encode()).hexdigest()
        vec = []
        for i in range(self.hash_dim):
            byte_val = int(h[i * 2:(i * 2) + 2], 16)
            vec.append((byte_val / 255.0) * 2 - 1)
        return vec

    def _extract_numeric(self, value):
        if value is None:
            return 0.0, False
        if isinstance(value, (int, float)):
            return float(value), True
        if isinstance(value, bool):
            return 1.0 if value else 0.0, True
        if isinstance(value, str):
            try:
                return float(value), True
            except (ValueError, TypeError):
                return 0.0, False
        if isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], (int, float)):
                return float(len(value)), True
            return float(len(value)), True
        if isinstance(value, dict):
            return float(len(value)), True
        return 0.0, False

    def build(self, data_field):
        key_presence = np.zeros(self.max_keys, dtype=np.float32)
        key_values = np.zeros(self.max_keys, dtype=np.float32)
        key_is_numeric = np.zeros(self.max_keys, dtype=np.float32)
        value_hashes = np.zeros((self.max_keys, self.hash_dim), dtype=np.float32)
        combined_hash = [0.0] * self.hash_dim
        item_count = 0.0

        if not data_field or not isinstance(data_field, list):
            return np.concatenate([key_presence, key_values, key_is_numeric,
                                   value_hashes.flatten(), combined_hash, [item_count]])

        key_names = []
        for item in data_field:
            if not isinstance(item, dict) or 'name' not in item:
                continue
            key_name = str(item['name'])
            value = item.get('value')
            self._register_key(key_name)
            key_names.append(key_name)
            item_count += 1

            if key_name in self.key_vocab:
                idx = self.key_vocab[key_name]
                key_presence[idx] = 1.0
                num_val, is_num = self._extract_numeric(value)
                key_values[idx] = num_val
                key_is_numeric[idx] = 1.0 if is_num else 0.0
                value_hashes[idx] = self._hash_value(str(value)[:200])

        if key_names:
            combined_hash = self._hash_value('|'.join(sorted(key_names)))

        return np.concatenate([key_presence, key_values, key_is_numeric,
                               value_hashes.flatten(), combined_hash, [item_count]])
