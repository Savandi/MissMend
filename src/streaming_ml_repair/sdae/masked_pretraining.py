import torch
import torch.nn as nn
import numpy as np
from collections import Counter

class MaskedActivityPretrainer:

    def __init__(self, latent_dim, mask_ratio=0.15):
        self.latent_dim = latent_dim
        self.mask_ratio = mask_ratio
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.class_weights = None
        self.classifier = None

    def fit_labels(self, labels):
        unique = sorted(set(labels))
        self.label_to_idx = {l: i for i, l in enumerate(unique)}
        self.idx_to_label = {i: l for l, i in self.label_to_idx.items()}

        counts = Counter(labels)
        total = sum(counts.values())
        n_classes = len(unique)
        weights = []
        for label in unique:
            w = total / (n_classes * counts[label])
            weights.append(w)
        self.class_weights = torch.tensor(weights, dtype=torch.float32)

        self.classifier = nn.Linear(self.latent_dim, n_classes)

    def encode_labels(self, labels):
        encoded = []
        for l in labels:
            if l not in self.label_to_idx:
                idx = len(self.label_to_idx)
                self.label_to_idx[l] = idx
                self.idx_to_label[idx] = l
                self._expand_classifier()
            encoded.append(self.label_to_idx[l])
        return torch.tensor(encoded, dtype=torch.long)

    def _expand_classifier(self):
        n_classes = len(self.label_to_idx)
        old = self.classifier
        self.classifier = nn.Linear(self.latent_dim, n_classes)
        if old is not None and old.out_features < n_classes:
            with torch.no_grad():
                self.classifier.weight[:old.out_features] = old.weight
                self.classifier.bias[:old.out_features] = old.bias
        counts = {l: 1 for l in self.label_to_idx}
        total = sum(counts.values())
        self.class_weights = torch.tensor(
            [total / (n_classes * counts.get(l, 1)) for l in sorted(self.label_to_idx, key=self.label_to_idx.get)],
            dtype=torch.float32
        )

    def compute_loss(self, z, labels_encoded):
        if self.classifier is None:
            return torch.tensor(0.0, device=z.device)
        logits = self.classifier(z)
        weights = self.class_weights.to(z.device)
        return nn.functional.cross_entropy(logits, labels_encoded, weight=weights)

    def mask_features(self, batch, cf_feature_count):
        masked = batch.clone()
        mask = torch.rand(batch.size(0)) < self.mask_ratio
        if mask.any():
            masked[mask, :cf_feature_count] = 0.0
        return masked, mask

    @property
    def num_classes(self):
        return len(self.label_to_idx)
