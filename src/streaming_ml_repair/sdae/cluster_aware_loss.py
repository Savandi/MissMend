import torch
import torch.nn.functional as F
from collections import defaultdict

class ClusterAwareLoss:

    def __init__(self, separation_weight=1.0, compactness_weight=0.5):
        self.separation_weight = separation_weight
        self.compactness_weight = compactness_weight

    def compute(self, embeddings, labels):
        unique_labels = list(set(labels))
        if len(unique_labels) < 2:
            return torch.tensor(0.0, device=embeddings.device)

        label_to_idx = defaultdict(list)
        for i, label in enumerate(labels):
            label_to_idx[label].append(i)

        centroids = {}
        for label, indices in label_to_idx.items():
            centroids[label] = embeddings[indices].mean(dim=0)

        separation_loss = torch.tensor(0.0, device=embeddings.device)
        pairs = 0
        centroid_list = list(centroids.values())
        for i in range(len(centroid_list)):
            for j in range(i + 1, len(centroid_list)):
                dist = torch.norm(centroid_list[i] - centroid_list[j], p=2)
                separation_loss = separation_loss + torch.exp(-dist)
                pairs += 1
        if pairs > 0:
            separation_loss /= pairs

        compactness_loss = torch.tensor(0.0, device=embeddings.device)
        for label, indices in label_to_idx.items():
            if len(indices) > 1:
                cluster_embs = embeddings[indices]
                centroid = centroids[label]
                dists = torch.norm(cluster_embs - centroid.unsqueeze(0), p=2, dim=1)
                compactness_loss = compactness_loss + dists.mean()
        compactness_loss = compactness_loss / len(unique_labels)

        return self.separation_weight * separation_loss + self.compactness_weight * compactness_loss
