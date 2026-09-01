from __future__ import annotations
import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class TemperatureScaler:

    def __init__(self):
        self.temperature = 1.0
        self._fitted = False

    def fit(self, logits: np.ndarray | torch.Tensor, targets: np.ndarray | torch.Tensor,
            lr: float = 0.01, max_iter: int = 200) -> None:
        logits_t = torch.as_tensor(logits, dtype=torch.float32)
        targets_t = torch.as_tensor(targets, dtype=torch.long)
        if logits_t.ndim != 2:
            raise ValueError(f"logits must be (N, V); got shape {tuple(logits_t.shape)}")
        if logits_t.shape[0] != targets_t.shape[0]:
            raise ValueError("logits and targets must share the leading dimension")
        if logits_t.shape[0] < 2:
            return

        T = nn.Parameter(torch.ones(1) * 1.0)
        optimiser = optim.LBFGS([T], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def closure():
            optimiser.zero_grad()
            loss = criterion(logits_t / T.clamp(min=1e-3), targets_t)
            loss.backward()
            return loss
        optimiser.step(closure)
        self.temperature = float(T.detach().clamp(min=1e-3).item())
        self._fitted = True

    def calibrate(self, logits: np.ndarray | torch.Tensor) -> np.ndarray:
        logits_arr = np.asarray(logits, dtype=np.float64)
        scaled = logits_arr / self.temperature
        scaled = scaled - scaled.max(axis=-1, keepdims=True)
        exp = np.exp(scaled)
        return exp / exp.sum(axis=-1, keepdims=True)

    def calibrated_argmax_prob(self, logits: np.ndarray | torch.Tensor) -> float:
        probs = self.calibrate(np.asarray(logits, dtype=np.float64).reshape(1, -1))[0]
        return float(probs.max())

    @property
    def is_fitted(self) -> bool:
        return self._fitted

class PlattScaler:

    def __init__(self):
        self.a = 1.0
        self.b = 0.0
        self._fitted = False

    def fit(self, confidences: Sequence[float], correctness: Sequence[int],
            lr: float = 0.1, max_iter: int = 500) -> None:
        x = torch.as_tensor(np.asarray(confidences, dtype=np.float32))
        y = torch.as_tensor(np.asarray(correctness, dtype=np.float32))
        if x.shape != y.shape:
            raise ValueError("confidences and correctness must have the same length")
        if x.shape[0] < 2:
            return

        unique = set(int(v) for v in y.tolist())
        if unique <= {0} or unique <= {1}:
            eps = 1e-3
            y = torch.clamp(y, min=eps, max=1.0 - eps)

        a = nn.Parameter(torch.ones(1))
        b = nn.Parameter(torch.zeros(1))
        optimiser = optim.LBFGS([a, b], lr=lr, max_iter=max_iter)

        def closure():
            optimiser.zero_grad()
            logits = a * x + b
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            return loss
        optimiser.step(closure)
        self.a = float(a.detach().item())
        self.b = float(b.detach().item())
        self._fitted = True

    def calibrate(self, conf: float) -> float:
        z = self.a * conf + self.b
        if z >= 0:
            return float(1.0 / (1.0 + math.exp(-z)))
        ez = math.exp(z)
        return float(ez / (1.0 + ez))

    @property
    def is_fitted(self) -> bool:
        return self._fitted

class IsotonicScaler:

    def __init__(self):
        self._iso = None
        self._fitted = False

    def fit(self, confidences: Sequence[float], correctness: Sequence[int]) -> None:
        from sklearn.isotonic import IsotonicRegression
        x = np.asarray(confidences, dtype=np.float64)
        y = np.asarray(correctness, dtype=np.float64)
        if x.shape != y.shape:
            raise ValueError("confidences and correctness must have the same length")
        if x.shape[0] < 2:
            return
        unique = set(int(v) for v in y.tolist())
        if unique <= {0} or unique <= {1}:
            return
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        iso.fit(x, y)
        self._iso = iso
        self._fitted = True

    def calibrate(self, conf: float) -> float:
        if not self._fitted:
            return float(conf)
        return float(self._iso.predict(np.asarray([conf], dtype=np.float64))[0])

    @property
    def is_fitted(self) -> bool:
        return self._fitted
