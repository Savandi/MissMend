"""Asymmetric calibrators for the hybrid framework's two confidence sources.

The sequence rescue head emits a softmax distribution over the activity
vocabulary; its argmax probability is calibrated with TEMPERATURE SCALING
(Guo et al., ICML 2017). Temperature scaling fits a single scalar T that
divides the logits before softmax: it is the simplest, most widely-adopted
softmax post-hoc calibrator and is exactly applicable here because the
head produces logits.

The cluster matcher emits a composite confidence
    conf = u^* . (1 - H(u) / H_max)^lambda . w_{k^*}
that is NOT a softmax probability and has no logits. Temperature scaling
is mis-specified for this signal. Instead the cluster confidence is mapped
to a calibrated probability via PLATT SCALING (Niculescu-Mizil & Caruana,
ICML 2005): a sigmoid map  P(correct | conf) = sigmoid(a * conf + b)  fit
against empirical correctness on a held-out calibration slice.

Both calibrators expose .fit(...) and .calibrate(...) and a no-op state
that returns the input unchanged (used before the first fit).
"""
from __future__ import annotations
import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class TemperatureScaler:
    """Single-parameter temperature scaling for a softmax classifier.

    fit() minimises NLL on a held-out (logits, target_id) set.
    calibrate() returns the calibrated softmax distribution for a batch
    of logits; calibrated_argmax_prob() returns just the argmax probability
    (i.e. the calibrated equivalent of the head's raw argmax confidence).
    """

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
    """Two-parameter sigmoid (Platt) scaling for a scalar confidence signal.

    Fits sigmoid(a * conf + b) against empirical correctness {0,1} via
    gradient descent on binary cross-entropy. Returns identity (the input
    confidence unchanged) before the first fit.
    """

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
    """Non-parametric monotonic calibration (Zadrozny & Elkan 2002).

    Wraps sklearn.isotonic.IsotonicRegression. Used as the non-parametric
    comparator in the calibration validation: if the score-correctness
    distortion is not sigmoidal, isotonic regression will fit it whereas
    Platt cannot. The comparison Brier(Platt) vs Brier(Isotonic) on a
    held-out half of H is the empirical test of whether Platt's sigmoidal
    assumption is justified for this score.
    """

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
