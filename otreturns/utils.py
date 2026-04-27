"""
Grid utilities and numerical helpers for optimal transport computations.
"""

from __future__ import annotations

import numpy as np


def uniform_grid(lo: float, hi: float, n: int) -> np.ndarray:
    """Return n evenly-spaced points in [lo, hi]."""
    return np.linspace(lo, hi, n)


def cost_matrix(x: np.ndarray, y: np.ndarray, p: int = 2) -> np.ndarray:
    """
    Compute the ground-cost matrix C_{ij} = |x_i - y_j|^p.

    Args:
        x: source support points, shape (n,)
        y: target support points, shape (m,)
        p: cost exponent

    Returns:
        C: (n, m) cost matrix
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.abs(x[:, None] - y[None, :]) ** p


def normalize_weights(w: np.ndarray) -> np.ndarray:
    """Normalize a non-negative weight vector to sum to 1."""
    w = np.asarray(w, dtype=float)
    s = w.sum()
    if s <= 0:
        raise ValueError("weights must sum to a positive value")
    return w / s


def quantile_grid(n: int) -> np.ndarray:
    """
    Return n quantile levels on (0, 1) excluding endpoints,
    using the midpoint convention u_i = (i - 0.5) / n.
    """
    return (np.arange(1, n + 1) - 0.5) / n


def smooth_labels(labels: np.ndarray, window: int) -> np.ndarray:
    """
    Apply a sliding-window majority vote to smooth a label sequence.

    Args:
        labels: integer label array of shape (T,)
        window: half-width of the smoothing window

    Returns:
        smoothed labels of the same shape
    """
    if window <= 0:
        return labels.copy()

    n = len(labels)
    smoothed = labels.copy()
    for t in range(n):
        lo = max(0, t - window)
        hi = min(n, t + window + 1)
        values, counts = np.unique(labels[lo:hi], return_counts=True)
        smoothed[t] = values[np.argmax(counts)]
    return smoothed


def check_weights(w: np.ndarray, name: str = "weights") -> np.ndarray:
    """Validate and return a normalised weight vector."""
    w = np.asarray(w, dtype=float)
    if np.any(w < 0):
        raise ValueError(f"{name} must be non-negative")
    return normalize_weights(w)
