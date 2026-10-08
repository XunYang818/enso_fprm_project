"""Forward delay coordinates, ported from the author's PhaSpaRecon.m (CC BY 4.0)."""
from __future__ import annotations

import numpy as np


def delay_embedding(reference, dimension: int = 3, delay: int = 3,
                    indices=None) -> tuple[np.ndarray, np.ndarray]:
    """Return rows [x[t], x[t+tau], ...] and their original zero-based t indices.

    This is PhaSpaRecon(s,tau,m).T. Do not reverse the delay direction or
    select samples according to complete target embedding windows.
    """
    x = np.asarray(reference, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Reference must be one-dimensional and fully observed")
    if not isinstance(dimension, int) or not isinstance(delay, int) or dimension < 1 or delay < 1:
        raise ValueError("dimension and delay must be positive integers")
    length = len(x) - (dimension - 1) * delay
    if length < 1:
        raise ValueError("Reference is too short for this embedding")
    if indices is None:
        idx = np.arange(length)
    else:
        idx = np.asarray(indices)
        if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer):
            raise ValueError("indices must be a one-dimensional integer array")
        if len(idx) == 0 or np.any(idx < 0) or np.any(idx >= length):
            raise ValueError("indices outside valid embedding range")
        if np.any(np.diff(idx) <= 0):
            raise ValueError("indices must be unique and strictly increasing")
    return x[idx[:, None] + delay * np.arange(dimension)[None, :]], idx.copy()
