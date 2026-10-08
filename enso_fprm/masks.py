"""Deterministic, shared, nested random masks and single contiguous gaps."""
import numpy as np

from .data import VARIABLES


def derived_seed(master: int, *keys: int) -> int:
    return int(np.random.SeedSequence([master, *keys]).generate_state(1)[0])


def missing_count(n: int, rate: float) -> int:
    if n < 2 or not 0 < rate < 1:
        raise ValueError("Need n >= 2 and 0 < missing rate < 1")
    k = int(np.floor(n * rate + 1e-9))
    if k == 0 or k >= n:
        raise ValueError("Mask must retain at least one observed and one hidden value")
    return k


def make_mask(n: int, rate: float, kind: str, target: str, seed: int,
              master: int = 20261008) -> tuple[np.ndarray, dict]:
    code = VARIABLES.index(target)
    hidden = np.zeros(n, dtype=bool)
    k = missing_count(n, rate)
    if kind == "random":
        # No reference variable or rate in the seed: shared references, nested rates.
        rng = np.random.default_rng(np.random.SeedSequence([master, code, seed, n, 0]))
        hidden[rng.permutation(n)[:k]] = True
        start = None
    elif kind == "block":
        rng = np.random.default_rng(np.random.SeedSequence([master, code, seed, n, 1, k]))
        start = int(rng.integers(0, n - k + 1))
        hidden[start:start + k] = True
    else:
        raise ValueError(f"Unknown mask kind: {kind}")
    return hidden, {"kind": kind, "n": n, "requested_rate": rate,
                    "actual_rate": k / n, "n_hidden": k, "block_start": start,
                    "target": target, "seed": seed, "master_seed": master}
