"""Seed-level summaries and paired bootstrap; targets are not independent repeats."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .masks import derived_seed

METRICS = ("rmse", "mae", "nrmse", "rho", "train_seconds", "validation_seconds",
           "predict_seconds", "total_seconds")
GROUPS = ["experiment", "reference", "mask_kind", "rate", "n", "target", "method"]


def validate_statistics_settings(iterations, seed) -> None:
    if type(iterations) is not int or iterations < 1:
        raise ValueError("bootstrap_iterations must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("statistics_seed must be a nonnegative integer")


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    for key, group in rows.groupby(GROUPS, dropna=False, sort=True):
        record = dict(zip(GROUPS, key))
        record.update(n_attempted=len(group), n_success=int((group.status == "ok").sum()),
                      n_failed=int((group.status != "ok").sum()))
        for metric in METRICS:
            values = pd.to_numeric(group.loc[group.status == "ok", metric], errors="coerce").dropna()
            record[metric + "_count"] = len(values)
            record[metric + "_mean"] = float(values.mean()) if len(values) else None
            record[metric + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
        output.append(record)
    return pd.DataFrame(output)


def seed_overall(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["experiment", "reference", "mask_kind", "rate", "n", "method", "seed"]
    result = []
    for key, group in rows.groupby(keys, sort=True):
        expected = group.expected_targets.iloc[0]
        good = group[(group.status == "ok") & group.nrmse.notna()]
        complete = group.target.nunique() == expected and good.target.nunique() == expected
        result.append({**dict(zip(keys, key)), "n_targets": expected,
                       "nrmse": float(good.nrmse.mean()) if complete else None,
                       "status": "ok" if complete else "incomplete_targets"})
    return pd.DataFrame(result)


def paired_bootstrap(differences, *, iterations: int = 10000, seed: int = 20261008) -> dict:
    validate_statistics_settings(iterations, seed)
    d = np.asarray(differences, dtype=float)
    if d.ndim != 1 or not np.isfinite(d).all() or iterations < 1:
        raise ValueError("Bootstrap needs finite paired differences and positive iterations")
    base = {"n_pairs": len(d), "delta_mean": None, "win_rate": None,
            "ci_lower": None, "ci_upper": None, "reason": None}
    if len(d) == 0:
        base["reason"] = "no complete seed pairs"
        return base
    base.update(delta_mean=float(d.mean()), win_rate=float(np.mean(d < 0)))
    if len(d) < 2:
        base["reason"] = "one seed cannot estimate mask variability"
        return base
    rng = np.random.default_rng(seed)
    samples = d[rng.integers(0, len(d), size=(iterations, len(d)))].mean(axis=1)
    lower, upper = np.quantile(samples, (0.025, 0.975))
    base.update(ci_lower=float(lower), ci_upper=float(upper))
    return base


def paired_comparisons(rows: pd.DataFrame, overall: pd.DataFrame,
                       iterations=10000, stats_seed=20261008) -> pd.DataFrame:
    validate_statistics_settings(iterations, stats_seed)
    output = []
    for source, level in ((rows, "target"), (overall, "overall")):
        groups = ["experiment", "reference", "mask_kind", "rate", "n"]
        if level == "target":
            groups += ["target"]
        for i, (key, group) in enumerate(source.groupby(groups, sort=True)):
            good = group[(group.status == "ok") & group.nrmse.notna()]
            pivot = good.pivot(index="seed", columns="method", values="nrmse")
            if not {"fprm", "multiscale_weighted"}.issubset(pivot.columns):
                continue
            pairs = pivot[["fprm", "multiscale_weighted"]].dropna()
            stats = paired_bootstrap((pairs.multiscale_weighted - pairs.fprm).to_numpy(),
                                     iterations=iterations, seed=derived_seed(stats_seed, i, level == "overall"))
            output.append({**dict(zip(groups, key)), "level": level,
                           "comparison": "multiscale_weighted_minus_fprm",
                           "paired_seeds": ",".join(map(str, pairs.index)), **stats})
    return pd.DataFrame(output)
