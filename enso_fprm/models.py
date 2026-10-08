"""Author-inspired FPRM and multiscale ensembles. No hidden truth enters this module.

The embedding, scalar labels, training-only predictor z-score, and GPR mapping
follow Wu et al.'s public MATLAB implementation, licensed under CC BY 4.0.
Target scaling, optimizer settings, and multiscale cross-validation are explicit
Python implementation choices; see REPRODUCTION.md for the differences.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from .embedding import delay_embedding
from .masks import derived_seed

MAIN_METHODS = ("mean", "linear", "gpr_raw", "fprm", "multiscale_equal", "multiscale_weighted")
ALL_METHODS = MAIN_METHODS + ("fprm_tau1", "fprm_tau6")


@dataclass(frozen=True)
class GPSettings:
    dimension: int = 3
    delays: tuple[int, ...] = (1, 3, 6)
    original_delay: int = 3
    folds: int = 3
    restarts: int = 2
    optimizer: bool = True
    alphas: tuple[float, ...] = (1e-8, 1e-7, 1e-6)

    def __post_init__(self):
        if self.dimension < 1 or self.original_delay not in self.delays:
            raise ValueError("original_delay must be included in delays; dimension >= 1")
        if len(set(self.delays)) != len(self.delays) or any(t < 1 for t in self.delays):
            raise ValueError("delays must be unique positive integers")
        if self.folds < 2 or self.restarts < 0 or not self.alphas or any(a <= 0 for a in self.alphas):
            raise ValueError("Invalid CV, restart, or numerical-jitter settings")


@dataclass
class RecoveryResult:
    recovered: np.ndarray | None
    train_seconds: float = 0.0
    validation_seconds: float = 0.0
    predict_seconds: float = 0.0
    metadata: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def total_seconds(self) -> float:
        return self.train_seconds + self.validation_seconds + self.predict_seconds


class ModelFailure(RuntimeError):
    pass


def standardize(values) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Standardization needs at least two finite training samples")
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0, ddof=1)
    std = np.where(np.ptp(values, axis=0) == 0, 1.0, std)
    return (values - mean) / std, mean, std


def fit_predict(x_train, y_train, x_predict, settings: GPSettings, seed: int):
    """Fit fresh fold-local scalers and a shared-length-scale SE Gaussian process."""
    start = perf_counter()
    zx, xm, xs = standardize(x_train)
    zy, ym, ys = standardize(y_train)
    events, failed_alphas = [], []
    model = None
    for alpha in settings.alphas:
        kernel = ConstantKernel(0.5, (1e-4, 1e4)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(0.01, (1e-8, 1e1))
        candidate = GaussianProcessRegressor(
            kernel=kernel, alpha=alpha, normalize_y=False,
            optimizer="fmin_l_bfgs_b" if settings.optimizer else None,
            n_restarts_optimizer=settings.restarts if settings.optimizer else 0,
            random_state=seed)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    candidate.fit(zx, zy)
                finally:
                    events.extend({"category": w.category.__name__, "message": str(w.message),
                                   "alpha": alpha} for w in caught)
            model = candidate
            break
        except np.linalg.LinAlgError as exc:
            failed_alphas.append({"alpha": alpha, "error": str(exc)})
    if model is None:
        raise ModelFailure(f"GPR factorization failed at all jitter values: {failed_alphas}")
    fit_seconds = perf_counter() - start
    start = perf_counter()
    pred = model.predict((np.asarray(x_predict) - xm) / xs) * ys + ym
    predict_seconds = perf_counter() - start
    if not np.isfinite(pred).all():
        raise ModelFailure("GPR returned non-finite predictions")
    info = {"kernel": str(model.kernel_), "kernel_theta": model.kernel_.theta.tolist(),
            "alpha": float(model.alpha), "factorization_retries": failed_alphas,
            "seed": seed, "warnings": events,
            "has_convergence_warning": any(e["category"] == ConvergenceWarning.__name__ for e in events),
            "x_mean": xm.tolist(), "x_std": xs.tolist(),
            "y_mean": float(ym), "y_std": float(ys),
            "n_train": len(y_train), "log_marginal_likelihood": float(model.log_marginal_likelihood_value_)}
    return np.asarray(pred), fit_seconds, predict_seconds, info


def validation_weights(features: dict[int, np.ndarray], observed_target: np.ndarray,
                       indices: np.ndarray, settings: GPSettings, seed: int):
    """Time-ordered 3-fold OOF errors; only the visible labels can set weights."""
    started = perf_counter()
    observed = np.flatnonzero(np.isfinite(observed_target))
    if len(observed) < settings.folds or min(len(observed) - len(f) for f in np.array_split(observed, settings.folds)) < 2:
        raise ModelFailure("Too few observed labels for the requested CV folds")
    folds = np.array_split(observed, settings.folds)
    mse, fold_info = [], []
    for delay, x in features.items():
        oof = np.empty(len(observed), dtype=float)
        mapping = {int(pos): j for j, pos in enumerate(observed)}
        for fold_id, held in enumerate(folds):
            train = np.setdiff1d(observed, held, assume_unique=True)
            pred, fit_s, pred_s, meta = fit_predict(
                x[train], observed_target[train], x[held], settings,
                derived_seed(seed, delay, fold_id, 1))
            oof[[mapping[int(pos)] for pos in held]] = pred
            fold_info.append({"delay": delay, "fold": fold_id,
                              "train_indices": indices[train].tolist(),
                              "validation_indices": indices[held].tolist(),
                              "fit_seconds": fit_s, "predict_seconds": pred_s, **meta})
        mse.append(float(np.mean((oof - observed_target[observed]) ** 2)))
    epsilon = 1e-8 * max(1.0, float(np.var(observed_target[observed], ddof=1)))
    # Scaled inverse MSE is algebraically identical and avoids huge intermediate values.
    denominators = np.asarray(mse) + epsilon
    inverse = denominators.min() / denominators
    weights = inverse / inverse.sum()
    return weights, perf_counter() - started, {
        "delays": list(features), "oof_mse": mse, "epsilon": epsilon,
        "weights": weights.tolist(), "folds": fold_info,
        "weight_source": "visible_labels_only"}


def _validate(reference, target, indices, settings):
    y = np.asarray(target, dtype=float)
    x, idx = delay_embedding(reference, settings.dimension, max(settings.delays), indices)
    if y.ndim != 1 or len(y) != len(idx) or np.isinf(y).any():
        raise ValueError("Target must match valid indices, with NaN for hidden labels")
    if np.isfinite(y).sum() < 2:
        raise ValueError("At least two observed target labels are required")
    return y, idx


def recover_bundle(reference, observed_target, indices, *, seed: int = 0,
                   settings: GPSettings | None = None,
                   methods=ALL_METHODS) -> dict[str, RecoveryResult]:
    """Recover several methods while sharing final views and recording their true costs.

    `observed_target` contains NaN at all hidden positions. The caller keeps truth
    exclusively in its evaluation layer. Indices are original month offsets.
    """
    settings = settings or GPSettings()
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(ALL_METHODS):
        raise ValueError("Unknown or duplicate recovery methods")
    # Single-view reproduction can use the full 416-month valid range.
    if set(methods) <= {"fprm"}:
        check = GPSettings(dimension=settings.dimension, delays=(settings.original_delay,),
                           original_delay=settings.original_delay, folds=settings.folds,
                           restarts=settings.restarts, optimizer=settings.optimizer, alphas=settings.alphas)
    else:
        check = settings
    y, idx = _validate(reference, observed_target, indices, check)
    known = np.flatnonzero(np.isfinite(y))
    hidden = np.flatnonzero(np.isnan(y))
    results, views, feature_views = {}, {}, {}

    def result_from_prediction(pred, train_s, predict_s, meta, validation_s=0.0):
        recovered = y.copy()
        recovered[hidden] = np.asarray(pred)[hidden]
        return RecoveryResult(recovered, train_s, validation_s, predict_s, meta)

    for method in ("mean", "linear"):
        if method not in methods:
            continue
        t = perf_counter()
        mean = float(y[known].mean()) if method == "mean" else None
        train_s = perf_counter() - t
        t = perf_counter()
        pred = np.full(len(y), mean) if method == "mean" else np.interp(idx, idx[known], y[known])
        results[method] = result_from_prediction(pred, train_s, perf_counter() - t,
                                                {"n_train": len(known), "boundary": "nearest_observation"})

    def view(delay, dimension):
        t = perf_counter()
        features, _ = delay_embedding(reference, dimension, delay, idx)
        prepare_s = perf_counter() - t
        pred, train_s, predict_s, meta = fit_predict(
            features[known], y[known], features, settings, derived_seed(seed, delay, dimension, 0))
        meta.update({"dimension": dimension, "delay": delay, "train_indices": idx[known].tolist()})
        return pred, train_s + prepare_s, predict_s, meta, features

    if "gpr_raw" in methods:
        try:
            pred, train_s, predict_s, meta, _ = view(1, 1)
            results["gpr_raw"] = result_from_prediction(pred, train_s, predict_s, meta)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            results["gpr_raw"] = RecoveryResult(None, error=f"{type(exc).__name__}: {exc}")
    need_multi = any(m.startswith("multiscale") for m in methods)
    required = set(settings.delays) if need_multi else set()
    required.update(settings.original_delay for m in methods if m == "fprm")
    required.update(tau for tau in (1, 6) if f"fprm_tau{tau}" in methods)
    failures = {}
    for delay in settings.delays:
        if delay not in required:
            continue
        try:
            pred, train_s, predict_s, meta, features = view(delay, settings.dimension)
            views[delay] = (pred, train_s, predict_s, meta)
            feature_views[delay] = features
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            failures[delay] = f"{type(exc).__name__}: {exc}"
    for method, delay in (("fprm", settings.original_delay), ("fprm_tau1", 1), ("fprm_tau6", 6)):
        if method not in methods:
            continue
        if delay in failures or delay not in views:
            results[method] = RecoveryResult(None, error=failures.get(delay, "Delay absent from settings"))
        else:
            results[method] = result_from_prediction(*views[delay])
    for method in ("multiscale_equal", "multiscale_weighted"):
        if method not in methods:
            continue
        if failures:
            results[method] = RecoveryResult(None, error=f"Required scale failed: {failures}")
            continue
        try:
            validation_s, weight_meta = 0.0, {}
            weights = np.ones(len(settings.delays)) / len(settings.delays)
            if method == "multiscale_weighted":
                weights, validation_s, weight_meta = validation_weights(feature_views, y, idx, settings, seed)
            t = perf_counter()
            pred = np.column_stack([views[d][0] for d in settings.delays]) @ weights
            fusion_s = perf_counter() - t
            meta = {"delays": list(settings.delays), "weights": weights.tolist(),
                    "views": {str(d): views[d][3] for d in settings.delays}, **weight_meta}
            results[method] = result_from_prediction(
                pred, sum(views[d][1] for d in settings.delays),
                sum(views[d][2] for d in settings.delays) + fusion_s, meta, validation_s)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            results[method] = RecoveryResult(None, error=f"{type(exc).__name__}: {exc}")
    return {m: results[m] for m in methods}


def recover(reference, observed_target, indices, *, method="fprm", seed=0, settings=None):
    """Single-method public interface. Returns a RecoveryResult, including any failure."""
    return recover_bundle(reference, observed_target, indices, methods=(method,),
                          seed=seed, settings=settings)[method]


def warmup():
    x = np.linspace(-1, 1, 16)[:, None]
    fit_predict(x, np.sin(x[:, 0]), x, GPSettings(restarts=0, optimizer=False), 0)
