"""Metrics on hidden positions only; undefined values have explicit reasons."""
import numpy as np


def evaluate(truth, recovered, hidden) -> dict:
    y, pred, mask = np.asarray(truth), np.asarray(recovered), np.asarray(hidden, dtype=bool)
    if y.ndim != 1 or y.shape != pred.shape or y.shape != mask.shape:
        raise ValueError("truth, recovered, and hidden must have equal 1D shapes")
    a, b = y[mask].astype(float), pred[mask].astype(float)
    if len(a) == 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Evaluation requires nonempty, finite hidden truth and predictions")
    err = b - a
    rmse = float(np.sqrt(np.mean(err ** 2)))
    result = {"rmse": rmse, "mae": float(np.mean(np.abs(err))),
              "n_hidden": len(a), "nrmse": None, "rho": None, "undefined": {}}
    if len(a) < 2:
        result["undefined"] = {"nrmse": "fewer than two hidden points",
                               "rho": "fewer than two hidden points"}
        return result
    std = float(np.std(a, ddof=1))
    truth_constant = np.ptp(a) == 0
    if not truth_constant and std > 0:
        result["nrmse"] = rmse / std
    else:
        result["undefined"]["nrmse"] = "hidden truth has zero sample standard deviation"
    if truth_constant or np.ptp(b) == 0:
        result["undefined"]["rho"] = "truth or predictions are constant"
    else:
        result["rho"] = float(np.corrcoef(a, b)[0, 1])
    return result
