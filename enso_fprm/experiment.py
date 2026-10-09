"""Experiment orchestration, guarded resume, and evaluation-only ground truth."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from time import perf_counter
import uuid

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits
import yaml

from .data import VARIABLES, load_enso, prepare_assets
from .io import PROC, artifact_path, canonical_hash, code_fingerprint, digest, environment_info, write_json
from .masks import derived_seed, make_mask, missing_count
from .metrics import evaluate
from .models import GPSettings, RecoveryResult, configured_methods, recover_bundle, warmup
from .plots import make_plots
from .statistics import METRICS, paired_comparisons, seed_overall, summarize, validate_statistics_settings

CASE_ROW_FIELDS = ("experiment", "reference", "target", "mask_kind", "rate", "seed", "n", "expected_targets")
ROW_FIELDS = set(CASE_ROW_FIELDS) | set(METRICS) | {
    "method", "case_id", "model_seed", "actual_rate", "n_observed", "n_hidden",
    "mask_sha256", "status", "error", "undefined"}


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    if not isinstance(cfg, dict) or cfg.get("schema_version") != 1:
        raise ValueError("Expected schema_version: 1 configuration")
    required = {"schema_version", "master_seed", "seeds", "rates", "reference", "alternate_reference",
                "gp", "bootstrap_iterations", "statistics_seed", "plot_seed", "experiments"}
    if set(cfg) != required:
        raise ValueError(f"Missing or unknown configuration keys: {set(cfg) ^ required}")
    if not cfg["rates"] or len(set(cfg["rates"])) != len(cfg["rates"]) or any(not 0 < r < 1 for r in cfg["rates"]):
        raise ValueError("rates must be distinct numbers between zero and one")
    if not cfg["seeds"] or any(type(s) is not int or s < 0 for s in cfg["seeds"]) or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("seeds must be distinct nonnegative integers")
    for name in ("master_seed", "plot_seed"):
        if type(cfg[name]) is not int or cfg[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if cfg["reference"] not in VARIABLES or cfg["alternate_reference"] not in VARIABLES:
        raise ValueError("Unknown reference variable")
    if cfg["reference"] == cfg["alternate_reference"]:
        raise ValueError("References must differ")
    allowed = {"main", "block", "reference_check", "reproduction"}
    if not cfg["experiments"] or set(cfg["experiments"]) - allowed or len(set(cfg["experiments"])) != len(cfg["experiments"]):
        raise ValueError("Unknown or duplicate experiment group")
    validate_statistics_settings(cfg["bootstrap_iterations"], cfg["statistics_seed"])
    gp_settings(cfg)


def gp_settings(cfg: dict) -> GPSettings:
    values = dict(cfg["gp"])
    for k in ("delays", "alphas"):
        if k in values:
            values[k] = tuple(values[k])
    return GPSettings(**values)


def _validate_planned_cases(planned: list[dict], gp: GPSettings) -> None:
    """Reject unusable cases before data preparation or creating run artifacts.

    Check only methods actually requested: reproduction needs neither the
    other delays nor the weighted ensemble's CV folds.
    """
    for case in planned:
        delays = set()
        for method in case["methods"]:
            if method == "fprm":
                delays.add(gp.original_delay)
            elif method.startswith("fprm_tau"):
                delays.add(int(method[len("fprm_tau"):]))
            elif method.startswith("multiscale"):
                delays.update(gp.delays)
        n = case["n"]
        valid_length = 422 - (gp.dimension - 1) * max(delays, default=0)
        context = (f"Invalid {case['experiment']} case for {case['target']} "
                   f"(rate={case['rate']}, reference_length=422, dimension={gp.dimension}, "
                   f"delays={sorted(delays)}, n={n}): ")
        if n < 2 or n > valid_length:
            raise ValueError(context + f"effective target length must be >= 2 and <= {valid_length}")
        try:
            n_hidden = missing_count(n, case["rate"])
        except ValueError as exc:
            raise ValueError(context + str(exc)) from exc
        n_observed = n - n_hidden
        if n_observed < 2:
            raise ValueError(context + f"at least two observed labels are required; got {n_observed}")
        if "multiscale_weighted" in case["methods"]:
            largest_fold = (n_observed + gp.folds - 1) // gp.folds
            if n_observed < gp.folds or n_observed - largest_fold < 2:
                raise ValueError(context + f"{n_observed} observed labels cannot support "
                                 f"{gp.folds} CV folds with at least two training labels per fold")


def cases(cfg: dict, suite: str):
    gp = gp_settings(cfg)
    comparison_methods = list(configured_methods(gp))
    n = 422 - (gp.dimension - 1) * max(gp.delays)
    if suite == "smoke":
        target = next(v for v in VARIABLES if v != cfg["reference"])
        planned = [{"experiment": "main", "reference": cfg["reference"], "target": target,
                 "rate": 0.5, "seed": 0, "mask_kind": "random", "n": n,
                 "methods": comparison_methods, "expected_targets": 1}]
        _validate_planned_cases(planned, gp)
        return planned
    result = []
    for experiment in cfg["experiments"]:
        ref = cfg["alternate_reference"] if experiment == "reference_check" else cfg["reference"]
        targets = [v for v in VARIABLES if v != ref]
        rates = [0.5] if experiment == "reproduction" else cfg["rates"]
        count = 422 - (gp.dimension - 1) * gp.original_delay if experiment == "reproduction" else n
        methods = ["fprm"] if experiment == "reproduction" else comparison_methods
        if experiment == "reference_check":
            methods = [m for m in comparison_methods if m == "fprm" or m.startswith("fprm_tau")]
            methods += ["multiscale_equal", "multiscale_weighted"]
        for target in targets:
            for rate in rates:
                for seed in cfg["seeds"]:
                    result.append({"experiment": experiment, "reference": ref, "target": target,
                                   "rate": rate, "seed": seed, "mask_kind": "block" if experiment == "block" else "random",
                                   "n": count, "methods": methods, "expected_targets": len(targets)})
    _validate_planned_cases(result, gp)
    return result


def case_id(case: dict) -> str:
    return (f"{case['experiment']}_{case['reference']}_{case['target']}_"
            f"r{round(case['rate'] * 1000):03d}_s{case['seed']:02d}_n{case['n']}_"
            + canonical_hash(case)[:8])


def validate_case_rows(record: dict, expected_case: dict | None = None) -> None:
    """Require one complete, correctly identified result per planned method."""
    case = record.get("case")
    if (not isinstance(case, dict) or not set(CASE_ROW_FIELDS).issubset(case)
            or not isinstance(case.get("methods"), list) or not case["methods"]
            or not all(isinstance(m, str) for m in case["methods"])
            or len(set(case["methods"])) != len(case["methods"])):
        raise ValueError("Completion record has an invalid case or method list")
    if expected_case is not None and case != expected_case:
        raise ValueError("Completion record does not match the planned case")
    rows = record.get("rows")
    if (not isinstance(rows, list) or not rows
            or not all(isinstance(row, dict) and ROW_FIELDS.issubset(row) for row in rows)):
        raise ValueError("Completion record is missing required method-result fields")
    methods = [row["method"] for row in rows]
    if (not all(isinstance(m, str) for m in methods)
            or len(methods) != len(set(methods)) or set(methods) != set(case["methods"])):
        raise ValueError("Completion record must contain exactly one result per planned method")
    for row in rows:
        if (any(row[k] != case[k] for k in CASE_ROW_FIELDS)
                or row["case_id"] != case_id(case)):
            raise ValueError("Method result does not match its case")
        if row["status"] not in ("ok", "failed") or not isinstance(row["undefined"], dict):
            raise ValueError("Method result has invalid status or undefined-metric reasons")
        for metric in METRICS:
            value = row[metric]
            nullable = row["status"] == "failed" or metric in ("rho", "nrmse")
            if value is None and nullable:
                continue
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not np.isfinite(value)):
                raise ValueError(f"Method result has invalid {metric}")


def _same_number(actual, expected) -> bool:
    if expected is None:
        return actual is None
    return (not isinstance(actual, bool) and isinstance(actual, (int, float))
            and np.isfinite(actual) and np.isclose(actual, expected, rtol=1e-10, atol=1e-12))


def validate_case_outputs(record: dict) -> None:
    """Cross-check result values against the checksummed prediction/parameter files.

    Read floats losslessly: rounding a constant prediction during CSV parsing can
    change whether Pearson correlation is defined. This is evaluation-only work;
    saved truth never enters model fitting or validation-weight estimation.
    """
    for filename, checksum in record["files"].items():
        if not Path(filename).is_file() or digest(Path(filename)) != checksum:
            raise ValueError("Output file is missing or its checksum differs")
    output_paths = [record[k] for k in ("predictions_file", "parameters_file", "mask_file")]
    if not all(filename in record["files"] for filename in output_paths):
        raise ValueError("Completion record is missing output checksums")
    table = pd.read_csv(record["predictions_file"], float_precision="round_trip")
    mask_table = pd.read_csv(record["mask_file"])
    case = record["case"]
    successful = [row for row in record["rows"] if row["status"] == "ok"]
    expected_columns = {"index", "month", "hidden", "truth", "observed_input",
                        *(row["method"] for row in successful)}
    if not expected_columns.issubset(table.columns):
        raise ValueError("Prediction table is missing required columns or methods")
    mask_columns = ["index", "month", "hidden"]
    if (len(table) != case["n"] or len(mask_table) != case["n"]
            or not set(mask_columns).issubset(mask_table.columns)
            or not table[mask_columns].equals(mask_table[mask_columns])
            or not np.array_equal(table["index"], np.arange(case["n"]))
            or not pd.api.types.is_bool_dtype(table.hidden)):
        raise ValueError("Prediction table does not match the case mask")
    hidden = table.hidden.to_numpy(dtype=bool)
    truth = table.truth.to_numpy(dtype=float)
    observed = table.observed_input.to_numpy(dtype=float)
    if (not np.isfinite(truth).all() or not np.isnan(observed[hidden]).all()
            or not np.array_equal(observed[~hidden], truth[~hidden])):
        raise ValueError("Prediction table has invalid truth or observed input")
    n_hidden = int(hidden.sum())
    parameters = json.loads(Path(record["parameters_file"]).read_text(encoding="utf-8"))
    if (not isinstance(parameters, dict) or parameters.get("case") != case
            or not isinstance(parameters.get("models"), dict)):
        raise ValueError("Parameter file does not match the case")
    expected_weights = {}
    for row in record["rows"]:
        if (type(row["n_hidden"]) is not int or row["n_hidden"] != n_hidden
                or type(row["n_observed"]) is not int or row["n_observed"] != len(hidden) - n_hidden
                or not _same_number(row["actual_rate"], n_hidden / len(hidden))
                or row["mask_sha256"] != record["files"][record["mask_file"]]):
            raise ValueError("Method result counts or mask identity differ from the prediction table")
        if row["status"] != "ok":
            continue
        prediction = table[row["method"]].to_numpy(dtype=float)
        if (not np.isfinite(prediction).all()
                or not np.array_equal(prediction[~hidden], observed[~hidden])):
            raise ValueError("Prediction table has non-finite values or altered observations")
        actual = evaluate(truth, prediction, hidden)
        for metric in ("rmse", "mae", "nrmse", "rho"):
            if not _same_number(row[metric], actual[metric]):
                raise ValueError(f"Method result {metric} differs from the prediction table")
        if row["undefined"] != actual["undefined"]:
            raise ValueError("Undefined-metric reasons differ from the prediction table")
        if not _same_number(row["total_seconds"], sum(row[k] for k in
                            ("train_seconds", "validation_seconds", "predict_seconds"))):
            raise ValueError("Method result total time differs from its phase times")
        if not row["method"].startswith("multiscale"):
            continue
        meta = parameters["models"][row["method"]]
        delays, weights = meta["delays"], meta["weights"]
        errors = meta["oof_mse"] if row["method"] == "multiscale_weighted" else [None] * len(delays)
        if (not delays or any(type(d) is not int or d < 1 for d in delays)
                or len(set(delays)) != len(delays)
                or len(weights) != len(delays) or len(errors) != len(delays)
                or any(not _same_number(w, w) or w < 0 for w in weights)
                or not _same_number(sum(weights), 1.0)
                or any(mse is not None and (not _same_number(mse, mse) or mse < 0) for mse in errors)):
            raise ValueError("Parameter file has invalid multiscale weights or OOF errors")
        for delay, weight, mse in zip(delays, weights, errors):
            expected_weights[(row["method"], delay)] = {
                "case_id": row["case_id"], "method": row["method"], "delay": delay,
                "weight": weight, "oof_mse": mse, **{k: case[k] for k in
                    ("seed", "target", "reference", "rate", "experiment")}}
    if not isinstance(record["weights"], list) or len(record["weights"]) != len(expected_weights):
        raise ValueError("Completion record is missing or has extra weight rows")
    seen = set()
    for row in record["weights"]:
        if not isinstance(row, dict) or type(row.get("delay")) is not int:
            raise ValueError("Completion record has an invalid weight row")
        key = (row["method"], row["delay"])
        if key not in expected_weights or key in seen:
            raise ValueError("Completion record has duplicate or unexpected weight rows")
        expected = expected_weights[key]
        if (not expected.keys() <= row.keys()
                or any(not _same_number(row[k], expected[k]) for k in ("weight", "oof_mse"))
                or any(row[k] != expected[k] for k in expected.keys() - {"weight", "oof_mse"})):
            raise ValueError("Completion record weights differ from the parameter file")
        seen.add(key)


def cached_record(folder: Path, signature: str, expected_case: dict | None = None):
    for path in sorted(folder.glob("attempt_*/completed.json"), reverse=True):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            required = {"signature", "case", "rows", "weights", "predictions_file",
                        "parameters_file", "mask_file", "files"}
            if not isinstance(record, dict) or not required.issubset(record):
                raise ValueError("Completion record is missing required fields")
            if (not isinstance(record["signature"], str) or not isinstance(record["case"], dict)
                    or not isinstance(record["rows"], list) or not record["rows"]
                    or not all(isinstance(row, dict) and "status" in row for row in record["rows"])
                    or not isinstance(record["weights"], list)
                    or not isinstance(record["files"], dict) or not record["files"]):
                raise ValueError("Completion record has invalid field types or empty outputs")
            output_paths = [record[k] for k in ("predictions_file", "parameters_file", "mask_file")]
            if not all(isinstance(f, str) and f in record["files"] for f in output_paths):
                raise ValueError("Completion record is missing output checksums")
            if record["signature"] != signature:
                continue
            validate_case_rows(record, expected_case)
            if any(row["status"] != "ok" for row in record["rows"]):
                continue
            validate_case_outputs(record)
            return record
        except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
            print(f"CACHE_SKIPPED={path}: {type(exc).__name__}: {exc}", flush=True)
    return None


def run_case(frame, case, cfg, gp, case_folder, signature):
    if case["reference"] == case["target"]:
        raise ValueError("Reference and target variables must differ")
    attempts = list(case_folder.glob("attempt_*"))
    number = max([int(p.name.split("_")[-1]) for p in attempts] or [0]) + 1
    folder = case_folder / f"attempt_{number:04d}"
    folder.mkdir(parents=True, exist_ok=False)
    n, target, reference = case["n"], case["target"], case["reference"]
    truth = frame[target].to_numpy()[:n].copy()
    hidden, mask_meta = make_mask(n, case["rate"], case["mask_kind"], target,
                                  case["seed"], cfg["master_seed"])
    y_observed = truth.copy()
    y_observed[hidden] = np.nan
    indices = np.arange(n)
    model_seed = derived_seed(cfg["master_seed"], VARIABLES.index(target),
                              VARIABLES.index(reference), case["seed"], n, int(case["rate"] * 1000),
                              int(case["mask_kind"] == "block"))
    start = perf_counter()
    try:
        # Full truth is deliberately absent from this model call.
        results = recover_bundle(frame[reference].to_numpy(), y_observed, indices,
                                 seed=model_seed, settings=gp, methods=case["methods"])
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        results = {m: RecoveryResult(None, error=f"{type(exc).__name__}: {exc}") for m in case["methods"]}
    wall_seconds = perf_counter() - start
    dates = frame.index[:n].astype(str)
    mask_table = pd.DataFrame({"index": indices, "month": dates, "hidden": hidden})
    mask_file = folder / "mask.csv"
    mask_table.to_csv(mask_file, index=False)
    start_gap = mask_meta["block_start"]
    if start_gap is not None:
        mask_meta.update(block_first_month=str(dates[start_gap]),
                         block_last_month=str(dates[start_gap + hidden.sum() - 1]))
    prediction_table = mask_table.copy()
    prediction_table["truth"] = truth
    prediction_table["observed_input"] = y_observed
    rows, parameters, weight_rows = [], {}, []
    for method, result in results.items():
        row = {**{k: case[k] for k in ("experiment", "reference", "target", "mask_kind", "rate", "seed", "n", "expected_targets")},
               "method": method, "case_id": case_id(case), "model_seed": model_seed,
               "actual_rate": mask_meta["actual_rate"], "n_observed": int((~hidden).sum()),
               "n_hidden": int(hidden.sum()), "mask_sha256": digest(mask_file),
               "train_seconds": result.train_seconds, "validation_seconds": result.validation_seconds,
               "predict_seconds": result.predict_seconds, "total_seconds": result.total_seconds,
               "status": "ok" if result.error is None else "failed", "error": result.error,
               "undefined": {}, "rmse": None, "mae": None, "nrmse": None, "rho": None}
        if result.error is None:
            if not np.array_equal(result.recovered[~hidden], y_observed[~hidden]):
                raise AssertionError("Recovery altered visible labels")
            row.update(evaluate(truth, result.recovered, hidden))
            prediction_table[method] = result.recovered
        parameters[method] = {"error": result.error, **result.metadata}
        if method.startswith("multiscale") and result.error is None:
            errors = result.metadata.get("oof_mse", [None] * len(gp.delays))
            weight_rows.extend({"case_id": case_id(case), "method": method, "delay": delay,
                                "weight": weight, "oof_mse": mse, "seed": case["seed"],
                                "target": target, "reference": reference, "rate": case["rate"],
                                "experiment": case["experiment"]}
                               for delay, weight, mse in zip(gp.delays, result.metadata["weights"], errors))
        rows.append(row)
    predictions_file = folder / "predictions.csv"
    prediction_table.to_csv(predictions_file, index=False)
    parameters_file = folder / "parameters.json"
    write_json(parameters_file, {"case": case, "mask": mask_meta, "models": parameters,
                                 "bundle_wall_seconds": wall_seconds,
                                 "timing_note": "Per-method accounted costs include shared view fits; not cache lookup time."})
    record = {"case": case, "signature": signature, "rows": rows, "weights": weight_rows,
              "predictions_file": str(predictions_file), "parameters_file": str(parameters_file),
              "mask_file": str(mask_file), "files": {str(p): digest(p) for p in
                                                        (mask_file, predictions_file, parameters_file)}}
    write_json(folder / "completed.json", record)
    return record


def aggregate(records, folder, cfg, plots=True, *, planned=None):
    expected = [r["case"] for r in records] if planned is None else planned
    if not records or len(records) != len(expected):
        raise ValueError("Aggregation requires exactly one record per planned case")
    if len({case_id(c) for c in expected}) != len(expected):
        raise ValueError("Aggregation received duplicate cases")
    for record, case in zip(records, expected):
        validate_case_rows(record, case)
        validate_case_outputs(record)
    expected_method_count = sum(len(c["methods"]) for c in expected)
    if sum(len(r["rows"]) for r in records) != expected_method_count:
        raise ValueError("Aggregation is missing planned method results")
    folder.mkdir(parents=True, exist_ok=False)
    metrics = pd.DataFrame([row for record in records for row in record["rows"]])
    flat = metrics.copy()
    flat["undefined"] = flat.undefined.map(lambda v: json.dumps(v, ensure_ascii=False))
    flat.to_csv(folder / "metrics.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(folder / "summary_by_target.csv", index=False)
    overall = seed_overall(metrics)
    overall.to_csv(folder / "overall_by_seed.csv", index=False)
    overall_summary = overall.groupby(["experiment", "reference", "mask_kind", "rate", "n", "method"], dropna=False).nrmse.agg(["count", "mean", "std"]).reset_index()
    overall_summary.to_csv(folder / "summary_overall.csv", index=False)
    paired = paired_comparisons(metrics, overall, cfg["bootstrap_iterations"], cfg["statistics_seed"])
    paired.to_csv(folder / "paired_statistics.csv", index=False)
    weights = [row for record in records for row in record["weights"]]
    pd.DataFrame(weights, columns=["case_id", "method", "delay", "weight", "oof_mse", "seed", "target", "reference", "rate", "experiment"]).to_csv(folder / "weights.csv", index=False)
    if plots:
        make_plots(summary, records, folder / "figures", cfg["plot_seed"], cfg["gp"]["original_delay"])
    checks = {"n_cases": len(records), "n_method_results": len(metrics),
              "n_expected_cases": len(expected), "n_expected_method_results": expected_method_count,
              "n_failed": int((metrics.status != "ok").sum()),
              "metric_undefined_counts": {k: int(metrics[k].isna().sum()) for k in ("rho", "nrmse")},
              "all_predictions_preserve_observed": True,
              "statistics_scope": "Variability over artificial masks on this fixed ENSO dataset only."}
    write_json(folder / "checks.json", checks)
    return checks


def run(cfg: dict, suite="smoke", resume: Path | None = None, plots=True, limit_cases=None) -> tuple[Path, dict]:
    validate_config(cfg)
    gp = gp_settings(cfg)
    planned = cases(cfg, suite)
    if limit_cases is not None:
        if limit_cases < 1:
            raise ValueError("limit_cases must be positive")
        planned = planned[:limit_cases]
    source = prepare_assets()
    frame = load_enso(source)
    env = environment_info()
    identity = {"config": cfg, "suite": suite, "data_sha256": digest(source),
                "code": code_fingerprint(), "environment": env,
                "threads": 1, "case_limit": limit_cases}
    signature = canonical_hash(identity)
    if resume:
        folder = artifact_path(resume)
        previous = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if previous["signature"] != signature:
            raise ValueError("Resume refused: data/config/code/environment/suite identity differs")
    else:
        tag = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        folder = PROC / "runs" / f"{suite}_{tag}_{uuid.uuid4().hex[:6]}"
        folder.mkdir(parents=True, exist_ok=False)
        write_json(folder / "manifest.json", {"signature": signature, "identity": identity,
                                               "case_count": len(planned), "created": datetime.now().astimezone().isoformat(),
                                               "source_workbook": str(source), "data_layout": frame.attrs, "gpr": cfg["gp"]})
        frame.to_csv(folder / "enso_selected.csv", index_label="month")
        (folder / "config_used.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    records = []
    print(f"RUN_DIR={folder}", flush=True)
    with threadpool_limits(limits=1):
        warmup()
        write_json(folder / f"threadpools_{uuid.uuid4().hex[:6]}.json", threadpool_info())
        for i, case in enumerate(planned, 1):
            case_folder = folder / "cases" / case_id(case)
            record = cached_record(case_folder, signature, case) if resume else None
            reused = record is not None
            if record is None:
                record = run_case(frame, case, cfg, gp, case_folder, signature)
            records.append(record)
            failed = sum(r["status"] != "ok" for r in record["rows"])
            print(f"[{i}/{len(planned)}] {case_id(case)} {'reused' if reused else 'finished'}; failed={failed}", flush=True)
    export = folder / f"exports_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    checks = aggregate(records, export, cfg, plots, planned=planned)
    write_json(export / "run_status.json", {"planned_cases": len(planned), "completed_cases": len(records),
                                            "signature": signature, "exports": str(export), **checks})
    print(f"EXPORT_DIR={export}", flush=True)
    print(json.dumps(checks, ensure_ascii=False), flush=True)
    return folder, checks
