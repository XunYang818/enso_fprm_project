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
from .masks import derived_seed, make_mask
from .models import ALL_METHODS, GPSettings, RecoveryResult, recover_bundle, warmup
from .plots import make_plots
from .statistics import METRICS, paired_comparisons, seed_overall, summarize


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("schema_version") != 1:
        raise ValueError("Expected schema_version: 1 configuration")
    required = {"schema_version", "master_seed", "seeds", "rates", "reference", "alternate_reference",
                "gp", "bootstrap_iterations", "statistics_seed", "plot_seed", "experiments"}
    if set(cfg) != required:
        raise ValueError(f"Missing or unknown configuration keys: {set(cfg) ^ required}")
    if not cfg["rates"] or len(set(cfg["rates"])) != len(cfg["rates"]) or any(not 0 < r < 1 for r in cfg["rates"]):
        raise ValueError("rates must be distinct numbers between zero and one")
    if not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"]) or any(not isinstance(s, int) or s < 0 for s in cfg["seeds"]):
        raise ValueError("seeds must be distinct nonnegative integers")
    if cfg["reference"] not in VARIABLES or cfg["alternate_reference"] not in VARIABLES:
        raise ValueError("Unknown reference variable")
    if cfg["reference"] == cfg["alternate_reference"]:
        raise ValueError("References must differ")
    allowed = {"main", "block", "reference_check", "reproduction"}
    if not cfg["experiments"] or set(cfg["experiments"]) - allowed or len(set(cfg["experiments"])) != len(cfg["experiments"]):
        raise ValueError("Unknown or duplicate experiment group")
    if cfg["bootstrap_iterations"] < 1:
        raise ValueError("bootstrap_iterations must be positive")
    gp_settings(cfg)
    return cfg


def gp_settings(cfg: dict) -> GPSettings:
    values = dict(cfg["gp"])
    for k in ("delays", "alphas"):
        if k in values:
            values[k] = tuple(values[k])
    return GPSettings(**values)


def cases(cfg: dict, suite: str):
    gp = gp_settings(cfg)
    n = 422 - (gp.dimension - 1) * max(gp.delays)
    if suite == "smoke":
        return [{"experiment": "main", "reference": cfg["reference"], "target": "NINO4",
                 "rate": 0.5, "seed": 0, "mask_kind": "random", "n": n,
                 "methods": list(ALL_METHODS), "expected_targets": 1}]
    result = []
    for experiment in cfg["experiments"]:
        ref = cfg["alternate_reference"] if experiment == "reference_check" else cfg["reference"]
        targets = [v for v in VARIABLES if v != ref]
        rates = [0.5] if experiment == "reproduction" else cfg["rates"]
        count = 422 - (gp.dimension - 1) * gp.original_delay if experiment == "reproduction" else n
        methods = ["fprm"] if experiment == "reproduction" else list(ALL_METHODS)
        if experiment == "reference_check":
            methods = ["fprm", "fprm_tau1", "fprm_tau6", "multiscale_equal", "multiscale_weighted"]
        for target in targets:
            for rate in rates:
                for seed in cfg["seeds"]:
                    result.append({"experiment": experiment, "reference": ref, "target": target,
                                   "rate": rate, "seed": seed, "mask_kind": "block" if experiment == "block" else "random",
                                   "n": count, "methods": methods, "expected_targets": len(targets)})
    return result


def case_id(case: dict) -> str:
    return (f"{case['experiment']}_{case['reference']}_{case['target']}_"
            f"r{round(case['rate'] * 1000):03d}_s{case['seed']:02d}_n{case['n']}_"
            + canonical_hash(case)[:8])


def cached_record(folder: Path, signature: str):
    for path in sorted(folder.glob("attempt_*/completed.json"), reverse=True):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["signature"] != signature or any(row["status"] != "ok" for row in record["rows"]):
            continue
        if all(Path(f).is_file() and digest(Path(f)) == checksum for f, checksum in record["files"].items()):
            return record
    return None


def run_case(frame, case, cfg, gp, case_folder, signature):
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
            from .metrics import evaluate
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


def aggregate(records, folder, cfg, plots=True):
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
        make_plots(summary, records, folder / "figures", cfg["plot_seed"])
    checks = {"n_cases": len(records), "n_method_results": len(metrics),
              "n_failed": int((metrics.status != "ok").sum()),
              "metric_undefined_counts": {k: int(metrics[k].isna().sum()) for k in ("rho", "nrmse")},
              "all_predictions_preserve_observed": True,
              "statistics_scope": "Variability over artificial masks on this fixed ENSO dataset only."}
    write_json(folder / "checks.json", checks)
    return checks


def run(cfg: dict, suite="smoke", resume: Path | None = None, plots=True, limit_cases=None) -> tuple[Path, dict]:
    gp = gp_settings(cfg)
    source = prepare_assets()
    frame = load_enso(source)
    env = environment_info()
    identity = {"config": cfg, "suite": suite, "data_sha256": digest(source),
                "code": code_fingerprint(), "environment": env,
                "threads": 1, "case_limit": limit_cases}
    signature = canonical_hash(identity)
    planned = cases(cfg, suite)
    if limit_cases is not None:
        if limit_cases < 1:
            raise ValueError("limit_cases must be positive")
        planned = planned[:limit_cases]
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
            record = cached_record(case_folder, signature) if resume else None
            reused = record is not None
            if record is None:
                record = run_case(frame, case, cfg, gp, case_folder, signature)
            records.append(record)
            failed = sum(r["status"] != "ok" for r in record["rows"])
            print(f"[{i}/{len(planned)}] {case_id(case)} {'reused' if reused else 'finished'}; failed={failed}", flush=True)
    export = folder / f"exports_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    checks = aggregate(records, export, cfg, plots)
    write_json(export / "run_status.json", {"planned_cases": len(planned), "completed_cases": len(records),
                                            "signature": signature, "exports": str(export), **checks})
    print(f"EXPORT_DIR={export}", flush=True)
    print(json.dumps(checks, ensure_ascii=False), flush=True)
    return folder, checks
