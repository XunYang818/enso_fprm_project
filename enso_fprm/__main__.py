"""python -m enso_fprm --suite smoke|full --config configs/default.yaml"""
import argparse
import os
from pathlib import Path


def main():
    # Set before importing NumPy / matplotlib. All caches stay in workspace/codex_proc.
    workspace = Path(__file__).resolve().parents[2]
    proc = workspace / "codex_proc" / "enso_fprm"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["MPLCONFIGDIR"] = str(proc / "cache" / "matplotlib")
    parser = argparse.ArgumentParser(description="Author-inspired ENSO FPRM recovery experiments")
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    parser.add_argument("--resume", type=Path, help="Existing run; refuses identity mismatches")
    parser.add_argument("--prepare", action="store_true", help="Fetch/verify data and original author source only")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--limit-cases", type=int, help="Explicit development subset; recorded in provenance")
    args = parser.parse_args()
    if args.prepare:
        from .data import prepare_assets, load_enso
        source = prepare_assets(include_source=True)
        data = load_enso(source)
        print(f"Verified {data.shape}: {data.index[0]} to {data.index[-1]}; {source}")
        return 0
    from .experiment import load_config, run
    _, checks = run(load_config(args.config), args.suite, args.resume, not args.no_plots, args.limit_cases)
    return 1 if checks["n_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
