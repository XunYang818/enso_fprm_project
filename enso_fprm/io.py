"""Artifact paths, strict JSON serialization, and provenance fingerprints."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
PROC = WORKSPACE / "codex_proc" / "enso_fprm"
PACKAGES = ("numpy", "scipy", "pandas", "scikit-learn", "matplotlib",
            "openpyxl", "PyYAML", "pytest", "threadpoolctl")


def digest(path: Path, algorithm: str = "sha256") -> str:
    return hashlib.new(algorithm, path.read_bytes()).hexdigest()


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False, default=json_default), encoding="utf-8")


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     default=json_default).encode("utf-8")).hexdigest()


def code_fingerprint() -> dict:
    files = sorted(PROJECT.glob("enso_fprm/*.py"))
    hashes = {str(p.relative_to(PROJECT)).replace("\\", "/"): digest(p) for p in files}
    return {"files": hashes, "sha256": canonical_hash(hashes)}


def environment_info() -> dict:
    versions = {p: importlib.metadata.version(p) for p in PACKAGES}
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "packages": versions,
            "thread_environment": {k: os.environ.get(k) for k in
                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}


def artifact_path(path: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to((WORKSPACE / "codex_proc").resolve()):
        raise ValueError("All generated artifacts must stay inside workspace/codex_proc")
    return path
