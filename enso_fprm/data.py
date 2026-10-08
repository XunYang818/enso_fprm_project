"""Read the author's ENSO workbook without repairing or silently dropping data."""
from __future__ import annotations

import hashlib
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .io import PROC, digest, write_json

ARTICLE = "https://doi.org/10.6084/m9.figshare.30446765.v3"
ASSETS = {
    "ENSO data.xlsx": {
        "url": "https://ndownloader.figshare.com/files/64540869",
        "md5": "fad680f9cea6557d3a3052a1328dc4bd",
        "sha256": "631f8046ee637c80a2ad28a20d8530bc56b3e5b1c5b613f21a47a3d33deb3e33"},
    "PhaSpaRecon.m": {
        "url": "https://ndownloader.figshare.com/files/64540899",
        "md5": "afb1cde2e407eb9725da6ff428af2267",
        "sha256": "c13dbd7dd5dd40f0436f28aaf16d414062660500d1afd14c6bbda151225cad8e"},
    "FPRM mian code.mlx": {
        "url": "https://ndownloader.figshare.com/files/64540893",
        "md5": "33f9655df4492ec20b184229497ca9da",
        "sha256": "f2afaae07de0d2c73474be3eaaefb8a8d0bdaac6ea6de9e0a98b485dd8e9260b"},
}
VARIABLES = ("NINO4", "NINO34", "NINO3", "NINO12")


def verify_asset(path: Path, spec: dict) -> None:
    for algorithm in ("md5", "sha256"):
        if digest(path, algorithm) != spec[algorithm]:
            raise ValueError(f"{algorithm} mismatch: {path}; existing file is preserved")


def prepare_assets(folder: Path = PROC / "data", include_source: bool = False) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name, spec in ASSETS.items():
        if not include_source and name != "ENSO data.xlsx":
            continue
        dest = folder / name
        if not dest.exists():
            request = urllib.request.Request(spec["url"], headers={"User-Agent": "enso-fprm/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response:
                content = response.read()
            for algorithm in ("md5", "sha256"):
                if hashlib.new(algorithm, content).hexdigest() != spec[algorithm]:
                    raise ValueError(f"Downloaded {name} failed {algorithm} verification")
            dest.write_bytes(content)
        verify_asset(dest, spec)
    write_json(folder / "sources.json", {
        "article": ARTICLE, "figshare_version": 3, "assets": ASSETS,
        "authors": "Tao Wu et al.", "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/"})
    return folder / "ENSO data.xlsx"


def load_enso(path: Path) -> pd.DataFrame:
    verify_asset(path, ASSETS["ENSO data.xlsx"])
    # The YEAR header is on row 11 in v3; detect it instead of assuming skiprows.
    raw = pd.read_excel(path, sheet_name=0, header=None, engine="openpyxl")
    matches = [i for i, value in enumerate(raw.iloc[:, 0])
               if str(value).strip().upper() == "YEAR"]
    if len(matches) != 1:
        raise ValueError("Workbook must contain exactly one YEAR header")
    header = matches[0]
    names = [re.sub(r"\s+", "", str(v)).upper() for v in raw.iloc[header]]
    expected = ["YEAR", "MONTH", "NINO4", "NINO34", "NINO3", "NINO1+2"]
    # v3's header physically splits "Month" across cells: "Mo", "nth NINO4".
    # Accept this exact documented label defect; numerical columns stay untouched.
    known_split_header = ["YEAR", "MO", "NTHNINO4", "NINO34", "NINO3", "NINO1+2"]
    if names not in (expected, known_split_header):
        raise ValueError(f"Unexpected workbook columns: {names}")
    data = raw.iloc[header + 1:].copy()
    data.columns = expected
    data = data.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Original workbook contains non-finite values")
    years, months = data.YEAR.to_numpy(), data.MONTH.to_numpy()
    if not ((years == np.floor(years)).all() and (months == np.floor(months)).all()):
        raise ValueError("YEAR and MONTH must be integers")
    if np.any((months < 1) | (months > 12)):
        raise ValueError("MONTH must be between 1 and 12")
    dates = pd.PeriodIndex.from_fields(year=years.astype(int), month=months.astype(int), freq="M")
    if not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("Original months must be unique and ordered")
    if not np.array_equal(dates.asi8, np.arange(dates.asi8[0], dates.asi8[-1] + 1)):
        raise ValueError("Original months must be continuous")
    frame = data[["NINO4", "NINO34", "NINO3", "NINO1+2"]].copy()
    frame.columns = VARIABLES
    frame.index = dates
    frame = frame.loc["1990-01":"2025-02"]
    if frame.shape != (422, 4) or not frame.index.equals(pd.period_range("1990-01", "2025-02", freq="M")):
        raise ValueError("Expected exactly 422 consecutive months, 1990-01 to 2025-02")
    frame = frame.astype(float)
    frame.attrs["original_header"] = names
    frame.attrs["header_mapping"] = dict(zip(names, ["YEAR", "MONTH", *VARIABLES]))
    return frame
