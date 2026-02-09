from __future__ import annotations

from pathlib import Path

import pandas as pd

from SeismoLab.io.loaders import load_csv_pair, load_csv_simple, load_mat_pair
from SeismoLab.models import RecordPair

_PAIR_SUFFIXES = [
    ("_x", "_y"),
    ("_e", "_n"),
    ("_h1", "_h2"),
]


def _group_csv_candidates(
    paths: list[Path], *, input_accel_unit: str = "mps2"
) -> tuple[list[RecordPair], set[Path]]:
    pairs: list[RecordPair] = []
    used: set[Path] = set()

    by_stem = {p.stem.lower(): p for p in paths}

    for stem, p in by_stem.items():
        if p in used:
            continue

        for sx, sy in _PAIR_SUFFIXES:
            if stem.endswith(sx):
                base = stem[: -len(sx)]
                py = by_stem.get(base + sy)
                if py and py not in used:
                    pairs.append(
                        load_csv_pair(p, py, pair_id=Path(base).name, accel_unit=input_accel_unit)
                    )
                    used.add(p)
                    used.add(py)
                break

    return pairs, used


def _try_load_csv_single(path: Path, *, input_accel_unit: str = "mps2") -> RecordPair | None:
    try:
        df = pd.read_csv(path, nrows=1)
        cols = {c.lower().strip() for c in df.columns}
        if {"t", "ax", "ay"}.issubset(cols) or {"time", "ax", "ay"}.issubset(cols):
            return load_csv_simple(path, accel_unit=input_accel_unit)
    except Exception:
        return None
    return None


def discover_pairs_from_paths(
    paths: list[Path], *, input_accel_unit: str = "mps2"
) -> list[RecordPair]:
    csv_paths = [p for p in paths if p.suffix.lower() == ".csv"]
    mat_paths = [p for p in paths if p.suffix.lower() == ".mat"]

    pairs: list[RecordPair] = [load_mat_pair(p, accel_unit=input_accel_unit) for p in mat_paths]

    pair_csv, used_csv = _group_csv_candidates(csv_paths, input_accel_unit=input_accel_unit)
    pairs.extend(pair_csv)

    for p in csv_paths:
        if p in used_csv:
            continue
        single = _try_load_csv_single(p, input_accel_unit=input_accel_unit)
        if single is not None:
            pairs.append(single)

    return pairs


def discover_pairs_in_folder(
    folder: Path, *, recursive: bool = False, input_accel_unit: str = "mps2"
) -> list[RecordPair]:
    pattern = "**/*" if recursive else "*"
    paths = [p for p in folder.glob(pattern) if p.is_file()]
    return discover_pairs_from_paths(paths, input_accel_unit=input_accel_unit)
