from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat

from SeismoLab.models import Record, RecordPair


def _accel_to_mps2(values: np.ndarray, accel_unit: str) -> np.ndarray:
    unit = accel_unit.lower().strip()
    if unit in {"mps2", "m/s2"}:
        return values
    if unit in {"gal", "gals"}:
        return values * 0.01
    msg = f"Unidad de aceleracion no soportada: {accel_unit}"
    raise ValueError(msg)


def load_csv_columns(
    path: str | Path,
    *,
    time_col: str,
    x_col: str,
    y_col: str,
    accel_unit: str = "mps2",
    delimiter: str = ",",
    pair_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> RecordPair:
    """Carga un CSV que contiene t, ax y ay usando nombres de columna explicitos."""
    p = Path(path)
    df = pd.read_csv(p, sep=delimiter)
    time = df[time_col].to_numpy(dtype=float)
    ax = _accel_to_mps2(df[x_col].to_numpy(dtype=float), accel_unit)
    ay = _accel_to_mps2(df[y_col].to_numpy(dtype=float), accel_unit)
    pid = pair_id or p.stem
    meta = {"source": str(p), "input_accel_unit": accel_unit, **(metadata or {})}
    rx = Record(record_id=f"{pid}_x", time=time, accel=ax, component="x", metadata=meta)
    ry = Record(record_id=f"{pid}_y", time=time, accel=ay, component="y", metadata=meta)
    return RecordPair(pair_id=pid, x=rx, y=ry, metadata=meta)


def load_csv_simple(
    path: str | Path,
    *,
    accel_unit: str = "mps2",
    delimiter: str = ",",
    pair_id: str | None = None,
) -> RecordPair:
    """Carga un CSV simple con columnas t, ax, ay (tolerante a mayusculas)."""
    p = Path(path)
    df = pd.read_csv(p, sep=delimiter)
    cols = {c.lower().strip(): c for c in df.columns}
    return load_csv_columns(
        p,
        time_col=cols.get("t", cols.get("time", "t")),
        x_col=cols.get("ax", cols.get("x", "ax")),
        y_col=cols.get("ay", cols.get("y", "ay")),
        accel_unit=accel_unit,
        delimiter=delimiter,
        pair_id=pair_id,
    )


def load_csv_pair(
    path_x: str | Path,
    path_y: str | Path,
    *,
    accel_unit: str = "mps2",
    delimiter: str = ",",
    time_col: str = "t",
    value_col_x: str = "ax",
    value_col_y: str = "ay",
    pair_id: str | None = None,
) -> RecordPair:
    """Carga dos CSV separados (t+ax y t+ay)."""
    px = Path(path_x)
    py = Path(path_y)
    dfx = pd.read_csv(px, sep=delimiter)
    dfy = pd.read_csv(py, sep=delimiter)

    tx = dfx[time_col].to_numpy(dtype=float)
    ty = dfy[time_col].to_numpy(dtype=float)
    if len(tx) != len(ty) or not np.allclose(tx, ty):
        msg = "Los archivos x/y no comparten el mismo vector de tiempo"
        raise ValueError(msg)

    ax = _accel_to_mps2(dfx[value_col_x].to_numpy(dtype=float), accel_unit)
    ay = _accel_to_mps2(dfy[value_col_y].to_numpy(dtype=float), accel_unit)
    pid = pair_id or px.stem.replace("_x", "").replace("_e", "")
    meta = {"source_x": str(px), "source_y": str(py), "input_accel_unit": accel_unit}
    rx = Record(record_id=f"{pid}_x", time=tx, accel=ax, component="x", metadata=meta)
    ry = Record(record_id=f"{pid}_y", time=tx, accel=ay, component="y", metadata=meta)
    return RecordPair(pair_id=pid, x=rx, y=ry, metadata=meta)


def _to_1d(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=float).squeeze()
    if out.ndim != 1:
        msg = "La variable MAT debe ser vectorial"
        raise ValueError(msg)
    return out


def load_mat_pair(
    path: str | Path,
    *,
    key_x: str = "acc_f_e",
    key_y: str = "acc_f_n",
    key_dt: str = "dt",
    accel_unit: str = "mps2",
    pair_id: str | None = None,
) -> RecordPair:
    """Carga un .mat con formato de acelerogramas filtrados E/N + dt."""
    p = Path(path)
    data = loadmat(p)

    ax = _accel_to_mps2(_to_1d(data[key_x]), accel_unit)
    ay = _accel_to_mps2(_to_1d(data[key_y]), accel_unit)
    dt = float(np.asarray(data[key_dt]).squeeze())
    time = np.arange(len(ax), dtype=float) * dt

    pid = pair_id or p.stem
    header = data.get("header")
    header_str = None
    if header is not None:
        header_str = " ".join(np.asarray(header).astype(str).ravel())

    meta = {
        "source": str(p),
        "mat_key_x": key_x,
        "mat_key_y": key_y,
        "dt": dt,
        "header": header_str,
        "input_accel_unit": accel_unit,
    }
    rx = Record(record_id=f"{pid}_x", time=time, accel=ax, component="x", metadata=meta)
    ry = Record(record_id=f"{pid}_y", time=time, accel=ay, component="y", metadata=meta)
    return RecordPair(pair_id=pid, x=rx, y=ry, metadata=meta)
