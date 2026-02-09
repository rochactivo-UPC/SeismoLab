from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from SeismoLab.models import RecordPair
from SeismoLab.preprocess.signal import accel_to_velocity_with_diagnostics
from SeismoLab.rotations.rotd import rotd_spectrum

G_STD = 9.80665


def _scale_pga(value: float, unit: str) -> float:
    if unit == "g":
        return value / G_STD
    if unit == "mps2":
        return value
    msg = f"Unidad PGA no soportada: {unit}"
    raise ValueError(msg)


def _scale_pgv(value: float, unit: str) -> float:
    if unit == "cms":
        return value * 100.0
    if unit == "mps":
        return value
    msg = f"Unidad PGV no soportada: {unit}"
    raise ValueError(msg)


def pair_ims_to_row(
    pair: RecordPair,
    *,
    pga_unit: str = "g",
    pgv_unit: str = "cms",
    rot_n_angles: int = 180,
    hp_corner_hz: float = 0.1,
    hp_order: int = 4,
    taper_fraction: float = 0.0,
    baseline_window_fraction: float = 0.05,
) -> dict[str, float | str | int | None]:
    ax = pair.x.accel
    ay = pair.y.accel
    dt = pair.dt

    vx, dx = accel_to_velocity_with_diagnostics(
        ax,
        dt,
        hp_corner_hz=hp_corner_hz,
        hp_order=hp_order,
        taper_fraction=taper_fraction,
        baseline_window_fraction=baseline_window_fraction,
    )
    vy, dy = accel_to_velocity_with_diagnostics(
        ay,
        dt,
        hp_corner_hz=hp_corner_hz,
        hp_order=hp_order,
        taper_fraction=taper_fraction,
        baseline_window_fraction=baseline_window_fraction,
    )

    pga_x = float(np.max(np.abs(ax)))
    pga_y = float(np.max(np.abs(ay)))
    pgv_x = float(np.max(np.abs(vx)))
    pgv_y = float(np.max(np.abs(vy)))

    pga_rotd = rotd_spectrum(ax, ay, n_angles=rot_n_angles)
    pgv_rotd = rotd_spectrum(vx, vy, n_angles=rot_n_angles)

    pga_vec = float(np.max(np.sqrt(ax**2 + ay**2)))
    pgv_vec = float(np.max(np.sqrt(vx**2 + vy**2)))

    source = pair.metadata.get("source") or pair.metadata.get("source_x")
    input_accel_unit = pair.metadata.get("input_accel_unit", "mps2")

    row: dict[str, float | str | int | None] = {
        "pair_id": pair.pair_id,
        "source": source,
        "input_accel_unit": input_accel_unit,
        "npts": pair.npts,
        "dt_s": dt,
        "pga_unit": pga_unit,
        "pgv_unit": pgv_unit,
        "pga_x": _scale_pga(pga_x, pga_unit),
        "pga_y": _scale_pga(pga_y, pga_unit),
        "pgv_x": _scale_pgv(pgv_x, pgv_unit),
        "pgv_y": _scale_pgv(pgv_y, pgv_unit),
        "pga_2d_max": _scale_pga(max(pga_x, pga_y), pga_unit),
        "pga_2d_geom_mean": _scale_pga(float(np.sqrt(pga_x * pga_y)), pga_unit),
        "pga_2d_vector_peak": _scale_pga(pga_vec, pga_unit),
        "pgv_2d_max": _scale_pgv(max(pgv_x, pgv_y), pgv_unit),
        "pgv_2d_geom_mean": _scale_pgv(float(np.sqrt(pgv_x * pgv_y)), pgv_unit),
        "pgv_2d_vector_peak": _scale_pgv(pgv_vec, pgv_unit),
        "pga_rotd50": _scale_pga(pga_rotd["RotD50"], pga_unit),
        "pga_rotd100": _scale_pga(pga_rotd["RotD100"], pga_unit),
        "pgv_rotd50": _scale_pgv(pgv_rotd["RotD50"], pgv_unit),
        "pgv_rotd100": _scale_pgv(pgv_rotd["RotD100"], pgv_unit),
        "diag_x_nan_pct": dx.nan_pct_input,
        "diag_y_nan_pct": dy.nan_pct_input,
        "diag_x_clipping_pct": dx.clipping_pct_input,
        "diag_y_clipping_pct": dy.clipping_pct_input,
        "diag_x_clipping_flag": dx.clipping_flag,
        "diag_y_clipping_flag": dy.clipping_flag,
        "diag_x_v_drift_end_start_mps": dx.drift_end_start_mps,
        "diag_y_v_drift_end_start_mps": dy.drift_end_start_mps,
        "diag_x_v_drift_slope_mps2": dx.drift_slope_mps2,
        "diag_y_v_drift_slope_mps2": dy.drift_slope_mps2,
    }
    return row


def pairs_to_dataframe(
    pairs: Iterable[RecordPair],
    *,
    pga_unit: str = "g",
    pgv_unit: str = "cms",
    rot_n_angles: int = 180,
    hp_corner_hz: float = 0.1,
    hp_order: int = 4,
    taper_fraction: float = 0.0,
    baseline_window_fraction: float = 0.05,
) -> pd.DataFrame:
    rows = [
        pair_ims_to_row(
            p,
            pga_unit=pga_unit,
            pgv_unit=pgv_unit,
            rot_n_angles=rot_n_angles,
            hp_corner_hz=hp_corner_hz,
            hp_order=hp_order,
            taper_fraction=taper_fraction,
            baseline_window_fraction=baseline_window_fraction,
        )
        for p in pairs
    ]
    return pd.DataFrame(rows)
