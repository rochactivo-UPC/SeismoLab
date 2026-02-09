from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.integrate import cumulative_trapezoid


@dataclass(slots=True)
class PGVDiagnostics:
    """Diagnosticos basicos del pipeline de preproceso para PGV."""

    nan_pct_input: float
    clipping_pct_input: float
    clipping_flag: bool
    drift_end_start_mps: float
    drift_slope_mps2: float


def _fill_nans_linear(a: np.ndarray) -> tuple[np.ndarray, float]:
    nan_mask = np.isnan(a)
    nan_pct = float(100.0 * np.mean(nan_mask))
    if not nan_mask.any():
        return a, nan_pct
    if nan_mask.all():
        msg = "La aceleracion contiene solo NaNs"
        raise ValueError(msg)
    idx = np.arange(a.size)
    out = a.copy()
    out[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], a[~nan_mask])
    return out, nan_pct


def _simple_clipping_pct(a: np.ndarray, *, rel_tol: float = 0.005) -> float:
    peak = float(np.max(np.abs(a)))
    if peak <= 0.0:
        return 0.0
    near_peak = np.abs(np.abs(a) - peak) <= (rel_tol * peak)
    return float(100.0 * np.mean(near_peak))


def _minimal_baseline_correction(a: np.ndarray, dt: float, *, window_fraction: float = 0.05) -> np.ndarray:
    n = len(a)
    nwin = max(1, int(round(n * window_fraction)))
    start_mean = float(np.mean(a[:nwin]))
    end_mean = float(np.mean(a[-nwin:]))
    t = np.arange(n, dtype=float) * dt
    baseline = np.linspace(start_mean, end_mean, n)
    return a - baseline


def accel_to_velocity_with_diagnostics(
    accel: np.ndarray,
    dt: float,
    *,
    hp_corner_hz: float = 0.1,
    hp_order: int = 4,
    taper_fraction: float = 0.0,
    baseline_window_fraction: float = 0.05,
    clipping_pct_threshold: float = 0.5,
) -> tuple[np.ndarray, PGVDiagnostics]:
    """Preproceso minimo para PGV + integracion, devolviendo diagnosticos."""
    a = np.asarray(accel, dtype=float)
    if a.ndim != 1:
        msg = "accel debe ser un vector 1D"
        raise ValueError(msg)
    if dt <= 0.0:
        msg = "dt debe ser positivo"
        raise ValueError(msg)
    if not 0.0 <= taper_fraction <= 1.0:
        msg = "taper_fraction debe estar entre 0 y 1"
        raise ValueError(msg)
    if not 0.0 < baseline_window_fraction <= 0.5:
        msg = "baseline_window_fraction debe estar en (0, 0.5]"
        raise ValueError(msg)

    a, nan_pct = _fill_nans_linear(a)
    clip_pct = _simple_clipping_pct(a)

    # Detrend por media y tendencia lineal.
    detrended = a - float(np.mean(a))
    detrended = signal.detrend(detrended, type="linear")

    # Baseline minima: linea entre medias de inicio/fin.
    corrected = _minimal_baseline_correction(
        detrended, dt, window_fraction=baseline_window_fraction
    )

    if taper_fraction > 0.0:
        corrected = corrected * signal.windows.tukey(len(corrected), alpha=taper_fraction)

    if hp_corner_hz > 0.0:
        nyq = 0.5 / dt
        wn = hp_corner_hz / nyq
        if not 0.0 < wn < 1.0:
            msg = "hp_corner_hz fuera de rango para el dt dado"
            raise ValueError(msg)
        b, aa = signal.butter(hp_order, wn, btype="highpass")
        corrected = signal.filtfilt(b, aa, corrected)

    vel = cumulative_trapezoid(corrected, dx=dt, initial=0.0)
    vel = signal.detrend(vel, type="linear")

    t = np.arange(len(vel), dtype=float) * dt
    slope, _intercept = np.polyfit(t, vel, 1)
    diag = PGVDiagnostics(
        nan_pct_input=nan_pct,
        clipping_pct_input=clip_pct,
        clipping_flag=clip_pct >= clipping_pct_threshold,
        drift_end_start_mps=float(vel[-1] - vel[0]),
        drift_slope_mps2=float(slope),
    )
    return vel, diag


def accel_to_velocity(
    accel: np.ndarray,
    dt: float,
    *,
    hp_corner_hz: float = 0.1,
    hp_order: int = 4,
    taper_fraction: float = 0.0,
    baseline_window_fraction: float = 0.05,
) -> np.ndarray:
    """Atajo compatible: devuelve solo v(t) del pipeline de PGV."""
    vel, _diag = accel_to_velocity_with_diagnostics(
        accel,
        dt,
        hp_corner_hz=hp_corner_hz,
        hp_order=hp_order,
        taper_fraction=taper_fraction,
        baseline_window_fraction=baseline_window_fraction,
    )
    return vel
