from __future__ import annotations

import numpy as np


def rotd_spectrum(x: np.ndarray, y: np.ndarray, *, n_angles: int = 180) -> dict[str, float]:
    """Calcula RotD50 y RotD100 de la serie pico-abs por rotaciones 0-180."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        msg = "x e y deben tener la misma forma"
        raise ValueError(msg)

    angles = np.linspace(0.0, np.pi, num=n_angles, endpoint=False)
    peaks = np.empty_like(angles)
    for i, th in enumerate(angles):
        rot = x * np.cos(th) + y * np.sin(th)
        peaks[i] = float(np.max(np.abs(rot)))

    return {
        "RotD50": float(np.percentile(peaks, 50)),
        "RotD100": float(np.max(peaks)),
    }
