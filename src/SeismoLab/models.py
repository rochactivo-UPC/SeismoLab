from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class Record:
    """Una componente de aceleracion en unidades internas m/s^2."""

    record_id: str
    time: np.ndarray
    accel: np.ndarray
    component: str
    unit_accel: str = "m/s2"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float)
        self.accel = np.asarray(self.accel, dtype=float)
        if self.time.ndim != 1 or self.accel.ndim != 1:
            msg = "time y accel deben ser arreglos 1D"
            raise ValueError(msg)
        if len(self.time) != len(self.accel):
            msg = "time y accel deben tener la misma longitud"
            raise ValueError(msg)
        if len(self.time) < 2:
            msg = "El registro debe tener al menos 2 muestras"
            raise ValueError(msg)

    @property
    def npts(self) -> int:
        return len(self.time)

    @property
    def dt(self) -> float:
        return float(np.median(np.diff(self.time)))


@dataclass(slots=True)
class RecordPair:
    """Par ortogonal de componentes X/Y o E/N."""

    pair_id: str
    x: Record
    y: Record
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.x.npts != self.y.npts:
            msg = "Las dos componentes deben tener el mismo numero de muestras"
            raise ValueError(msg)
        if not np.allclose(self.x.time, self.y.time):
            msg = "Las dos componentes deben compartir el mismo vector de tiempo"
            raise ValueError(msg)

    @property
    def time(self) -> np.ndarray:
        return self.x.time

    @property
    def dt(self) -> float:
        return self.x.dt

    @property
    def npts(self) -> int:
        return self.x.npts
