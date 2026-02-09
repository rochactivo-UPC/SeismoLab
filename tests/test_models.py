import numpy as np

from SeismoLab.models import Record, RecordPair


def test_record_pair_validates_time_alignment() -> None:
    t = np.array([0.0, 0.1, 0.2])
    rx = Record("id_x", t, np.array([0.0, 1.0, 0.0]), "x")
    ry = Record("id_y", t, np.array([0.0, -1.0, 0.0]), "y")
    pair = RecordPair("id", rx, ry)
    assert pair.npts == 3
    assert pair.dt == 0.1
