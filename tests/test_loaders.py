from pathlib import Path

import numpy as np
import pandas as pd

from SeismoLab.io.loaders import load_csv_columns, load_csv_pair, load_csv_simple


def test_load_csv_simple(tmp_path: Path) -> None:
    p = tmp_path / "pair.csv"
    pd.DataFrame({"t": [0.0, 0.1], "ax": [1.0, 2.0], "ay": [3.0, 4.0]}).to_csv(p, index=False)

    pair = load_csv_simple(p)
    assert pair.pair_id == "pair"
    assert np.isclose(pair.x.accel.max(), 2.0)


def test_load_csv_pair(tmp_path: Path) -> None:
    px = tmp_path / "ev_x.csv"
    py = tmp_path / "ev_y.csv"
    pd.DataFrame({"t": [0.0, 0.1], "ax": [0.0, 1.0]}).to_csv(px, index=False)
    pd.DataFrame({"t": [0.0, 0.1], "ay": [0.0, -1.0]}).to_csv(py, index=False)

    pair = load_csv_pair(px, py)
    assert pair.pair_id == "ev"


def test_load_csv_columns(tmp_path: Path) -> None:
    p = tmp_path / "custom.csv"
    pd.DataFrame({"time_s": [0.0, 0.1], "east": [1.0, 2.0], "north": [2.0, 1.0]}).to_csv(
        p, index=False
    )

    pair = load_csv_columns(p, time_col="time_s", x_col="east", y_col="north")
    assert pair.npts == 2


def test_load_csv_simple_with_gal_input_unit(tmp_path: Path) -> None:
    p = tmp_path / "pair_gal.csv"
    pd.DataFrame({"t": [0.0, 0.1], "ax": [100.0, -100.0], "ay": [50.0, -50.0]}).to_csv(
        p, index=False
    )

    pair = load_csv_simple(p, accel_unit="gal")
    assert np.isclose(pair.x.accel[0], 1.0)
    assert np.isclose(pair.y.accel[0], 0.5)
