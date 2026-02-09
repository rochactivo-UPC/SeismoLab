import numpy as np
from scipy.integrate import cumulative_trapezoid

from SeismoLab.ims.metrics import pair_ims_to_row
from SeismoLab.models import Record, RecordPair
from SeismoLab.preprocess.signal import accel_to_velocity_with_diagnostics


def test_pair_ims_keys() -> None:
    t = np.arange(0.0, 2.0, 0.01)
    ax = np.sin(2 * np.pi * 1.0 * t)
    ay = np.cos(2 * np.pi * 1.0 * t)
    pair = RecordPair(
        "sine",
        Record("sine_x", t, ax, "x"),
        Record("sine_y", t, ay, "y"),
    )
    row = pair_ims_to_row(pair)
    for key in [
        "pga_x",
        "pgv_x",
        "pga_rotd50",
        "pgv_rotd100",
        "pga_2d_vector_peak",
        "diag_x_nan_pct",
        "diag_x_clipping_flag",
        "diag_x_v_drift_end_start_mps",
    ]:
        assert key in row
    assert row["pga_x"] > 0.0


def test_sine_pga_pgv_known_values() -> None:
    dt = 0.005
    t = np.arange(0.0, 20.0, dt)
    freq_hz = 1.0
    amp = 2.0  # m/s2
    ax = amp * np.sin(2 * np.pi * freq_hz * t)
    ay = 0.5 * ax

    pair = RecordPair(
        "sine_known",
        Record("sine_known_x", t, ax, "x"),
        Record("sine_known_y", t, ay, "y"),
    )
    row = pair_ims_to_row(
        pair,
        pga_unit="mps2",
        pgv_unit="mps",
        hp_corner_hz=0.0,
        taper_fraction=0.0,
    )

    expected_pga_x = amp
    expected_pgv_x = amp / (2 * np.pi * freq_hz)
    assert np.isclose(row["pga_x"], expected_pga_x, rtol=0.01)
    assert np.isclose(row["pgv_x"], expected_pgv_x, rtol=0.05)


def test_offset_signal_preprocessing_reduces_velocity_drift() -> None:
    dt = 0.01
    t = np.arange(0.0, 40.0, dt)
    acc = 0.2 * np.sin(2 * np.pi * 0.8 * t) + 0.05 + 0.0005 * t

    raw_vel = cumulative_trapezoid(acc, dx=dt, initial=0.0)
    raw_drift = abs(float(raw_vel[-1] - raw_vel[0]))

    vel, diag = accel_to_velocity_with_diagnostics(
        acc,
        dt,
        hp_corner_hz=0.1,
        taper_fraction=0.1,
    )
    proc_drift = abs(diag.drift_end_start_mps)

    assert len(vel) == len(acc)
    assert diag.nan_pct_input == 0.0
    assert proc_drift < raw_drift * 0.2


def test_diagnostics_capture_nans_and_simple_clipping() -> None:
    dt = 0.01
    acc = np.zeros(1000)
    acc[100:130] = 5.0
    acc[500] = np.nan

    _vel, diag = accel_to_velocity_with_diagnostics(acc, dt, hp_corner_hz=0.1)
    assert diag.nan_pct_input > 0.0
    assert diag.clipping_pct_input > 0.0
    assert diag.clipping_flag
