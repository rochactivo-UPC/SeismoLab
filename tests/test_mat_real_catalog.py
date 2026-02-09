from pathlib import Path
import os

import pytest

from SeismoLab.catalog.discovery import discover_pairs_in_folder
from SeismoLab.ims.metrics import pair_ims_to_row
from SeismoLab.io.loaders import load_mat_pair
from SeismoLab.preprocess.signal import accel_to_velocity_with_diagnostics

MAT_CATALOG = Path(r"C:\Users\rocha\Documents\ETABSlab\data\mat")
DEFAULT_TARGET_MAT = MAT_CATALOG / "199803281115AFIR.mat"


def _mat_files() -> list[Path]:
    if not MAT_CATALOG.exists():
        return []
    return sorted(MAT_CATALOG.glob("*.mat"))


@pytest.mark.skipif(not MAT_CATALOG.exists(), reason="Catalogo MAT no disponible en esta maquina")
def test_load_real_mat_file() -> None:
    mats = _mat_files()
    if not mats:
        pytest.skip("No hay archivos .mat en el catalogo real")

    pair = load_mat_pair(mats[0])
    assert pair.npts > 10
    assert pair.dt > 0.0
    assert pair.x.component == "x"
    assert pair.y.component == "y"


@pytest.mark.skipif(not MAT_CATALOG.exists(), reason="Catalogo MAT no disponible en esta maquina")
def test_discover_and_compute_ims_from_real_mat_catalog() -> None:
    pairs = discover_pairs_in_folder(MAT_CATALOG)
    if not pairs:
        pytest.skip("No se pudieron descubrir pares MAT en el catalogo")

    row = pair_ims_to_row(pairs[0])
    for key in ["pair_id", "pga_x", "pga_y", "pgv_x", "pgv_y", "pga_rotd50", "pgv_rotd100"]:
        assert key in row

    summary_keys = [
        "pair_id",
        "dt_s",
        "npts",
        "pga_x",
        "pga_y",
        "pga_rotd50",
        "pga_rotd100",
        "pgv_x",
        "pgv_y",
        "pgv_rotd50",
        "pgv_rotd100",
    ]
    print("\nIM summary (first discovered pair):")
    for k in summary_keys:
        print(f"  {k}: {row.get(k)}")


@pytest.mark.skipif(not MAT_CATALOG.exists(), reason="Catalogo MAT no disponible en esta maquina")
def test_single_mat_full_compute_and_plot() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target = Path(os.getenv("SEISMOLAB_TEST_MAT_FILE", str(DEFAULT_TARGET_MAT)))
    if not target.exists():
        pytest.skip(f"Archivo objetivo no existe: {target}")

    pair = load_mat_pair(target, accel_unit="gal")
    row = pair_ims_to_row(pair, pga_unit="g", pgv_unit="cms", hp_corner_hz=0.1, taper_fraction=0.05)

    vx, dx = accel_to_velocity_with_diagnostics(
        pair.x.accel,
        pair.dt,
        hp_corner_hz=0.1,
        taper_fraction=0.05,
    )
    vy, dy = accel_to_velocity_with_diagnostics(
        pair.y.accel,
        pair.dt,
        hp_corner_hz=0.1,
        taper_fraction=0.05,
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(pair.time, pair.x.accel, label="acc_x (m/s2)", linewidth=1.0)
    axes[0].plot(pair.time, pair.y.accel, label="acc_y (m/s2)", linewidth=1.0)
    axes[0].set_ylabel("Aceleracion")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(pair.time, vx, label="vel_x (m/s)", linewidth=1.0)
    axes[1].plot(pair.time, vy, label="vel_y (m/s)", linewidth=1.0)
    axes[1].set_ylabel("Velocidad")
    axes[1].set_xlabel("Tiempo [s]")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    out_dir = Path(os.getenv("SEISMOLAB_PLOT_DIR", "tests/_artifacts"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{pair.pair_id}_signals.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    print(f"\nArchivo MAT analizado: {target}")
    print(f"Grafica guardada en: {out_png}")
    print("\nParametros IM/diagnosticos calculados:")
    for key in sorted(row):
        print(f"  {key}: {row[key]}")

    print("\nDiagnosticos directos preprocess (por componente):")
    print(f"  x.nan_pct={dx.nan_pct_input}, x.clip_pct={dx.clipping_pct_input}, x.drift={dx.drift_end_start_mps}")
    print(f"  y.nan_pct={dy.nan_pct_input}, y.clip_pct={dy.clipping_pct_input}, y.drift={dy.drift_end_start_mps}")

    assert out_png.exists()
    assert row["pga_x"] is not None
    assert row["pgv_x"] is not None
