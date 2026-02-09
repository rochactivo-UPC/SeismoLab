from __future__ import annotations

import argparse
from pathlib import Path

from SeismoLab.catalog.discovery import discover_pairs_from_paths, discover_pairs_in_folder
from SeismoLab.ims.metrics import pairs_to_dataframe
from SeismoLab.utils.ui import select_files_dialog, select_folder_dialog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="imcalc",
        description="Calcula IMs 1D/2D de pares ortogonales y exporta parquet.",
    )
    parser.add_argument("catalog_folder", nargs="?", help="Carpeta del catalogo")
    parser.add_argument("--out", default="results.parquet", help="Salida parquet")
    parser.add_argument("--pga-unit", choices=["g", "mps2"], default="g")
    parser.add_argument("--pgv-unit", choices=["cms", "mps"], default="cms")
    parser.add_argument("--input-accel-unit", choices=["mps2", "gal"], default="mps2")
    parser.add_argument("--hp-corner-hz", type=float, default=0.1)
    parser.add_argument("--hp-order", type=int, default=4)
    parser.add_argument("--taper-fraction", type=float, default=0.0)
    parser.add_argument("--baseline-window-fraction", type=float, default=0.05)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--rot-angles", type=int, default=180)
    parser.add_argument("--use-gui-files", action="store_true")
    parser.add_argument("--use-gui-folder", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.use_gui_files:
        files = select_files_dialog()
        pairs = discover_pairs_from_paths(files, input_accel_unit=args.input_accel_unit)
    else:
        folder = Path(args.catalog_folder) if args.catalog_folder else None
        if args.use_gui_folder:
            folder = select_folder_dialog()
        if folder is None:
            parser.error("Debes indicar <carpeta_catalogo> o usar --use-gui-folder/--use-gui-files")
        pairs = discover_pairs_in_folder(
            folder,
            recursive=args.recursive,
            input_accel_unit=args.input_accel_unit,
        )

    if not pairs:
        parser.error("No se encontraron pares validos en la entrada")

    df = pairs_to_dataframe(
        pairs,
        pga_unit=args.pga_unit,
        pgv_unit=args.pgv_unit,
        rot_n_angles=args.rot_angles,
        hp_corner_hz=args.hp_corner_hz,
        hp_order=args.hp_order,
        taper_fraction=args.taper_fraction,
        baseline_window_fraction=args.baseline_window_fraction,
    )
    out = Path(args.out)
    df.to_parquet(out, index=False)
    print(f"Generado: {out} ({len(df)} pares)")


if __name__ == "__main__":
    main()
