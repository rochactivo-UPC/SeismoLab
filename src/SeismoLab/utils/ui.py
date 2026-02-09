from __future__ import annotations

from pathlib import Path


def select_files_dialog() -> list[Path]:
    """Abre ventana para seleccionar archivos de entrada."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    names = filedialog.askopenfilenames(
        title="Selecciona archivos CSV/MAT",
        filetypes=[("Acelerogramas", "*.csv *.mat"), ("Todos", "*.*")],
    )
    root.destroy()
    return [Path(p) for p in names]


def select_folder_dialog() -> Path | None:
    """Abre ventana para seleccionar carpeta de catalogo."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Selecciona carpeta de catalogo")
    root.destroy()
    return Path(folder) if folder else None
