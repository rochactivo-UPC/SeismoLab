# SeismoLab

MVP para explorar catalogos de pares ortogonales de acelerogramas y calcular IMs para analisis masivos.

## Alcance actual

- Modelo de datos reutilizable: `Record` y `RecordPair`
- Loaders:
  - CSV simple (`t,ax,ay` en un archivo)
  - CSV en dos archivos (`t+ax`, `t+ay`)
  - CSV generico por columnas
  - MAT (`.mat`) con estructura tipo:
    - `acc_f_e`, `acc_f_n`, `dt`, `header` (opcional)
- Preproceso minimo para PGV:
  - detrend (constante + lineal)
  - baseline correction minima (linea entre medias inicio/fin)
  - filtro Butterworth pasa-altas configurable (default 0.1 Hz)
  - taper Tukey opcional
  - integracion por trapecios
- IMs:
  - 1D: PGA/PGV por componente
  - 2D no rotacional: max, media geometrica, vectorial instantanea
  - 2D rotacional: RotD50 y RotD100 para PGA y PGV
- CLI:
  - `imcalc <carpeta_catalogo> --out results.parquet`

## Instalacion

```bash
pip install -e .
pip install -e .[dev]
pre-commit install
```

## Uso CLI

```bash
imcalc data/catalogo --out results.parquet
```

Opciones principales:

- `--input-accel-unit mps2|gal` (default `mps2`)
- `--pga-unit g|mps2` (default `g`)
- `--pgv-unit cms|mps` (default `cms`)
- `--recursive` para busqueda recursiva
- `--use-gui-files` abrir ventana para seleccionar archivos
- `--use-gui-folder` abrir ventana para seleccionar carpeta
- `--hp-corner-hz` frecuencia de corte pasa-altas (default `0.1`)
- `--hp-order` orden del filtro Butterworth (default `4`)
- `--taper-fraction` fraccion Tukey en `[0,1]` (default `0.0`)
- `--baseline-window-fraction` ventana relativa para baseline minima (default `0.05`)

Si tus acelerogramas estan en `gal`, usa:

```bash
imcalc data/catalogo --input-accel-unit gal --out results.parquet
```

## Ejemplo en Python

```python
from pathlib import Path
from SeismoLab.catalog.discovery import discover_pairs_in_folder
from SeismoLab.ims.metrics import pair_ims_to_row

pairs = discover_pairs_in_folder(Path("data/catalogo"))
row = pair_ims_to_row(pairs[0], pga_unit="g", pgv_unit="cms")
print(row)
```

## Estructura

```text
src/SeismoLab/
  io/
  preprocess/
  ims/
  rotations/
  catalog/
  cli/
  utils/
tests/
```

## Supuestos y diagnosticos

- Unidades internas de aceleracion/velocidad: `m/s2` y `m/s`.
- Si entrada esta en `gal`, se convierte internamente con `1 gal = 0.01 m/s2`.
- PGA se calcula como `max(abs(a))` por componente.
- PGV se calcula desde velocidad integrada tras preproceso minimo.
- NaNs: se rellenan por interpolacion lineal antes del filtro.
- Clipping simple: porcentaje de muestras cercanas al pico absoluto.
- Diagnosticos exportados por componente:
  - `diag_*_nan_pct`
  - `diag_*_clipping_pct`
  - `diag_*_clipping_flag`
  - `diag_*_v_drift_end_start_mps`
  - `diag_*_v_drift_slope_mps2`
