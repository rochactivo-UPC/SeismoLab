from pathlib import Path

from SeismoLab.catalog.discovery import discover_pairs_in_folder
from SeismoLab.ims.metrics import pairs_to_dataframe

pairs = discover_pairs_in_folder(Path("data/catalogo"), recursive=True)
df = pairs_to_dataframe(pairs, pga_unit="g", pgv_unit="cms")
print(df.head())
