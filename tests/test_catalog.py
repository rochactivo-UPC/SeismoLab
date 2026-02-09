from pathlib import Path

import pandas as pd

from SeismoLab.catalog.discovery import discover_pairs_in_folder


def test_discover_pairs(tmp_path: Path) -> None:
    pd.DataFrame({"t": [0.0, 0.1], "ax": [0.0, 1.0]}).to_csv(tmp_path / "ev_x.csv", index=False)
    pd.DataFrame({"t": [0.0, 0.1], "ay": [0.0, -1.0]}).to_csv(tmp_path / "ev_y.csv", index=False)

    pairs = discover_pairs_in_folder(tmp_path)
    assert len(pairs) == 1
    assert pairs[0].pair_id == "ev"
