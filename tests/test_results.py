import pandas as pd
import pytest

from m5knobs import results
from tests.fixtures.make_fake_results import CUTS, make


def test_history_missing_points_to_force_download(tmp_path):
    with pytest.raises(SystemExit, match="download_results.py --force"):
        results.history(str(tmp_path))


def test_fixture_has_history(tmp_path):
    res = make(str(tmp_path / "results"))
    h = results.history(res)
    assert h["ds"].max() == CUTS[0]
    assert h.groupby("unique_id")["ds"].nunique().eq(60).all()
    assert set(h["unique_id"]) == set(results.actuals(res)["unique_id"])
