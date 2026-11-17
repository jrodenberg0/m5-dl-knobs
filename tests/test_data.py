import numpy as np
import pandas as pd
import pytest

from m5knobs.data import pick_examples, weekly_from_daily, write_history

CUT = pd.Timestamp("2016-01-02")


def _panel():
    rng = np.random.default_rng(0)
    ds = pd.date_range(end=CUT, periods=60, freq="W-SAT")
    rows = []
    for i in range(30):
        zero_share = [0.0, 0.1, 0.6, 0.7, 0.95][i % 5]
        y = rng.poisson(5 + i, len(ds)).astype(float)
        y[rng.random(len(ds)) < zero_share] = 0.0
        rows.append(pd.DataFrame({"unique_id": f"s{i}", "ds": ds, "y": y}))
    return pd.concat(rows, ignore_index=True)


def test_pick_examples_deterministic():
    p = _panel()
    a, b = pick_examples(p, CUT), pick_examples(p, CUT)
    assert a == b and len(a) == 2 and a[0] != b[1]
    last = p[p.ds > CUT - pd.Timedelta(weeks=52)]
    zs = last.groupby("unique_id")["y"].apply(lambda s: (s == 0).mean())
    assert zs[a[0]] < 0.2
    assert 0.5 <= zs[a[1]] <= 0.8


@pytest.mark.slow
def test_load_panel_fixed():
    from m5knobs.data import load_panel
    a, b = load_panel(5000), load_panel(5000)
    assert set(a.unique_id) == set(b.unique_id)
    assert a.unique_id.nunique() == 5000


def test_weekly_from_daily_wsat_sums_trims_and_keeps_partial():
    ds = pd.date_range("2016-01-01", "2016-01-24", freq="D")  # Fri .. Sun
    a = pd.DataFrame({"unique_id": "a", "ds": ds, "y": 1.0})
    b = pd.DataFrame({"unique_id": "b", "ds": ds, "y": [0.0] * 10 + [2.0] * (len(ds) - 10)})
    w = weekly_from_daily(pd.concat([a, b]))
    wa = w[w.unique_id == "a"].set_index("ds")["y"]
    assert list(wa.index) == list(pd.to_datetime(["2016-01-02", "2016-01-09", "2016-01-16", "2016-01-23", "2016-01-30"]))
    assert list(wa) == [2.0, 7.0, 7.0, 7.0, 1.0]  # Fri+Sat, full weeks, final partial (Sun only)
    wb = w[w.unique_id == "b"]
    assert wb["y"].iloc[0] > 0 and wb.ds.min() == pd.Timestamp("2016-01-16")  # leading zero weeks dropped
    assert list(w.columns) == ["unique_id", "ds", "y"]


def test_write_history_window(tmp_path):
    ds = pd.date_range(end=CUT + pd.Timedelta(weeks=10), periods=80, freq="W-SAT")
    p = pd.concat([pd.DataFrame({"unique_id": u, "ds": ds, "y": 1.0}) for u in ("b", "a")], ignore_index=True)
    out = write_history(p, CUT, str(tmp_path / "history.parquet"))
    h = pd.read_parquet(out)
    assert list(h.columns) == ["unique_id", "ds", "y"] and h["y"].dtype == "float32"
    assert h["ds"].max() == CUT and (h["ds"] <= CUT).all()
    assert h.groupby("unique_id")["ds"].nunique().eq(60).all()
    assert h["ds"].min() == CUT - pd.Timedelta(weeks=59)
    assert h.equals(h.sort_values(["unique_id", "ds"]).reset_index(drop=True))
