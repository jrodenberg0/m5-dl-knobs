import pandas as pd

from prepare_data import weekly_from_daily


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
