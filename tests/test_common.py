from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from common import fit_trace, per_series, pick_examples, summarize

CUT = pd.Timestamp("2016-01-02")


def _frames(y, yhat, uid="a"):
    ds = pd.date_range(CUT + pd.Timedelta(weeks=1), periods=len(y), freq="W-SAT")
    actual = pd.DataFrame({"unique_id": uid, "ds": ds, "y": y, "cutoff": CUT})
    fc = pd.DataFrame({"unique_id": uid, "ds": ds, "yhat": yhat, "cutoff": CUT})
    return fc, actual


def test_per_series_constant_actual_gives_nan():
    fc1, a1 = _frames([0.0, 0.0, 0.0], [1.0, 2.0, 3.0], "zero")
    fc2, a2 = _frames([1.0, 5.0, 2.0], [1.0, 4.0, 2.0], "live")
    ps = per_series(pd.concat([fc1, fc2]), pd.concat([a1, a2]))
    z = ps.set_index("unique_id").loc["zero"]
    assert np.isnan(z["corr"]) and np.isnan(z["amp_ratio"])
    s = summarize(ps)
    assert np.isfinite(s["wape"])
    assert np.isfinite(s["corr_median"])


def test_per_series_missing_forecast_scored_as_zero():
    fc, a = _frames([3.0, 4.0], [1.0, 1.0])
    ps = per_series(fc.iloc[:0], a)
    row = ps.iloc[0]
    assert row["yhat_sum"] == 0.0
    assert row["abs_err"] == row["y_sum"] == 7.0


def test_summarize_values():
    fc, a = _frames([10.0, 10.0], [5.0, 15.0])
    s = summarize(per_series(fc, a))
    assert s["wape"] == pytest.approx(50.0)
    assert s["bias"] == pytest.approx(0.0)


def test_fit_trace_drops_step0_keeps_nan():
    model = SimpleNamespace(train_trajectories=[(0, 2.0), (1, 1.5), (2, 1.0)],
                            valid_trajectories=[(0, 9.0), (2, 1.7)])
    w = SimpleNamespace(_gn=SimpleNamespace(norms=[1.0, float("nan"), 3.0], thresholds=[1.0, 1.0]),
                        nf=SimpleNamespace(models=[model]),
                        _lr=SimpleNamespace(lr=[1e-4, 2e-4, 3e-4]))
    tr = fit_trace(w)
    assert list(tr["step"]) == [1, 2, 3]
    assert np.isnan(tr.loc[tr.step == 2, "grad_norm"].iloc[0])
    assert tr["val_loss"].notna().tolist() == [False, True, False]
    assert tr.loc[tr.step == 2, "val_loss"].iloc[0] == pytest.approx(1.7)
    assert tr["lr"].tolist() == pytest.approx([1e-4, 2e-4, 3e-4])
    assert tr["train_loss"].tolist() == pytest.approx([2.0, 1.5, 1.0])


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
    from common import load_panel
    a, b = load_panel(5000), load_panel(5000)
    assert set(a.unique_id) == set(b.unique_id)
    assert a.unique_id.nunique() == 5000


def test_clip_thr_aligned_after_nonfinite():
    model = SimpleNamespace(train_trajectories=[(0, 2.0), (1, 1.5), (2, 1.0)], valid_trajectories=[])
    w = SimpleNamespace(_gn=SimpleNamespace(norms=[1.0, float("nan"), 3.0], thresholds=[1.0, 1.5]),
                        nf=SimpleNamespace(models=[model]), _lr=SimpleNamespace(lr=[1e-4] * 3))
    thr = fit_trace(w)["clip_thr"].tolist()
    assert thr[0] == 1.0 and np.isnan(thr[1]) and thr[2] == 1.5


def test_flat_forecast_with_varying_actuals_has_zero_corr():
    fc, a = _frames([1.0, 5.0, 2.0], [0.0, 0.0, 0.0])
    row = per_series(fc, a).iloc[0]
    assert row["corr"] == 0.0 and row["amp_ratio"] == 0.0


def test_pick_seasonal_finds_seasonal_series():
    from common import pick_seasonal
    rng = np.random.default_rng(1)
    ds = pd.date_range(end=CUT, periods=200, freq="W-SAT")
    rows = []
    for i in range(20):
        base = rng.poisson(20, len(ds)).astype(float)
        if i == 7:
            base = 20 + 15 * np.sin(2 * np.pi * np.arange(len(ds)) / 52) + rng.normal(0, 1, len(ds))
        rows.append(pd.DataFrame({"unique_id": f"s{i}", "ds": ds, "y": base}))
    p = pd.concat(rows, ignore_index=True)
    a = pick_seasonal(p, CUT, exclude=["s0"])
    assert a == pick_seasonal(p, CUT, exclude=["s0"])
    assert a == "s7"
