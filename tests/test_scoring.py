from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from m5knobs.scoring import fit_trace, is_hurt, label_row, paired_bootstrap_p, per_series, row_verdict, summarize

CUT = pd.Timestamp("2016-01-02")


def _wape(df):
    return df["abs_err"].sum() / df["y_sum"].sum()


def _ps(n=500, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.gamma(2.0, 10.0, n)
    return pd.DataFrame({"unique_id": [f"s{i}" for i in range(n)], "abs_err": rng.gamma(2.0, 3.0, n) + shift, "y_sum": y})


def test_is_hurt_needs_both():
    assert is_hurt(3.0, 0.0, 0.01, higher_is_worse=True)
    assert not is_hurt(3.0, 0.0, 0.2, higher_is_worse=True)
    assert not is_hurt(0.0, 3.0, 0.001, higher_is_worse=True)


def test_is_hurt_direction():
    assert is_hurt(0.2, 0.5, 0.01, higher_is_worse=False)
    assert not is_hurt(0.5, 0.2, 0.01, higher_is_worse=False)


def test_bootstrap_identical_is_not_significant():
    a = _ps()
    assert paired_bootstrap_p(_wape, a, a.copy()) >= 0.5


def test_bootstrap_clear_shift_significant():
    b = _ps()
    a = b.copy()
    a["abs_err"] += 5.0
    assert paired_bootstrap_p(_wape, a, b) < 0.01


def test_bootstrap_aligns_on_unique_id():
    b = _ps()
    a = b.copy()
    a["abs_err"] += 5.0
    assert paired_bootstrap_p(_wape, a.sample(frac=1, random_state=1), b) < 0.01


@pytest.mark.parametrize("hurt,label", [
    (dict(insample_mae=False, corr=False, amp_ratio=False, abs_bias=False), "no clear effect on M5"),
    (dict(insample_mae=True, corr=False, amp_ratio=False, abs_bias=False), "underfit"),
    (dict(insample_mae=False, corr=False, amp_ratio=True, abs_bias=False), "underfit"),
    (dict(insample_mae=False, corr=False, amp_ratio=False, abs_bias=True), "biased"),
    (dict(insample_mae=False, corr=True, amp_ratio=False, abs_bias=True), "both"),
])
def test_label_row(hurt, label):
    assert label_row(hurt) == label


def _run(n=400, err_shift=0.0, flat=False, bias=0.0, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"s{i}" for i in range(n)]
    y = rng.gamma(2.0, 10.0, n)
    ps = pd.DataFrame({"unique_id": ids, "cutoff": pd.Timestamp("2016-01-02"), "abs_err": rng.gamma(2, 3, n) + err_shift, "y_sum": y,
                       "yhat_sum": y * (1 + bias) + rng.normal(0, 1, n),
                       "corr": rng.uniform(0.0, 0.3, n) if flat else rng.uniform(0.5, 0.8, n),
                       "amp_ratio": rng.uniform(0.1, 0.3, n) if flat else rng.uniform(0.7, 1.0, n)})
    ins = pd.DataFrame({"unique_id": ids, "abs_err": rng.gamma(2, 3, n) + err_shift, "y_abs": y})
    return dict(ps=ps, ins=ins)


def test_row_verdict_flat_mistake_is_underfit():
    v = row_verdict(_run(flat=True, err_shift=4.0), _run())
    assert v["label"] == "underfit"
    assert v["insample_mae"]["hurt"] and v["amp_ratio"]["hurt"]
    assert {"insample_mae", "corr", "amp_ratio", "abs_bias", "label"} <= set(v)


def test_row_verdict_biased_mistake():
    v = row_verdict(_run(bias=-0.3), _run())
    assert v["label"] == "biased"


def test_row_verdict_same_is_no_effect():
    v = row_verdict(_run(), _run())
    assert v["label"] == "no clear effect on M5"


def test_effect_floor_blocks_tiny_wape_gap():
    hurt = dict(insample_mae=True, corr=False, amp_ratio=True, abs_bias=False)
    assert label_row(hurt, wape_gap=0.07, abs_bias_gap_pts=0.0) == "no clear effect on M5"
    assert label_row(hurt, wape_gap=0.22, abs_bias_gap_pts=0.0) == "underfit"


def test_effect_floor_blocks_tiny_bias_gap():
    hurt = dict(insample_mae=False, corr=False, amp_ratio=False, abs_bias=True)
    assert label_row(hurt, wape_gap=0.0, abs_bias_gap_pts=0.5) == "no clear effect on M5"
    assert label_row(hurt, wape_gap=0.0, abs_bias_gap_pts=6.8) == "biased"


def test_row_verdict_reports_wape_gap():
    v = row_verdict(_run(flat=True, err_shift=4.0), _run())
    assert "wape_gap" in v and "abs_bias_gap_pts" in v


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
