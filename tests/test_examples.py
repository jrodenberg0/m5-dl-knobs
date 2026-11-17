import json
import os

import numpy as np
import pandas as pd
import pytest

from m5knobs import examples

CUT = pd.Timestamp("2016-01-02")


def _row(uid, **kw):
    r = dict(unique_id=uid, weeks=10, y=100.0, f_wrong=100.0, f_right=100.0, err_wrong=20.0, err_right=20.0,
             mean_hist=10.0, zero_share_hist=0.0, max_zero_run_hist=0, yearly_repeat=0.0, amp_wrong=1.0, amp_right=1.0,
             shape_gain=0.0, zero_share_fc=0.0, max_zero_run_fc=0, zeros_y52=0, corr52_wrong=0.0, corr52_right=0.0,
             origins=3, origins_wrong_below_right=0, origins_wrong_below_y=0, low_weeks_hist=0, low_weeks_fc=0, n_hist=60)
    r.update(kw)
    return r


def _table(rows):
    t = pd.DataFrame(rows)
    t["gain"] = t["err_wrong"] - t["err_right"]
    t["rel_change"] = (t["f_wrong"] - t["f_right"]).abs() / np.maximum((t["f_wrong"] + t["f_right"]) / 2, 1)
    return t


def _filler(n=6, **kw):
    return [_row(f"z{i}", **kw) for i in range(n)]


# ---------- pair_table ----------

def _write_arm(res, arm, fc):
    d = os.path.join(res, "runs", f"{arm}__s1")
    os.makedirs(d, exist_ok=True)
    fc.to_parquet(os.path.join(d, "forecasts.parquet"))
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump({}, f)


def _pair_results(tmp_path):
    res = str(tmp_path / "results")
    os.makedirs(os.path.join(res, "runs"))
    fds = pd.date_range(CUT + pd.Timedelta(weeks=1), periods=13, freq="W-SAT")
    ya = np.array([0, 2, 4, 6, 1, 3, 5, 0, 0, 2, 9, 1, 1], float)
    yd = np.arange(13, dtype=float)
    ids = ["a"] * 13 + ["b"] * 13 + ["c"] * 13 + ["d"] * 13
    act = pd.DataFrame({"unique_id": ids, "ds": list(fds) * 4,
                        "y": np.r_[ya, np.ones(13), np.full(13, 3.0), yd], "cutoff": CUT})
    act.to_parquet(os.path.join(res, "actuals.parquet"))
    # a: 60 weeks, the forecast weeks a year earlier sold 2*ya + 1 (acf52 = 1); d: 52 weeks (acf52 NaN); b: none
    hds = pd.date_range(end=CUT, periods=60, freq="W-SAT")
    ha = np.zeros(60)
    ha[[0, 1, 2, 3, 20, 21, 22]] = 7.0  # zero runs: 16 weeks (4..19) is the longest
    ha = pd.Series(ha, index=hds)
    ha[fds - pd.Timedelta(weeks=52)] = 2 * ya + 1
    hist = pd.concat([pd.DataFrame({"unique_id": "a", "ds": hds, "y": ha.to_numpy()}),
                      pd.DataFrame({"unique_id": "c", "ds": hds[-10:], "y": 1.0}),
                      pd.DataFrame({"unique_id": "d", "ds": hds[-52:], "y": np.arange(52.0) % 13})], ignore_index=True)
    hist.to_parquet(os.path.join(res, "history.parquet"))
    base = act[["unique_id", "ds", "cutoff"]]
    # wrong arm: all four series plus one week past the actuals (partial week, must be dropped)
    extra = pd.DataFrame({"unique_id": ["a"], "ds": [fds[-1] + pd.Timedelta(weeks=1)], "cutoff": [CUT]})
    w = pd.concat([base, extra], ignore_index=True)
    w["yhat"] = np.r_[np.ones(13), np.full(13, 2.0), np.full(13, 3.0), yd, 99.0]
    _write_arm(res, "wrong", w)
    # right arm: no forecast for c -> c excluded
    r = base[base.unique_id != "c"].copy()
    r["yhat"] = np.r_[ya, np.ones(13), yd]
    _write_arm(res, "right", r)
    return res, ha.to_numpy(), ya


def test_pair_table_inner_join_and_values(tmp_path):
    res, ha, ya = _pair_results(tmp_path)
    t = examples.pair_table(res, "wrong", "right").set_index("unique_id")
    assert list(t.index) == ["a", "b", "d"]
    a = t.loc["a"]
    assert a.weeks == 13 and a.y == ya.sum() and a.f_wrong == 13 and a.f_right == ya.sum()
    err = np.abs(1 - ya).sum()
    assert a.err_wrong == pytest.approx(err) and a.err_right == 0 and a.gain == pytest.approx(err)
    assert a.rel_change == pytest.approx(err / ((13 + ya.sum()) / 2))  # weekly abs differences over mean volume
    assert a.amp_wrong == 0 and a.amp_right == pytest.approx(1)
    assert a.mean_hist == pytest.approx(ha.mean()) and a.zero_share_hist == pytest.approx((ha == 0).mean())
    assert a.yearly_repeat == pytest.approx(1.0)
    assert a.corr52_right == pytest.approx(1.0) and np.isnan(a.corr52_wrong)  # flat forecast: undefined
    assert a.zero_share_fc == pytest.approx(3 / 13) and a.max_zero_run_fc == 2 and a.zeros_y52 == 0
    assert a.shape_gain == pytest.approx(np.abs(ya.mean() - ya).sum())  # wrong rescaled to the actual mean; right exact
    assert (a.origins, a.origins_wrong_below_right, a.origins_wrong_below_y) == (1, 1, 1)
    b = t.loc["b"]
    assert np.isnan(b.mean_hist) and np.isnan(b.zero_share_hist) and np.isnan(b.yearly_repeat)
    assert np.isnan(b.amp_wrong)  # flat actuals -> undefined, as scoring._amp
    assert np.isnan(t.loc["d", "yearly_repeat"])  # 52 weeks of history: under 53 -> NaN


def test_history_features(tmp_path):
    res, ha, _ = _pair_results(tmp_path)
    f = examples.history_features(pd.read_parquet(os.path.join(res, "history.parquet"))).set_index("unique_id")
    run, best = 0, 0
    for v in ha:
        run = run + 1 if v == 0 else 0
        best = max(best, run)
    assert f.loc["a", "max_zero_run_hist"] == best and f.loc["c", "max_zero_run_hist"] == 0
    assert f.loc["d", "n_hist"] == 52 and f.loc["d", "zero_share_hist"] == pytest.approx(4 / 52)


# ---------- rules ----------

def test_overforecast_fix_needs_wrong_above_right_and_actual():
    t = _table(_filler() + [
        _row("pick", f_wrong=150, f_right=110, err_wrong=60, err_right=15),
        _row("stockout", max_zero_run_hist=20, f_wrong=150, f_right=110, err_wrong=90, err_right=15),
        _row("under_actual", f_wrong=90, f_right=80, err_wrong=90, err_right=20),     # bigger gain, wrong < y
        _row("below_right", f_wrong=120, f_right=130, err_wrong=90, err_right=10),     # bigger gain, wrong < right
        _row("slow", mean_hist=1, f_wrong=200, f_right=100, err_wrong=200, err_right=0),  # below median volume
    ])
    p = examples.pick_overforecast_fix(t)
    assert p.uid == "pick" and p.rank == 1 and p.n_candidates == 1


def _seasonal(**kw):
    r = dict(yearly_repeat=0.9, corr52_wrong=0.0, corr52_right=0.5, amp_right=0.8, err_wrong=50, err_right=40)
    r.update(kw)
    return r


def test_seasonal_gain_fix_follows_last_year():
    rows = [_row(f"z{i}", yearly_repeat=0.1 * i) for i in range(8)]
    rows += [_row("pick", **_seasonal(corr52_right=0.8)),
             _row("weaker", **_seasonal(corr52_right=0.4)),
             _row("stockout_fc", **_seasonal(corr52_right=0.9, zero_share_fc=0.1)),
             _row("stockout_y52", **_seasonal(corr52_right=0.9, zeros_y52=2)),
             _row("long_gap", **_seasonal(corr52_right=0.9, max_zero_run_hist=20)),
             _row("fix_flat", **_seasonal(corr52_right=0.9, amp_right=0.3)),
             _row("no_gain", **_seasonal(corr52_right=0.9, err_right=60)),
             _row("weak_season", **_seasonal(yearly_repeat=0.0, corr52_right=0.9)),
             _row("no_hist", **_seasonal(corr52_right=0.9, yearly_repeat=np.nan, mean_hist=np.nan))]
    p = examples.pick_seasonal_gain(_table(rows))
    assert p.uid == "pick" and p.n_candidates == 2


def test_flatter_wrong_scores_shape_not_level():
    flat = dict(amp_wrong=0.2, amp_right=0.9, err_wrong=60, err_right=20)
    t = _table(_filler() + [
        _row("pick", **flat, shape_gain=30),
        _row("level_only", **{**flat, "err_wrong": 200}, shape_gain=5),            # big gain, from level
        _row("stockout_fc", **flat, shape_gain=90, max_zero_run_fc=5),
        _row("gap_hist", **flat, shape_gain=90, max_zero_run_hist=5),
        _row("barely_flatter", **{**flat, "amp_wrong": 0.85}, shape_gain=90),
        _row("amp_nan", **{**flat, "amp_wrong": np.nan, "amp_right": np.nan}, shape_gain=90),
    ])
    assert examples.pick_flatter_wrong(t).uid == "pick"


def _loss(**kw):
    r = dict(zero_share_hist=0.4, zero_share_fc=0.4, max_zero_run_hist=3, max_zero_run_fc=3, mean_hist=1.0, y=100,
             f_wrong=40, f_right=100, origins_wrong_below_right=3, origins_wrong_below_y=3)
    r.update(kw)
    return r


def test_underforecast_wrong_intermittent_moderate_every_origin():
    t = _table(_filler() + [
        _row("pick", **_loss(f_wrong=30)),                       # closes 100% of 70
        _row("less_closed", **_loss(f_right=80)),                # 40 of 60
        _row("same_share_smaller", **_loss(f_wrong=50)),         # 100% of 50: tie on share, less closed
        _row("high_volume", **_loss(mean_hist=50, f_wrong=0)),
        _row("one_origin", **_loss(f_wrong=0, origins_wrong_below_right=2)),
        _row("fc_steady", **_loss(f_wrong=0, zero_share_fc=0.0)),
        _row("stockout", **_loss(f_wrong=0, max_zero_run_fc=10)),
        _row("small", **_loss(y=40, f_wrong=0, f_right=40)),
    ])
    p = examples.pick_underforecast_wrong(t)
    assert p.uid == "pick" and p.n_candidates == 3


def test_floor_picks_mostly_zero_series_forecast_at_floor():
    slow = dict(zero_share_hist=0.8, mean_hist=0.2, weeks=10, y=2)
    t = _table(_filler() + [
        _row("pick", **slow, f_wrong=10, f_right=2, err_wrong=10, err_right=2),
        _row("above_floor", **slow, f_wrong=40, f_right=2, err_wrong=38, err_right=2),    # 4 a week: not the floor
        _row("below_floor", **slow, f_wrong=5, f_right=0, err_wrong=5, err_right=2),
        _row("no_sales", **{**slow, "y": 0}, f_wrong=10, f_right=0, err_wrong=10, err_right=0),
        _row("selling", **{**slow, "y": 30}, f_wrong=10, f_right=30, err_wrong=20, err_right=0),  # 3 a week
        _row("steady", zero_share_hist=0.1, weeks=10, y=2, f_wrong=10, f_right=0, err_wrong=30, err_right=0),
    ])
    p = examples.pick_floor(t)
    assert p.uid == "pick" and p.n_candidates == 1
    # pattern over the 5 mostly-zero series: forecast >= 0.9 a week while sales < 0.5 a week (pick, above_floor, no_sales)
    assert (p.n_pattern, p.n_pattern_base, p.n_population) == (3, 5, 12)
    assert p.pattern_share == 3 / 5 and "50% zero weeks" in p.pattern_base_text


def test_typical_is_median_rel_change():
    rows = [_row(f"s{i}", f_wrong=100 + 10 * i, f_right=100) for i in range(5)]
    p = examples.pick_typical(_table(rows))
    assert p.uid == "s2" and p.rank == 3 and p.n_candidates == 5


def test_segment_rules_respect_mask():
    t = _table(_filler() + [
        _row("dl_pick", err_wrong=50, err_right=10),
        _row("gbm_pick", err_wrong=10, err_right=40),
        _row("outside", err_wrong=500, err_right=0),
        _row("outside2", err_wrong=0, err_right=500),
    ])
    mask = ~t.unique_id.str.startswith("outside")
    p = examples.pick_right_wins_segment(t, mask)
    assert p.uid == "dl_pick" and p.n_population == 8 and p.n_pattern == 1
    assert examples.pick_wrong_wins_segment(t, mask).uid == "gbm_pick"
    by_uid = pd.Series(mask.to_numpy(), index=t.unique_id)  # a mask keyed by unique_id works too
    assert examples.pick_right_wins_segment(t, by_uid).uid == "dl_pick"
    with pytest.raises(ValueError):
        examples.pick_right_wins_segment(t, pd.Series(True, index=range(100, 110)))


def test_empty_candidates_fall_back_to_typical():
    t = _table([_row(f"s{i}", f_wrong=100 + 10 * i, f_right=100) for i in range(5)])  # nobody intermittent
    p = examples.pick_underforecast_wrong(t)
    assert p.rule.startswith("no series matched") and "typical series instead" in p.rule
    assert p.uid == examples.pick_typical(t).uid


def test_no_improvement_falls_back_to_typical():
    t = _table(_filler(err_wrong=10, err_right=20) + [_row("x", **_seasonal(err_wrong=10, err_right=30))])
    p = examples.pick_seasonal_gain(t)
    assert p.rule.startswith("no series matched")


def test_counts_exact():
    t = _table([
        _row("helped", y=100, err_wrong=50, err_right=30, f_wrong=150, f_right=120),   # gain 20 > 10, changed
        _row("hurt", y=100, err_wrong=50, err_right=70, f_wrong=150, f_right=140),     # gain -20, rel 10/145 < .1
        _row("small_gain", y=100, err_wrong=50, err_right=45, f_wrong=150, f_right=145),
        _row("low_volume", y=50, err_wrong=50, err_right=0, f_wrong=150, f_right=100),  # y not > 50: not counted
    ])
    p = examples.pick_typical(t)
    assert (p.n_candidates, p.n_population, p.n_counted, p.n_helped, p.n_hurt, p.n_changed) == (4, 4, 3, 1, 1, 1)


def test_counts_do_not_depend_on_the_rule_filter():
    # only "pick" over-forecasts; the hurt series do not match the rule but must still be counted
    t = _table(_filler() + [
        _row("pick", y=100, f_wrong=150, f_right=110, err_wrong=60, err_right=15),
        _row("hurt1", y=100, f_wrong=90, f_right=60, err_wrong=10, err_right=40),
        _row("hurt2", y=100, f_wrong=90, f_right=60, err_wrong=10, err_right=40),
    ])
    p = examples.pick_overforecast_fix(t)
    assert p.uid == "pick" and p.n_candidates == 1
    assert (p.n_helped, p.n_hurt, p.n_pattern, p.n_population) == (1, 2, 1, 9)
    assert p.pattern_share == pytest.approx(1 / 9)


def test_pick_reports_wmape_and_abs_bias():
    t = _table(_filler() + [_row("pick", y=200, f_wrong=300, f_right=180, err_wrong=120, err_right=40)])
    p = examples.pick_overforecast_fix(t)
    assert p.uid == "pick"
    assert (p.wmape_wrong, p.wmape_right) == (pytest.approx(60), pytest.approx(20))
    assert (p.absbias_wrong, p.absbias_right) == (pytest.approx(50), pytest.approx(10))


def test_bias_summary_over_intermittent_series():
    t = _table(_filler() + [_row("i1", zero_share_hist=0.5, y=100, f_wrong=50, f_right=90),
                            _row("i2", zero_share_hist=0.4, y=100, f_wrong=110, f_right=100)])
    b = examples.bias_summary(t)
    assert b["n"] == 2 and b["share_wrong_below_right"] == 0.5
    assert b["bias_wrong"] == pytest.approx(-20) and b["bias_right"] == pytest.approx(-5)


def test_pick_right_closer_total_picks_largest_total_miss_cut():
    rows = _filler(6, mean_hist=1.0) + [
        _row("big_wmape", mean_hist=10.0, y=100.0, f_wrong=100.0, f_right=100.0, err_wrong=40.0, err_right=10.0),
        _row("closer", mean_hist=10.0, y=100.0, f_wrong=70.0, f_right=98.0, err_wrong=31.0, err_right=30.0),
        _row("smaller_cut", mean_hist=10.0, y=100.0, f_wrong=85.0, f_right=99.0),
        _row("not_close", mean_hist=10.0, y=100.0, f_wrong=10.0, f_right=60.0),
        _row("level_shift", mean_hist=3.0, y=100.0, f_wrong=50.0, f_right=100.0),
        _row("outside", mean_hist=10.0, y=100.0, f_wrong=10.0, f_right=100.0),
        _row("stockout", mean_hist=10.0, y=100.0, f_wrong=10.0, f_right=100.0, max_zero_run_fc=4),
        _row("near_zero", mean_hist=10.0, y=100.0, f_wrong=10.0, f_right=100.0, low_weeks_fc=1),
        _row("new_item", mean_hist=10.0, y=100.0, f_wrong=10.0, f_right=100.0, n_hist=12)]
    t = _table(rows)
    mask = pd.Series(t["unique_id"] != "outside", index=t.index)
    p = examples.pick_right_closer_total(t, mask)
    assert p.uid == "closer" and p.rank == 1 and p.n_population == len(t) - 1


def test_pair_table_counts_near_zero_weeks(tmp_path):
    from tests.fixtures.make_fake_results import make
    res = make(str(tmp_path / "r"))
    h = pd.read_parquet(os.path.join(res, "history.parquet"))
    h.loc[(h.unique_id == "s9") & (h.ds == h.ds.max()), "y"] = 0.0
    h.to_parquet(os.path.join(res, "history.parquet"))
    t = examples.pair_table(res, "f2_noclip", "f2_clip").set_index("unique_id")
    assert t.loc["s9", "low_weeks_hist"] >= 1 and (t["low_weeks_fc"] >= 0).all()


def test_wins_segment_rules_need_full_history():
    rows = _filler(6, mean_hist=1.0) + [
        _row("new_item", mean_hist=10.0, err_wrong=10.0, err_right=500.0, n_hist=12),
        _row("stockout", mean_hist=10.0, err_wrong=10.0, err_right=400.0, max_zero_run_hist=9),
        _row("level_shift", mean_hist=50.0, err_wrong=10.0, err_right=300.0),
        _row("old_item", mean_hist=10.0, err_wrong=10.0, err_right=100.0, n_hist=52),
        _row("dl_old", mean_hist=10.0, err_wrong=100.0, err_right=10.0, n_hist=52),
        _row("dl_new", mean_hist=10.0, err_wrong=900.0, err_right=10.0, n_hist=20)]
    t = _table(rows)
    everyone = pd.Series(True, index=t.index)
    p = examples.pick_wrong_wins_segment(t, everyone)
    assert p.uid == "old_item" and p.text(p.pattern_text, ("LightGBM", "N-HiTS")) == "have the lower error with LightGBM"
    assert examples.pick_right_wins_segment(t, everyone).uid == "dl_old"
