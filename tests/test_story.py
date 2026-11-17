import pandas as pd
import pytest

from m5knobs import examples, results, scoring, story
from tests.fixtures.make_fake_results import make


@pytest.fixture
def res(tmp_path):
    return make(str(tmp_path / "results"))


def test_compare_line_uses_repo_scoring(res):
    t = story.compare_line("f2_noclip", "f2_clip", ("clip off", "AutoClip"), res)
    for arm in ("f2_noclip", "f2_clip"):
        s = scoring.summarize(results.load_run(arm, res)["ps"])
        assert f"WMAPE {s['wape']:.2f}, |bias| {abs(s['bias']):.1f}%" in t
    assert "**clip off**" in t and "**AutoClip**" in t and "seed noise" in t


def test_typical_line_top_case_and_typical(res):
    t = examples.pair_table(res, "f2_noclip", "f2_clip")
    top = examples.pick_overforecast_fix(t)
    if not top.rule.startswith("no series matched"):
        assert "rule's top case (rank 1 of" in story.typical_line(top, ("a", "b"))
    typ = examples.pick_typical(t)
    line = story.typical_line(typ, ("a", "b"))
    assert "typical series, not an extreme one" in line and f"{typ.n_pattern:,} of the {typ.n_population:,} series both runs forecast" in line
    assert "within 10% of each other" in line


def test_summary_table_has_every_setting(res):
    t = story.summary_table(res)
    assert list(t["arm"]) == [a for _, a, _ in story.SETTINGS]
    assert {"WMAPE", "bias %", "|bias| %"} <= set(t.columns)
    assert (t["|bias| %"] == t["bias %"].abs()).all()


def _table():
    rows = []
    for seg, kind, share, (w_l, b_l), (w_d, b_d, pw, pb) in [
            ("Q4 fastest", "volume quartile", 0.7, (40.0, -6.0), (39.9, -0.5, 0.30, 0.001)),
            ("Q1 slowest", "volume quartile", 0.05, (75.8, -15.0), (75.0, -20.0, 0.01, 0.001)),
            ("Q2", "volume quartile", 0.1, (60.0, -9.0), (59.9, -9.5, 0.02, 0.40)),
            ("FOODS_1", "department", 0.08, (47.9, -4.3), (49.2, -0.5, 0.001, 0.002)),
            ("FOODS_2", "department", 0.14, (45.3, -12.5), (44.3, -9.0, 0.001, 0.001))]:
        rows.append(dict(segment_kind=kind, segment=seg, vol_share=share, model="LightGBM", wape=w_l, bias=b_l, abs_bias=abs(b_l),
                         p_wape=float("nan"), p_abs_bias=float("nan")))
        rows.append(dict(segment_kind=kind, segment=seg, vol_share=share, model=story.DL, wape=w_d, bias=b_d, abs_bias=abs(b_d),
                         p_wape=pw, p_abs_bias=pb))
    return pd.DataFrame(rows)


def test_where_bullets_use_segment_p():
    b = story.where_bullets(_table())
    assert b[0] == ("N-HiTS Huber beats LightGBM on WMAPE at p < 0.05 within the segment: FOODS_2 (14% of volume), -1.0, "
                    "volume Q1 slowest (5% of volume), -0.8 and volume Q2 (10% of volume), -0.1.")
    assert b[1] == "LightGBM beats N-HiTS Huber on WMAPE at p < 0.05 within the segment: FOODS_1 (8% of volume), +1.3."
    assert "volume Q4 fastest (70% of volume), -5.5pt" in b[2] and "FOODS_1 (8% of volume), -3.8pt" in b[2]
    assert b[3].startswith("LightGBM beats N-HiTS Huber on |bias| at p < 0.05 within the segment: volume Q1 slowest")
    assert b[4] == "No clear WMAPE difference (p ≥ 0.05) in 1 of 5 segments: volume Q4 fastest (70% of volume)."
    assert b[5].startswith("Of the clear WMAPE differences, volume Q2 (10% of volume) is within the all-series seed-noise bar")


def test_dl_lead_mirrors_lightgbm_lead_and_none():
    r = story.dl_lead(_table())
    assert (r["segment_kind"], r["segment"]) == ("department", "FOODS_2") and r["d_wmape"] == pytest.approx(-1.0)
    t = _table()
    t.loc[t.model == story.DL, "p_wape"] = 0.5
    assert story.dl_lead(t) is None and story.lightgbm_lead(t) is None


def test_lightgbm_lead_picks_largest_wmape_lead():
    r = story.lightgbm_lead(_table())
    assert (r["segment_kind"], r["segment"]) == ("department", "FOODS_1") and r["d_wmape"] == pytest.approx(1.3)


def test_group_metrics_and_concentration(res):
    t = examples.pair_table(res, "f4_lgbm", "f3_fix")
    mask = pd.Series(True, index=t.index)
    g = story.group_metrics(t, mask)
    assert g["n"] == len(t) and g["vol_share"] == pytest.approx(1.0)
    assert g["wmape_wrong"] == pytest.approx(100 * t.err_wrong.sum() / t.y.sum())
    c = story.concentration(t, mask, k=3)
    assert c["lead"] == pytest.approx(g["wmape_right"] - g["wmape_wrong"])


def test_slow_seller_table_and_spikes(res):
    n, vol, tab = story.slow_seller_table(res, rate=4.0)
    assert list(tab.index) == list(story.segments.MODELS) and 0 <= vol <= 1
    assert "WMAPE on these series" in tab.columns
    sp = story.val_spikes(res)
    assert set(sp) == {"lr1e2", "lr3e2", "nofilter"} and all(v["on"] >= 1 for v in sp.values())


def test_gap_table_and_claim4(res):
    g = story.segments.gap_rows(res, n_boot=20)
    assert list(story.gap_table(g)["model"]) == list(story.segments.MODELS)
    assert story.claim4(g).startswith("Claim 4") and "N-HiTS Huber" in story.claim4(g)


def test_bias_caveat_only_when_wrong_has_smaller_abs_bias():
    L = ("patience 250", "patience 1,000")
    assert story.bias_caveat(dict(wmape=45.0, bias=-5.0), dict(wmape=45.5, bias=-3.0), L, "all series") == ""
    t = story.bias_caveat(dict(wmape=45.93, bias=-0.5), dict(wmape=45.27, bias=-3.6), L, "all series")
    assert t.startswith("On all series, patience 250 has the smaller |bias| (0.5% vs 3.6%). Both under-forecast")
    assert "sit higher" in t and "WMAPE 45.93 vs 45.27" in t
    o = story.bias_caveat(dict(wmape=45.0, bias=1.0), dict(wmape=45.3, bias=-3.6), ("MinMax", "RevIN"), "all series")
    assert "opposite directions" in o and "not worse either" in o


def test_group_line_names_split_metrics():
    t = pd.DataFrame(dict(unique_id=["a", "b"], y=[100.0, 100.0], f_wrong=[96.0, 96.0], f_right=[100.0, 100.0],
                          err_wrong=[40.0, 40.0], err_right=[45.0, 45.0], gain=[-5.0, -5.0]))
    line = story.group_line(t, pd.Series(True, index=t.index), ("LightGBM", "N-HiTS Huber"), "Seg")
    assert "The result splits by metric: LightGBM has the lower WMAPE (+5.00); N-HiTS Huber has the smaller |bias| (-4.0pt)" in line


def test_example_line_flags_either_metric():
    from m5knobs.examples import Pick
    p = Pick(uid="x", rule="", rank=1, n_candidates=1, n_pattern=0, pattern_share=0.0, n_population=1, n_counted=1,
             n_helped=0, n_hurt=0, n_changed=0, wmape_wrong=54.1, wmape_right=59.9, absbias_wrong=18.0, absbias_right=1.5)
    t = story.example_line(p, ("LightGBM", "N-HiTS Huber"))
    assert "On this series LightGBM has the lower WMAPE: the example was chosen by the rule above" in t
    p.wmape_right = 50.0
    assert "On this series" not in story.example_line(p, ("LightGBM", "N-HiTS Huber"))


def test_spike_text_names_autoclip_spikes():
    sp = {"a": dict(off=1.0, on=1754.4, off_recovered=True, on_recovered=True),
          "b": dict(off=1.0, on=1.0, off_recovered=True, on_recovered=True)}
    first, second = story.spike_text(sp, {"a": "LR 10x", "b": "filter off"})
    assert "LR 10x: the AutoClip run spiked to 1,754.4× its best and recovered" in first and "filter off: no spike" in first
    assert second.startswith("The larger spikes were in the AutoClip runs (LR 10x)")
    sp["a"].update(off=50.0, on=1.0)
    assert story.spike_text(sp, {"a": "LR 10x", "b": "filter off"})[1] == ""


def test_example_line_favouring_first_side():
    from m5knobs.examples import Pick
    p = Pick(uid="x", rule="", rank=1, n_candidates=1, n_pattern=0, pattern_share=0.0, n_population=1, n_counted=1,
             n_helped=0, n_hurt=0, n_changed=0, wmape_wrong=33.6, wmape_right=43.6, absbias_wrong=13.1, absbias_right=26.8)
    assert "On this series" not in story.example_line(p, ("LightGBM", "N-HiTS Huber"), favours="wrong")
    p.absbias_right = 5.0
    assert "On this series N-HiTS Huber has the smaller |bias|" in story.example_line(p, ("LightGBM", "N-HiTS Huber"), favours="wrong")
