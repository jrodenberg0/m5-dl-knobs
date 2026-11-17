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
    assert "typical series, not an extreme one" in line and f"{typ.n_pattern:,} of {typ.n_population:,}" in line


def test_summary_table_has_every_setting(res):
    t = story.summary_table(res)
    assert list(t["arm"]) == [a for _, a, _ in story.SETTINGS]
    assert {"WMAPE", "bias %", "|bias| %"} <= set(t.columns)
    assert (t["|bias| %"] == t["bias %"].abs()).all()


def _table():
    rows = []
    for seg, kind, share, (w_l, b_l), (w_d, b_d) in [
            ("Q4 fastest", "volume quartile", 0.7, (40.0, -6.0), (39.9, -0.5)),
            ("Q1 slowest", "volume quartile", 0.05, (75.8, -15.0), (75.0, -20.0)),
            ("FOODS_1", "department", 0.08, (47.9, -4.3), (49.2, -0.5))]:
        rows.append(dict(segment_kind=kind, segment=seg, vol_share=share, model="LightGBM", wape=w_l, bias=b_l, abs_bias=abs(b_l)))
        rows.append(dict(segment_kind=kind, segment=seg, vol_share=share, model=story.DL, wape=w_d, bias=b_d, abs_bias=abs(b_d)))
    return pd.DataFrame(rows)


def test_where_bullets_sorts_segments_by_noise():
    b = story.where_bullets(_table())
    assert b[0] == "N-HiTS Huber beats LightGBM on WMAPE beyond seed noise (±0.2): volume Q1 slowest (5% of volume), -0.8."
    assert b[1] == "LightGBM beats N-HiTS Huber on WMAPE beyond seed noise (±0.2): FOODS_1 (8% of volume), +1.3."
    assert "volume Q4 fastest (70% of volume), -5.5pt" in b[2] and "FOODS_1 (8% of volume), -3.8pt" in b[2]
    assert b[3].startswith("LightGBM beats N-HiTS Huber on |bias| beyond seed noise (±1pt): volume Q1 slowest")
    assert b[4] == "WMAPE within seed noise in 1 of 3 segments: volume Q4 fastest (70% of volume)."


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
