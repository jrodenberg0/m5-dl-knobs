import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from m5knobs import results, scoring
from m5knobs.data import SEED_NOISE
from tests.fixtures.make_fake_results import make


@pytest.fixture
def res(tmp_path):
    return make(str(tmp_path / "results"))


def _rescore(res, arms):
    act = results.actuals(res)
    for a in arms:
        d = os.path.join(res, "runs", f"{a}__s1")
        scoring.per_series(pd.read_parquet(os.path.join(d, "forecasts.parquet")), act).to_parquet(os.path.join(d, "per_series.parquet"))


def test_segment_table_kinds_labels_and_overall(res):
    from m5knobs import segments
    t = segments.segment_table(res)
    assert list(t.columns[:7]) == ["segment_kind", "segment", "n_series", "vol_share", "model", "wape", "bias"]
    assert list(dict.fromkeys(t.segment_kind)) == list(segments.KINDS)
    assert list(dict.fromkeys(t.model)) == list(segments.MODELS)
    assert list(t[(t.segment_kind == "volume quartile") & (t.model == "LightGBM")].segment) == list(segments.QUARTILES)
    assert set(t[t.segment_kind == "zero weeks"].segment) <= set(segments.ZERO_LABELS)
    assert set(t[t.segment_kind == "department"].segment) == {f"s{i}" for i in range(20)}  # fixture ids have one part
    lg = t[(t.model == "LightGBM") & (t.segment_kind == "origin")]
    ps = results.load_run("f4_lgbm", res)["ps"]
    s = scoring.summarize(ps)
    # origins partition the volume: shares sum to 1 and the volume-weighted segment WMAPE is the overall WMAPE
    assert lg.vol_share.sum() == pytest.approx(1)
    assert (lg.vol_share * lg.wape).sum() == pytest.approx(s["wape"])
    assert np.allclose(t.abs_bias, t.bias.abs())
    assert t.wape.notna().all()


def test_quartiles_use_last_52_history_weeks_and_skip_series_without_history(res):
    from m5knobs import segments
    hp = os.path.join(res, "history.parquet")
    h = pd.read_parquet(hp)
    h = h[h.unique_id != "s19"]  # s19: the fastest seller, no history before the first origin
    # s0: 8 huge weeks before the 52-week window; they must not move it out of the slowest quartile
    early = h.ds < h.ds.max() - pd.Timedelta(weeks=51)
    h.loc[early & (h.unique_id == "s0"), "y"] = 1e6
    h.to_parquet(hp)
    seg = segments.series_segments(res)
    assert seg.set_index("unique_id").loc["s0", "volume quartile"] == "Q1 slowest"
    assert seg.set_index("unique_id").loc["s19", ["volume quartile", "zero weeks"]].isna().all()
    assert seg.set_index("unique_id").loc["s19", "department"] == "s19"
    t = segments.segment_table(res, {"LightGBM": "f4_lgbm"})
    act = results.actuals(res)
    q = t[t.segment_kind == "volume quartile"]
    assert q.vol_share.sum() == pytest.approx(1 - act[act.unique_id == "s19"].y.sum() / act.y.sum())
    assert q.n_series.sum() == 19


def test_zero_volume_segment_dropped(res):
    from m5knobs import segments
    p = os.path.join(res, "actuals.parquet")
    act = pd.read_parquet(p)
    act.loc[act.unique_id == "s0", "y"] = 0.0
    act.to_parquet(p)
    _rescore(res, segments.MODELS.values())
    t = segments.segment_table(res)
    assert "s0" not in set(t[t.segment_kind == "department"].segment)
    assert t.wape.notna().all() and np.isfinite(t.bias).all()


def test_overall_gap_matches_summaries(res):
    from m5knobs import segments
    g = segments.overall_gap(res, "f3_fix", "f4_lgbm", n_boot=50)
    a, b = (scoring.summarize(results.load_run(x, res)["ps"]) for x in ("f3_fix", "f4_lgbm"))
    assert g["wape_a"] == pytest.approx(a["wape"]) and g["wape_b"] == pytest.approx(b["wape"])
    assert g["gap"] == pytest.approx(a["wape"] - b["wape"])
    assert g["bias_a"] == pytest.approx(a["bias"]) and g["bias_b"] == pytest.approx(b["bias"])
    assert g["abs_bias_gap"] == pytest.approx(abs(a["bias"]) - abs(b["bias"]))
    assert 0 <= g["p"] <= 1 and 0 <= g["p_abs_bias"] <= 1


def test_claim_text_judged_on_noise():
    from m5knobs import segments
    base = dict(wape_a=45.27, wape_b=45.35, p=0.4, bias_a=-3.6, bias_b=-7.4, abs_bias_gap=-3.8, p_abs_bias=0.0)
    hold = segments.claim_text(dict(base, gap=SEED_NOISE["wape"]))
    assert "N-HiTS Huber" in hold and "f3_fix" in hold and hold.endswith("holds.")
    worse = segments.claim_text(dict(base, gap=SEED_NOISE["wape"] + 0.01))
    assert worse.endswith("does not hold.")


def test_draw_segments(res, tmp_path):
    from m5knobs import segments
    from m5knobs.figures import segments as fig
    t = segments.segment_table(res)
    out = fig.draw(t, list(segments.MODELS), "story_segments", str(tmp_path / "fig"))
    assert [os.path.basename(p) for p in out] == ["story_segments.png", "story_segments.svg"]
    assert Image.open(out[0]).size[0] >= 1200
    svg = open(out[1]).read()
    assert "WMAPE" in svg and "|bias|" in svg and "N-HiTS Huber, best of 5 ES" in svg
    assert f"±{SEED_NOISE['wape']}" in svg
    assert "p &lt; 0.05 within the segment" in svg or "p < 0.05 within the segment" in svg
    again = fig.draw(t, list(segments.MODELS), "story_segments", str(tmp_path / "fig2"))
    assert open(again[1], "rb").read() == open(out[1], "rb").read()


def test_report_has_segment_section(res, tmp_path):
    from m5knobs import report
    out = str(tmp_path / "F.md")
    report.write(results_dir=res, out=out, n_boot=20)
    text = open(out).read()
    assert "## Deep learning vs LightGBM by segment" in text
    assert "best of 5 early-stopping settings" in text
    sec = text.split("## Deep learning vs LightGBM by segment")[1]
    claim = [ln for ln in sec.splitlines() if "claim 4" in ln.lower()]
    assert len(claim) == 1 and "f3_fix" in claim[0] and "N-HiTS Huber" in claim[0]
    assert claim[0].endswith(("Verdict: holds.", "Verdict: does not hold."))
    for k in ("volume quartile", "zero weeks", "department", "origin"):
        assert f"### By {k}" in sec


def test_report_without_history_skips_segment_section(res, tmp_path):
    from m5knobs import report
    os.remove(os.path.join(res, "history.parquet"))
    out = str(tmp_path / "F.md")
    report.write(results_dir=res, out=out, n_boot=20)
    assert "by segment" not in open(out).read()


def test_segment_p_matches_paired_bootstrap(res):
    from m5knobs import results, scoring, segments
    a, b = (scoring._series_level(results.load_run(x, res)["ps"]) for x in ("f3_fix", "f4_lgbm"))
    pw, pb = segments.segment_p(a, b, n_boot=200)
    assert pw == scoring.paired_bootstrap_p(segments._wape, a, b, n_boot=200)
    assert pb == scoring.paired_bootstrap_p(segments._abs_bias, a, b, n_boot=200)


def test_segment_table_has_p_for_deep_models_only(res):
    from m5knobs import segments
    t = segments.segment_table(res, n_boot=50)
    assert t.loc[t.model == segments.BASE, "p_wape"].isna().all()
    assert t.loc[t.model != segments.BASE, ["p_wape", "p_abs_bias"]].apply(lambda c: c.between(0, 1)).all().all()
