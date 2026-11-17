import os

import pytest
from PIL import Image

from m5knobs import examples, results, scoring
from tests.fixtures.make_fake_results import make


@pytest.fixture
def res(tmp_path):
    return make(str(tmp_path / "results"))


def _draw(res, out_dir, wrong="f2_noclip", right="f2_clip"):
    from m5knobs.figures import compare
    pick = examples.pick_overforecast_fix(examples.pair_table(res, wrong, right))
    return compare.draw(res, wrong, right, pick, ("clip off", "AutoClip"), "Clipping example", "cmp_test", out_dir=out_dir)


def test_renders_png_and_svg(res, tmp_path):
    out = _draw(res, str(tmp_path / "fig"))
    assert [os.path.basename(p) for p in out] == ["cmp_test.png", "cmp_test.svg"]
    assert Image.open(out[0]).size[0] >= 1200


def test_skips_errored_arm(res, tmp_path, capsys):
    from m5knobs.figures import compare
    d = os.path.join(res, "runs", "f2_clip__s1")
    os.remove(os.path.join(d, "DONE"))
    open(os.path.join(d, "ERROR.txt"), "w").write("Traceback\nRuntimeError: boom\n")
    pick = examples.pick_typical(examples.pair_table(res, "f2_noclip", "f3_fix"))
    out = compare.draw(res, "f2_noclip", "f2_clip", pick, ("a", "b"), "t", "cmp_skip", out_dir=str(tmp_path / "fig"))
    assert out == []
    assert "skipped: f2_clip did not finish" in capsys.readouterr().out
    assert not os.path.exists(tmp_path / "fig" / "cmp_skip.png")


def test_metrics_text_from_summarize_and_pick(res, tmp_path):
    out = _draw(res, str(tmp_path / "fig"))
    svg = open(out[1]).read()
    for arm in ("f2_noclip", "f2_clip"):
        s = scoring.summarize(results.load_run(arm, res)["ps"])
        assert f"{s['wape']:.1f}%" in svg
        assert f"{abs(s['bias']):.1f}%" in svg
    pick = examples.pick_overforecast_fix(examples.pair_table(res, "f2_noclip", "f2_clip"))
    for v in (pick.wmape_wrong, pick.wmape_right, pick.absbias_wrong, pick.absbias_right):
        assert f"{v:.1f}%" in svg
    assert "WMAPE" in svg and "|bias|" in svg
    assert f"rank {pick.rank} of {pick.n_candidates:,}" in svg
    assert f"{pick.n_helped:,} improved" in svg


def test_svg_reproducible(res, tmp_path):
    a = open(_draw(res, str(tmp_path / "a"))[1], "rb").read()
    b = open(_draw(res, str(tmp_path / "b"))[1], "rb").read()
    assert a == b


def test_series_without_history_starts_at_forecast_weeks(res, tmp_path):
    import pandas as pd
    from m5knobs.figures import compare
    p = os.path.join(res, "history.parquet")
    h = pd.read_parquet(p)
    pick = examples.pick_typical(examples.pair_table(res, "f2_noclip", "f2_clip"))
    h[h.unique_id != pick.uid].to_parquet(p)
    out = compare.draw(res, "f2_noclip", "f2_clip", pick, ("a", "b"), "t", "cmp_nohist", out_dir=str(tmp_path / "fig"))
    assert len(out) == 2


def test_population_panel_sums_only_those_series(res, tmp_path):
    from m5knobs.figures import compare
    pop = ["s1", "s2", "s3"]
    pick = examples.pick_typical(examples.pair_table(res, "f2_noclip", "f2_clip"))
    out = compare.draw(res, "f2_noclip", "f2_clip", pick, ("clip off", "AutoClip"), "t", "cmp_pop",
                       out_dir=str(tmp_path / "fig"), population=pop, population_label="Three series")
    svg = open(out[1]).read()
    assert "Three series: 3 series summed" in svg
    for arm in ("f2_noclip", "f2_clip"):
        ps = results.load_run(arm, res)["ps"]
        s = scoring.summarize(ps[ps.unique_id.isin(pop)])
        assert f"{s['wape']:.1f}%" in svg and f"{abs(s['bias']):.1f}%" in svg



def test_pct_keeps_large_values_narrow():
    from m5knobs.figures.compare import _pct
    assert [_pct(45.23), _pct(99.94), _pct(1236.31), _pct(205.2)] == ["45.2%", "99.9%", "1,236%", "205%"]
