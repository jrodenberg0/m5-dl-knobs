import os

import pandas as pd
import pytest
from PIL import Image

from tests.fixtures.make_fake_results import make


@pytest.fixture
def res(tmp_path):
    return make(str(tmp_path / "results"))


def test_all_cards_render(res, tmp_path):
    import fig_cards
    paths = fig_cards.main(results_dir=res, out_dir=str(tmp_path / "fig"), n_boot=20)
    names = {os.path.basename(p) for p in paths}
    for k in ("clipping", "input_size", "early_stopping", "loss", "scaler"):
        assert f"card_{k}.png" in names and f"card_{k}.svg" in names
        assert Image.open(os.path.join(tmp_path, "fig", f"card_{k}.png")).size[0] >= 1200


def test_headline_no_effect_makes_no_recommendation():
    import fig_cards
    card = fig_cards.CARDS[4]
    h = fig_cards.headline(card, {"label": "no clear effect on M5"}, dict(wape=45.75, bias=-0.6), dict(wape=45.68, bias=-3.4))
    assert "no measurable difference" in h and card["action"] not in h


def test_headline_reports_moved_metric():
    import fig_cards
    card = fig_cards.CARDS[0]
    h = fig_cards.headline(card, {"label": "biased"}, dict(wape=48.0, bias=4.8), dict(wape=46.9, bias=0.1))
    assert card["action"] in h and "+4.8%" in h and "+0.1%" in h


def test_read_lists_only_hurt_measures():
    import fig_cards
    v = {"label": "underfit", "wape_gap": 1.5, "abs_bias_gap_pts": -2.2,
         "insample_mae": {"hurt": True}, "corr": {"hurt": False}, "amp_ratio": {"hurt": True}, "abs_bias": {"hurt": False}}
    r = fig_cards.what_moved(v, "13 weeks")
    assert "fit to the training history" in r and "flatter" in r and "tracking" not in r


def test_card_knobs_name_neuralforecast_params():
    import fig_cards
    knobs = {c["key"]: c["knob"] for c in fig_cards.CARDS}
    assert knobs == {"clipping": "gradient_clip_val", "input_size": "input_size",
                     "early_stopping": "early_stop_patience_steps × val_check_steps", "loss": "loss", "scaler": "scaler_type"}


def test_f4_headline_from_table():
    import pandas as pd
    import fig_f4_table
    t = pd.DataFrame([dict(model="LGBM (baseline)", wape=46.49, gap_vs_lgbm=None, read="baseline"),
                      dict(model="N-HiTS", wape=46.84, gap_vs_lgbm=0.35, read="worse (not significant)"),
                      dict(model="TCN", wape=45.59, gap_vs_lgbm=-0.90, read="better (significant)")])
    h = fig_f4_table.headline_text(t)
    assert "TCN beat LGBM by 0.90 WAPE" in h and "N-HiTS" in h


def test_stress_and_sweep_figures_render(res, tmp_path):
    import fig_stress
    import fig_es_sweep
    p1 = fig_stress.main(results_dir=res, out_dir=str(tmp_path / "fig"), n_boot=20)
    p2 = fig_es_sweep.main(results_dir=res, out_dir=str(tmp_path / "fig"))
    for p in p1 + p2:
        assert os.path.exists(p)
    assert Image.open(os.path.join(tmp_path, "fig", "card_clipping_stress.png")).size[0] >= 1200
    assert Image.open(os.path.join(tmp_path, "fig", "card_early_stopping_sweep.png")).size[0] >= 1200


def test_sweep_headline_names_best_window():
    import fig_es_sweep
    import pandas as pd
    t = pd.DataFrame({"window": [250, 500, 1000, 2000, 4000], "wape": [46.0, 45.6, 45.3, 45.1, 45.15], "bias": [-1, -2, -3, -3.5, -4]})
    h = fig_es_sweep.headline_text(t, base=1000)
    assert "2,000" in h


def test_stress_figure_shows_crash_as_result(res, tmp_path):
    import json
    import pandas as pd
    import fig_stress
    d = os.path.join(res, "runs", "st_lr3e2_noclip__s1")
    os.remove(os.path.join(d, "DONE"))
    open(os.path.join(d, "ERROR.txt"), "w").write("Traceback...\nException: Loss is NaN, training stopped.\n")
    fig_stress.main(results_dir=res, out_dir=str(tmp_path / "fig"), n_boot=20)
    out = json.load(open(os.path.join(res, "stress_outcomes.json")))
    assert out["lr3e2"]["outcome"] == "rescued"


def test_sweep_headline_mentions_bias_tradeoff():
    import pandas as pd
    import fig_es_sweep
    t = pd.DataFrame({"window": [250, 500, 1000, 2000, 4000], "wape": [45.71, 45.71, 45.15, 45.05, 44.92],
                      "bias": [-1.3, -0.3, -3.4, -5.6, -5.4]})
    h = fig_es_sweep.headline_text(t, base=1000)
    assert "4,000" in h and "bias" in h and "-5.4%" in h


def test_stress_headline_counts_divergence_separately():
    import fig_stress
    h = fig_stress.headline({"a": dict(outcome="diverged"), "b": dict(outcome="diverged"), "c": dict(outcome="no effect")})
    assert "diverged" in h and "helped" not in h


def _run(wape, bias, best, val, gmax=10.0, gmed=1.0):
    tr = pd.DataFrame({"step": [50 * (i + 1) for i in range(len(val))], "val_loss": val, "grad_norm": 1.0, "clip_thr": None})
    return dict(ok=True, trace=tr, summary=dict(wape=wape, bias=bias, fold_stats=[dict(best_step=best, grad_norm_max=gmax, grad_norm_median=gmed)]))


def test_stress_divergence_read_from_validation_path():
    import fig_stress
    off = _run(46.9, -5.3, 300, [10, 9.9, 10, 30, 10, 101, 101])
    on = _run(50.3, 9.1, 450, [10, 9.9, 10.1, 29.0, 10, 9.98, 9.98])  # transient spike, recovered
    tag, text = fig_stress.outcome(off, on, {"label": "no clear effect on M5"}, {"label": "both"})
    assert tag == "stabilized" and "blew up" in text and "46.9" in text
    both = _run(50.0, 5.0, 650, [10, 9.9, 44, 10, 96, 96])
    tag2, _ = fig_stress.outcome(both, _run(49.0, 6.0, 200, [10, 9.9, 100, 96]), {"label": "underfit"}, None)
    assert tag2 == "diverged"
    tag3, _ = fig_stress.outcome(_run(45.6, -5.5, 4100, [10.8, 9.9, 9.85]), _run(45.5, -5.6, 4100, [10.8, 9.9, 9.85]),
                                 {"label": "no clear effect on M5"}, {"label": "no clear effect on M5"})
    assert tag3 == "no effect"


def test_stress_non_nan_error_is_not_a_result(res):
    import fig_stress
    d = os.path.join(res, "runs", "st_lr3e2_noclip__s1")
    os.remove(os.path.join(d, "DONE"))
    open(os.path.join(d, "ERROR.txt"), "w").write("Traceback...\nRuntimeError: MPS backend out of memory\n")
    with pytest.raises(SystemExit, match="not a training result"):
        fig_stress._load("st_lr3e2_noclip", res)


def test_what_moved_reports_bias_tradeoff_of_fix():
    import fig_cards
    v = {"label": "underfit", "wape_gap": 0.65, "abs_bias_gap_pts": -3.2,
         "insample_mae": {"hurt": True}, "corr": {"hurt": False}, "amp_ratio": {"hurt": False}, "abs_bias": {"hurt": False}}
    r = fig_cards.what_moved(v, "250 steps", ms=dict(bias=-0.5), fs=dict(bias=-3.6))
    assert "trade-off" in r.lower() and "-0.5%" in r and "-3.6%" in r


def test_no_unsupported_max_steps_claim():
    import fig_cards
    assert all("max_steps" not in c["try_effect"] for c in fig_cards.CARDS)
