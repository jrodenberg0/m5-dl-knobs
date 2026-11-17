import os

import pandas as pd
import pytest
from PIL import Image

from tests.fixtures.make_fake_results import make


@pytest.fixture
def res(tmp_path):
    return make(str(tmp_path / "results"))


def _png_ok(path, min_w=1200):
    assert os.path.exists(path), path
    assert os.path.exists(path.replace(".png", ".svg"))
    assert Image.open(path).size[0] >= min_w


def test_all_cards_render(res, tmp_path):
    from m5knobs.figures import cards
    paths = cards.draw(results_dir=res, out_dir=str(tmp_path / "fig"), n_boot=20)
    names = {os.path.basename(p) for p in paths}
    for k in ("clipping", "input_size", "early_stopping", "loss", "scaler"):
        assert f"card_{k}.png" in names and f"card_{k}.svg" in names
        assert Image.open(os.path.join(tmp_path, "fig", f"card_{k}.png")).size[0] >= 1200


def test_headline_no_effect_makes_no_recommendation():
    from m5knobs.figures import cards
    card = cards.CARDS[4]
    h = cards.headline(card, {"label": "no clear effect on M5"}, dict(wape=45.75, bias=-0.6), dict(wape=45.68, bias=-3.4))
    assert "no measurable difference" in h and card["action"] not in h


def test_headline_reports_moved_metric():
    from m5knobs.figures import cards
    card = cards.CARDS[0]
    h = cards.headline(card, {"label": "biased"}, dict(wape=48.0, bias=4.8), dict(wape=46.9, bias=0.1))
    assert card["action"] in h and "+4.8%" in h and "+0.1%" in h


def test_read_lists_only_hurt_measures():
    from m5knobs.figures import cards
    v = {"label": "underfit", "wape_gap": 1.5, "abs_bias_gap_pts": -2.2,
         "insample_mae": {"hurt": True}, "corr": {"hurt": False}, "amp_ratio": {"hurt": True}, "abs_bias": {"hurt": False}}
    r = cards.what_moved(v, "13 weeks")
    assert "fit to the training history" in r and "flatter" in r and "tracking" not in r


def test_card_knobs_name_neuralforecast_params():
    from m5knobs.figures import cards
    knobs = {c["key"]: c["knob"] for c in cards.CARDS}
    assert knobs == {"clipping": "callbacks=[AutoClip(clip_percentile)]", "input_size": "input_size",
                     "early_stopping": "early_stop_patience_steps × val_check_steps", "loss": "loss", "scaler": "scaler_type"}


def test_f4_headline_from_table():
    from m5knobs.figures import architectures
    t = pd.DataFrame([dict(model="LGBM (baseline)", wape=46.49, gap_vs_lgbm=None, read="baseline"),
                      dict(model="N-HiTS", wape=46.84, gap_vs_lgbm=0.35, read="worse (not significant)"),
                      dict(model="TCN", wape=45.59, gap_vs_lgbm=-0.90, read="better (significant)")])
    h = architectures.headline_text(t)
    assert "TCN beat LGBM by 0.90 WAPE" in h and "N-HiTS" in h


def test_stress_and_sweep_figures_render(res, tmp_path):
    from m5knobs.figures import stress
    from m5knobs.figures import early_stopping
    p1 = stress.draw(results_dir=res, out_dir=str(tmp_path / "fig"), n_boot=20)
    p2 = early_stopping.draw(results_dir=res, out_dir=str(tmp_path / "fig"))
    for p in p1 + p2:
        assert os.path.exists(p)
    assert Image.open(os.path.join(tmp_path, "fig", "card_clipping_stress.png")).size[0] >= 1200
    assert Image.open(os.path.join(tmp_path, "fig", "card_early_stopping_sweep.png")).size[0] >= 1200


def test_sweep_headline_names_best_window():
    from m5knobs.figures import early_stopping
    t = pd.DataFrame({"window": [250, 500, 1000, 2000, 4000], "wape": [46.0, 45.6, 45.3, 45.1, 45.15], "bias": [-1, -2, -3, -3.5, -4]})
    h = early_stopping.headline_text(t, base=1000)
    assert "2,000" in h


def test_stress_figure_shows_crash_as_result(res, tmp_path):
    import json
    from m5knobs.figures import stress
    d = os.path.join(res, "runs", "st_lr3e2_noclip__s1")
    os.remove(os.path.join(d, "DONE"))
    open(os.path.join(d, "ERROR.txt"), "w").write("Traceback...\nException: Loss is NaN, training stopped.\n")
    stress.draw(results_dir=res, out_dir=str(tmp_path / "fig"), n_boot=20)
    out = json.load(open(os.path.join(res, "stress_outcomes.json")))
    assert out["lr3e2"]["outcome"] == "rescued"


def test_sweep_headline_mentions_bias_tradeoff():
    from m5knobs.figures import early_stopping
    t = pd.DataFrame({"window": [250, 500, 1000, 2000, 4000], "wape": [45.71, 45.71, 45.15, 45.05, 44.92],
                      "bias": [-1.3, -0.3, -3.4, -5.6, -5.4]})
    h = early_stopping.headline_text(t, base=1000)
    assert "4,000" in h and "bias" in h and "-5.4%" in h


def test_stress_headline_counts_divergence_separately():
    from m5knobs.figures import stress
    h = stress.headline({"a": dict(outcome="diverged"), "b": dict(outcome="diverged"), "c": dict(outcome="no effect")})
    assert "diverged" in h and "helped" not in h


def _run(wape, bias, best, val, gmax=10.0, gmed=1.0):
    tr = pd.DataFrame({"step": [50 * (i + 1) for i in range(len(val))], "val_loss": val, "grad_norm": 1.0, "clip_thr": None})
    return dict(ok=True, trace=tr, summary=dict(wape=wape, bias=bias, fold_stats=[dict(best_step=best, grad_norm_max=gmax, grad_norm_median=gmed)]))


def test_stress_divergence_read_from_validation_path():
    from m5knobs.figures import stress
    off = _run(46.9, -5.3, 300, [10, 9.9, 10, 30, 10, 101, 101])
    on = _run(50.3, 9.1, 450, [10, 9.9, 10.1, 29.0, 10, 9.98, 9.98])  # transient spike, recovered
    tag, text = stress.outcome(off, on, {"label": "no clear effect on M5"}, {"label": "both"})
    assert tag == "stabilized" and "blew up" in text and "46.9" in text
    both = _run(50.0, 5.0, 650, [10, 9.9, 44, 10, 96, 96])
    tag2, _ = stress.outcome(both, _run(49.0, 6.0, 200, [10, 9.9, 100, 96]), {"label": "underfit"}, None)
    assert tag2 == "diverged"
    tag3, _ = stress.outcome(_run(45.6, -5.5, 4100, [10.8, 9.9, 9.85]), _run(45.5, -5.6, 4100, [10.8, 9.9, 9.85]),
                                 {"label": "no clear effect on M5"}, {"label": "no clear effect on M5"})
    assert tag3 == "no effect"


def test_stress_non_nan_error_is_not_a_result(res):
    from m5knobs.figures import stress
    d = os.path.join(res, "runs", "st_lr3e2_noclip__s1")
    os.remove(os.path.join(d, "DONE"))
    open(os.path.join(d, "ERROR.txt"), "w").write("Traceback...\nRuntimeError: MPS backend out of memory\n")
    with pytest.raises(SystemExit, match="not a training result"):
        stress._load("st_lr3e2_noclip", res)


def test_what_moved_reports_bias_tradeoff_of_fix():
    from m5knobs.figures import cards
    v = {"label": "underfit", "wape_gap": 0.65, "abs_bias_gap_pts": -3.2,
         "insample_mae": {"hurt": True}, "corr": {"hurt": False}, "amp_ratio": {"hurt": False}, "abs_bias": {"hurt": False}}
    r = cards.what_moved(v, "250 steps", ms=dict(bias=-0.5), fs=dict(bias=-3.6))
    assert "trade-off" in r.lower() and "-0.5%" in r and "-3.6%" in r


def test_no_unsupported_max_steps_claim():
    from m5knobs.figures import cards
    assert all("max_steps" not in c["try_effect"] for c in cards.CARDS)


def test_f4_renders(res, tmp_path):
    from m5knobs.figures import architectures
    architectures.draw(results_dir=res, out_dir=str(tmp_path / "fig"), n_boot=50)
    _png_ok(str(tmp_path / "fig" / "f4_table.png"), min_w=1000)
    t = pd.read_csv(os.path.join(res, "f4_table.csv"))
    assert list(t["model"]) == ["LGBM (baseline)", "N-HiTS", "TCN"]
    assert t["p_vs_lgbm"].dropna().between(0, 1).all()


def _toy_data(n=400, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    x = np.log1p(rng.poisson(2.0, n) * (rng.random(n) < 0.5))
    y = rng.poisson(np.exp(-0.5 + 0.8 * x)).astype(float)
    y[rng.random(n) < 0.02] *= 30  # a few spikes, like promo weeks
    return pd.DataFrame({"x": x, "y": y})


def test_toy_clip_paths(tmp_path):
    import numpy as np
    from m5knobs import toy
    traj, surf = toy.main(steps=30, data=_toy_data(), results_dir=str(tmp_path))
    assert set(traj["run"]) == {"unclipped", "clipped"}
    assert (traj.groupby("run").size() == 31).all()
    step_len = traj.sort_values(["run", "step"]).groupby("run")[["a", "b"]].diff().pow(2).sum(axis=1).pow(0.5)
    mx = step_len.groupby(traj.sort_values(["run", "step"])["run"]).max()
    assert mx["clipped"] <= mx["unclipped"] + 1e-12
    assert np.isfinite(surf["loss"]).all()
    assert os.path.exists(os.path.join(str(tmp_path), "toy_paths.parquet"))


def test_f1_renders(tmp_path):
    from m5knobs.figures import toy as toy_fig
    from m5knobs import toy
    toy.main(steps=30, data=_toy_data(), results_dir=str(tmp_path))
    toy_fig.draw(results_dir=str(tmp_path), out_dir=str(tmp_path / "fig"))
    _png_ok(str(tmp_path / "fig" / "f1_toy.png"), min_w=1000)


NONE = "no clear effect on M5"
BAD_GRAMMAR = (", and and", " and and", "  ", " ,", "both.", "in in ", "was was", " ;")


def _clean(t):
    assert not any(b in t for b in BAD_GRAMMAR), t
    return t


def test_join_words():
    from m5knobs.figures.style import join_words
    assert [join_words([]), join_words(["a"]), join_words(["a", "b"]), join_words(["a", "b", "c"])] == ["", "a", "a and b", "a, b and c"]


def test_label_text_spells_out_both():
    from m5knobs.figures.style import label_text
    assert [label_text(x) for x in ("both", "underfit", "biased", NONE)] == \
        ["underfit and biased", "underfit", "biased", NONE]


def test_try_lines_make_no_data_or_cost_claims():
    from m5knobs.figures import cards
    for c in cards.CARDS:
        for k in ("try_effect", "try_none"):
            t = c[k].lower()
            bad = ("on m5", " here", "didn't", "shows where", "measurably", "cost", "cheap", "saves", "reasonable", "fine default",
                   "other likelihood", "poisson")
            assert not any(w in t for w in bad), (c["key"], k, c[k])


def _v(label, wg, bg, **hurt):
    h = {k: {"hurt": hurt.get(k, False)} for k in ("insample_mae", "corr", "amp_ratio", "abs_bias")}
    return dict(h, label=label, wape_gap=wg, abs_bias_gap_pts=bg)


def test_what_moved_no_effect_branches():
    from m5knobs.figures import cards
    ms, fs = dict(wape=45.12, bias=-0.6), dict(wape=45.27, bias=-3.6)
    assert _clean(cards.what_moved(_v(NONE, 0.05, 0.3), "x", ms, fs)) == \
        "Nothing moved beyond seed noise (WAPE Δ +0.05, |bias| Δ +0.3pt)."
    assert _clean(cards.what_moved(_v(NONE, -0.15, -3.0), "x", ms, fs)) == \
        "Moved beyond seed noise without meeting the verdict rule: |bias| was 3.0pt smaller with x (bias -0.6% vs -3.6%)."
    assert _clean(cards.what_moved(_v(NONE, -0.5, 2.0), "x", ms, fs)) == \
        ("Moved beyond seed noise without meeting the verdict rule: WAPE was 0.50 lower with x (45.12 vs 45.27) "
         "and |bias| was 2.0pt larger with x (bias -0.6% vs -3.6%).")


def test_what_moved_effect_lists_hurt_measures_with_and():
    from m5knobs.figures import cards
    r = cards.what_moved(_v("underfit", 1.3, 0.0, insample_mae=True, corr=True, amp_ratio=True), "input_size = 13")
    assert _clean(r) == ("With input_size = 13, these got worse: fit to the training history, week-to-week tracking and how much "
                         "the forecast moves, which went flatter (WAPE +1.30).")
    assert ") (" not in r


def test_card_verdict_text_is_the_one_way_card_verdict():
    from m5knobs.figures import cards
    v = dict(label="both", wape_gap=0.5, abs_bias_gap_pts=2.0, wape=[45.0, 44.5], bias=[-3.0, -1.0], moved="M.")
    t = cards.verdict_text(cards.CARDS[0], v)
    assert _clean(t) == ("Compared: `clip off` (without) vs `AutoClip, 10th percentile` (with). **Verdict on the 'without' run: "
                         "underfit and biased.** WAPE 45.00 → 44.50, bias -3.0% → -1.0%. M.")


def _V(labels, gaps):
    from m5knobs.figures import cards
    return {c["key"]: dict(label=l, wape_gap=g) for c, l, g in zip(cards.CARDS, labels, gaps)}


def test_card_claims_mixed_case_exact():
    from m5knobs.figures import cards
    c = cards.claims(_V([NONE, "underfit", "underfit", "biased", NONE], [0.05, 1.3, 0.65, 0.18, -0.15]))
    assert _clean(c["handful"]) == (
        "3 of 5 cards gave the wrong value a verdict of underfit or biased: Input window, Early stopping and Loss function. "
        "Largest WAPE gap among them: +1.30 (Input window). "
        "Verdict on the claim, read as 'for most of the tested settings, the wrong value measurably hurts': holds.")
    assert _clean(c["underfit"]) == (
        "2 of 5 cards gave the wrong value a verdict of underfit: Input window and Early stopping. "
        "Verdict on the claim, read as 'more than half of the cards are underfit': does not hold.")


def test_card_claims_none_and_one_and_all():
    from m5knobs.figures import cards
    c = cards.claims(_V([NONE] * 5, [-0.5, 0.0, 0.0, 0.0, 0.1]))
    assert _clean(c["handful"]) == ("0 of 5 cards gave the wrong value a verdict of underfit or biased. No card met the verdict "
                                    "rule. Verdict on the claim, read as 'for most of the tested settings, the wrong value "
                                    "measurably hurts': does not hold.")
    assert c["underfit"].startswith("0 of 5 cards gave the wrong value a verdict of underfit. Verdict")
    c1 = cards.claims(_V(["both", NONE, NONE, NONE, NONE], [0.4, 0.0, 0.0, 0.0, 0.9]))
    assert "1 of 5 cards gave the wrong value a verdict of underfit or biased: Gradient clipping. Largest WAPE gap among them: +0.40" \
        in _clean(c1["handful"]) and "underfit: Gradient clipping." in c1["underfit"]
    ca = cards.claims(_V(["underfit"] * 5, [0.3] * 5))
    assert ca["handful"].endswith(": holds.") and ca["underfit"].endswith(": holds.")


def _O(**kw):
    return {k: dict(outcome=o, label=lab) for k, (o, lab) in kw.items()}


def test_stress_conclusion_mixed_exact():
    from m5knobs.figures import stress
    O = _O(lr1e2=("stabilized", NONE), lr3e2=("diverged", "underfit"), nofilter=("no effect", NONE))
    assert _clean(stress.conclusion(NONE, O)) == "\n".join([
        "- At the recipe's settings, the card's verdict on clip off: no clear effect on M5.",
        "- Learning rate 10× (1e-2): clip off diverged, AutoClip did not.",
        "- Learning rate 30× (3e-2): both runs diverged.",
        "- Short-history windows let in: no measurable difference.",
        "- Verdict on the claim: **mixed** (recipe card: no clear effect on M5; "
        "in favour of clipping: clip off diverged under the 10× learning rate)."])


def test_stress_conclusion_holds_does_not_hold_and_against():
    from m5knobs.figures import stress
    calm = _O(lr1e2=("no effect", NONE), lr3e2=("diverged", NONE))
    assert _clean(stress.conclusion("both", calm)).endswith(
        "- Verdict on the claim: **holds** (recipe card: underfit and biased).")
    assert "the card's verdict on clip off: underfit and biased." in stress.conclusion("both", calm)
    assert _clean(stress.conclusion(NONE, calm)).endswith(
        "- Verdict on the claim: **does not hold** (recipe card: no clear effect on M5).")
    bad = _O(lr1e2=("clip worse", NONE), lr3e2=("clip failed", None), nofilter=("rescued", None))
    t = _clean(stress.conclusion("underfit", bad))
    assert "- Learning rate 10× (1e-2): neither diverged; the clipped forecast was worse." in t
    assert "- Learning rate 30× (3e-2): only AutoClip crashed." in t
    assert "- Short-history windows let in: clip off crashed, AutoClip finished." in t
    assert t.endswith("**mixed** (recipe card: underfit; in favour of clipping: clip off crashed with the short-history filter off; "
                      "against clipping: the clipped forecast was worse under the 10× learning rate and AutoClip crashed under "
                      "the 30× learning rate).")
    for tag in ("clip worse", "clip failed", "rescued", "stabilized"):
        assert tag not in t
    helped = _O(lr1e2=("helped", "both"), lr3e2=("clip diverged", NONE))
    t = _clean(stress.conclusion(NONE, helped))
    assert "- Learning rate 10× (1e-2): neither diverged; the verdict on clip off was underfit and biased." in t
    assert t.endswith("**mixed** (recipe card: no clear effect on M5; in favour of clipping: clip off scored worse under the 10× "
                      "learning rate; against clipping: AutoClip diverged under the 30× learning rate).")


def test_stress_describe_has_no_raw_tags():
    from m5knobs.figures import stress
    assert stress.describe(dict(outcome="clip worse", label=NONE)) == "neither diverged; the clipped forecast was worse"
    assert stress.describe(dict(outcome="helped", label="both")) == "neither diverged; the verdict on clip off was underfit and biased"
    assert stress.describe(dict(outcome="both failed", label=None)) == "both runs crashed"


def test_f4_headline_and_claim_text():
    from m5knobs.figures import architectures
    t = pd.DataFrame([dict(model="LGBM (baseline)", wape=45.0, gap_vs_lgbm=None, read="baseline"),
                      dict(model="N-HiTS", wape=45.5, gap_vs_lgbm=0.5, read="worse (not significant)"),
                      dict(model="TCN", wape=45.0, gap_vs_lgbm=0.0, read="within seed noise"),
                      dict(model="X", wape=44.0, gap_vs_lgbm=-1.0, read="better (significant)")])
    h = architectures.headline_text(t)
    assert _clean(h) == ("N-HiTS trailed LGBM by 0.50 WAPE (not significant), TCN was level with LGBM and X beat LGBM by 1.00 WAPE")
    assert _clean(architectures.claim_text(t)) == "Level with or better than LGBM on WAPE: TCN and X. Verdict on the claim: holds."
    assert _clean(architectures.claim_text(t.iloc[:2])) == \
        "No deep model was level with or better than LGBM on WAPE. Verdict on the claim: does not hold."


def test_card_claims_size_relative_to_with_run():
    from m5knobs.figures import cards
    V = _V([NONE, "underfit", NONE, NONE, NONE], [0.0, 1.3, 0.0, 0.0, 0.0])
    V["input_size"]["wape"] = [46.57, 45.27]
    assert "Largest WAPE gap among them: +1.30 (Input window, 2.9% of the 'with' run's WAPE)." in _clean(cards.claims(V)["handful"])


def test_stress_best_checkpoint_reported_when_clip_off_scored_better():
    from m5knobs.figures import stress
    O = {"lr1e2": dict(outcome="stabilized", label=NONE, off=46.9, on=50.3), "nofilter": dict(outcome="no effect", label=NONE)}
    t = _clean(stress.conclusion(NONE, O))
    assert "- Learning rate 10× (1e-2): clip off diverged, AutoClip did not, though clip off's best checkpoint, restored before it " \
           "diverged, scored better." in t
    assert t.endswith("in favour of clipping: clip off diverged under the 10× learning rate, though its best checkpoint, restored "
                      "before it diverged, scored better).")
    O["lr1e2"].update(off=50.3, on=46.9)
    assert "scored better" not in stress.conclusion(NONE, O)


def test_stress_outcome_text_names_runs():
    from m5knobs.figures import stress
    off = _run(46.9, -5.3, 300, [10, 9.9, 10, 30, 10, 101, 101], gmax=1000.0)
    on = _run(50.3, 9.1, 450, [10, 9.9, 10.1, 29.0, 10, 9.98, 9.98])
    _, text = stress.outcome(off, on, {"label": NONE}, None)
    assert "scored better: WAPE 46.9 clip off vs 50.3 AutoClip" in text and "median gradient norm" in text
    assert "typical" not in text


def test_stress_rules_table_covers_every_outcome():
    from m5knobs.figures import stress
    t = stress.rules_table()
    assert len(t) == len(stress.WHAT_HAPPENED) == len(stress.RULES)
    assert set(stress.RULES) == set(stress.WHAT_HAPPENED)
    assert (t["counts"] == "for clipping").sum() == len(stress.FOR_CLIPPING)
    assert (t["counts"] == "against clipping").sum() == len(stress.AGAINST_CLIPPING)
