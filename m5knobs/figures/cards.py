"""One "setting card" per NeuralForecast hyperparameter: the knob, the two runs compared, and the verdict rule on the wrong value.
Verdicts and "what moved" text are generated from the results; only the knob explanations and "try" lines are fixed text."""
import json
import os

from m5knobs import paths, results
from m5knobs.data import SEED_NOISE
from m5knobs.figures.style import join_words
from m5knobs.scoring import row_verdict

CARDS = [
    dict(key="clipping", name="Gradient clipping", action="Clip your gradients", knob="callbacks=[AutoClip(clip_percentile)]",
         source="Lightning callback from m5knobs.model, passed in the model's callbacks. Lightning's gradient_clip_val is the "
                "fixed-threshold alternative.",
         what="Before each weight update, if the gradient's total size is above a threshold, it is scaled down to that threshold; the "
              "direction is kept. AutoClip moves the threshold every step to the 10th percentile of the gradient sizes seen so far.",
         why="One promo week or launch spike in a batch can produce a very large Tweedie gradient and drag the whole fit with it.",
         mistake="f2_noclip", fix="f2_clip", m_label="clip off", f_label="AutoClip, 10th percentile", loss="Tweedie (rho 1.5)", diag="grad",
         try_effect="Turn it on for Tweedie: AutoClip at the 10th percentile, or a fixed gradient_clip_val.",
         try_none="Keep AutoClip on, and test with and without clipping again if you raise the learning rate."),
    dict(key="input_size", name="Input window", action="Show the model a full year", knob="input_size",
         source="NeuralForecast model argument: the length of history in each training window.",
         what="Each training example is input_size past weeks followed by the next h weeks to predict. The network only ever sees that "
              "window, so anything longer than it (a yearly peak, last year's holiday) is invisible.",
         why="Weekly retail demand repeats yearly. A 13-week window can't see last year, so the model falls back to the recent level.",
         mistake="f3_input13", fix="f3_fix", m_label="input_size = 13", f_label="input_size = 52", loss="Huber δ5", diag=None,
         try_effect="Start at one season (52 weekly, 12 monthly) or two, and shorten only if you have evidence.",
         try_none="Start at one season and shorten only with evidence from your own data."),
    dict(key="early_stopping", name="Early stopping", action="Give early stopping room", knob="early_stop_patience_steps × val_check_steps",
         source="NeuralForecast model arguments. Patience counts validation checks, so the window in optimizer steps is the product.",
         what="Every val_check_steps steps the model scores the held-out last h weeks. Training stops when that score hasn't improved "
              "for early_stop_patience_steps checks in a row, and the best weights are restored.",
         why="Validation loss on intermittent demand is noisy and often plateaus before it improves again; a short window quits early.",
         mistake="f3_patience250", fix="f3_fix_noguard", m_label="5 × 50 = 250 steps", f_label="20 × 50 = 1,000 steps", loss="Huber δ5",
         diag="loss",
         try_effect="Allow about 1,000 optimizer steps without improvement before stopping; a longer window can trade WMAPE for bias, "
                    "so check both.",
         try_none="Set the window to about 1,000 steps, and check validation loss for plateaus before shortening it."),
    dict(key="loss", name="Loss function", action="With a point loss, use Huber, not MAE", knob="loss",
         source="NeuralForecast model argument, e.g. loss=MAE() or loss=HuberLoss(delta=5). Computed on the scaled target.",
         what="MAE is minimized by the median of each forecast's distribution. Huber is squared for small errors and absolute for large "
              "ones, so it sits between the median and the mean; delta (in scaled units) sets where it switches.",
         why="On intermittent series the median week is often zero, so a median forecast under-calls total volume.",
         mistake="f3_mae", fix="f3_fix", m_label="loss = MAE()", f_label="loss = HuberLoss(delta=5)", loss="", diag="total",
         try_effect="Pick the loss by how you're scored: if total volume (bias) matters, avoid pure MAE; tune Huber's delta.",
         try_none="Check bias, not just WMAPE, when you pick the loss for your own data."),
    dict(key="scaler", name="Input scaling", action="Use RevIN scaling", knob="scaler_type",
         source="NeuralForecast model argument: how each window is normalized before the network sees it.",
         what="'revin' subtracts each window's mean and divides by its standard deviation, then reverses that on the output. "
              "'minmax' squeezes each window into [0, 1] using its min and max.",
         why="One spike sets a window's max, so MinMax squashes every other week toward zero. Note: Huber δ5 acts like MSE on [0, 1].",
         mistake="f3_minmax", fix="f3_fix", m_label="scaler_type = 'minmax'", f_label="scaler_type = 'revin'", loss="Huber δ5", diag=None,
         try_effect="Default to RevIN (or robust); avoid MinMax for spiky demand.",
         try_none="Compare RevIN, robust and MinMax on your own data before settling on one."),
]

NONE = "no clear effect on M5"
HURT_WORDS = {"insample_mae": "fit to the training history", "corr": "week-to-week tracking",
              "amp_ratio": "how much the forecast moves, which went flatter"}


def what_moved(v, m_label, ms=None, fs=None) -> str:
    """What changed between the two runs, in numbers. 'Nothing moved' only when both gaps are inside the seed-noise bars. The
    verdict rule only looks for harm from the wrong value, so a gap in its favour is stated here in words and numbers."""
    wg, bg = v["wape_gap"], v["abs_bias_gap_pts"]
    if v["label"] == NONE:
        if abs(wg) <= SEED_NOISE["wape"] and abs(bg) <= SEED_NOISE["bias"]:
            return f"Nothing moved beyond seed noise (WMAPE Δ {wg:+.2f}, |bias| Δ {bg:+.1f}pt)."
        both = ms is not None and fs is not None
        moved = []
        if abs(wg) > SEED_NOISE["wape"]:
            moved.append(f"WMAPE was {abs(wg):.2f} {'higher' if wg > 0 else 'lower'} with {m_label}"
                         + (f" ({ms['wape']:.2f} vs {fs['wape']:.2f})" if both else ""))
        if abs(bg) > SEED_NOISE["bias"]:
            moved.append(f"|bias| was {abs(bg):.1f}pt {'larger' if bg > 0 else 'smaller'} with {m_label}"
                         + (f" (bias {ms['bias']:+.1f}% vs {fs['bias']:+.1f}%)" if both else ""))
        return f"Moved beyond seed noise without meeting the verdict rule: {join_words(moved)}."
    trade = ""
    if bg < -SEED_NOISE["bias"]:
        trade = (f" Trade-off: bias moved from {ms['bias']:+.1f}% to {fs['bias']:+.1f}% with the fix (|bias| {-bg:.1f}pt "
                 "larger)." if ms and fs else f" Trade-off: |bias| was {-bg:.1f}pt smaller with {m_label}.")
    hurt = [HURT_WORDS[k] for k in HURT_WORDS if v[k]["hurt"]] if v["label"] in ("underfit", "both") else []
    parts = []
    if hurt:
        parts.append(f"With {m_label}, these got worse: " + join_words(hurt) + f" (WMAPE +{wg:.2f}).")
    if v["label"] in ("biased", "both"):
        parts.append(f"Total-volume bias was {bg:.1f}pt larger.")
    return " ".join(parts) + trade


def _claim_verdict(k, n) -> str:
    return "holds" if 2 * k > n else "mixed" if 2 * k == n else "does not hold"


def claims(V) -> dict:
    """Computed verdicts on two practitioner claims, from the one-way card verdicts (the label on each 'without' run, as on the
    figures): most settings matter (underfit, biased or both) and wrong settings mostly cause underfitting."""
    n = len(CARDS)
    eff = [c for c in CARDS if V[c["key"]]["label"] != NONE]
    under = [c for c in CARDS if V[c["key"]]["label"] in ("underfit", "both")]
    names = lambda cs: f": {join_words(c['name'] for c in cs)}." if cs else "."  # noqa: E731
    if eff:
        top = max(eff, key=lambda c: abs(V[c["key"]]["wape_gap"]))
        vt = V[top["key"]]
        rel = f", {100 * abs(vt['wape_gap']) / vt['wape'][1]:.1f}% of the 'with' run's WMAPE" if vt.get("wape") else ""
        size = f" Largest WMAPE gap among them: {vt['wape_gap']:+.2f} ({top['name']}{rel})."
    else:
        size = " No card met the verdict rule."
    return dict(
        handful=(f"{len(eff)} of {n} cards gave the wrong value a verdict of underfit or biased{names(eff)}{size} "
                 f"Verdict on the claim, read as 'for most of the tested settings, the wrong value measurably hurts': "
                 f"{_claim_verdict(len(eff), n)}."),
        underfit=(f"{len(under)} of {n} cards gave the wrong value a verdict of underfit{names(under)} "
                  f"Verdict on the claim, read as 'more than half of the cards are underfit': {_claim_verdict(len(under), n)}."))


def verdicts(results_dir=paths.RESULTS, n_boot=2000) -> dict:
    """The verdict rule on every card's 'without' run (row_verdict), written to results/card_verdicts.json for FINDINGS.md."""
    out = {}
    for card in CARDS:
        results.require([card["mistake"], card["fix"]], results_dir)
        m, f = results.load_run(card["mistake"], results_dir), results.load_run(card["fix"], results_dir)
        v = row_verdict(dict(ps=m["ps"], ins=m["ins"]), dict(ps=f["ps"], ins=f["ins"]), n_boot=n_boot)
        out[card["key"]] = dict(label=v["label"], wape_gap=v["wape_gap"], abs_bias_gap_pts=v["abs_bias_gap_pts"],
                                wape=[m["summary"]["wape"], f["summary"]["wape"]], bias=[m["summary"]["bias"], f["summary"]["bias"]],
                                moved=what_moved(v, card["m_label"], m["summary"], f["summary"]),
                                try_line=card["try_none"] if v["label"] == NONE else card["try_effect"])
    with open(os.path.join(results_dir, "card_verdicts.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out
