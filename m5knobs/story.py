"""Text and table helpers for notebooks/m5_deep_learning_story.ipynb: every number in its prose comes from results/ through here
or through an examples.Pick, so the notebook cells stay short."""
import numpy as np
import pandas as pd

from m5knobs import examples, paths, results, scoring, segments
from m5knobs.data import SEED_NOISE
from m5knobs.examples import CHANGE_FRAC, HELP_FRAC, HELP_MIN_Y, Pick
from m5knobs.figures.style import join_words

# (setting, arm, how it was fit): every run the story compares, in story order
SETTINGS = [
    ("N-HiTS Tweedie, clip off", "f2_noclip", "fit once"),
    ("N-HiTS Tweedie, AutoClip (the recipe)", "f2_clip", "fit once"),
    ("N-HiTS Huber, input_size=13", "f3_input13", "fit once"),
    ("N-HiTS Huber, patience 250 steps, no min-steps guard", "f3_patience250", "fit once"),
    ("N-HiTS Huber, patience 1,000 steps, no min-steps guard", "f3_fix_noguard", "fit once"),
    ("N-HiTS MAE loss", "f3_mae", "fit once"),
    ("N-HiTS Huber, MinMax scaler", "f3_minmax", "fit once"),
    ("N-HiTS Huber (pre-set recipe)", "f3_fix", "fit once"),
    ("N-HiTS Huber, best of 5 early-stopping settings (hindsight)", "es_p4000", "fit once"),
    ("N-HiTS Tweedie", "f4_nhits", "refit every origin"),
    ("TCN Tweedie", "f4_tcn", "refit every origin"),
    ("LightGBM", "f4_lgbm", "refit every origin"),
]
DL = "N-HiTS Huber"
STORY_KINDS = ("volume quartile", "zero weeks")
DETAIL_KINDS = ("department", "origin")


def metrics(arm: str, results_dir: str = paths.RESULTS) -> dict:
    """WMAPE, signed bias and |bias| (all %) over every series, week and origin, from the repo's scoring."""
    s = scoring.summarize(results.load_run(arm, results_dir)["ps"])
    return dict(wmape=s["wape"], bias=s["bias"], abs_bias=abs(s["bias"]))


def _side(b: float) -> str:
    return "over-forecast" if b > 0 else "under-forecast"


def compare_line(arm_wrong: str, arm_right: str, labels, results_dir: str = paths.RESULTS) -> str:
    """Both settings' WMAPE and |bias| over all series, and the differences against the seed-noise bars."""
    w, r = metrics(arm_wrong, results_dir), metrics(arm_right, results_dir)
    one = lambda lab, m: f"**{lab}**: WMAPE {m['wmape']:.2f}, |bias| {m['abs_bias']:.1f}% ({_side(m['bias'])})"  # noqa: E731
    return (f"All series: {one(labels[0], w)} · {one(labels[1], r)}. Going from {labels[0]} to {labels[1]}: WMAPE "
            f"{r['wmape'] - w['wmape']:+.2f}, |bias| {r['abs_bias'] - w['abs_bias']:+.1f}pt (seed noise ±{SEED_NOISE['wape']} "
            f"WMAPE, ±{SEED_NOISE['bias']:.0f}pt |bias|). " + bias_caveat(w, r, labels, "all series")).strip()


def bias_caveat(w: dict, r: dict, labels, scope: str, verdict: bool = True) -> str:
    """When the mistaken setting has the smaller |bias|, say so and why it can happen. w, r: dicts with wmape and bias (%).
    Empty otherwise."""
    if abs(w["bias"]) >= abs(r["bias"]):
        return ""
    if (w["bias"] > 0) == (r["bias"] > 0):
        why = (f"Both {_side(r['bias'])} total volume, and the {labels[0]} forecasts happen to sit "
               f"{'higher' if w['bias'] > r['bias'] else 'lower'}, which lands closer to actual sales in total.")
    else:
        why = (f"They miss in opposite directions ({labels[0]} {_side(w['bias'])}s by {abs(w['bias']):.1f}%, {labels[1]} "
               f"{_side(r['bias'])}s by {abs(r['bias']):.1f}%), and the {labels[0]} miss is the smaller one.")
    tail = (f" Its week-by-week error is still larger (WMAPE {w['wmape']:.2f} vs {r['wmape']:.2f}): a model can fit worse and "
            "still land closer in total by chance" + (", which is why the verdict looks at fit and WMAPE, and |bias| separately."
                                                     if verdict else ".")
            if w["wmape"] > r["wmape"] else
            f" Its WMAPE is not worse either ({w['wmape']:.2f} vs {r['wmape']:.2f}).")
    return f"On {scope}, {labels[0]} has the smaller |bias| ({abs(w['bias']):.1f}% vs {abs(r['bias']):.1f}%). {why}{tail}"


def typical_line(pick: Pick, labels) -> str:
    """How typical the example is: its rank under the rule and how many series show the same pattern."""
    if pick.rule.startswith("median") or "typical series instead" in pick.rule:
        lead = (f"The example ({pick.uid}) is a typical series, not an extreme one: the median forecast change "
                f"(rank {pick.rank:,} of {pick.n_candidates:,}).")
        if "typical series instead" in pick.rule:
            lead = "No series matched the rule, so the figure shows a typical one. " + lead
    else:
        lead = f"The example ({pick.uid}) is the rule's top case (rank {pick.rank} of {pick.n_candidates:,} candidates)."
    if pick.n_hurt > pick.n_helped:
        mixed = f" More series got clearly worse than better with {labels[1]} on their own error."
    elif pick.n_hurt >= pick.n_helped / 2:
        mixed = " Large per-series changes go both ways, so the all-series result is a net effect, not a uniform one."
    else:
        mixed = ""
    base = pick.n_pattern_base or pick.n_population
    what = pick.pattern_base_text or "series both runs forecast"
    pattern = pick.text(pick.pattern_text, labels) or "show the pattern"
    return (f"{lead} {pick.n_pattern:,} of the {base:,} {what} ({pick.pattern_share:.0%}) {pattern}. "
            f"Of the {pick.n_counted:,} series selling more than {HELP_MIN_Y} units, {pick.n_helped:,} got more than "
            f"{HELP_FRAC:.0%} better with {labels[1]} and {pick.n_hurt:,} got worse by as much; on {pick.n_changed:,} of them the "
            f"two forecasts differ by more than {CHANGE_FRAC:.0%}.{mixed}")


def example_line(pick: Pick, labels, favours: str = "right") -> str:
    """The example's own WMAPE and |bias|. favours: the side the rule picks the example for ("right" = labels[1], or
    "wrong" = labels[0]); says so when the other side is better on either metric on this series."""
    w, r = (pick.wmape_wrong, pick.absbias_wrong), (pick.wmape_right, pick.absbias_right)
    o_w, o_b = (w if favours == "right" else r)
    f_w, f_b = (r if favours == "right" else w)
    other = labels[0] if favours == "right" else labels[1]
    notes = ([f"{other} has the smaller |bias|"] if o_b < f_b else []) + ([f"{other} has the lower WMAPE"] if o_w < f_w else [])
    note = (f" On this series {join_words(notes)}: the example was chosen by the rule above, which is about a different "
            "measure." if notes else "")
    return (f"On {pick.uid}: {labels[0]} WMAPE {pick.wmape_wrong:.1f}, |bias| {pick.absbias_wrong:.1f}%; "
            f"{labels[1]} WMAPE {pick.wmape_right:.1f}, |bias| {pick.absbias_right:.1f}%.{note}")


def summary_table(results_dir: str = paths.RESULTS) -> pd.DataFrame:
    """Every model and setting in the story: WMAPE, signed bias and |bias| over all series, weeks and origins."""
    rows = []
    for name, arm, fit in SETTINGS:
        m = metrics(arm, results_dir)
        rows.append({"model / setting": name, "arm": arm, "fit": fit, "WMAPE": round(m["wmape"], 2),
                     "bias %": round(m["bias"], 1), "|bias| %": round(m["abs_bias"], 1)})
    return pd.DataFrame(rows)


ALPHA = 0.05  # per-segment paired bootstrap threshold


def segment_gaps(table: pd.DataFrame, model: str = DL) -> pd.DataFrame:
    """One row per segment: model minus LightGBM on WMAPE and |bias| (points), the per-segment paired bootstrap p of each
    gap, and which side each metric favours at p < ALPHA within the segment ("no clear difference" otherwise)."""
    a = table[table.model == model].set_index(["segment_kind", "segment"])
    b = table[table.model == segments.BASE].set_index(["segment_kind", "segment"])
    g = pd.DataFrame({"vol_share": b["vol_share"], "wmape_dl": a["wape"], "wmape_lgbm": b["wape"], "d_wmape": a["wape"] - b["wape"],
                      "p_wmape": a["p_wape"], "absbias_dl": a["abs_bias"], "absbias_lgbm": b["abs_bias"],
                      "d_absbias": a["abs_bias"] - b["abs_bias"], "p_absbias": a["p_abs_bias"]})
    read = lambda d, p: "deep learning" if p < ALPHA and d < 0 else "LightGBM" if p < ALPHA and d > 0 else "no clear difference"  # noqa: E731
    g["wmape_read"] = [read(d, p) for d, p in zip(g["d_wmape"], g["p_wmape"])]
    g["bias_read"] = [read(d, p) for d, p in zip(g["d_absbias"], g["p_absbias"])]
    return g.reset_index()


SERIES_KINDS = ("volume quartile", "zero weeks", "department")


def lightgbm_lead(table: pd.DataFrame, kinds=SERIES_KINDS, model: str = DL):
    """The series segment (not origin) where LightGBM's WMAPE lead over the model is largest among leads with p < ALPHA
    within the segment; None if there is none."""
    g = segment_gaps(table, model)
    g = g[g.segment_kind.isin(kinds) & (g.wmape_read == "LightGBM")]
    return None if g.empty else g.sort_values("d_wmape", ascending=False).iloc[0].to_dict()


def dl_lead(table: pd.DataFrame, kinds=SERIES_KINDS, model: str = DL):
    """Mirror of lightgbm_lead: the series segment where the model's WMAPE lead over LightGBM is largest, p < ALPHA."""
    g = segment_gaps(table, model)
    g = g[g.segment_kind.isin(kinds) & (g.wmape_read == "deep learning")]
    return None if g.empty else g.sort_values("d_wmape").iloc[0].to_dict()


def _name(r) -> str:
    fmt = {"volume quartile": "volume {}", "zero weeks": "{} zero weeks", "department": "{}", "origin": "origin {}"}
    return f"{fmt[r['segment_kind']].format(r['segment'])} ({r['vol_share']:.0%} of volume)"


def where_bullets(table: pd.DataFrame, model: str = DL) -> list:
    """Markdown bullets: the segments where the model beats or loses to LightGBM on WMAPE and on |bias|, at p < ALPHA within
    the segment (paired bootstrap over its series; seed noise is not in that p)."""
    g = segment_gaps(table, model)
    out = []
    for col, read, metric, unit in (("d_wmape", "wmape_read", "WMAPE", ""), ("d_absbias", "bias_read", "|bias|", "pt")):
        for side, verb in (("deep learning", f"{model} beats LightGBM"), ("LightGBM", "LightGBM beats " + model)):
            s = g[g[read] == side]
            if s.empty:
                out.append(f"{verb} on {metric} at p < {ALPHA} within the segment: in no segment.")
                continue
            items = [f"{_name(r)}, {r[col]:+.1f}{unit}" for _, r in s.sort_values(col, key=abs, ascending=False).iterrows()]
            out.append(f"{verb} on {metric} at p < {ALPHA} within the segment: {join_words(items)}.")
    ties = g[g.wmape_read == "no clear difference"]
    out.append(f"No clear WMAPE difference (p ≥ {ALPHA}) in {len(ties)} of {len(g)} segments: "
               f"{join_words(_name(r) for _, r in ties.iterrows())}.")
    small = g[(g.wmape_read != "no clear difference") & (g.d_wmape.abs() <= SEED_NOISE["wape"])]
    if len(small):
        out.append(f"Of the clear WMAPE differences, {join_words(_name(r) for _, r in small.iterrows())} "
                   f"{'is' if len(small) == 1 else 'are'} within the all-series seed-noise bar (±{SEED_NOISE['wape']}); the "
                   "segment p covers series-to-series variation, not seed-to-seed, so treat those as likely ties.")
    return out


def group_metrics(t: pd.DataFrame, mask) -> dict:
    """WMAPE and bias (%) of both forecasts in a pair_table over the series in mask, the group's share of volume, and the
    share of its series where the second forecast has the lower error."""
    m = examples._mask(t, mask)
    g = t[m]
    y = g["y"].sum()
    return dict(n=len(g), vol_share=y / t["y"].sum(), wmape_wrong=100 * g["err_wrong"].sum() / y,
                wmape_right=100 * g["err_right"].sum() / y, bias_wrong=100 * (g["f_wrong"].sum() / y - 1),
                bias_right=100 * (g["f_right"].sum() / y - 1), share_right_better=float((g["gain"] > 0).mean()))


def group_line(t: pd.DataFrame, mask, labels, name: str) -> str:
    g = group_metrics(t, mask)
    return (f"{name}, counting only the series and weeks both models forecast (so the numbers differ slightly from the "
            f"segment table): {g['n']:,} series, {g['vol_share']:.0%} of volume. {labels[0]} WMAPE {g['wmape_wrong']:.2f}, |bias| "
            f"{abs(g['bias_wrong']):.1f}% ({_side(g['bias_wrong'])}); {labels[1]} WMAPE {g['wmape_right']:.2f}, |bias| "
            f"{abs(g['bias_right']):.1f}% ({_side(g['bias_right'])}). Series by series it is mixed: {labels[1]} has the lower "
            f"error on {g['share_right_better']:.0%} of them. " + _split(g, labels)
            + bias_caveat(dict(wmape=g["wmape_wrong"], bias=g["bias_wrong"]), dict(wmape=g["wmape_right"], bias=g["bias_right"]),
                          labels, f"these {g['n']:,} series", verdict=False)).strip()


def _split(g: dict, labels) -> str:
    """Which side each metric favours beyond seed noise, in words (ties named)."""
    dw = g["wmape_right"] - g["wmape_wrong"]
    db = abs(g["bias_right"]) - abs(g["bias_wrong"])
    side = lambda d, noise: labels[1] if d < -noise else labels[0] if d > noise else None  # noqa: E731
    sw, sb = side(dw, SEED_NOISE["wape"]), side(db, SEED_NOISE["bias"])
    w = f"{sw} has the lower WMAPE ({dw:+.2f})" if sw else f"WMAPE is a tie (within ±{SEED_NOISE['wape']}: {dw:+.2f})"
    b = f"{sb} has the smaller |bias| ({db:+.1f}pt)" if sb else f"|bias| is a tie (within ±{SEED_NOISE['bias']:.0f}pt)"
    lead = "The result splits by metric: " if sw and sb and sw != sb else "By metric: "
    return f"{lead}{w}; {b} (differences are {labels[1]} minus {labels[0]}). "


def slow_seller_table(results_dir: str = paths.RESULTS, models: dict = None, rate: float = 1.0) -> tuple:
    """Series that sold under `rate` a week over the year before the first origin: (count, share of volume, table of
    bias, median weekly forecast and share of weekly forecasts under 0.9 units per model)."""
    models = models or segments.MODELS
    seg = segments.series_segments(results_dir)
    slow = set(seg.loc[seg["mean_hist"] < rate, "unique_id"])
    rows, vol = [], None
    for name, arm in models.items():
        r = results.load_run(arm, results_dir)
        ps, fc = r["ps"][r["ps"].unique_id.isin(slow)], r["fc"][r["fc"].unique_id.isin(slow)]
        vol = ps["y_sum"].sum() / r["ps"]["y_sum"].sum()
        rows.append({"model": name, "WMAPE on these series": round(100 * ps["abs_err"].sum() / ps["y_sum"].sum(), 1),
                     "bias on these series %": round(100 * (ps["yhat_sum"].sum() / ps["y_sum"].sum() - 1), 1),
                     "median weekly forecast": round(float(fc["yhat"].median()), 2),
                     "weekly forecasts under 0.9 units %": round(100 * float((fc["yhat"] < 0.9).mean()), 1)})
    return len(slow), vol, pd.DataFrame(rows).set_index("model")


def gap_table(g: pd.DataFrame) -> pd.DataFrame:
    """segments.gap_rows output, rounded, for display."""
    pf = lambda p: "" if pd.isna(p) else "<0.001" if p < 0.001 else f"{p:.3f}"  # noqa: E731
    return pd.DataFrame({"model": g["model"], "arm": g["arm"], "WMAPE": g["wape"].round(2), "bias %": g["bias"].round(1),
                         "|bias| %": g["abs_bias"].round(1), "Δ WMAPE vs LightGBM": g["gap"].round(2), "p (WMAPE)": g["p"].map(pf),
                         "Δ |bias| vs LightGBM": g["abs_bias_gap"].round(1), "p (|bias|)": g["p_abs_bias"].map(pf)})


def claim4(g: pd.DataFrame) -> str:
    """segments.claim_text on the N-HiTS Huber row of segments.gap_rows (the same call FINDINGS.md makes)."""
    gi = g.set_index("model")
    h, b = gi.loc[DL], gi.loc[segments.BASE]
    return segments.claim_text(dict(wape_a=h.wape, wape_b=b.wape, gap=h.gap, p=h.p, bias_a=h.bias, bias_b=b.bias,
                                    p_abs_bias=h.p_abs_bias))


def concentration(t: pd.DataFrame, mask, k: int = 10) -> dict:
    """How much of the first arm's error advantage inside mask comes from its k largest per-series advantages: their share
    of the total advantage, and the first arm's WMAPE lead (points) without them."""
    g = t[examples._mask(t, mask)]
    adv = (g["err_right"] - g["err_wrong"]).sort_values(ascending=False)
    top, y = adv.head(k).sum(), g["y"].sum()
    return dict(k=k, share=top / adv.sum() if adv.sum() > 0 else float("nan"), lead=100 * adv.sum() / y,
                lead_without=100 * (adv.sum() - top) / (y - g.loc[adv.head(k).index, "y"].sum()))


def val_spikes(results_dir: str = paths.RESULTS) -> dict:
    """Per stressor and run (off / on): largest validation loss divided by the best one, and whether the run recovered
    (did not diverge under stress.diverged: last validation loss at most 2x the best, no crash)."""
    from m5knobs.experiments import STRESSORS
    from m5knobs.figures import stress
    out = {}
    for k in STRESSORS:
        out[k] = {}
        for side, arm in (("off", f"st_{k}_noclip"), ("on", f"st_{k}_clip")):
            run = stress._load(arm, results_dir)
            tr = run.get("trace")
            v = tr["val_loss"][np.isfinite(tr["val_loss"])] if tr is not None else pd.Series(dtype=float)
            out[k][side] = float(v.max() / v.min()) if len(v) else float("nan")
            out[k][side + "_recovered"] = not stress.diverged(run)
    return out


SPIKE = 2.0  # a validation-loss peak above 2x the run's best counts as a spike


def spike_text(sp: dict, names: dict) -> tuple:
    """(sentence naming the runs whose validation loss spiked above SPIKE x best and whether they recovered,
    sentence on what that means for the clipping claim, or '' when no AutoClip run spiked more than its clip-off pair)."""
    parts, on_worse = [], []
    for k, v in sp.items():
        spiked = [(side, lab) for side, lab in (("off", "clip off"), ("on", "AutoClip")) if v[side] > SPIKE]
        if not spiked:
            parts.append(f"{names[k]}: no spike in either run")
            continue
        parts.append(f"{names[k]}: " + join_words(
            f"the {lab} run spiked to {v[side]:,.1f}× its best and " + ("recovered" if v[side + '_recovered'] else "did not recover")
            for side, lab in spiked))
        if v["on"] > SPIKE and v["on"] > v["off"]:
            on_worse.append(names[k])
    first = f"Validation-loss spikes (peak above {SPIKE:g}× the run's best): " + "; ".join(parts) + "."
    second = ""
    if on_worse:
        second = (f"The larger spikes were in the AutoClip runs ({join_words(on_worse)}), not the unclipped ones. So the "
                  "stress test does not show clipping preventing instability; clipping's case here rests on the recipe "
                  "verdict and on where each run ended (WMAPE), not on calmer training.")
    return first, second


segment_name = _name  # public name for the notebook: "volume Q4 fastest (69% of volume)"
