"""Clipping stress test: Tweedie N-HiTS pushed until gradients misbehave (stressors fixed in experiments.STRESSORS), clip off vs AutoClip.
A run that crashed (e.g. NaN loss) is shown as a result. Reads results/ only."""
import json
import os

import numpy as np
import pandas as pd

from m5knobs import paths, results
from m5knobs.experiments import STRESSORS
from m5knobs.figures import style
from m5knobs.figures.style import COLORS, join_words, label_text, plt
from m5knobs.scoring import row_verdict

LABELS = {"lr1e2": "Learning rate 10× (1e-2)", "lr3e2": "Learning rate 30× (3e-2)",
          "nofilter": "Short-history windows let in\n(training_data_availability_threshold = [0, 0])"}


def _load(arm, results_dir):
    d = os.path.join(results_dir, "runs", f"{arm}__s1")
    if os.path.exists(os.path.join(d, "DONE")):
        return dict(ok=True, **results.load_run(arm, results_dir))
    if os.path.exists(os.path.join(d, "ERROR.txt")):
        err = [ln for ln in open(os.path.join(d, "ERROR.txt")).read().strip().splitlines() if ln.strip()][-1]
        if "nan" not in err.lower():
            raise SystemExit(f"{arm}__s1 failed for a reason that is not a training result ({err[:80]}); rerun it")
        tr = pd.read_parquet(os.path.join(d, "trace.parquet")) if os.path.exists(os.path.join(d, "trace.parquet")) else None
        return dict(ok=False, error=err.split(":", 1)[-1].strip()[:80], trace=tr)
    raise SystemExit(f"missing results: {arm}__s1 (run scripts/run_experiments.py --group stress first)")


def _fs(run):
    return (run.get("summary", {}).get("fold_stats") or [{}])[0]


def diverged(run) -> bool:
    """Crashed, or validation loss ended above 2x its best (a spike it recovered from doesn't count). Rule set after seeing results."""
    if not run["ok"]:
        return True
    tr = run.get("trace")
    if tr is None:
        return False
    v = tr.dropna(subset=["val_loss"])
    v = v[np.isfinite(v["val_loss"])].reset_index(drop=True)
    if v.empty:
        return False
    return bool(v["val_loss"].iloc[-1] > 2 * v["val_loss"].min())


def outcome(off, on, v, v_rev=None):
    """(tag, plain-English line) for one stressor. v: clip-off judged as the mistake; v_rev: clip-on judged as the mistake."""
    if not off["ok"] and on["ok"]:
        return "rescued", "clipping rescued training (without it, training crashed with a NaN loss)"
    if not off["ok"] and not on["ok"]:
        return "both failed", "training crashed (NaN loss) with and without clipping"
    if off["ok"] and not on["ok"]:
        return "clip failed", "training crashed (NaN loss) only with clipping"
    d_off, d_on = diverged(off), diverged(on)
    wo, wn = off["summary"]["wape"], on["summary"]["wape"]
    if d_off and d_on:
        tag, text = "diverged", ("training diverged with and without clipping; the WAPE gap reflects which early checkpoint was "
                                 "restored, not clipping")
    elif d_off:
        vo = off["trace"]["val_loss"]
        tag = "stabilized"
        text = (f"clipping kept training stable: without it, validation loss blew up to {np.nanmax(vo[np.isfinite(vo)]):.0f} "
                f"(best {np.nanmin(vo):.1f})")
        text += (f"; even so, the unclipped run's best checkpoint (restored before it diverged) scored better: WAPE {wo:.1f} clip off "
                 f"vs {wn:.1f} AutoClip" if wo < wn else f", and the clipped run also scored better (WAPE {wn:.1f} AutoClip vs {wo:.1f} clip off)")
    elif d_on:
        tag, text = "clip diverged", "training diverged only with clipping"
    elif v["label"] != "no clear effect on M5":
        tag, text = "helped", {"underfit": "clipping improved the fit", "biased": "clipping removed bias",
                               "both": "clipping improved the fit and removed bias"}[v["label"]]
    elif v_rev is not None and v_rev["label"] != "no clear effect on M5":
        tag, text = "clip worse", "clipping made the forecast worse"
    else:
        tag, text = "no effect", "no measurable difference"
    fo, fn = _fs(off), _fs(on)
    if fo.get("grad_norm_median") and fn.get("grad_norm_median"):
        ro, rn = fo["grad_norm_max"] / fo["grad_norm_median"], fn["grad_norm_max"] / fn["grad_norm_median"]
        if ro > 2 * rn:
            text += (f"; clipping shrank the largest gradient spike from {ro:,.0f}× to {rn:,.0f}× the median gradient norm (norms are "
                     "recorded before clipping)")
    return tag, text


def headline(outcomes) -> str:
    n = len(outcomes)
    c = {}
    for r in outcomes.values():
        c[r["outcome"]] = c.get(r["outcome"], 0) + 1
    words = [("stabilized", "clipping prevented divergence in {}"), ("rescued", "clipping rescued a crashed run in {}"),
             ("helped", "clipping helped in {}"), ("diverged", "training diverged either way in {}"), ("both failed", "both crashed in {}"),
             ("clip diverged", "clipping diverged in {}"), ("clip failed", "clipping crashed in {}"), ("clip worse", "clipping hurt in {}"),
             ("no effect", "no difference in {}")]
    parts = [w.format(c[k]) for k, w in words if c.get(k)]
    return f"Stress test ({n} setups): " + join_words(parts)


NONE = "no clear effect on M5"
FOR_CLIPPING = ("stabilized", "rescued", "helped")
AGAINST_CLIPPING = ("clip diverged", "clip failed", "clip worse")
AS_EVIDENCE = {"stabilized": "clip off diverged", "rescued": "clip off crashed", "helped": "clip off scored worse",
               "clip diverged": "AutoClip diverged", "clip failed": "AutoClip crashed", "clip worse": "the clipped forecast was worse"}
WHERE = {"lr1e2": "under the 10× learning rate", "lr3e2": "under the 30× learning rate", "nofilter": "with the short-history filter off"}
WHAT_HAPPENED = {"stabilized": "clip off diverged, AutoClip did not", "rescued": "clip off crashed, AutoClip finished",
                 "helped": "neither diverged; the verdict on clip off was {label}", "diverged": "both runs diverged",
                 "both failed": "both runs crashed", "clip diverged": "only AutoClip diverged", "clip failed": "only AutoClip crashed",
                 "clip worse": "neither diverged; the clipped forecast was worse", "no effect": "no measurable difference"}


RULES = {"stabilized": "clip off diverged and AutoClip did not",
         "rescued": "clip off crashed (NaN loss) and AutoClip finished",
         "helped": "neither run diverged, and the verdict rule (notebook 01) gives clip off underfit, biased or both",
         "diverged": "both runs diverged",
         "both failed": "both runs crashed",
         "clip diverged": "only AutoClip diverged",
         "clip failed": "only AutoClip crashed",
         "clip worse": "neither run diverged, clip off shows no clear effect, and the same rule applied with AutoClip as the wrong "
                       "value gives underfit, biased or both",
         "no effect": "neither run diverged and the rule finds no effect either way"}
BEST_CKPT = "though clip off's best checkpoint, restored before it diverged, scored better"


def _off_better(o) -> bool:
    return o["outcome"] == "stabilized" and o.get("off") is not None and o.get("on") is not None and o["off"] < o["on"]


def rules_table() -> pd.DataFrame:
    """Each stressor outcome, when it applies, and which side of the clipping claim it counts for."""
    side = lambda t: "for clipping" if t in FOR_CLIPPING else "against clipping" if t in AGAINST_CLIPPING else "neither"  # noqa: E731
    return pd.DataFrame([dict(outcome=WHAT_HAPPENED[t].format(label="underfit, biased or both"), applies_when=RULES[t], counts=side(t))
                         for t in WHAT_HAPPENED])


def describe(o) -> str:
    """One stressor's outcome (an entry of stress_outcomes.json) in plain words."""
    return WHAT_HAPPENED[o["outcome"]].format(label=label_text(o.get("label") or NONE)) + (f", {BEST_CKPT}" if _off_better(o) else "")


def conclusion(label, outcomes) -> str:
    """Markdown bullets: the verdict on 'clipping is non-negotiable for Tweedie', from existing labels only (label: the clipping
    card's verdict on clip off; outcomes: stress_outcomes.json tags). Holds if clip off is underfit/biased at the recipe's settings
    and no stressor went against clipping; does not hold if neither the recipe card nor any stressor favours clipping; else mixed."""
    lines = [f"- At the recipe's settings, the card's verdict on clip off: {label_text(label)}."]
    for k, o in outcomes.items():
        lines.append(f"- {LABELS[k].splitlines()[0] if k in LABELS else k}: {describe(o)}.")
    pro = [f"{AS_EVIDENCE[o['outcome']]} {WHERE.get(k, 'in ' + k)}" + (f", though its best checkpoint, restored before it diverged, "
           "scored better" if _off_better(o) else "") for k, o in outcomes.items() if o["outcome"] in FOR_CLIPPING]
    con = [f"{AS_EVIDENCE[o['outcome']]} {WHERE.get(k, 'in ' + k)}" for k, o in outcomes.items() if o["outcome"] in AGAINST_CLIPPING]
    if label != NONE and not con:
        verdict = "holds"
    elif label == NONE and not pro:
        verdict = "does not hold"
    else:
        verdict = "mixed"
    why = [f"recipe card: {label_text(label)}"] + ([f"in favour of clipping: {join_words(pro)}"] if pro else []) + \
          ([f"against clipping: {join_words(con)}"] if con else [])
    lines.append(f"- Verdict on the claim: **{verdict}** ({'; '.join(why)}).")
    return "\n".join(lines)


def _grad(ax, off, on):
    for run, side in ((off, "mistake"), (on, "fix")):
        tr = run.get("trace")
        if tr is None or tr.empty:
            continue
        g = tr["grad_norm"].replace([np.inf, -np.inf], np.nan)
        ax.plot(tr["step"], g, color=COLORS[side], lw=0.3, alpha=0.3)
        ax.plot(tr["step"], g.rolling(50, min_periods=1, center=True).median(), color=COLORS[side], lw=1.3)
        bad = tr["step"][~np.isfinite(tr["grad_norm"])]
        if len(bad):
            ax.axvline(bad.iloc[0], color=COLORS[side], lw=0.8, ls=":")
            ax.text(bad.iloc[0], 0.95, " non-finite", transform=ax.get_xaxis_transform(), color=COLORS[side], fontsize=6.5, va="top")
        if side == "fix" and tr["clip_thr"].notna().any():
            ax.plot(tr["step"], tr["clip_thr"], color=COLORS["actual"], lw=0.8, ls=(0, (4, 2)))
        if not run["ok"]:
            ax.plot([tr["step"].iloc[-1]], [np.nanmax(g) if np.isfinite(np.nanmax(g)) else 1], "X", color=COLORS[side], ms=7)
    ax.set_yscale("log")
    ax.set_ylabel("gradient size", fontsize=7)


def _val(ax, off, on):
    for run, side in ((off, "mistake"), (on, "fix")):
        tr = run.get("trace")
        if tr is None:
            continue
        v = tr.dropna(subset=["val_loss"])
        v = v[np.isfinite(v["val_loss"])]
        if len(v):
            ax.plot(v["step"], v["val_loss"], "o-", color=COLORS[side], ms=1.6, lw=1.0)
    ax.set_ylabel("validation loss", fontsize=7)


def _nums(ax, off, on, text):
    ax.set_axis_off()
    for y, run, side, lab in ((0.86, off, "mistake", "clip off"), (0.56, on, "fix", "AutoClip")):
        s = (f"WAPE {run['summary']['wape']:.1f} · bias {run['summary']['bias']:+.1f}%" if run["ok"] else f"crashed: {run['error']}")
        ax.text(0, y, lab, transform=ax.transAxes, fontsize=7.5, color=COLORS[side], fontweight="bold", va="top")
        ax.text(0, y - 0.11, s, transform=ax.transAxes, fontsize=7.2, color="#222222", va="top")
    import textwrap
    ax.text(0, 0.24, "\n".join(textwrap.wrap(text, 44)), transform=ax.transAxes, fontsize=6.8, fontweight="bold", va="top",
            linespacing=1.25)


def draw(results_dir=paths.RESULTS, out_dir=paths.FIGURES, n_boot=2000):
    style.apply()
    keys = list(STRESSORS)
    fig, axes = plt.subplots(len(keys), 3, figsize=(style.WIDTH, 1.75 * len(keys) + 1.3),
                             gridspec_kw=dict(width_ratios=[1.25, 1.1, 0.9], hspace=0.65, wspace=0.32, left=0.1, right=0.99, top=0.83))
    outcomes = {}
    for i, k in enumerate(keys):
        off, on = _load(f"st_{k}_noclip", results_dir), _load(f"st_{k}_clip", results_dir)
        both = off["ok"] and on["ok"]
        v = row_verdict(dict(ps=off["ps"], ins=off["ins"]), dict(ps=on["ps"], ins=on["ins"]), n_boot=n_boot) if both else {"label": None}
        v_rev = (row_verdict(dict(ps=on["ps"], ins=on["ins"]), dict(ps=off["ps"], ins=off["ins"]), n_boot=n_boot)
                 if both and v["label"] == "no clear effect on M5" else None)
        tag, text = outcome(off, on, v, v_rev)
        outcomes[k] = dict(outcome=tag, text=text, label=v["label"], off=off.get("summary", {}).get("wape"), on=on.get("summary", {}).get("wape"))
        _grad(axes[i, 0], off, on)
        _val(axes[i, 1], off, on)
        _nums(axes[i, 2], off, on, text)
        axes[i, 0].set_title(LABELS[k], loc="left", fontsize=8.3, fontweight="bold")
        for a in axes[i, :2]:
            a.set_xlabel("optimizer step", fontsize=6.5)
    title = headline(outcomes)
    n = results.actuals(results_dir)["unique_id"].nunique()
    fig.text(0.01, 0.985, title, fontsize=13.5, fontweight="bold", va="top")
    fig.text(0.01, 0.94, f"STRESS TEST, not the recipe. N-HiTS, Tweedie loss, {n:,} M5 weekly series, single seed. Each row pushes one setting\n"
             "past the recipe; red = clip off, blue = AutoClip (10th percentile), dashed = clip threshold. Stressors were fixed before any run;\n"
             "the divergence rule (validation loss ending above 2× its best) was set after seeing the results.",
             fontsize=8, color="#555555", va="top", linespacing=1.4)
    json.dump(outcomes, open(os.path.join(results_dir, "stress_outcomes.json"), "w"), indent=1)
    return style.save(fig, "card_clipping_stress", out_dir)

