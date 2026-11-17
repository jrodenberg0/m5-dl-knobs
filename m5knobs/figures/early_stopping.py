"""Early-stopping study: WAPE and bias across patience window, val_size, val_check_steps and restore_best (arms group 'es').
Every arm shares one LR schedule (one-cycle over 10,000 steps) so only the stopping rule differs. Reads results/ only."""
import json
import os

import pandas as pd

from m5knobs import paths, results
from m5knobs.data import SEED_NOISE
from m5knobs.figures import style
from m5knobs.figures.style import COLORS, plt

SWEEPS = [
    ("Patience window (steps)", "early_stop_patience_steps × val_check_steps",
     [("250", "es_p250"), ("500", "es_p500"), ("1,000", "es_p1000"), ("2,000", "es_p2000"), ("4,000", "es_p4000")]),
    ("Validation tail (weeks)", "val_size", [("13", "es_p1000"), ("26", "es_val26"), ("52", "es_val52")]),
    ("Check every N steps (window fixed at 1,000)", "val_check_steps", [("25", "es_check25"), ("50", "es_p1000"), ("100", "es_check100")]),
    ("Weights kept at the end", "restore_best", [("best (restored)", "es_p1000"), ("last (stop step)", "es_norestore")]),
]


def headline_text(t: pd.DataFrame, base: int = 1000) -> str:
    b = t.loc[t["wape"].idxmin()]
    bw = t.loc[t["window"] == base, "wape"].iloc[0]
    if int(b["window"]) == base:
        return f"Early stopping: the {base:,}-step patience window was the best of those tried"
    bb = t.loc[t["window"] == base, "bias"].iloc[0]
    note = " (within seed noise)" if bw - b["wape"] < SEED_NOISE["wape"] else ""
    trade = (f", but bias {bb:+.1f}% → {b['bias']:+.1f}%" if abs(b["bias"]) - abs(bb) > SEED_NOISE["bias"] else "")
    return f"More patience: {int(b['window']):,} steps gave WAPE {b['wape']:.2f} vs {bw:.2f} at {base:,}{note}{trade}"


def _summary(arm, results_dir):
    s = results.load_run(arm, results_dir)["summary"]
    fs = (s.get("fold_stats") or [{}])[0]
    return dict(wape=s["wape"], bias=s["bias"], best=fs.get("best_step"), stop=fs.get("stop_step"), cap=bool(fs.get("hit_cap")))


def draw(results_dir=paths.RESULTS, out_dir=paths.FIGURES):
    arms_needed = sorted({a for _, _, xs in SWEEPS for _, a in xs})
    results.require(arms_needed, results_dir)
    style.apply()
    S = {a: _summary(a, results_dir) for a in arms_needed}
    base = S["es_p1000"]
    fig, axes = plt.subplots(2, 2, figsize=(style.WIDTH, 6.6), gridspec_kw=dict(hspace=0.95, wspace=0.28, top=0.8, bottom=0.12, left=0.09, right=0.98))
    for ax, (title, knob, xs) in zip(axes.ravel(), SWEEPS):
        x = list(range(len(xs)))
        w = [S[a]["wape"] for _, a in xs]
        ax.axhspan(base["wape"] - SEED_NOISE["wape"], base["wape"] + SEED_NOISE["wape"], color=COLORS["fix"], alpha=0.10, lw=0)
        ax.plot(x, w, "-", color=COLORS["muted"], lw=1.0)
        for xi, (lab, a) in zip(x, xs):
            s = S[a]
            col = COLORS["fix"] if a == "es_p1000" else COLORS["actual"]
            ax.plot([xi], [s["wape"]], "o", color=col, ms=6, mec="white", zorder=5)
            ax.annotate(f"{s['wape']:.2f}", (xi, s["wape"]), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=6.8, color=col,
                        fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{lab}\nbias {S[a]['bias']:+.1f}%\nstop {int(S[a]['stop'] or 0):,}" + (" (cap)" if S[a]["cap"] else "")
                            for lab, a in xs], fontsize=6.6, linespacing=1.3)
        for tl in ax.get_xticklabels():
            tl.set_color("#333333")
        lo, hi = min(w), max(w)
        pad = max(0.35, (hi - lo) * 0.6)
        ax.set_ylim(lo - pad * 0.6, hi + pad * 0.8)
        ax.set_xlim(-0.5, len(xs) - 0.5)
        ax.set_title(title, loc="left", fontsize=8.8, fontweight="bold", pad=15)
        ax.text(0, 1.025, knob, transform=ax.transAxes, fontsize=6.6, family="monospace", color=COLORS["muted"], va="bottom")
        ax.set_ylabel("WAPE", fontsize=7)
    t = pd.DataFrame([dict(window=int(lab.replace(",", "")), wape=S[a]["wape"], bias=S[a]["bias"]) for lab, a in SWEEPS[0][2]])
    n = results.actuals(results_dir)["unique_id"].nunique()
    fig.text(0.01, 0.985, headline_text(t), fontsize=12.5, fontweight="bold", va="top")
    fig.text(0.01, 0.94, f"N-HiTS, Huber δ5, {n:,} M5 weekly series, single seed. One setting changes per panel; every run uses the same\n"
             f"learning-rate schedule (one-cycle over 10,000 steps, no minimum-steps guard). Blue dot = base (1,000-step window, 13-week\n"
             f"validation, check every 50 steps, best weights restored); shaded band = base ± seed noise ({SEED_NOISE['wape']} WAPE).",
             fontsize=7.8, color="#555555", va="top", linespacing=1.4)
    json.dump(S, open(os.path.join(results_dir, "es_sweep.json"), "w"), indent=1)
    return style.save(fig, "card_early_stopping_sweep", out_dir)

