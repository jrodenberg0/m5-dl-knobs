"""F4: LGBM vs N-HiTS vs TCN results table as an image. Reads results/ only."""
import os

import pandas as pd

from m5knobs import paths, results
from m5knobs.data import H, NFOLDS, SEED_NOISE
from m5knobs.figures import style
from m5knobs.figures.style import COLORS, join_words, plt
from m5knobs.scoring import _series_level, paired_bootstrap_p

ROWS = [("LGBM (baseline)", "f4_lgbm"), ("N-HiTS", "f4_nhits"), ("TCN", "f4_tcn")]


def _wape(d):
    return 100 * d["abs_err"].sum() / d["y_sum"].sum()


def table(results_dir=paths.RESULTS, n_boot=2000) -> pd.DataFrame:
    results.require([a for _, a in ROWS], results_dir)
    runs = {a: results.load_run(a, results_dir) for _, a in ROWS}
    base = _series_level(runs["f4_lgbm"]["ps"])
    out = []
    for name, a in ROWS:
        s, ser = runs[a]["summary"], _series_level(runs[a]["ps"])
        gap = s["wape"] - runs["f4_lgbm"]["summary"]["wape"]
        p = None if a == "f4_lgbm" else paired_bootstrap_p(_wape, ser, base, n_boot=n_boot)
        read = "baseline" if p is None else ("within seed noise" if abs(gap) < SEED_NOISE["wape"] else
                                            ("better" if gap < 0 else "worse") + (" (significant)" if p < 0.05 else " (not significant)"))
        out.append(dict(model=name, wape=s["wape"], bias=s["bias"], gap_vs_lgbm=None if p is None else gap, p_vs_lgbm=p, read=read))
    return pd.DataFrame(out)


def headline_text(t: pd.DataFrame) -> str:
    base = t.iloc[0]["model"].split(" (")[0]
    parts = []
    for r in t.iloc[1:].itertuples():
        if r.read.startswith("better (significant)"):
            parts.append(f"{r.model} beat {base} by {-r.gap_vs_lgbm:.2f} WAPE")
        elif r.read.startswith("worse (significant)"):
            parts.append(f"{r.model} trailed {base} by {r.gap_vs_lgbm:.2f} WAPE")
        elif r.read.startswith("worse"):
            parts.append(f"{r.model} trailed {base} by {r.gap_vs_lgbm:.2f} WAPE (not significant)")
        elif r.read.startswith("better"):
            parts.append(f"{r.model} led {base} by {-r.gap_vs_lgbm:.2f} WAPE (not significant)")
        else:
            parts.append(f"{r.model} was level with {base}")
    return join_words(parts)


def claim_text(t: pd.DataFrame) -> str:
    """Verdict on 'deep learning can match gradient-boosted trees': holds if any deep model reads 'within seed noise' or 'better'."""
    base = t.iloc[0]["model"].split(" (")[0]
    level = [r.model for r in t.iloc[1:].itertuples() if r.read.startswith("better") or r.read == "within seed noise"]
    if level:
        return f"Level with or better than {base} on WAPE: {join_words(level)}. Verdict on the claim: holds."
    return f"No deep model was level with or better than {base} on WAPE. Verdict on the claim: does not hold."


def draw(results_dir=paths.RESULTS, out_dir=paths.FIGURES, n_boot=2000):
    t = table(results_dir, n_boot)
    t.to_csv(os.path.join(results_dir, "f4_table.csv"), index=False)
    style.apply()
    n = results.load_run("f4_lgbm", results_dir)["ps"]["unique_id"].nunique()
    cols = [("Model", 0.02, "left"), ("WAPE", 0.32, "right"), ("Bias", 0.44, "right"), ("Δ vs LGBM", 0.62, "right"),
            ("p", 0.71, "right"), ("Read", 0.74, "left")]
    fig, ax = plt.subplots(figsize=(style.WIDTH, 0.55 * len(t) + 1.2))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(len(t) + 0.2, -1.1)
    for c, x, ha in cols:
        ax.text(x, -0.55, c, ha=ha, va="center", fontsize=8.5, color=COLORS["muted"], fontweight="bold")
    ax.axhline(-0.15, color="#cccccc", lw=0.8)
    best = t["wape"].idxmin()
    for i, r in t.iterrows():
        y = i + 0.35
        if i % 2 == 1:
            ax.axhspan(i - 0.15, i + 0.85, color="#f6f6f6", zorder=0)
        w = "bold" if i == best else "normal"
        ax.text(0.02, y, r["model"], va="center", fontsize=10, fontweight=w)
        ax.text(0.32, y, f"{r['wape']:.2f}", ha="right", va="center", fontsize=10, fontweight=w)
        ax.text(0.44, y, f"{r['bias']:+.1f}%", ha="right", va="center", fontsize=10)
        if pd.notna(r["gap_vs_lgbm"]):
            col = COLORS["fix"] if r["gap_vs_lgbm"] < -SEED_NOISE["wape"] else COLORS["mistake"] if r["gap_vs_lgbm"] > SEED_NOISE["wape"] else COLORS["muted"]
            ax.text(0.62, y, f"{r['gap_vs_lgbm']:+.2f}", ha="right", va="center", fontsize=10, color=col, fontweight="bold")
            ax.text(0.71, y, "<0.001" if r["p_vs_lgbm"] < 0.001 else f"{r['p_vs_lgbm']:.3f}", ha="right", va="center", fontsize=9)
        ax.text(0.74, y, r["read"], va="center", fontsize=8, color=COLORS["muted"])
    ax.axhline(len(t) - 0.15, color="#cccccc", lw=0.8)
    ax.text(0.02, len(t) + 0.15,
            f"M5 weekly, {n:,} item-store series, rolling origin ({NFOLDS} folds × {H} weeks, refit each fold; the partial final week is not scored), "
            f"single seed,\nTweedie loss for all. p = paired bootstrap over series. 'Within seed noise' = |Δ| < {SEED_NOISE['wape']} WAPE (earlier M5 seed spread).\n"
            "LGBM: factory weekly recipe, direct (one model per lead), lags 1–26, rolling 4/13/26/52, calendar features, stock mlforecast transforms.",
            fontsize=6.5, color=COLORS["muted"], va="top")
    style.headline(fig, headline_text(t), "Same series, same folds, same Tweedie loss. Lower WAPE is better; bias = total forecast vs total actual.",
                   y=0.95)
    return style.save(fig, "f4_table", out_dir)

