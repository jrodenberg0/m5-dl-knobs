"""Overlay figure for one setting change: an example series (picked by m5knobs.examples) and all series summed.

Each panel: actual sales (black), the mistaken setting's forecast (red) and the fix's (blue) per origin, shaded gap,
dotted origin lines. A metrics column right of each panel gives forecast-week totals, WMAPE and |bias| for both
forecasts; the footer gives the selection rule, the example's rank and how common the pattern is.
"""
import os
import textwrap

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from m5knobs import examples, paths, results, scoring
from m5knobs.figures import style

HIST_WEEKS = 30
KEY = ["unique_id", "ds", "cutoff"]


def _done(results_dir, arm):
    return os.path.exists(os.path.join(results_dir, "runs", f"{arm}__s1", "DONE"))


def _overlay(ax, actual, fw, fr):
    """actual: ds, y. fw / fr: cutoff, ds, yhat. Returns every plotted y value (for the y-limits)."""
    C = style.COLORS
    ax.plot(actual["ds"], actual["y"], color=C["actual"], lw=1.2)
    vals = [actual["y"]]
    for c, g in fw.groupby("cutoff"):
        g = g.sort_values("ds")
        h = fr[fr["cutoff"] == c].sort_values("ds")
        both = g.merge(h, on="ds", suffixes=("_w", "_r"))
        ax.fill_between(both["ds"].to_numpy(), both["yhat_w"].to_numpy(), both["yhat_r"].to_numpy(),
                        color=C["mistake"], alpha=0.16, lw=0)
        ax.plot(g["ds"], g["yhat"], color=C["mistake"], lw=1.9)
        ax.axvline(c, color=C["muted"], lw=0.6, ls=":")
        vals.append(g["yhat"])
    for _, h in fr.groupby("cutoff"):
        h = h.sort_values("ds")
        ax.plot(h["ds"], h["yhat"], color=C["fix"], lw=1.9)
        vals.append(h["yhat"])
    v = pd.concat(vals).astype(float)
    lo, hi = float(v.min()), float(v.max())
    pad = 0.06 * (hi - lo if hi > lo else max(abs(hi), 1.0))
    ax.set_ylim(max(lo - pad, 0) if lo >= 0 else lo - pad, hi + pad)
    loc = mdates.AutoDateLocator(maxticks=8)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def _total(v):
    return f"{v / 1e6:,.2f}M" if v >= 1e6 else f"{v:,.0f}"


def _pct(v) -> str:
    """Percent with one decimal below 100, whole numbers with thousands separators from 100 up (keeps columns narrow)."""
    return f"{v:.1f}%" if abs(v) < 100 else f"{v:,.0f}%"


def _metrics(ax, heading, labels, totals, wmape, absbias):
    """Metrics column: rows actual / wrong / right; columns total, WMAPE, |bias|. ax is an empty side axes."""
    C = style.COLORS
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 1.0, heading, fontsize=8, color="#555555", va="top")
    cols = (0.58, 0.79, 1.0)
    two_lines = max(len(lab) for lab in labels) > 10  # long labels get their own line, numbers under them
    y0, dy = 0.84, 0.16 if two_lines else 0.10
    for x, h in zip(cols, ("total", "WMAPE", "|bias|")):
        ax.text(x, y0, h, fontsize=7.5, color="#555555", ha="right", va="center")
    rows = [("actual", C["actual"], totals[0], None, None),
            (labels[0], C["mistake"], totals[1], wmape[0], absbias[0]),
            (labels[1], C["fix"], totals[2], wmape[1], absbias[1])]
    for i, (lab, col, tot, w, b) in enumerate(rows, start=1):
        y = y0 - 0.10 - (i - 1) * dy
        ax.plot([0.0, 0.07], [y, y], color=col, lw=1.2 if i == 1 else 1.9, solid_capstyle="butt")
        ax.text(0.10, y, lab, fontsize=8, color=col, va="center")
        yn = y - 0.065 if two_lines and i > 1 else y
        cells = (_total(tot), "" if w is None else _pct(w), "" if b is None else _pct(b))
        for x, c in zip(cols, cells):
            ax.text(x, yn, c, fontsize=8, color="#222222", ha="right", va="center")


def _footer(pick: examples.Pick, labels) -> str:
    rule = textwrap.fill(f"Example chosen by rule: {pick.text(pick.rule, labels)}.", 160)
    base = pick.n_pattern_base or pick.n_population
    what = pick.pattern_base_text or "series both forecasts cover"
    pattern = textwrap.fill(f"{pick.uid} is rank {pick.rank} of {pick.n_candidates:,} candidates. {pick.n_pattern:,} of "
                            f"{base:,} {what} ({pick.pattern_share:.0%}) "
                            f"{pick.text(pick.pattern_text, labels) or 'show this pattern'}.", 160)
    return "\n".join([
        f"Shaded = gap between {labels[0]} and {labels[1]}. Dotted lines = forecast origins. "
        "WMAPE = sum of absolute errors / sum of actual sales; |bias| = |total forecast / total actual - 1|.",
        rule,
        pattern,
        f"Of the {pick.n_counted:,} series with >{examples.HELP_MIN_Y} units: {pick.n_helped:,} improved with {labels[1]} by "
        f">{examples.HELP_FRAC:.0%} of their error, {pick.n_hurt:,} got worse by as much.",
    ])


def draw(results_dir, arm_wrong: str, arm_right: str, pick: examples.Pick, labels, title: str, name: str,
         out_dir: str = paths.FIGURES, population=None, population_label: str = None) -> list:
    """Overlay figure; [] (with a printed message) if either arm did not finish.

    population: optional collection of unique_ids; the bottom panel then sums only those series (titled population_label)
    instead of all series."""
    for arm in (arm_wrong, arm_right):
        if not _done(results_dir, arm):
            print(f"skipped: {arm} did not finish")
            return []
    style.apply()
    act = results.actuals(results_dir)
    rw, rr = results.load_run(arm_wrong, results_dir), results.load_run(arm_right, results_dir)
    # forecast weeks without actual sales (none are scored) are left out
    weeks = act[["ds", "cutoff"]].drop_duplicates()
    fw, fr = (r["fc"][KEY + ["yhat"]].merge(weeks, on=["ds", "cutoff"]) for r in (rw, rr))

    fig = plt.figure(figsize=(9.2, 7.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.5, 1], hspace=0.42, wspace=0.08,
                          left=0.08, right=0.99, top=0.9, bottom=0.08)
    ax1, side1 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax2, side2 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # example series: history before the first origin, then the forecast-week actuals
    u = pick.uid
    a_u = act[act["unique_id"] == u]
    sw, sr = fw[fw["unique_id"] == u], fr[fr["unique_id"] == u]
    hist = results.history(results_dir)
    first = act["cutoff"].min()
    h_u = hist[(hist["unique_id"] == u) & (hist["ds"] > first - pd.Timedelta(weeks=HIST_WEEKS))]
    line = pd.concat([h_u[["ds", "y"]], a_u[["ds", "y"]]]).drop_duplicates("ds").sort_values("ds")
    _overlay(ax1, line, sw, sr)
    ax1.set_title(f"Example series {u}", loc="left")
    ax1.set_ylabel("units / week")
    m = a_u[KEY + ["y"]].merge(sw.rename(columns={"yhat": "f_w"}), on=KEY).merge(sr.rename(columns={"yhat": "f_r"}), on=KEY)
    _metrics(side1, "this series, forecast weeks", labels, (m["y"].sum(), m["f_w"].sum(), m["f_r"].sum()),
             (pick.wmape_wrong, pick.wmape_right), (pick.absbias_wrong, pick.absbias_right))

    # all series summed, or the population's series
    ps_w, ps_r = rw["ps"], rr["ps"]
    if population is not None:
        keep = set(population)
        act = act[act["unique_id"].isin(keep)]
        fw, fr = fw[fw["unique_id"].isin(keep)], fr[fr["unique_id"].isin(keep)]
        ps_w, ps_r = ps_w[ps_w["unique_id"].isin(keep)], ps_r[ps_r["unique_id"].isin(keep)]
    tot = act.groupby("ds", as_index=False)["y"].sum()
    tw = fw.groupby(["cutoff", "ds"], as_index=False)["yhat"].sum()
    tr = fr.groupby(["cutoff", "ds"], as_index=False)["yhat"].sum()
    _overlay(ax2, tot, tw, tr)
    n = act["unique_id"].nunique()
    ax2.set_title(f"{population_label}: {n:,} series summed" if population is not None else f"All {n:,} series summed", loc="left")
    ax2.set_ylabel("units / week, summed")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1e3:,.0f}k" if abs(v) >= 1e4 else f"{v:,.0f}"))
    sw_, sr_ = scoring.summarize(ps_w), scoring.summarize(ps_r)
    _metrics(side2, "these series, forecast weeks" if population is not None else "all series, forecast weeks", labels,
             (ps_w["y_sum"].sum(), ps_w["yhat_sum"].sum(), ps_r["yhat_sum"].sum()),
             (sw_["wape"], sr_["wape"]), (abs(sw_["bias"]), abs(sr_["bias"])))

    style.headline(fig, title, f"{labels[0]} (red) vs {labels[1]} (blue), forecast from each origin.", y=0.94)
    fig.text(0.01, 0.0, _footer(pick, labels), fontsize=7.5, color="#555555", va="top", linespacing=1.45)
    return style.save(fig, name, out_dir)
