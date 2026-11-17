"""Deep learning vs LightGBM by segment: WMAPE and |bias| of each deep model minus LightGBM's, with the seed-noise band."""
import numpy as np
import pandas as pd

from m5knobs import paths
from m5knobs.data import SEED_NOISE
from m5knobs.figures import style
from m5knobs.figures.style import COLORS, plt
from m5knobs.segments import BASE, HIST_WEEKS, KINDS

PALETTE = ["#2e86ab", "#93c6dd", "#d1495b", "#edae49", "#6b4c9a", "#3b8b5a"]
ROW_TITLES = {"volume quartile": f"By volume quartile (mean weekly sales, {HIST_WEEKS} weeks before the first origin)",
              "zero weeks": f"By share of zero-sales weeks (same {HIST_WEEKS} weeks)",
              "department": "By department", "origin": "By forecast origin"}
PANELS = (("wape", "WMAPE", SEED_NOISE["wape"]), ("abs_bias", "|bias|", SEED_NOISE["bias"]))


def draw(table: pd.DataFrame, models: list, name: str, out_dir: str = paths.FIGURES, kinds=KINDS,
         title: str = "Deep learning vs LightGBM by segment", subtitle: str = None) -> list:
    """table: segments.segment_table output; models: labels in table.model, LightGBM (segments.BASE) among them.
    One row per segment kind, two panels: WMAPE and |bias| of each other model minus LightGBM's (points), grouped by segment."""
    style.apply()
    others = [m for m in models if m != BASE]
    nseg = [table[(table.segment_kind == k) & (table.model == BASE)].shape[0] for k in kinds]
    fig, axes = plt.subplots(len(kinds), 2, figsize=(style.WIDTH, 0.42 * sum(nseg) + 0.95 * len(kinds)), squeeze=False,
                             sharey="row", gridspec_kw=dict(height_ratios=nseg, hspace=0.6, wspace=0.06))
    step = 0.8 / len(others)
    for r, (row, kind) in enumerate(zip(axes, kinds)):
        t = table[table.segment_kind == kind]
        base = t[t.model == BASE].set_index("segment")
        segs = list(base.index)
        y = np.arange(len(segs))
        for ax, (col, label, noise) in zip(row, PANELS):
            ax.axvspan(-noise, noise, color="#d9d9d9", zorder=0, lw=0)
            ax.axvline(0, color="#777777", lw=0.7, zorder=1)
            for i, m in enumerate(others):
                d = t[t.model == m].set_index("segment").reindex(segs)
                yy, dd = y - 0.4 + step * (i + 0.5), (d[col] - base[col]).to_numpy()
                ax.barh(yy, dd, step * 0.9, color=PALETTE[i], zorder=2, label=m)
                pcol = "p_" + col
                if pcol in d:
                    for yi, di, pi in zip(yy, dd, d[pcol].to_numpy()):
                        if pi < 0.05:
                            ax.text(di, yi, " *" if di >= 0 else "* ", ha="left" if di >= 0 else "right", va="center",
                                    fontsize=7, color="#333333", zorder=3)
            ax.grid(axis="y", visible=False)
            ax.tick_params(axis="y", length=0)
            lo, hi = ax.get_xlim()
            pad = 0.06 * (hi - lo)
            ax.set_xlim(min(lo, -noise - pad), max(hi, noise + pad))
            if r == len(kinds) - 1:
                ax.set_xlabel(f"{label} minus {BASE} (points)")
        row[0].set_yticks(y)
        row[0].set_yticklabels([f"{s}  ·  {base.at[s, 'vol_share']:.0%} of volume\n{BASE}: WMAPE {base.at[s, 'wape']:.1f}, "
                                f"|bias| {base.at[s, 'abs_bias']:.1f}%" for s in segs], fontsize=6.8, linespacing=1.3)
        row[0].set_ylim(len(segs) - 0.5, -0.5)
        for ax, (_, label, _) in zip(row, PANELS):
            ax.set_title(label, loc="left", fontsize=8, color="#555555", pad=3)
    fig.subplots_adjust(left=0.33, right=0.99)  # segment labels sit inside the figure, under the left-aligned titles
    inch = 1 / fig.get_figheight()
    for row, kind in zip(axes, kinds):
        fig.text(0.01, row[0].get_position().y1 + 0.24 * inch, ROW_TITLES.get(kind, kind), fontsize=9, fontweight="bold",
                 ha="left", va="bottom")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#d9d9d9"))
    labels.append(f"{BASE} ± all-series seed noise (±{SEED_NOISE['wape']} WMAPE, ±{SEED_NOISE['bias']:.0f}pt |bias|)")
    top = axes[0, 0].get_position().y1 + 0.45 * inch
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, top), ncol=3, fontsize=7.3, handlelength=1.2,
               columnspacing=1.2)
    subtitle = subtitle or ("Each bar: a deep model's WMAPE or |bias| minus LightGBM's on the same series and weeks. "
                            "Left of 0 = better than LightGBM.\n* = p < 0.05 within the segment (paired bootstrap over its "
                            "series). Grey band = all-series seed noise; smaller segments are noisier.\n\"Best of 5 ES\" = best "
                            "of 5 early-stopping settings, picked with hindsight. N-HiTS Huber: fit once; Tweedie: refit every origin.")
    style.headline(fig, title, subtitle, y=top + 0.42 * inch)
    for ax in axes.flat:
        ax.tick_params(axis="x", colors=COLORS["muted"])
    return style.save(fig, name, out_dir)
