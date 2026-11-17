"""F1: what a clipped vs unclipped update does, on a real 2-weight Tweedie fit (m5knobs.toy). Reads results/ only."""
import os

import numpy as np
import pandas as pd

from m5knobs import paths
from m5knobs.figures import style
from m5knobs.figures.style import COLORS, plt


def draw(results_dir=paths.RESULTS, out_dir=paths.FIGURES):
    p = os.path.join(results_dir, "toy_paths.parquet")
    if not os.path.exists(p):
        raise SystemExit("missing results: toy_paths.parquet (run scripts/toy_clip.py first)")
    style.apply()
    traj = pd.read_parquet(p)
    s = np.load(os.path.join(results_dir, "toy_surface.npz"))
    fig = plt.figure(figsize=(style.WIDTH, 4.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1], hspace=0.45, wspace=0.28)
    ax = fig.add_subplot(gs[:, 0])
    L = s["loss"]
    levels = np.quantile(L[np.isfinite(L)], np.linspace(0.0, 0.97, 16))
    ax.contourf(s["a"], s["b"], L, levels=levels, cmap="Greys_r", alpha=0.35)
    ax.contour(s["a"], s["b"], L, levels=levels, colors="white", linewidths=0.4)
    i, j = np.unravel_index(np.nanargmin(L), L.shape)
    ax.plot(s["a"][i, j], s["b"][i, j], marker="*", color=COLORS["actual"], ms=9, zorder=6)
    ax.text(s["a"][i, j], s["b"][i, j] - 0.06 * (s["b"].max() - s["b"].min()), "lowest loss", fontsize=7, ha="center", va="top", bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85), zorder=8)
    ax.text(0.97, 0.97, "unclipped", transform=ax.transAxes, color=COLORS["mistake"], fontsize=8.5, fontweight="bold", ha="right", va="top")
    ax.text(0.97, 0.91, "clipped", transform=ax.transAxes, color=COLORS["fix"], fontsize=8.5, fontweight="bold", ha="right", va="top")
    for run, side in (("unclipped", "mistake"), ("clipped", "fix")):
        r = traj[traj.run == run].sort_values("step")
        ax.plot(r["a"], r["b"], "-o", color=COLORS[side], lw=1.0, ms=1.8, alpha=0.9, zorder=4)
    u = traj[traj.run == "unclipped"].sort_values("step").reset_index(drop=True)
    jump = np.hypot(u["a"].diff(), u["b"].diff())
    k = int(jump.idxmax())
    ax.annotate(f"one spiky batch:\nstep {k}, grad norm {u.loc[k, 'grad_norm']:,.0f}", xy=(u.loc[k, "a"], u.loc[k, "b"]),
                xytext=(0.04, 0.06), textcoords="axes fraction", fontsize=7, color=COLORS["mistake"],
                arrowprops=dict(arrowstyle="->", color=COLORS["mistake"], lw=0.8))
    st = traj[traj.step == 0].iloc[0]
    ax.plot(st["a"], st["b"], "s", color=COLORS["actual"], ms=5, zorder=6)
    ax.text(st["a"], st["b"], "  start", fontsize=7, va="center")
    ax.set_xlabel("weight a (intercept)")
    ax.set_ylabel("weight b (slope on last week's sales)")
    ax.grid(False)

    ag = fig.add_subplot(gs[0, 1])
    al = fig.add_subplot(gs[1, 1], sharex=ag)
    for run, side in (("unclipped", "mistake"), ("clipped", "fix")):
        r = traj[(traj.run == run) & (traj.step > 0)]
        al.plot(r["step"], r["loss"], color=COLORS[side], lw=1.2)
    r = traj[(traj.run == "clipped") & (traj.step > 0)]
    ag.plot(r["step"], r["grad_norm"], color=COLORS["muted"], lw=0.6)
    ag.plot(r["step"], r["clip_thr"], color=COLORS["fix"], lw=1.2)
    ag.text(0.98, 0.04, "clip threshold", transform=ag.transAxes, color=COLORS["fix"], fontsize=7, ha="right", va="bottom",
            bbox=dict(fc="white", ec="none", pad=0.5))
    ag.text(0.02, 0.97, "raw grad norm, clipped run", transform=ag.transAxes, color=COLORS["muted"], fontsize=6.5,
            va="top", bbox=dict(fc="white", ec="none", pad=0.5))
    ag.set_yscale("log")
    ag.set_title("Gradient norm per step", loc="left", fontsize=8.5)
    al.set_title("Full-data loss", loc="left", fontsize=8.5)
    al.set_xlabel("step")
    n = int(s["n_rows"])
    style.headline(fig, "Clipping caps the step size, not the direction",
                   f"Real fit: Tweedie GLM with 2 weights on {n:,} M5 item-weeks. Same start, same mini-batches (size {int(s['batch'])}), "
                   f"lr {float(s['lr'])}.\nOnly difference: AutoClip at the 10th percentile of past gradient norms.", y=0.9)
    return style.save(fig, "f1_toy", out_dir)

