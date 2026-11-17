"""One "setting card" per NeuralForecast hyperparameter: what the knob does, without vs with on real M5 runs, what moved.
Headlines and reads are generated from the results; only the knob explanations and "try" lines are fixed text. Reads results/ only."""
import json
import os
import textwrap

import numpy as np
import pandas as pd

import style
from common import SEED_NOISE
from style import COLORS, TINT, plt
from verdict import row_verdict

CARDS = [
    dict(key="clipping", name="Gradient clipping", action="Clip your gradients", knob="gradient_clip_val",
         source="PyTorch Lightning trainer argument, passed through by NeuralForecast. We use AutoClip on top of it.",
         what="Before each weight update, if the gradient's total size is above a threshold, it is scaled down to that threshold; the "
              "direction is kept. AutoClip moves the threshold every step to the 10th percentile of the gradient sizes seen so far.",
         why="One promo week or launch spike in a batch can produce a very large Tweedie gradient and drag the whole fit with it.",
         mistake="f2_noclip", fix="f2_clip", m_label="clip off", f_label="AutoClip, 10th percentile", loss="Tweedie (rho 1.5)", diag="grad",
         try_effect="Turn it on for likelihood losses (Tweedie, Poisson, NB); AutoClip saves you guessing a fixed gradient_clip_val.",
         try_none="It didn't move the average on M5 at the recipe's learning rate; the stress test shows where it changes training."),
    dict(key="input_size", name="Input window", action="Show the model a full year", knob="input_size",
         source="NeuralForecast model argument: the length of history in each training window.",
         what="Each training example is input_size past weeks followed by the next h weeks to predict. The network only ever sees that "
              "window, so anything longer than it (a yearly peak, last year's holiday) is invisible.",
         why="Weekly retail demand repeats yearly. A 13-week window can't see last year, so the model falls back to the recent level.",
         mistake="f3_input13", fix="f3_fix", m_label="input_size = 13", f_label="input_size = 52", loss="Huber δ5", diag=None,
         try_effect="Start at one season (52 weekly, 12 monthly) or two, and shorten only if you have evidence.",
         try_none="One season is a safe start; on M5 the shorter window didn't measurably hurt."),
    dict(key="early_stopping", name="Early stopping", action="Give early stopping room", knob="early_stop_patience_steps × val_check_steps",
         source="NeuralForecast model arguments. Patience counts validation checks, so the window in optimizer steps is the product.",
         what="Every val_check_steps steps the model scores the held-out last h weeks. Training stops when that score hasn't improved "
              "for early_stop_patience_steps checks in a row, and the best weights are restored.",
         why="Validation loss on intermittent demand is noisy and often plateaus before it improves again; a short window quits early.",
         mistake="f3_patience250", fix="f3_fix_noguard", m_label="5 × 50 = 250 steps", f_label="20 × 50 = 1,000 steps", loss="Huber δ5",
         diag="loss",
         try_effect="Allow ~1,000 optimizer steps without improvement before stopping, and watch bias: longer training pushed it down here.",
         try_none="A 1,000-step window costs little extra time; on M5 the short one didn't measurably hurt."),
    dict(key="loss", name="Loss function", action="Use Huber, not MAE", knob="loss",
         source="NeuralForecast model argument, e.g. loss=MAE() or loss=HuberLoss(delta=5). Computed on the scaled target.",
         what="MAE is minimized by the median of each forecast's distribution. Huber is squared for small errors and absolute for large "
              "ones, so it sits between the median and the mean; delta (in scaled units) sets where it switches.",
         why="On intermittent series the median week is often zero, so a median forecast under-calls total volume.",
         mistake="f3_mae", fix="f3_fix", m_label="loss = MAE()", f_label="loss = HuberLoss(delta=5)", loss="", diag="total",
         try_effect="Pick the loss by how you're scored: if total volume (bias) matters, avoid pure MAE; tune Huber's delta.",
         try_none="On M5 the two losses ended up close; check bias, not just WAPE, on your own data."),
    dict(key="scaler", name="Input scaling", action="Use RevIN scaling", knob="scaler_type",
         source="NeuralForecast model argument: how each window is normalized before the network sees it.",
         what="'revin' subtracts each window's mean and divides by its standard deviation, then reverses that on the output. "
              "'minmax' squeezes each window into [0, 1] using its min and max.",
         why="One spike sets a window's max, so MinMax squashes every other week toward zero. Note: Huber δ5 acts like MSE on [0, 1].",
         mistake="f3_minmax", fix="f3_fix", m_label="scaler_type = 'minmax'", f_label="scaler_type = 'revin'", loss="Huber δ5", diag=None,
         try_effect="Default to RevIN (or robust); avoid MinMax for spiky demand.",
         try_none="RevIN is a fine default; on M5 MinMax wasn't measurably worse."),
]

HURT_WORDS = {"insample_mae": "fit to the training history", "corr": "week-to-week tracking",
              "amp_ratio": "how much the forecast moves (it went flatter)"}
ROW_NAMES = ["steady seller", "intermittent", "seasonal"]


def headline(card, v, ms, fs) -> str:
    lab = v["label"]
    if lab == "no clear effect on M5":
        return f"{card['name']}: no measurable difference on M5"
    w = f"WAPE {ms['wape']:.1f} → {fs['wape']:.1f}"
    b = f"bias {ms['bias']:+.1f}% → {fs['bias']:+.1f}%"
    return f"{card['action']}: " + {"underfit": w, "biased": b, "both": f"{w}, {b}"}[lab]


def what_moved(v, m_label, ms=None, fs=None) -> str:
    trade = ""
    if v["abs_bias_gap_pts"] < -SEED_NOISE["bias"]:
        trade = (f" Trade-off: bias moved from {ms['bias']:+.1f}% to {fs['bias']:+.1f}% with the fix (|bias| {-v['abs_bias_gap_pts']:.1f}pt "
                 "larger)." if ms and fs else f" Trade-off: |bias| was {-v['abs_bias_gap_pts']:.1f}pt smaller with {m_label}.")
    if v["label"] == "no clear effect on M5":
        return f"Nothing moved beyond seed noise (WAPE Δ {v['wape_gap']:+.2f})." + trade
    hurt = [HURT_WORDS[k] for k in HURT_WORDS if v[k]["hurt"]] if v["label"] in ("underfit", "both") else []
    parts = []
    if hurt:
        parts.append(f"With {m_label}, these got worse: " + "; ".join(hurt) + f" (WAPE +{v['wape_gap']:.2f}).")
    if v["label"] in ("biased", "both"):
        parts.append(f"Total-volume bias was {v['abs_bias_gap_pts']:.1f}pt larger.")
    return " ".join(parts) + trade


def _wrap(ax, x, y, text, width, **kw):
    ax.text(x, y, "\n".join(textwrap.wrap(text, width)), transform=ax.transAxes, va="top", **kw)


KNOB_BLOCKS = lambda c: [  # noqa: E731  (text, wrap width, fontsize, style kwargs, gap after in points)
    ("THE KNOB", 200, 7, dict(color=COLORS["muted"], fontweight="bold"), 3),
    (c["knob"], 200, 10.5, dict(family="monospace", fontweight="bold"), 3),
    (c["source"], 140, 7, dict(color=COLORS["muted"]), 7),
    ("What it does: " + c["what"], 128, 7.8, dict(color="#222222"), 5),
    ("Why it matters here: " + c["why"], 128, 7.8, dict(color="#222222", style="italic"), 0)]


def knob_height_pt(card) -> float:
    return 6 + sum(len(textwrap.wrap(t, w)) * fs * 1.28 + gap for t, w, fs, _, gap in KNOB_BLOCKS(card))


def _knob_box(ax, card):
    ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc="#f4f4f1", ec="none", zorder=0))
    y = -8.0
    for text, width, fs, kw, gap in KNOB_BLOCKS(card):
        lines = textwrap.wrap(text, width)
        ax.annotate("\n".join(lines), xy=(0.015, 1), xycoords="axes fraction", xytext=(0, y), textcoords="offset points",
                    fontsize=fs, va="top", linespacing=1.25, **kw)
        y -= len(lines) * fs * 1.32 + gap


def _diag(ax, card, run, side, xmax):
    tr = run["trace"]
    if card["diag"] == "grad":
        g = tr["grad_norm"].replace([np.inf, -np.inf], np.nan)
        ax.plot(tr["step"], g, color=COLORS[side], lw=0.35, alpha=0.35)
        ax.plot(tr["step"], g.rolling(50, min_periods=1, center=True).median(), color=COLORS[side], lw=1.3)
        if tr["clip_thr"].notna().any():
            ax.plot(tr["step"], tr["clip_thr"], color=COLORS["actual"], lw=0.9, ls=(0, (4, 2)))
            ax.text(0.98, 0.06, "clip threshold (dashed): anything above is scaled down", transform=ax.transAxes, fontsize=6.3,
                    ha="right", bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))
        ax.set_yscale("log")
        ax.set_ylabel("gradient size", fontsize=7)
    elif card["diag"] == "loss":
        v = tr.dropna(subset=["val_loss"])
        ax.plot(v["step"], v["val_loss"], "o-", color=COLORS[side], ms=1.8, lw=1.0)
        best = v.loc[v["val_loss"].idxmin()]
        stop = tr["step"].max()
        ax.axvspan(best["step"], stop, color=COLORS[side], alpha=0.12, lw=0)
        ax.plot([best["step"]], [best["val_loss"]], "o", color=COLORS[side], ms=5, mec="white")
        ax.text(0.98, 0.92, f"best {int(best['step']):,} · stopped {int(stop):,}", transform=ax.transAxes, fontsize=6.8, ha="right",
                va="top", color=COLORS[side], fontweight="bold", bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))
        ax.set_xlim(0, xmax * 1.03)
        ax.set_ylabel("validation loss", fontsize=7)
    ax.set_xlabel("optimizer step", fontsize=6.5)


def _total(ax, act, fc, side):
    a = act.groupby("ds")["y"].sum()
    ax.plot(a.index, a.values, color=COLORS["actual"], lw=1.0)
    f = act[["unique_id", "ds", "cutoff"]].merge(fc, on=["unique_id", "ds", "cutoff"], how="left").fillna({"yhat": 0})
    for _, g in f.groupby("cutoff"):
        t = g.groupby("ds")["yhat"].sum()
        ax.plot(t.index, t.values, color=COLORS[side], lw=1.8)
    ax.set_ylabel("units / week,\nall series", fontsize=7)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(bymonth=[1, 4, 7, 10]))


def _series(ax, uid, hist, fc, side, cut0):
    h = hist[(hist.unique_id == uid) & (hist.ds > cut0 - pd.Timedelta(weeks=26))]
    ax.plot(h["ds"], h["y"], color=COLORS["actual"], lw=0.85)
    for _, g in fc[fc.unique_id == uid].sort_values("ds").groupby("cutoff"):
        ax.plot(g["ds"], g["yhat"], color=COLORS[side], lw=1.7)
    ax.axvline(cut0, color=COLORS["muted"], lw=0.5, ls=":")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(bymonth=[1, 7]))


def _numbers(ax, ms, fs, v):
    ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc="white", ec="#dddddd", lw=0.8))
    lab = v["label"]
    col = COLORS["muted"] if lab == "no clear effect on M5" else COLORS["mistake"]
    for x, head, val, c in [(0.02, "WAPE (without → with)", f"{ms['wape']:.1f} → {fs['wape']:.1f}", "#111111"),
                            (0.28, "Bias (without → with)", f"{ms['bias']:+.1f}% → {fs['bias']:+.1f}%", "#111111"),
                            (0.56, "Verdict on the 'without' run", {"both": "underfit + biased"}.get(lab, lab.replace(" on M5", "")), col),
                            (0.82, "Seed noise (earlier M5)", f"±{SEED_NOISE['wape']} WAPE, ±{SEED_NOISE['bias']:.0f}pt", COLORS["muted"])]:
        ax.text(x, 0.76, head, transform=ax.transAxes, fontsize=6.5, color=COLORS["muted"], va="center")
        ax.text(x, 0.30, val, transform=ax.transAxes, fontsize=10 if c == "#111111" else 8.5, color=c, fontweight="bold", va="center")


class _Layout:
    """Stack blocks top-down in inches; returns axes placed in figure fractions."""
    W, LEFT, RIGHT, GAP = style.WIDTH, 0.75, 0.15, 0.2

    def __init__(self, heights):
        self.H = sum(heights)
        self.fig = plt.figure(figsize=(self.W, self.H))
        self.y = self.H

    def take(self, h):
        self.y -= h
        return self.y

    def ax(self, y, h, col=None, full=False):
        if full:
            x0, w = 0.15, self.W - 0.3
        else:
            cw = (self.W - self.LEFT - self.RIGHT - self.GAP) / 2
            x0, w = (self.LEFT if col == 0 else self.LEFT + cw + self.GAP), cw
        return self.fig.add_axes([x0 / self.W, y / self.H, w / self.W, h / self.H])


def build(card, results_dir=style.RESULTS, n_boot=2000):
    style.require([card["mistake"], card["fix"]], results_dir)
    style.apply()
    m, f = style.load_run(card["mistake"], results_dir), style.load_run(card["fix"], results_dir)
    v = row_verdict(dict(ps=m["ps"], ins=m["ins"]), dict(ps=f["ps"], ins=f["ins"]), n_boot=n_boot)
    act = style.actuals(results_dir)
    ex = style.examples3(results_dir)
    hist = style.example_history(results_dir)
    hist = pd.concat([hist, act[act.unique_id.isin(ex)][["unique_id", "ds", "y"]]]).drop_duplicates(["unique_id", "ds"])
    cut0, n = act["cutoff"].min(), act["unique_id"].nunique()

    try_line = card["try_none"] if v["label"] == "no clear effect on M5" else card["try_effect"]
    moved = textwrap.wrap("What moved: " + what_moved(v, card["m_label"], m["summary"], f["summary"]), 128)
    trys = textwrap.wrap("Try on your data: " + try_line, 128)
    H_HEAD, H_KNOB = 1.0, knob_height_pt(card) / 72 + 0.12
    H_COLT, H_DIAG, H_ROW, H_TICK = 0.28, 0.95, 0.72, 0.32
    H_NUM, H_READ = 0.52, (len(moved) + len(trys)) * 0.15 + 0.2
    heights = [H_HEAD, H_KNOB, 0.15, H_COLT] + ([H_DIAG, 0.42] if card["diag"] else []) + [H_ROW + H_TICK] * len(ex) + \
              [0.05, H_NUM, 0.15, H_READ]
    L = _Layout(heights)
    fig = L.fig

    loss = f", {card['loss']} loss" if card["loss"] else ""
    y = L.take(H_HEAD)
    fig.text(0.15 / L.W, (y + H_HEAD - 0.12) / L.H, headline(card, v, m["summary"], f["summary"]), fontsize=13.5, fontweight="bold",
             va="top", color="#111111")
    fig.text(0.15 / L.W, (y + H_HEAD - 0.50) / L.H,
             f"N-HiTS{loss}, {n:,} M5 weekly series, single seed. Only this setting differs between the two columns.\n"
             "Black = actuals; colored = 13-week forecasts from 3 origins (one fit at the first origin). Numbers cover all series.",
             fontsize=8, color="#555555", va="top", linespacing=1.4)
    y = L.take(H_KNOB)
    _knob_box(L.ax(y, H_KNOB, full=True), card)
    L.take(0.15)
    y = L.take(H_COLT)
    for j, (side, lab) in enumerate([("mistake", "Without: " + card["m_label"]), ("fix", "With: " + card["f_label"])]):
        a = L.ax(y, H_COLT, col=j)
        a.set_axis_off()
        a.text(0, 0.2, lab, fontsize=9.5, color=COLORS[side], fontweight="bold", transform=a.transAxes)

    if card["diag"]:
        y = L.take(H_DIAG)
        pair = []
        for j, (run, side) in enumerate([(m, "mistake"), (f, "fix")]):
            a = L.ax(y, H_DIAG, col=j)
            a.set_facecolor(TINT[side])
            if card["diag"] == "total":
                _total(a, act, run["fc"], side)
            else:
                _diag(a, card, run, side, max(m["trace"]["step"].max(), f["trace"]["step"].max()))
            pair.append(a)
        lo, hi = min(p.get_ylim()[0] for p in pair), max(p.get_ylim()[1] for p in pair)
        for p in pair:
            p.set_ylim(lo, hi)
        pair[1].tick_params(labelleft=False)
        pair[1].set_ylabel("")
        L.take(0.42)
    for i, uid in enumerate(ex):
        y = L.take(H_ROW + H_TICK) + H_TICK
        pair = []
        for j, (run, side) in enumerate([(m, "mistake"), (f, "fix")]):
            a = L.ax(y, H_ROW, col=j)
            a.set_facecolor(TINT[side])
            _series(a, uid, hist, run["fc"], side, cut0)
            pair.append(a)
        lo, hi = min(p.get_ylim()[0] for p in pair), max(p.get_ylim()[1] for p in pair)
        for p in pair:
            p.set_ylim(lo, hi)
        pair[0].set_ylabel(ROW_NAMES[i], fontsize=7.5)
        pair[1].tick_params(labelleft=False)
    L.take(0.05)
    y = L.take(H_NUM)
    _numbers(L.ax(y, H_NUM, full=True), m["summary"], f["summary"], v)
    L.take(0.15)
    y = L.take(H_READ)
    ra = L.ax(y, H_READ, full=True)
    ra.set_axis_off()
    ra.annotate("\n".join(moved), xy=(0, 1), xycoords="axes fraction", va="top", fontsize=7.8, color="#222222", linespacing=1.3)
    ra.annotate("\n".join(trys), xy=(0, 1), xycoords="axes fraction", xytext=(0, -(len(moved) * 7.8 * 1.3 + 6)),
                textcoords="offset points", va="top", fontsize=7.8, color="#222222", fontweight="bold", linespacing=1.3)
    return fig, v


def main(results_dir=style.RESULTS, out_dir=style.FIGURES, n_boot=2000):
    paths, verdicts = [], {}
    for card in CARDS:
        fig, v = build(card, results_dir, n_boot)
        verdicts[card["key"]] = dict(label=v["label"], wape_gap=v["wape_gap"], abs_bias_gap_pts=v["abs_bias_gap_pts"])
        paths += style.save(fig, f"card_{card['key']}", out_dir)
    with open(os.path.join(results_dir, "card_verdicts.json"), "w") as fh:
        json.dump(verdicts, fh, indent=1)
    return paths


if __name__ == "__main__":
    print("\n".join(main()))
