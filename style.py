"""Shared figure style (DESIGN.md "B" direction) and results loading for the fig_*.py scripts."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("RESULTS_DIR", os.path.join(HERE, "results"))
FIGURES = os.environ.get("FIG_DIR", os.path.join(HERE, "figures"))
COLORS = dict(mistake="#d1495b", fix="#2e86ab", actual="#222222", muted="#8a8a8a", grid="#e6e6e6")
TINT = dict(mistake="#fbeff1", fix="#eef5f9")
WIDTH = 7.0  # inches; at dpi 200 -> 1400 px


def apply():
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],  # list = per-glyph fallback (Helvetica lacks →, ≈, Δ)
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#bbbbbb", "axes.linewidth": 0.6,
        "axes.grid": True, "grid.color": COLORS["grid"], "grid.linewidth": 0.5, "xtick.color": "#555555", "ytick.color": "#555555",
        "legend.frameon": False, "svg.fonttype": "none",
    })


def save(fig, name: str, out_dir: str = FIGURES) -> list:
    os.makedirs(out_dir, exist_ok=True)
    paths = [os.path.join(out_dir, f"{name}.{ext}") for ext in ("png", "svg")]
    fig.savefig(paths[0], dpi=200, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)
    return paths


def headline(fig, title: str, subtitle: str = "", y: float = 1.0):
    """Title and subtitle above y (figure fraction); spacing in inches so short and tall figures look the same."""
    inch = 1 / fig.get_figheight()
    sub_lines = subtitle.count("\n") + 1 if subtitle else 0
    if subtitle:
        fig.text(0.01, y + 0.10 * inch, subtitle, fontsize=8.5, color="#555555", ha="left", va="bottom", linespacing=1.35)
    fig.text(0.01, y + (0.16 + 0.17 * sub_lines) * inch, title, fontsize=13, fontweight="bold", color="#111111", ha="left", va="bottom")


def require(arm_ids, results_dir: str = RESULTS, seeds=(1,)):
    missing = [f"{a}__s{s}" for a in arm_ids for s in seeds if not os.path.exists(os.path.join(results_dir, "runs", f"{a}__s{s}", "DONE"))]
    if missing:
        raise SystemExit("missing results: " + ", ".join(missing) + " (run run_experiments.py first)")


def load_run(arm_id: str, results_dir: str = RESULTS, seed: int = 1) -> dict:
    d = os.path.join(results_dir, "runs", f"{arm_id}__s{seed}")
    rd = lambda f: pd.read_parquet(os.path.join(d, f)) if os.path.exists(os.path.join(d, f)) else None  # noqa: E731
    return dict(summary=json.load(open(os.path.join(d, "summary.json"))), trace=rd("trace.parquet"), ps=rd("per_series.parquet"),
                ins=rd("insample.parquet"), fc=rd("forecasts.parquet"))


def examples(results_dir: str = RESULTS) -> list:
    return json.load(open(os.path.join(results_dir, "examples.json")))


def actuals(results_dir: str = RESULTS) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(results_dir, "actuals.parquet"))


def examples3(results_dir: str = RESULTS) -> list:
    """[steady seller, intermittent, seasonal]: the runner's two rule-picked series plus common.pick_seasonal (cached)."""
    p = os.path.join(results_dir, "examples3.json")
    if not os.path.exists(p):
        import common
        act = actuals(results_dir)
        panel = common.load_panel(0)
        panel = panel[panel.unique_id.isin(set(act["unique_id"]))]
        ex = examples(results_dir)
        ex = ex + [common.pick_seasonal(panel, act["cutoff"].min(), exclude=ex)]
        json.dump(ex, open(p, "w"))
        panel[panel.unique_id.isin(ex)].to_parquet(os.path.join(results_dir, "examples_history.parquet"))
    return json.load(open(p))


def example_history(results_dir: str = RESULTS) -> pd.DataFrame:
    """Full history of the example series (cached in results/ on first use)."""
    p = os.path.join(results_dir, "examples_history.parquet")
    ex = examples3(results_dir) if os.path.exists(os.path.join(results_dir, "examples3.json")) else examples(results_dir)
    if not os.path.exists(p) or not set(ex) <= set(pd.read_parquet(p, columns=["unique_id"]).unique_id):
        import common
        panel = common.load_panel(0)
        panel[panel.unique_id.isin(ex)].to_parquet(p)
    return pd.read_parquet(p)


def smooth(s: pd.Series, n: int = 50) -> pd.Series:
    return s.rolling(n, min_periods=1, center=True).mean()
