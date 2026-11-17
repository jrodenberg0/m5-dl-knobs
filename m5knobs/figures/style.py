"""Shared figure style: colors, rcParams, title block, save; plus two text helpers for generated prose."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from m5knobs import paths  # noqa: E402

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
        "svg.hashsalt": "m5-dl-knobs",  # stable clip-path ids, so re-drawn SVGs only differ when the figure does
    })


def save(fig, name: str, out_dir: str = paths.FIGURES) -> list:
    os.makedirs(out_dir, exist_ok=True)
    out = [os.path.join(out_dir, f"{name}.{ext}") for ext in ("png", "svg")]
    fig.savefig(out[0], dpi=200, bbox_inches="tight", pad_inches=0.12, facecolor="white", metadata={"Software": None})
    fig.savefig(out[1], bbox_inches="tight", pad_inches=0.12, facecolor="white", metadata={"Date": None})
    plt.close(fig)
    return out


def headline(fig, title: str, subtitle: str = "", y: float = 1.0):
    """Title and subtitle above y (figure fraction); spacing in inches so short and tall figures look the same."""
    inch = 1 / fig.get_figheight()
    sub_lines = subtitle.count("\n") + 1 if subtitle else 0
    if subtitle:
        fig.text(0.01, y + 0.10 * inch, subtitle, fontsize=8.5, color="#555555", ha="left", va="bottom", linespacing=1.35)
    fig.text(0.01, y + (0.16 + 0.17 * sub_lines) * inch, title, fontsize=13, fontweight="bold", color="#111111", ha="left", va="bottom")


def smooth(s: pd.Series, n: int = 50) -> pd.Series:
    return s.rolling(n, min_periods=1, center=True).mean()


def join_words(items) -> str:
    """'a', 'a and b', 'a, b and c' ('' for none)."""
    items = [str(x) for x in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1] if items else ""


def label_text(label: str) -> str:
    """A verdict label as plain text ('both' -> 'underfit and biased')."""
    return {"both": "underfit and biased"}.get(label, label)
