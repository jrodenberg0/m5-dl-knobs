"""Pre-registered labeling rule for the mistakes grid (DESIGN.md, single-seed version): a measure is hurt when it moves in the
worse direction and a paired bootstrap over series gives two-sided p < 0.05."""
import numpy as np
import pandas as pd

from common import SEED_NOISE

ALPHA = 0.05
LABELS = ("underfit", "biased", "both", "no clear effect on M5")


def paired_bootstrap_p(stat_fn, a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> float:
    """a, b: one row per series (mistake, fix), aligned on unique_id. Resamples series with replacement."""
    a = a.set_index("unique_id")
    b = b.set_index("unique_id")
    ids = a.index.intersection(b.index)
    a, b = a.loc[ids].reset_index(drop=True), b.loc[ids].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(ids), len(ids))
        d[i] = stat_fn(a.iloc[idx]) - stat_fn(b.iloc[idx])
    return float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))


def is_hurt(mistake_val: float, fix_val: float, p: float, higher_is_worse: bool) -> bool:
    gap = mistake_val - fix_val if higher_is_worse else fix_val - mistake_val
    return bool(gap > 0 and p < ALPHA)


def label_row(hurt: dict, wape_gap: float = np.inf, abs_bias_gap_pts: float = np.inf) -> str:
    """Effect floor (added by Jack after the single-seed results): underfit also needs the WAPE gap to exceed prior seed noise,
    biased needs |bias| to worsen by more than the prior bias noise."""
    under = (hurt["insample_mae"] or hurt["corr"] or hurt["amp_ratio"]) and wape_gap > SEED_NOISE["wape"]
    biased = hurt["abs_bias"] and abs_bias_gap_pts > SEED_NOISE["bias"]
    return "both" if under and biased else "underfit" if under else "biased" if biased else "no clear effect on M5"


def _series_level(ps: pd.DataFrame) -> pd.DataFrame:
    """Folds collapsed per series: error and volume sums, mean corr / amp_ratio over the folds where they're defined."""
    return ps.groupby("unique_id", as_index=False).agg(abs_err=("abs_err", "sum"), y_sum=("y_sum", "sum"),
                                                       yhat_sum=("yhat_sum", "sum"), corr=("corr", "mean"),
                                                       amp_ratio=("amp_ratio", "mean"))


MEASURES = {  # name: (frame key, statistic, higher_is_worse)
    "insample_mae": ("ins", lambda d: d["abs_err"].sum() / d["y_abs"].sum(), True),
    "corr": ("ps", lambda d: float(np.nanmedian(d["corr"])), False),
    "amp_ratio": ("ps", lambda d: float(np.nanmedian(d["amp_ratio"])), False),
    "abs_bias": ("ps", lambda d: abs(d["yhat_sum"].sum() / d["y_sum"].sum() - 1), True),
}


def row_verdict(mistake_run: dict, fix_run: dict, n_boot: int = 2000) -> dict:
    """Runs: dict(ps=per_series frame, ins=insample frame). Returns per-measure gap/p/hurt and the row label."""
    frames = {k: (_series_level(mistake_run["ps"]), _series_level(fix_run["ps"])) for k in ("ps",)}
    frames["ins"] = (mistake_run["ins"], fix_run["ins"])
    out = {}
    for name, (key, fn, worse_up) in MEASURES.items():
        m, f = frames[key]
        mv, fv = fn(m), fn(f)
        p = paired_bootstrap_p(fn, m, f, n_boot=n_boot)
        out[name] = dict(mistake=float(mv), fix=float(fv), gap=float(mv - fv), p=p, hurt=is_hurt(mv, fv, p, worse_up))
    m, f = frames["ps"]
    wape = lambda d: 100 * d["abs_err"].sum() / d["y_sum"].sum()  # noqa: E731
    out["wape_gap"] = float(wape(m) - wape(f))
    out["abs_bias_gap_pts"] = float(100 * (out["abs_bias"]["mistake"] - out["abs_bias"]["fix"]))
    out["label"] = label_row({k: out[k]["hurt"] for k in MEASURES}, out["wape_gap"], out["abs_bias_gap_pts"])
    return out
