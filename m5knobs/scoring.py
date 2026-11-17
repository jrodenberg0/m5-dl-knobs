"""Scoring: per-step training traces, per-series forecast errors, and the labeling rule that turns a mistake/fix pair into a verdict.

Labeling rule for the setting cards (single-seed version): a measure is hurt when it moves in the worse direction
and a paired bootstrap over series gives two-sided p < 0.05.
"""
import numpy as np
import pandas as pd

from m5knobs.data import H, SEED_NOISE

ALPHA = 0.05
LABELS = ("underfit", "biased", "both", "no clear effect on M5")


def fit_trace(w) -> pd.DataFrame:
    """One row per optimizer step of the last fit (1-based). Lightning's step-0 validation sanity check is dropped."""
    m = w.nf.models[0]
    norms = np.asarray(w._gn.norms, dtype=float)
    n = len(norms)
    tr = pd.DataFrame({"step": np.arange(1, n + 1), "grad_norm": norms})
    th = np.full(n, np.nan)
    if len(w._gn.thresholds):  # AutoClip appends a threshold only on steps with a finite norm
        pos = np.flatnonzero(np.isfinite(norms))[:len(w._gn.thresholds)]
        th[pos] = np.asarray(w._gn.thresholds, dtype=float)[:len(pos)]
    tr["clip_thr"] = th
    lr = np.full(n, np.nan)
    lr[:min(n, len(w._lr.lr))] = w._lr.lr[:n]
    tr["lr"] = lr
    train = {int(g) + 1: float(v) for g, v in m.train_trajectories}
    valid = {int(g): float(v) for g, v in m.valid_trajectories if g > 0}
    tr["train_loss"] = tr["step"].map(train)
    tr["val_loss"] = tr["step"].map(valid)
    return tr


def insample_mae_raw(w, train_df: pd.DataFrame) -> pd.DataFrame:
    """Per-series in-sample absolute error in raw units, from non-overlapping H-step in-sample windows of the last fit."""
    ins = w.nf.predict_insample(step_size=H)
    col = next(c for c in ins.columns if c not in ("unique_id", "ds", "cutoff", "y"))
    real = train_df[["unique_id", "ds", "y"]].rename(columns={"y": "y_true"})
    m = ins[["unique_id", "ds", col]].merge(real, on=["unique_id", "ds"], how="inner")
    m["abs_err"] = (m[col].clip(lower=0) - m["y_true"]).abs()
    m["y_abs"] = m["y_true"].abs()
    return m.groupby("unique_id", as_index=False)[["abs_err", "y_abs"]].sum()


def _corr(f, a):
    """Undefined (NaN) when actuals are flat; a flat forecast against moving actuals tracks nothing, so 0 (not dropped)."""
    if np.std(a) == 0:
        return np.nan
    if np.std(f) == 0:
        return 0.0
    return float(np.corrcoef(f, a)[0, 1])


def _amp(f, a):
    sa = np.std(a)
    return np.nan if sa == 0 else float(np.std(f) / sa)


def per_series(fc: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Per (series, cutoff) errors over the horizon. Missing forecasts count as 0; corr/amp_ratio NaN when undefined."""
    m = actual.merge(fc[["unique_id", "ds", "cutoff", "yhat"]], on=["unique_id", "ds", "cutoff"], how="left")
    m["yhat"] = m["yhat"].fillna(0.0)
    m["e"] = (m["yhat"] - m["y"]).abs()
    rows = []
    for (uid, cut), g in m.groupby(["unique_id", "cutoff"], sort=False):
        y, f = g["y"].to_numpy(float), g["yhat"].to_numpy(float)
        rows.append((uid, cut, g["e"].sum(), y.sum(), f.sum(), _corr(f, y), _amp(f, y)))
    return pd.DataFrame(rows, columns=["unique_id", "cutoff", "abs_err", "y_sum", "yhat_sum", "corr", "amp_ratio"])


def summarize(ps: pd.DataFrame) -> dict:
    ysum = ps["y_sum"].sum()
    return dict(wape=float(100 * ps["abs_err"].sum() / ysum), bias=float(100 * (ps["yhat_sum"].sum() / ysum - 1)),
                corr_median=float(np.nanmedian(ps["corr"])), amp_ratio_median=float(np.nanmedian(ps["amp_ratio"])))


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
