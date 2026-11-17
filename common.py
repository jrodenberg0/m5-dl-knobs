"""Shared data loading, model factory, per-step traces and scoring for the experiments."""
import functools
import os

import numpy as np
import pandas as pd

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Tweedie predictions sample torch.poisson, which MPS lacks

H, FREQ, NFOLDS = 13, "W-SAT", 3
N_SERIES = int(os.environ.get("N_SERIES", 30490))  # the full M5 panel
PANEL_SEED = 0
SEEDS = [1]
SEED_NOISE = dict(wape=0.2, bias=1.0)  # single-seed noise from the author's earlier multi-seed M5 runs (3 seeds, same model family), quoted in captions
HERE = os.path.dirname(os.path.abspath(__file__))
M5_PARQUET = os.environ.get("M5_WEEKLY", os.path.join(HERE, "data", "m5_weekly.parquet"))
RECIPE = dict(input_size=52, max_steps=5000, lr_max=1e-3, clip_percentile=10.0, batch_size=128, windows_batch_size=1024, restore_best=True,
              loss="tweedie", tweedie_rho=1.5)  # plus knobs.Knobs defaults: NAdam, one-cycle LR, RevIN, patience 20 x 50 steps
if os.environ.get("SMOKE"):
    RECIPE.update(max_steps=100, val_check_steps=20)


@functools.lru_cache(maxsize=1)
def _full() -> pd.DataFrame:
    df = pd.read_parquet(M5_PARQUET)[["unique_id", "ds", "y"]]
    df["unique_id"] = df["unique_id"].astype(str)
    df["ds"] = pd.to_datetime(df["ds"])
    return df.sort_values(["unique_id", "ds"]).reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def _cutoffs():
    maxd = _full()["ds"].max()
    return [maxd - pd.Timedelta(weeks=H * (NFOLDS - i)) for i in range(NFOLDS)]


def __getattr__(name):  # CUTOFFS is lazy so helpers import without reading the parquet
    if name == "CUTOFFS":
        return _cutoffs()
    raise AttributeError(name)


def load_panel(n_series: int = N_SERIES, panel_seed: int = PANEL_SEED) -> pd.DataFrame:
    df = _full()
    ids = df["unique_id"].unique()
    if n_series and n_series < len(ids):
        keep = np.random.default_rng(panel_seed).choice(ids, n_series, replace=False)
        df = df[df["unique_id"].isin(keep)]
    return df.reset_index(drop=True)


def drop_leading_zeros(df: pd.DataFrame) -> pd.DataFrame:
    started = df["y"].ne(0).groupby(df["unique_id"]).cummax()
    return df[started].reset_index(drop=True)


def make_nf(arch: str, seed: int, **overrides):
    """The article's recipe (RECIPE) with per-arm overrides, as a knobs.Knobs model on the configured accelerator."""
    import knobs
    kw = {**RECIPE, **{k: v for k, v in overrides.items() if k != "log_model"}}
    return knobs.Knobs(arch=arch, h=H, freq=FREQ, seed=seed, accelerator=os.environ.get("ACCEL", "mps"), **kw)


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


def pick_examples(panel: pd.DataFrame, cutoff) -> list:
    """Two example series fixed by rule: nearest the median 52-week volume among dense series (zero share < 0.2) and among
    intermittent series (zero share in [0.5, 0.8]), last 52 weeks before the cutoff."""
    last = panel[(panel["ds"] <= cutoff) & (panel["ds"] > cutoff - pd.Timedelta(weeks=52))]
    g = last.groupby("unique_id")["y"]
    st = pd.DataFrame({"zs": g.apply(lambda s: (s == 0).mean()), "vol": g.sum()}).sort_index()
    out = []
    for band in (st[st.zs < 0.2], st[(st.zs >= 0.5) & (st.zs <= 0.8)]):
        med = band["vol"].median()
        out.append(str((band["vol"] - med).abs().sort_values(kind="mergesort").index[0]))
    return out


def pick_seasonal(panel: pd.DataFrame, cutoff, exclude=()) -> str:
    """Third example series, fixed by rule: among dense series (zero share < 0.2 over the last 104 weeks, >= 156 weeks of
    history) in the top decile of lag-52 autocorrelation, the one nearest the median volume."""
    hist = panel[(panel["ds"] <= cutoff) & (panel["ds"] > cutoff - pd.Timedelta(weeks=156))]
    rows = []
    for uid, g in hist.groupby("unique_id", sort=True):
        y = g.sort_values("ds")["y"].to_numpy(float)
        if uid in exclude or len(y) < 156 or (y[-104:] == 0).mean() >= 0.2 or np.std(y) == 0:
            continue
        a, b = y[52:] - y[52:].mean(), y[:-52] - y[:-52].mean()
        den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        rows.append((uid, float((a * b).sum() / den) if den > 0 else 0.0, y[-52:].sum()))
    st = pd.DataFrame(rows, columns=["uid", "ac52", "vol"])
    top = st[st["ac52"] >= st["ac52"].quantile(0.9)]
    return str(top.loc[(top["vol"] - top["vol"].median()).abs().sort_values(kind="mergesort").index[0], "uid"])
