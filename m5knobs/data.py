"""The weekly M5 panel: building it from daily data, loading it, the forecast cutoffs, and the rule-picked example series.

Source: datasetsforecast's M5 loader (Nixtla's public mirror of the competition files, including the evaluation-phase test
days, so the data runs through 2016-06-19). The Kaggle download alone (sales_train_evaluation.csv) ends 4 weeks earlier and
would shift every cutoff, so it won't reproduce these numbers.

Weekly = calendar weeks ending Saturday (W-SAT), summed per item-store series; each series starts at its first non-zero week.
The final week holds only one day (Sunday 2016-06-19); it is kept here and excluded from scoring by m5knobs.experiments.actuals.

Build: python3 scripts/prepare_data.py [--data-dir data]  ->  data/m5_weekly.parquet
"""
import argparse
import functools
import os

import numpy as np
import pandas as pd

from m5knobs import paths

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Tweedie predictions sample torch.poisson, which MPS lacks

H, FREQ, NFOLDS = 13, "W-SAT", 3
N_SERIES = int(os.environ.get("N_SERIES", 30490))  # the full M5 panel
PANEL_SEED = 0
SEEDS = [1]
SEED_NOISE = dict(wape=0.2, bias=1.0)  # single-seed noise from the author's earlier multi-seed M5 runs (3 seeds, same model family), quoted in captions
M5_PARQUET = paths.M5_WEEKLY
RECIPE = dict(input_size=52, max_steps=5000, lr_max=1e-3, clip_percentile=10.0, batch_size=128, windows_batch_size=1024, restore_best=True,
              loss="tweedie", tweedie_rho=1.5)  # plus model.Knobs defaults: NAdam, one-cycle LR, RevIN, patience 20 x 50 steps
if os.environ.get("SMOKE"):
    RECIPE.update(max_steps=100, val_check_steps=20)


def weekly_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily[["unique_id", "ds", "y"]].copy()
    d["unique_id"] = d["unique_id"].astype(str)
    d["ds"] = pd.to_datetime(d["ds"]).dt.to_period("W-SAT").dt.end_time.dt.normalize()
    w = d.groupby(["unique_id", "ds"], as_index=False, sort=True)["y"].sum()
    started = w["y"].ne(0).groupby(w["unique_id"]).cummax()
    return w[started].reset_index(drop=True)[["unique_id", "ds", "y"]]


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


def main():
    """Download M5 via datasetsforecast and write data/m5_weekly.parquet."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=paths.DATA)
    a = ap.parse_args()
    from datasetsforecast.m5 import M5
    daily, _, _ = M5.load(a.data_dir)
    w = weekly_from_daily(daily)
    out = os.path.join(a.data_dir, "m5_weekly.parquet")
    w.to_parquet(out, index=False)
    print(f"{out}: {w.unique_id.nunique():,} series, {w.ds.min().date()} .. {w.ds.max().date()}, {len(w):,} rows")
