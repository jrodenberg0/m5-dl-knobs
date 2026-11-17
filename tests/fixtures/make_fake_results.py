"""Tiny synthetic results/ tree with the same files the experiment runner writes, for figure tests."""
import json
import os

import numpy as np
import pandas as pd

from m5knobs import scoring
from m5knobs.experiments import ARMS

CUTS = [pd.Timestamp("2016-01-02") + pd.Timedelta(weeks=13 * i) for i in range(3)]


def make(results_dir, n_series=20, steps=50, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"s{i}" for i in range(n_series)]
    ds = pd.date_range(CUTS[0] - pd.Timedelta(weeks=60), CUTS[-1] + pd.Timedelta(weeks=13), freq="W-SAT")
    panel = pd.concat([pd.DataFrame({"unique_id": u, "ds": ds, "y": rng.poisson(3 + i, len(ds)).astype(float)})
                       for i, u in enumerate(ids)], ignore_index=True)
    act = pd.concat([panel[(panel.ds > c) & (panel.ds <= c + pd.Timedelta(weeks=13))].assign(cutoff=c) for c in CUTS], ignore_index=True)
    os.makedirs(os.path.join(results_dir, "runs"), exist_ok=True)
    act.to_parquet(os.path.join(results_dir, "actuals.parquet"))
    panel[panel.unique_id.isin(["s3", "s7", "s5"])].to_parquet(os.path.join(results_dir, "examples_history.parquet"))
    with open(os.path.join(results_dir, "examples3.json"), "w") as f:
        json.dump(["s3", "s7", "s5"], f)
    with open(os.path.join(results_dir, "examples.json"), "w") as f:
        json.dump(["s3", "s7"], f)
    for a in ARMS:
        d = os.path.join(results_dir, "runs", f"{a.id}__s1")
        os.makedirs(d, exist_ok=True)
        noise = 3.0 if a.id in ("f2_noclip", "f3_mae") else 1.0
        fc = act[["unique_id", "ds", "cutoff"]].copy()
        fc["yhat"] = np.clip(act["y"].to_numpy() + rng.normal(0, noise, len(fc)), 0, None)
        fc.to_parquet(os.path.join(d, "forecasts.parquet"))
        ps = scoring.per_series(fc, act)
        ps.to_parquet(os.path.join(d, "per_series.parquet"))
        summ = scoring.summarize(ps)
        summ["fold_stats"] = [{"best_step": 30.0, "stop_step": 45.0}]
        if a.arch != "LGBM":
            g = np.abs(rng.normal(1.0, 0.3, steps))
            g[10] = np.nan
            g[20] = 40.0
            val = np.full(steps, np.nan)
            val[9::10] = np.linspace(2, 1, len(val[9::10]))
            tr = pd.DataFrame({"step": np.arange(1, steps + 1), "grad_norm": g,
                               "clip_thr": np.nan if a.overrides.get("clip_percentile", 10) is None else 0.8,
                               "lr": np.linspace(1e-4, 1e-3, steps), "train_loss": np.linspace(3, 1, steps) + rng.normal(0, .1, steps),
                               "val_loss": val})
            tr.to_parquet(os.path.join(d, "trace.parquet"))
            ins = pd.DataFrame({"unique_id": ids, "abs_err": rng.gamma(2, noise, n_series), "y_abs": rng.gamma(2, 10, n_series)})
            ins.to_parquet(os.path.join(d, "insample.parquet"))
            summ["insample_mae"] = float(ins.abs_err.sum() / ins.y_abs.sum())
        with open(os.path.join(d, "summary.json"), "w") as f:
            json.dump(summ, f)
        open(os.path.join(d, "DONE"), "w").close()
    return results_dir
