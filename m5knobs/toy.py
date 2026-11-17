"""The 2-weight Tweedie toy fit behind F1.

A real 2-weight Tweedie fit on M5 rows, trained twice from the same start on the same mini-batches, unclipped vs
AutoClip (p10 of the running grad-norm history, the same rule as model.AutoClip).

Model: log mu = a + b * x, x = log1p(last week's sales). Loss: Tweedie (rho 1.5) negative log-likelihood up to constants,
2*y/sqrt(mu) + 2*sqrt(mu). Writes results/toy_paths.parquet and results/toy_surface.npz.
"""
import json
import os

import numpy as np
import pandas as pd

from m5knobs import paths

RHO, LR, BATCH, START, CLIP_P = 1.5, 0.05, 8, (-2.0, 2.0), 10.0
ETA_MAX = 20.0


def m5_rows(results_dir, n_series=50, seed=0) -> pd.DataFrame:
    from m5knobs import data
    panel = data.load_panel(5000)  # fixed 5,000-series sample (panel_seed 0), independent of N_SERIES
    panel = panel[panel.ds <= data.CUTOFFS[0]]
    ex = json.load(open(os.path.join(results_dir, "examples.json")))[1]
    others = [u for u in panel.unique_id.unique() if u != ex]
    ids = [ex] + list(np.random.default_rng(seed).choice(others, n_series - 1, replace=False))
    p = data.drop_leading_zeros(panel[panel.unique_id.isin(ids)]).sort_values(["unique_id", "ds"])
    p["x"] = np.log1p(p.groupby("unique_id")["y"].shift(1))
    return p.dropna(subset=["x"])[["x", "y"]].reset_index(drop=True)


def _mu(a, b, x):
    return np.exp(np.minimum(a + b * x, ETA_MAX))


def loss(a, b, x, y):
    mu = _mu(a, b, x)
    return float(np.mean(y * mu ** (1 - RHO) / (RHO - 1) + mu ** (2 - RHO) / (2 - RHO)))


def grad(a, b, x, y):
    mu = _mu(a, b, x)
    d = -y * mu ** (1 - RHO) + mu ** (2 - RHO)  # dL/d(eta)
    return np.array([d.mean(), (d * x).mean()])


def train(x, y, batches, clip: bool):
    th = np.array(START, dtype=float)
    hist, rows = [], [dict(step=0, a=th[0], b=th[1], grad_norm=np.nan, clip_thr=np.nan, loss=loss(*th, x, y))]
    for t, idx in enumerate(batches, 1):
        g = grad(*th, x[idx], y[idx])
        n = float(np.linalg.norm(g))
        thr = np.nan
        if clip and np.isfinite(n):
            hist.append(n)
            hist.sort()
            thr = hist[int(round(CLIP_P / 100 * (len(hist) - 1)))]
            if n > thr > 0:
                g = g * thr / n
        th = th - LR * g
        rows.append(dict(step=t, a=th[0], b=th[1], grad_norm=n, clip_thr=thr, loss=loss(*th, x, y)))
    return pd.DataFrame(rows)


def main(steps=300, data=None, results_dir=paths.RESULTS, seed=0):
    os.makedirs(results_dir, exist_ok=True)
    d = data if data is not None else m5_rows(results_dir)
    x, y = d["x"].to_numpy(float), d["y"].to_numpy(float)
    batches = np.random.default_rng(seed).integers(0, len(x), (steps, BATCH))
    traj = pd.concat([train(x, y, batches, False).assign(run="unclipped"), train(x, y, batches, True).assign(run="clipped")],
                      ignore_index=True)
    fin = traj[np.isfinite(traj.a) & np.isfinite(traj.b)]
    lo_a, hi_a = np.clip([fin.a.min(), fin.a.max()], -8, 8)
    lo_b, hi_b = np.clip([fin.b.min(), fin.b.max()], -4, 5)
    pa, pb = 0.15 * (hi_a - lo_a) + 0.3, 0.15 * (hi_b - lo_b) + 0.3
    A, B = np.meshgrid(np.linspace(lo_a - pa, hi_a + pa, 120), np.linspace(lo_b - pb, hi_b + pb, 120))
    L = np.vectorize(lambda a, b: loss(a, b, x, y))(A, B)
    traj.to_parquet(os.path.join(results_dir, "toy_paths.parquet"))
    np.savez(os.path.join(results_dir, "toy_surface.npz"), a=A, b=B, loss=L, lr=LR, batch=BATCH, n_rows=len(x))
    return traj, dict(a=A, b=B, loss=L)
