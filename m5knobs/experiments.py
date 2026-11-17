"""Every run behind the figures (the Arm list; F3 row pairs in F3_ROWS) and the runner that trains them.

The runner saves raw results under results/runs/<arm>__s<seed>/ (DONE written last, so reruns skip finished pairs and redo
half-written ones).

Usage: python3 scripts/run_experiments.py [--group f2|f3|f4|all] [--arms id,...] [--seeds 1] [--results-dir results]
"""
import argparse
import json
import logging
import os
import shutil
import time
import traceback
import warnings
from dataclasses import dataclass, field

import pandas as pd

from m5knobs import data, paths, scoring


@dataclass(frozen=True)
class Arm:
    id: str
    group: str
    arch: str  # NeuralForecast class name, or "LGBM"
    overrides: dict = field(default_factory=dict)
    refit_each_fold: bool = False
    insample: bool = True  # in-sample error is the slowest step (~10 min at 30k); only the verdicts use it


HUBER = dict(loss="huber", huber_delta=5.0)

ARMS = [
    Arm("f2_noclip", "f2", "NHITS", dict(clip_percentile=None)),
    Arm("f2_clip", "f2", "NHITS", {}),
    Arm("f3_fix", "f3", "NHITS", dict(HUBER)),
    Arm("f3_fix_noguard", "f3", "NHITS", dict(HUBER, min_steps=0)),
    Arm("f3_patience250", "f3", "NHITS", dict(HUBER, min_steps=0, early_stop_patience_steps=5, val_check_steps=50)),
    Arm("f3_input13", "f3", "NHITS", dict(HUBER, input_size=13)),
    Arm("f3_minmax", "f3", "NHITS", dict(HUBER, scaler_type="minmax")),
    Arm("f3_mae", "f3", "NHITS", dict(loss="mae")),
    Arm("f4_lgbm", "f4", "LGBM", {}, True),
    Arm("f4_nhits", "f4", "NHITS", {}, True),
    Arm("f4_tcn", "f4", "TCN", {}, True),
]

# Clipping stress test (Tweedie): stressors fixed before any result. Each pair differs only in clipping.
STRESSORS = {"lr1e2": dict(lr_max=1e-2), "lr3e2": dict(lr_max=3e-2), "nofilter": dict(availability_threshold=[0.0, 0.0])}
for _k, _o in STRESSORS.items():
    ARMS += [Arm(f"st_{_k}_noclip", "stress", "NHITS", dict(_o, clip_percentile=None)), Arm(f"st_{_k}_clip", "stress", "NHITS", dict(_o))]

# Early-stopping study (Huber d5): one schedule for every arm (one-cycle over 10,000 steps, no minimum-steps guard), so only
# the stopping rule differs. Base = 1,000-step window (20 checks x 50 steps), 13-week validation tail, best weights restored.
ES_BASE = dict(HUBER, max_steps=10000, min_steps=0)
ARMS += [Arm(f"es_p{w}", "es", "NHITS", dict(ES_BASE, early_stop_patience_steps=w // 50, val_check_steps=50), insample=False)
         for w in (250, 500, 1000, 2000, 4000)]
ARMS += [Arm("es_val26", "es", "NHITS", dict(ES_BASE, early_stop_patience_steps=20, val_size=26), insample=False),
         Arm("es_val52", "es", "NHITS", dict(ES_BASE, early_stop_patience_steps=20, val_size=52), insample=False),
         Arm("es_check25", "es", "NHITS", dict(ES_BASE, early_stop_patience_steps=40, val_check_steps=25), insample=False),
         Arm("es_check100", "es", "NHITS", dict(ES_BASE, early_stop_patience_steps=10, val_check_steps=100), insample=False),
         Arm("es_norestore", "es", "NHITS", dict(ES_BASE, early_stop_patience_steps=20, restore_best=False), insample=False)]

# (row label, loss label, mistake arm, fix arm, training losses on the same scale in both arms)
F3_ROWS = [
    ("No gradient clipping", "Tweedie", "f2_noclip", "f2_clip", True),
    ("Early-stop window 250 steps", "Huber δ5", "f3_patience250", "f3_fix_noguard", True),
    ("Input window 13 weeks", "Huber δ5", "f3_input13", "f3_fix", False),  # RevIN stats over 13 vs 52 weeks
    ("MinMax scaler", "Huber δ5 (≈ MSE on MinMax scale)", "f3_minmax", "f3_fix", False),  # δ5 never caps errors in [0, 1] units
    ("MAE loss", "MAE vs Huber δ5", "f3_mae", "f3_fix", False),
]


def by_id(arm_id: str) -> Arm:
    return next(a for a in ARMS if a.id == arm_id)


def make_nf(arch: str, seed: int, **overrides):
    """The recipe (data.RECIPE) with per-arm overrides, as a model.Knobs model on the configured accelerator."""
    from m5knobs import model  # imported here so the arm table and scoring load without torch
    kw = {**data.RECIPE, **{k: v for k, v in overrides.items() if k != "log_model"}}
    return model.Knobs(arch=arch, h=data.H, freq=data.FREQ, seed=seed, accelerator=os.environ.get("ACCEL", "mps"), **kw)


def load_data():
    return data.load_panel(), data.CUTOFFS


def actuals(panel, cutoffs):
    """Horizon actuals per cutoff. The final week is dropped when it is partial (M5 daily ends on a Sunday, so the last
    Saturday-ending week holds one day): total under half the median of the 4 weeks before it."""
    tot = panel.groupby("ds")["y"].sum().sort_index()
    if len(tot) > 4 and tot.iloc[-1] < 0.5 * tot.iloc[-5:-1].median():
        panel = panel[panel.ds < tot.index[-1]]
    return pd.concat([panel[(panel.ds > c) & (panel.ds <= c + pd.Timedelta(weeks=data.H))].assign(cutoff=c) for c in cutoffs],
                     ignore_index=True)[["unique_id", "ds", "y", "cutoff"]]


def _snap(panel, cut):
    return data.drop_leading_zeros(panel[panel.ds <= cut][["unique_id", "ds", "y"]])


def _train_nf(arm, seed, panel, cutoffs):
    fcs, trace, ins, stats = [], None, None, []
    w, seen = None, None
    for i, cut in enumerate(cutoffs):
        snap = _snap(panel, cut)
        if i == 0 or arm.refit_each_fold:
            w = make_nf(arm.arch, seed, log_model=False, **arm.overrides)
            try:
                w.fit(snap)
            except Exception as e:  # e.g. NeuralForecast's "Loss is NaN": keep the steps that ran, the failure is a result
                if i == 0:
                    e.partial_trace = scoring.fit_trace(w)
                raise
            stats.append(w.fold_stats[-1] if w.fold_stats else {})
            if i == 0:
                trace, ins = scoring.fit_trace(w), (scoring.insample_mae_raw(w, snap) if arm.insample else None)
            seen = set(snap.unique_id)
            fc = w.predict(h=data.H)
        else:
            fc = w.predict(h=data.H, predict_df=snap[snap.unique_id.isin(seen)])
        fcs.append(fc.assign(cutoff=cut))
    return dict(forecasts=pd.concat(fcs, ignore_index=True), trace=trace, insample=ins, fold_stats=stats)


def _train_lgbm(arm, seed, panel, cutoffs):
    from m5knobs.baselines import make_lgbm
    fcs, stats = [], []
    for cut in cutoffs:
        t0 = time.time()
        mlf = make_lgbm(seed, data.FREQ)
        mlf.fit(_snap(panel, cut), static_features=[], max_horizon=data.H)  # direct: recursive compounded on fold 2 (FINDINGS)
        fcs.append(mlf.predict(h=data.H).assign(cutoff=cut))
        stats.append({"fit_seconds": time.time() - t0})
    return dict(forecasts=pd.concat(fcs, ignore_index=True), trace=None, insample=None, fold_stats=stats)


def train_arm(arm, seed, panel, cutoffs):
    return (_train_lgbm if arm.arch == "LGBM" else _train_nf)(arm, seed, panel, cutoffs)


def _save(d, res, act):
    fc = res["forecasts"][["unique_id", "ds", "cutoff", "yhat"]].copy()
    fc["yhat"] = fc["yhat"].clip(lower=0)
    ps = scoring.per_series(fc, act)
    summ = scoring.summarize(ps)
    if res["insample"] is not None:
        summ["insample_mae"] = float(res["insample"]["abs_err"].sum() / res["insample"]["y_abs"].sum())
        res["insample"].to_parquet(os.path.join(d, "insample.parquet"))
    if res["trace"] is not None:
        res["trace"].to_parquet(os.path.join(d, "trace.parquet"))
    fc.to_parquet(os.path.join(d, "forecasts.parquet"))
    ps.to_parquet(os.path.join(d, "per_series.parquet"))
    summ["fold_stats"] = res["fold_stats"]
    summ["accelerator"] = os.environ.get("ACCEL", "mps")
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1, default=float)
    open(os.path.join(d, "DONE"), "w").close()
    return summ


def rescore(results_dir=paths.RESULTS):
    """Re-score every finished run from its saved forecasts against results/actuals.parquet (no retraining)."""
    act = pd.read_parquet(os.path.join(results_dir, "actuals.parquet"))
    for name in sorted(os.listdir(os.path.join(results_dir, "runs"))):
        d = os.path.join(results_dir, "runs", name)
        if not os.path.exists(os.path.join(d, "DONE")):
            continue
        ps = scoring.per_series(pd.read_parquet(os.path.join(d, "forecasts.parquet")), act)
        with open(os.path.join(d, "summary.json")) as f:
            summ = json.load(f)
        summ.update(scoring.summarize(ps))
        ps.to_parquet(os.path.join(d, "per_series.parquet"))
        with open(os.path.join(d, "summary.json"), "w") as f:
            json.dump(summ, f, indent=1, default=float)
        print(f"[rescored] {name} wape={summ['wape']:.2f} bias={summ['bias']:.2f}", flush=True)


def manifest() -> dict:
    """What every run in one results dir must share; the device may differ per run and is recorded in each summary."""
    return dict(n_series=int(data.N_SERIES), smoke=bool(os.environ.get("SMOKE")), recipe=json.dumps(data.RECIPE, sort_keys=True),
                tweedie_link="neuralforecast+zero_guard")


def check_manifest(results_dir):
    p = os.path.join(results_dir, "manifest.json")
    m = manifest()
    if os.path.exists(p):
        old = json.load(open(p))
        if old != m:
            raise SystemExit(f"manifest mismatch in {results_dir}: it holds runs made with {old}, now {m}. Use a fresh --results-dir.")
    else:
        os.makedirs(results_dir, exist_ok=True)
        json.dump(m, open(p, "w"), indent=1)


def run(arm_ids, seeds=(1,), results_dir=paths.RESULTS):
    check_manifest(results_dir)
    panel, cutoffs = load_data()
    os.makedirs(os.path.join(results_dir, "runs"), exist_ok=True)
    act = actuals(panel, cutoffs)
    if not os.path.exists(os.path.join(results_dir, "actuals.parquet")):  # several runner processes may share one results dir
        act.to_parquet(os.path.join(results_dir, "actuals.parquet"))
    if not os.path.exists(os.path.join(results_dir, "examples.json")):
        with open(os.path.join(results_dir, "examples.json"), "w") as f:
            json.dump(data.pick_examples(panel, cutoffs[0]), f)
    for arm_id in arm_ids:
        arm = by_id(arm_id)
        for seed in seeds:
            name = f"{arm.id}__s{seed}"
            d = os.path.join(results_dir, "runs", name)
            if os.path.exists(os.path.join(d, "DONE")):
                print(f"[skip] {name} done", flush=True)
                continue
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d)
            t0 = time.time()
            print(f"[run] {name}", flush=True)
            try:
                res = train_arm(arm, seed, panel, cutoffs)
                s = _save(d, res, act)
                print(f"[done] {name} wape={s['wape']:.2f} bias={s['bias']:.2f} {time.time() - t0:.0f}s", flush=True)
            except Exception as e:
                with open(os.path.join(d, "ERROR.txt"), "w") as f:
                    f.write(traceback.format_exc())
                if getattr(e, "partial_trace", None) is not None:
                    e.partial_trace.to_parquet(os.path.join(d, "trace.parquet"))
                print(f"[error] {name}, see {d}/ERROR.txt", flush=True)


def main():
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default=None, help="comma-separated: " + ", ".join(sorted({a.group for a in ARMS})) + ", or all")
    ap.add_argument("--arms", default=None)
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--results-dir", default=paths.RESULTS)
    ap.add_argument("--rescore", action="store_true", help="rewrite actuals.parquet, re-score finished runs, train nothing")
    a = ap.parse_args()
    if a.rescore:
        check_manifest(a.results_dir)
        panel, cutoffs = load_data()
        actuals(panel, cutoffs).to_parquet(os.path.join(a.results_dir, "actuals.parquet"))
        return rescore(a.results_dir)
    if a.arms:
        ids = a.arms.split(",")
    else:
        gs = (a.group or "all").split(",")
        bad = set(gs) - {x.group for x in ARMS} - {"all"}
        if bad:
            ap.error(f"unknown group(s): {', '.join(sorted(bad))}")
        ids = [x.id for x in ARMS if "all" in gs or x.group in gs]
    run(ids, seeds=[int(s) for s in a.seeds.split(",")], results_dir=a.results_dir)
