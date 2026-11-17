"""Readers for the results/ tree the experiment runner writes (runs, actuals, example series)."""
import json
import os

import pandas as pd

from m5knobs import paths


GET_RESULTS = "run `python scripts/download_results.py` to fetch the finished runs, or train them with scripts/run_experiments.py"


def check_available(results_dir: str = paths.RESULTS):
    """Stop with instructions when results_dir has no finished runs."""
    runs = os.path.join(results_dir, "runs")
    if not (os.path.isdir(runs) and any(os.path.exists(os.path.join(runs, r, "DONE")) for r in os.listdir(runs))):
        raise SystemExit(f"no finished runs in {results_dir}: {GET_RESULTS}")


def require(arm_ids, results_dir: str = paths.RESULTS, seeds=(1,)):
    missing = [f"{a}__s{s}" for a in arm_ids for s in seeds if not os.path.exists(os.path.join(results_dir, "runs", f"{a}__s{s}", "DONE"))]
    if missing:
        raise SystemExit("missing results: " + ", ".join(missing) + f" ({GET_RESULTS})")


def load_run(arm_id: str, results_dir: str = paths.RESULTS, seed: int = 1) -> dict:
    d = os.path.join(results_dir, "runs", f"{arm_id}__s{seed}")
    rd = lambda f: pd.read_parquet(os.path.join(d, f)) if os.path.exists(os.path.join(d, f)) else None  # noqa: E731
    return dict(summary=json.load(open(os.path.join(d, "summary.json"))), trace=rd("trace.parquet"), ps=rd("per_series.parquet"),
                ins=rd("insample.parquet"), fc=rd("forecasts.parquet"))


def examples(results_dir: str = paths.RESULTS) -> list:
    return json.load(open(os.path.join(results_dir, "examples.json")))


def actuals(results_dir: str = paths.RESULTS) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(results_dir, "actuals.parquet"))


def examples3(results_dir: str = paths.RESULTS) -> list:
    """[steady seller, intermittent, seasonal]: the runner's two rule-picked series plus data.pick_seasonal (cached)."""
    p = os.path.join(results_dir, "examples3.json")
    if not os.path.exists(p):
        from m5knobs import data
        act = actuals(results_dir)
        panel = data.load_panel(0)
        panel = panel[panel.unique_id.isin(set(act["unique_id"]))]
        ex = examples(results_dir)
        ex = ex + [data.pick_seasonal(panel, act["cutoff"].min(), exclude=ex)]
        json.dump(ex, open(p, "w"))
        panel[panel.unique_id.isin(ex)].to_parquet(os.path.join(results_dir, "examples_history.parquet"))
    return json.load(open(p))


def example_history(results_dir: str = paths.RESULTS) -> pd.DataFrame:
    """Full history of the example series (cached in results/ on first use)."""
    p = os.path.join(results_dir, "examples_history.parquet")
    ex = examples3(results_dir) if os.path.exists(os.path.join(results_dir, "examples3.json")) else examples(results_dir)
    if not os.path.exists(p) or not set(ex) <= set(pd.read_parquet(p, columns=["unique_id"]).unique_id):
        from m5knobs import data
        panel = data.load_panel(0)
        panel[panel.unique_id.isin(ex)].to_parquet(p)
    return pd.read_parquet(p)
