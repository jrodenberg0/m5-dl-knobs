"""Readers for the results/ tree the experiment runner writes (runs, actuals, history)."""
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


def actuals(results_dir: str = paths.RESULTS) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(results_dir, "actuals.parquet"))


def history(results_dir: str = paths.RESULTS) -> pd.DataFrame:
    """Every series' 60 weeks up to and including the first origin (unique_id, ds, y); shipped from results-v2 on."""
    p = os.path.join(results_dir, "history.parquet")
    if not os.path.exists(p):
        raise SystemExit(f"{p} is missing (results from before results-v2): re-run `python scripts/download_results.py --force`")
    return pd.read_parquet(p)
