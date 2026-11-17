"""Builds the weekly M5 panel used by every experiment from the public M5 data.

Source: datasetsforecast's M5 loader (Nixtla's public mirror of the competition files, including the evaluation-phase test
days, so the data runs through 2016-06-19). The Kaggle download alone (sales_train_evaluation.csv) ends 4 weeks earlier and
would shift every cutoff, so it won't reproduce the article's numbers.

Weekly = calendar weeks ending Saturday (W-SAT), summed per item-store series; each series starts at its first non-zero week.
The final week holds only one day (Sunday 2016-06-19); it is kept here and excluded from scoring by run_experiments.py.

Usage: python3 prepare_data.py [--data-dir data]  ->  data/m5_weekly.parquet
"""
import argparse
import os

import pandas as pd


def weekly_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily[["unique_id", "ds", "y"]].copy()
    d["unique_id"] = d["unique_id"].astype(str)
    d["ds"] = pd.to_datetime(d["ds"]).dt.to_period("W-SAT").dt.end_time.dt.normalize()
    w = d.groupby(["unique_id", "ds"], as_index=False, sort=True)["y"].sum()
    started = w["y"].ne(0).groupby(w["unique_id"]).cummax()
    return w[started].reset_index(drop=True)[["unique_id", "ds", "y"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    a = ap.parse_args()
    from datasetsforecast.m5 import M5
    daily, _, _ = M5.load(a.data_dir)
    w = weekly_from_daily(daily)
    out = os.path.join(a.data_dir, "m5_weekly.parquet")
    w.to_parquet(out, index=False)
    print(f"{out}: {w.unique_id.nunique():,} series, {w.ds.min().date()} .. {w.ds.max().date()}, {len(w):,} rows")


if __name__ == "__main__":
    main()
