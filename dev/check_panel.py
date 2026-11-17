"""Dev check (not shipped logic): the public build must equal the panel the original experiments used."""
import os, sys
import pandas as pd
a = pd.read_parquet("data/m5_weekly.parquet")
b = pd.read_parquet(os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/m5_weekly.parquet"))
norm = lambda d: d.assign(unique_id=d.unique_id.astype(str), ds=pd.to_datetime(d.ds), y=d.y.astype("float64")).sort_values(["unique_id", "ds"]).reset_index(drop=True)[["unique_id", "ds", "y"]]
a, b = norm(a), norm(b)
pd.testing.assert_frame_equal(a, b)
print(f"EXACT MATCH: {len(a):,} rows, {a.unique_id.nunique():,} series, y total {a.y.sum():,.0f}")
