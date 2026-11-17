"""Compare two results dirs on the parity arms: WAPE, bias, best/stop step, forecast differences. Gate: |dWAPE|<0.05, |dbias|<0.2pt."""
import json, os, sys
import numpy as np, pandas as pd
A, B = sys.argv[1], sys.argv[2]
rows, ok = [], True
for arm in ["f2_clip", "f2_noclip", "f3_fix"]:
    sa, sb = (json.load(open(os.path.join(d, "runs", f"{arm}__s1", "summary.json"))) for d in (A, B))
    fa, fb = (pd.read_parquet(os.path.join(d, "runs", f"{arm}__s1", "forecasts.parquet")) for d in (A, B))
    m = fa.merge(fb, on=["unique_id", "ds", "cutoff"], suffixes=("_a", "_b"))
    dw, db = sa["wape"] - sb["wape"], sa["bias"] - sb["bias"]
    good = abs(dw) < 0.05 and abs(db) < 0.2
    ok &= good
    rows.append(dict(arm=arm, wape_a=sa["wape"], wape_b=sb["wape"], d_wape=dw, bias_a=sa["bias"], bias_b=sb["bias"], d_bias=db,
                     best_a=sa["fold_stats"][0].get("best_step"), best_b=sb["fold_stats"][0].get("best_step"),
                     stop_a=sa["fold_stats"][0].get("stop_step"), stop_b=sb["fold_stats"][0].get("stop_step"),
                     max_abs_dfc=float((m.yhat_a - m.yhat_b).abs().max()), corr_fc=float(np.corrcoef(m.yhat_a, m.yhat_b)[0, 1]), pass_=good))
t = pd.DataFrame(rows)
print(t.to_markdown(index=False, floatfmt=".4f"))
print("PARITY", "PASS" if ok else "FAIL")
