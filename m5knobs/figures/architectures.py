"""Architecture comparison (refit at every origin, Tweedie): LGBM vs N-HiTS vs TCN, with paired bootstrap p. Reads results/ only."""
import os

import pandas as pd

from m5knobs import paths, results
from m5knobs.data import SEED_NOISE
from m5knobs.scoring import _series_level, paired_bootstrap_p

ROWS = [("LGBM (baseline)", "f4_lgbm"), ("N-HiTS", "f4_nhits"), ("TCN", "f4_tcn")]


def _wape(d):
    return 100 * d["abs_err"].sum() / d["y_sum"].sum()


def table(results_dir=paths.RESULTS, n_boot=2000) -> pd.DataFrame:
    results.require([a for _, a in ROWS], results_dir)
    runs = {a: results.load_run(a, results_dir) for _, a in ROWS}
    base = _series_level(runs["f4_lgbm"]["ps"])
    out = []
    for name, a in ROWS:
        s, ser = runs[a]["summary"], _series_level(runs[a]["ps"])
        gap = s["wape"] - runs["f4_lgbm"]["summary"]["wape"]
        p = None if a == "f4_lgbm" else paired_bootstrap_p(_wape, ser, base, n_boot=n_boot)
        read = "baseline" if p is None else ("within seed noise" if abs(gap) < SEED_NOISE["wape"] else
                                            ("better" if gap < 0 else "worse") + (" (significant)" if p < 0.05 else " (not significant)"))
        out.append(dict(model=name, wape=s["wape"], bias=s["bias"], gap_vs_lgbm=None if p is None else gap, p_vs_lgbm=p, read=read))
    return pd.DataFrame(out)


def write_table(results_dir=paths.RESULTS, n_boot=2000) -> pd.DataFrame:
    """table(), written to results/f4_table.csv for FINDINGS.md."""
    t = table(results_dir, n_boot)
    t.to_csv(os.path.join(results_dir, "f4_table.csv"), index=False)
    return t
