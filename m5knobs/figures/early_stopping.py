"""Early-stopping study: WAPE and bias across patience window, val_size, val_check_steps and restore_best (arms group 'es').
Every arm shares one LR schedule (one-cycle over 10,000 steps) so only the stopping rule differs. Reads results/ only."""
import json
import os

import pandas as pd

from m5knobs import paths, results
from m5knobs.data import SEED_NOISE

SWEEPS = [
    ("Patience window (steps)", "early_stop_patience_steps × val_check_steps",
     [("250", "es_p250"), ("500", "es_p500"), ("1,000", "es_p1000"), ("2,000", "es_p2000"), ("4,000", "es_p4000")]),
    ("Validation tail (weeks)", "val_size", [("13", "es_p1000"), ("26", "es_val26"), ("52", "es_val52")]),
    ("Check every N steps (window fixed at 1,000)", "val_check_steps", [("25", "es_check25"), ("50", "es_p1000"), ("100", "es_check100")]),
    ("Weights kept at the end", "restore_best", [("best (restored)", "es_p1000"), ("last (stop step)", "es_norestore")]),
]


def headline_text(t: pd.DataFrame, base: int = 1000) -> str:
    b = t.loc[t["wape"].idxmin()]
    bw = t.loc[t["window"] == base, "wape"].iloc[0]
    if int(b["window"]) == base:
        return f"Early stopping: the {base:,}-step patience window was the best of those tried"
    bb = t.loc[t["window"] == base, "bias"].iloc[0]
    note = " (within seed noise)" if bw - b["wape"] < SEED_NOISE["wape"] else ""
    trade = (f", but bias {bb:+.1f}% → {b['bias']:+.1f}%" if abs(b["bias"]) - abs(bb) > SEED_NOISE["bias"] else "")
    return f"More patience: {int(b['window']):,} steps gave WMAPE {b['wape']:.2f} vs {bw:.2f} at {base:,}{note}{trade}"


def _summary(arm, results_dir):
    s = results.load_run(arm, results_dir)["summary"]
    fs = (s.get("fold_stats") or [{}])[0]
    return dict(wape=s["wape"], bias=s["bias"], best=fs.get("best_step"), stop=fs.get("stop_step"), cap=bool(fs.get("hit_cap")))


def sweep(results_dir=paths.RESULTS) -> dict:
    """WMAPE, bias, best/stop step and cap per early-stopping arm, written to results/es_sweep.json for FINDINGS.md."""
    arms_needed = sorted({a for _, _, xs in SWEEPS for _, a in xs})
    results.require(arms_needed, results_dir)
    S = {a: _summary(a, results_dir) for a in arms_needed}
    with open(os.path.join(results_dir, "es_sweep.json"), "w") as fh:
        json.dump(S, fh, indent=1)
    return S


def patience_table(S: dict) -> pd.DataFrame:
    """The five patience windows of the sweep: window (steps), WMAPE, bias."""
    return pd.DataFrame([dict(window=int(lab.replace(",", "")), arm=a, wape=S[a]["wape"], bias=S[a]["bias"])
                         for lab, a in SWEEPS[0][2]])
