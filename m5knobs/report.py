"""Writes FINDINGS.md: the numbers behind every figure, read from results/ (draw the m5knobs.figures modules first for verdicts/tables)."""
import glob
import json
import os

import numpy as np
import pandas as pd

from m5knobs import paths
from m5knobs.experiments import ARMS
from m5knobs.data import SEED_NOISE


def _done(a, rd):
    return os.path.exists(os.path.join(rd, "runs", f"{a}__s1", "DONE"))


def _grad(tr):
    g = tr["grad_norm"].replace([np.inf, -np.inf], np.nan)
    return dict(median=np.nanmedian(g), max=np.nanmax(g), ratio=np.nanmax(g) / np.nanmedian(g), nonfinite=int((~np.isfinite(tr["grad_norm"])).sum()))


DERIVED = ("card_verdicts.json", "stress_outcomes.json", "es_sweep.json", "f4_table.csv")


def check_derived_fresh(results_dir=paths.RESULTS):
    """Stop unless every file the story notebook derives exists and is newer than the newest finished run."""
    done = [os.path.getmtime(p) for p in glob.glob(os.path.join(results_dir, "runs", "*", "DONE"))]
    newest = max(done, default=0.0)
    bad = [f for f in DERIVED if not os.path.exists(os.path.join(results_dir, f))
           or os.path.getmtime(os.path.join(results_dir, f)) < newest]
    if bad:
        raise SystemExit(f"run notebooks/m5_deep_learning_story.ipynb first: missing or older than the newest finished run in results/: {', '.join(bad)}")


def run_status(results_dir=paths.RESULTS) -> pd.DataFrame:
    """One row per arm (seed 1): status done/error/missing and, for finished runs, the summary numbers."""
    rows = []
    for a in ARMS:
        d = os.path.join(results_dir, "runs", f"{a.id}__s1")
        s = json.load(open(os.path.join(d, "summary.json"))) if _done(a.id, results_dir) else {}
        fs = (s.get("fold_stats") or [{}])[0]
        rows.append(dict(arm=a.id, study=a.group,
                         status="done" if s else "error" if os.path.exists(os.path.join(d, "ERROR.txt")) else "missing",
                         wape=s.get("wape"), bias=s.get("bias"),
                         device=("cpu" if a.arch == "LGBM" else s.get("accelerator")) if s else None,
                         stop_step=fs.get("stop_step", ""), best_step=fs.get("best_step", ""),
                         hit_cap=fs.get("hit_cap"), fit_seconds=fs.get("fit_seconds", 0)))
    df = pd.DataFrame(rows)
    for c in ("device", "stop_step", "best_step", "hit_cap", "fit_seconds"):
        df[c] = pd.Series([r[c] for r in rows], dtype=object)
    return df


def unforecast(results_dir=paths.RESULTS):
    """Series with no sales before an origin (no neural forecast, scored as 0): per-origin counts and volume, or None."""
    act_p, fc_p = os.path.join(results_dir, "actuals.parquet"), os.path.join(results_dir, "runs", "f2_clip__s1", "forecasts.parquet")
    if not (os.path.exists(act_p) and os.path.exists(fc_p)):
        return None
    act, fc = pd.read_parquet(act_p), pd.read_parquet(fc_p, columns=["unique_id", "ds", "cutoff"])
    m = act.merge(fc.assign(has=1), on=["unique_id", "ds", "cutoff"], how="left")
    miss = m[m["has"].isna()]
    per = miss.groupby("cutoff")["unique_id"].nunique()
    return dict(per_min=per.min(), per_max=per.max(), units=miss["y"].sum(), total=act["y"].sum())


def _segment_section(results_dir, n_boot) -> list:
    """Deep learning vs LightGBM: overall gaps with bootstrap p, claim 4 judged on N-HiTS Huber (f3_fix), and per-segment tables.
    Empty when history.parquet (results-v2) or one of the compared runs is missing."""
    from m5knobs import segments
    if not (os.path.exists(os.path.join(results_dir, "history.parquet")) and all(_done(a, results_dir) for a in segments.MODELS.values())):
        return []
    M = segments.MODELS
    g = segments.gap_rows(results_dir, n_boot=n_boot)
    pfmt = lambda p: "" if pd.isna(p) else "<0.001" if p < 0.001 else f"{p:.3f}"  # noqa: E731
    dfmt = lambda d: "" if pd.isna(d) else f"{d:+.2f}"  # noqa: E731
    L = ["## Deep learning vs LightGBM by segment", "",
         "WMAPE = Σ|error| / Σ actual (the WAPE above); bias = Σ forecast / Σ actual − 1, and |bias| its size. Same series, weeks and "
         "scoring for every model. LightGBM (`f4_lgbm`) is refit at every origin; N-HiTS Huber (`f3_fix`) is the pre-set recipe, fit "
         f"once at the first origin. N-HiTS Huber, best of 5 ES (`{M['N-HiTS Huber, best of 5 ES']}`) is the best of 5 early-stopping "
         "settings from the early-stopping study, picked after seeing the results. The Tweedie rows are the deep models of the "
         "architecture comparison. Δ and p are against LightGBM (paired bootstrap over series).", "",
         "| model | arm | WMAPE | Δ WMAPE | p | bias | \\|bias\\| | Δ \\|bias\\| | p |", "|---|---|---|---|---|---|---|---|---|"]
    for r in g.itertuples():
        L.append(f"| {r.model} | {r.arm} | {r.wape:.2f} | {dfmt(r.gap)} | {pfmt(r.p)} | {r.bias:+.1f}% | {r.abs_bias:.1f}% | "
                 f"{dfmt(r.abs_bias_gap)} | {pfmt(r.p_abs_bias)} |")
    gi = g.set_index("model")
    huber = gi.loc["N-HiTS Huber"]
    L += ["", "- " + segments.claim_text(dict(wape_a=huber.wape, wape_b=gi.loc[segments.BASE, "wape"], gap=huber.gap, p=huber.p,
                                                bias_a=huber.bias, bias_b=gi.loc[segments.BASE, "bias"], p_abs_bias=huber.p_abs_bias)),
          "- With the stock Tweedie loss: " + "; ".join(
              f"{m} {gi.loc[m, 'gap']:+.2f} WMAPE vs LightGBM ({segments.p_text(gi.loc[m, 'p'])}), bias {gi.loc[m, 'bias']:+.1f}%"
              for m in M if "Tweedie" in m) + f" (LightGBM {gi.loc[segments.BASE, 'bias']:+.1f}%).", ""]
    seg = segments.series_segments(results_dir)
    act = pd.read_parquet(os.path.join(results_dir, "actuals.parquet"), columns=["unique_id", "y"])
    nohist = set(seg.loc[seg["volume quartile"].isna(), "unique_id"])
    L += [f"Segments: volume quartile = quartile of mean weekly sales over the {segments.HIST_WEEKS} weeks before the first origin "
          f"(equal series counts, Q1 slowest); zero weeks = share of those weeks with no sales; department = first two parts of the "
          f"item id; origin = forecast origin. {len(nohist):,} series have no sales history in that window "
          f"({100 * act.loc[act.unique_id.isin(nohist), 'y'].sum() / act['y'].sum():.2f}% of actual volume): they are left out of the "
          "volume and zero-week segments and kept in department and origin. Segments with no actual sales are dropped. "
          "Cells: WMAPE / bias.", ""]
    t = segments.segment_table(results_dir)
    for kind in segments.KINDS:
        k = t[t.segment_kind == kind]
        L += [f"### By {kind}", "", "| segment | series | share of volume | " + " | ".join(M) + " |", "|---" * (3 + len(M)) + "|"]
        for s, d in k.groupby("segment", sort=False):
            d = d.set_index("model")
            L.append(f"| {s} | {d['n_series'].iloc[0]:,} | {100 * d['vol_share'].iloc[0]:.1f}% | "
                     + " | ".join(f"{d.at[m, 'wape']:.2f} / {d.at[m, 'bias']:+.1f}%" for m in M) + " |")
        L.append("")
    return L


def write(results_dir=paths.RESULTS, out=None, n_boot=2000):
    out = out or paths.FINDINGS
    L = ["# Findings: training settings for deep learning on M5", "",
         "Generated by `notebooks/m5_deep_learning_story.ipynb` from `results/`. Single seed (1) for every arm; "
         f"earlier M5 seed noise ≈ {SEED_NOISE['wape']} WAPE, ±{SEED_NOISE['bias']:.0f}pt bias (author's earlier 3-seed M5 runs).", ""]

    L += ["## Run status", "", "| arm | status | device | WAPE | bias | stop step | best step | hit cap | fit s |", "|---|---|---|---|---|---|---|---|---|"]
    for r in run_status(results_dir).to_dict("records"):
        if r["status"] == "done":
            L.append(f"| {r['arm']} | done | {r['device'] or 'mps*'} | {r['wape']:.2f} | {r['bias']:+.2f} | {r['stop_step']} | {r['best_step']} | "
                     f"{'yes' if r['hit_cap'] else ''} | {r['fit_seconds']:.0f} |")
        else:
            L.append(f"| {r['arm']} | {'ERROR' if r['status'] == 'error' else 'missing'} | | | | | | | |")
    L += ["", "\\* device not recorded in the summary (runs made before the runner recorded it); they ran in the MPS job. LightGBM always runs "
          "on CPU.", ""]
    u = unforecast(results_dir)
    if u:
        L += [f"Series with no sales before an origin get no neural forecast and are scored as 0: {u['per_min']}–{u['per_max']} series per origin, "
              f"{u['units']:,.0f} of {u['total']:,.0f} actual units ({100 * u['units'] / u['total']:.2f}% of volume), "
              "the same for every neural run.", ""]

    cv = os.path.join(results_dir, "card_verdicts.json")
    if os.path.exists(cv):
        from m5knobs.figures.cards import CARDS
        V = json.load(open(cv))
        L += ["## Setting cards (N-HiTS, one setting changed vs the recipe)", "",
              "Verdict = worse direction + paired bootstrap over series (p < 0.05) + effect floor (WAPE gap > "
              f"{SEED_NOISE['wape']} for underfit, |bias| worse by > {SEED_NOISE['bias']:.0f}pt for biased).", "",
              "| card | knob | without → with | loss | WAPE | bias | verdict on 'without' |", "|---|---|---|---|---|---|---|"]
        for c in CARDS:
            if c["key"] not in V or not (_done(c["mistake"], results_dir) and _done(c["fix"], results_dir)):
                continue
            m, f = (json.load(open(os.path.join(results_dir, "runs", f"{a}__s1", "summary.json"))) for a in (c["mistake"], c["fix"]))
            L.append(f"| {c['name']} | `{c['knob']}` | {c['m_label']} → {c['f_label']} | {c['loss'] or 'MAE / Huber δ5'} | "
                     f"{m['wape']:.2f} → {f['wape']:.2f} | {m['bias']:+.1f}% → {f['bias']:+.1f}% | {V[c['key']]['label']} |")
        L += ["", "Caveats:",
              "- The effect floor was added after seeing single-seed results: with thousands of series the bootstrap alone flags gaps far "
              "inside seed noise.",
              "- In-sample error is MAE in raw units at the restored best weights; the MAE-loss arm optimizes that same metric.",
              "- MinMax card: Huber δ5 is computed in scaled units, so on MinMax's [0, 1] scale it never caps errors (acts as MSE).",
              "- Card forecasts at origins 2 and 3 come from the origin-1 fit (no refit); the architecture table refits at every origin.", ""]

    so = os.path.join(results_dir, "stress_outcomes.json")
    if os.path.exists(so):
        O = json.load(open(so))
        L += ["## Clipping stress test (Tweedie, stressors fixed before any run)", "",
              "| stressor | clip off: WAPE / bias / best step / spike max÷median | AutoClip: same | outcome |", "|---|---|---|---|"]
        for k, r in O.items():
            cells = []
            for a in (f"st_{k}_noclip", f"st_{k}_clip"):
                pth = os.path.join(results_dir, "runs", f"{a}__s1", "summary.json")
                if os.path.exists(pth):
                    sm = json.load(open(pth))
                    fs = (sm.get("fold_stats") or [{}])[0]
                    cells.append(f"{sm['wape']:.2f} / {sm['bias']:+.1f}% / {int(fs.get('best_step') or 0):,} / "
                                 f"{fs.get('grad_norm_max', float('nan')) / fs.get('grad_norm_median', float('nan')):,.0f}×")
                else:
                    cells.append("crashed")
            L.append(f"| {k} | {cells[0]} | {cells[1]} | {r['outcome']} |")
        L += ["", "A run 'diverged' when its validation loss ended above 2× its best, i.e. it didn't recover (or it crashed with a NaN loss). "
              "This rule was set after seeing the results; the stressors themselves were fixed before any run. When both runs diverge, "
              "the WAPE gap reflects which early checkpoint was restored, so it isn't read as a clipping effect.", ""]

    es = os.path.join(results_dir, "es_sweep.json")
    if os.path.exists(es):
        E = json.load(open(es))
        L += ["## Early-stopping study (Huber δ5, one-cycle over 10,000 steps for every arm, no minimum-steps guard)", "",
              "| arm | WAPE | bias | best step | stop step | hit cap |", "|---|---|---|---|---|---|"]
        from m5knobs.figures.early_stopping import SWEEPS
        order = list(dict.fromkeys(a for _, _, xs in SWEEPS for _, a in xs))
        for a in order:
            r = E[a]
            L.append(f"| {a} | {r['wape']:.2f} | {r['bias']:+.1f}% | {int(r['best'] or 0):,} | {int(r['stop'] or 0):,} | {'yes' if r['cap'] else ''} |")
        L.append("")

    tp = os.path.join(results_dir, "toy_paths.parquet")
    if os.path.exists(tp):
        p = pd.read_parquet(tp)
        L += ["## Toy fit (real 2-weight Tweedie GLM)", ""]
        for run, g in p[p.step > 0].groupby("run"):
            steps = np.hypot(g.a.diff(), g.b.diff())
            L.append(f"- {run}: final loss {g.loss.iloc[-1]:.3f}, min {g.loss.min():.3f}, loss std over last 100 steps "
                     f"{g.loss.iloc[-100:].std():.3f}, max step length {steps.max():.3f}, grad max/median {g.grad_norm.max() / g.grad_norm.median():.1f}×")
        L.append("")

    tp = os.path.join(results_dir, "f4_table.csv")
    if os.path.exists(tp):
        t = pd.read_csv(tp)
        L += ["## Architecture comparison (refit at every origin, Tweedie)", "", t.to_markdown(index=False, floatfmt=".3f"),
              "", "LightGBM: direct (one model per lead), lags 1–26, rolling mean/std/max/min over 4/13/26/52 weeks, expanding mean, calendar "
              "features.", ""]

    L += _segment_section(results_dir, n_boot)

    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    return out

