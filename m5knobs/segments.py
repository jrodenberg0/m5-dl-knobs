"""Deep learning vs LightGBM by segment: WMAPE and bias per volume quartile, zero-week share, department and origin.

Segments come from each series' last HIST_WEEKS weeks of history before the first origin (results/history.parquet).
Series with no history in that window (they start selling later) get no volume quartile or zero-week segment and are
left out of those two kinds; they stay in the department and origin segments.
"""
import numpy as np
import pandas as pd

from m5knobs import paths, results, scoring
from m5knobs.data import SEED_NOISE
from m5knobs.scoring import _series_level, paired_bootstrap_p

MODELS = {"LightGBM": "f4_lgbm", "N-HiTS Huber": "f3_fix", "N-HiTS Huber, best of 5 ES": "es_p4000",
          "N-HiTS Tweedie": "f4_nhits", "TCN Tweedie": "f4_tcn"}
BASE = "LightGBM"
HIST_WEEKS = 52  # one year: a full seasonal cycle, the same window for every series
KINDS = ("volume quartile", "zero weeks", "department", "origin")
QUARTILES = ("Q1 slowest", "Q2", "Q3", "Q4 fastest")
ZERO_LABELS = ("<10%", "10–50%", ">50%")


def _zero_label(s: float) -> str:
    return ZERO_LABELS[0] if s < 0.10 else ZERO_LABELS[1] if s <= 0.50 else ZERO_LABELS[2]


def series_segments(results_dir: str = paths.RESULTS) -> pd.DataFrame:
    """unique_id, mean_hist, zero_share_hist, volume quartile, zero weeks, department for every series in actuals."""
    h = results.history(results_dir)
    h = h[h["ds"] > h["ds"].max() - pd.Timedelta(weeks=HIST_WEEKS)]
    g = h.groupby("unique_id")["y"]
    f = pd.DataFrame({"mean_hist": g.mean(), "zero_share_hist": h["y"].eq(0).groupby(h["unique_id"]).mean()})
    f["volume quartile"] = pd.qcut(f["mean_hist"].rank(method="first"), 4, labels=list(QUARTILES)).astype(object)
    f["zero weeks"] = f["zero_share_hist"].map(_zero_label)
    ids = pd.Index(sorted(results.actuals(results_dir)["unique_id"].unique()), name="unique_id")
    f = f.reindex(ids).reset_index()
    f["department"] = f["unique_id"].str.split("_").str[:2].str.join("_")
    return f


def _order(kind: str, segs) -> list:
    fixed = {"volume quartile": QUARTILES, "zero weeks": ZERO_LABELS}.get(kind)
    return [s for s in fixed if s in set(segs)] if fixed else sorted(segs)


def segment_table(results_dir: str = paths.RESULTS, arms: dict = None, n_boot: int = 2000) -> pd.DataFrame:
    """Long format: segment_kind, segment, n_series, vol_share (of all actual volume), model, wape, bias, abs_bias (%),
    p_wape, p_abs_bias.

    Scored from each run's per_series.parquet (missing forecasts count as 0, as in every summary). Segments with no
    actual volume (WMAPE undefined) are dropped. p_wape / p_abs_bias: paired bootstrap over the segment's series of the
    model's WMAPE / |bias| minus LightGBM's (NaN for LightGBM itself, or with n_boot=0).
    """
    arms = arms or MODELS
    results.require(list(arms.values()), results_dir)
    seg = series_segments(results_dir)
    loaded = {}
    for model, arm in arms.items():
        ps = results.load_run(arm, results_dir)["ps"].merge(seg, on="unique_id", how="left")
        ps["origin"] = pd.to_datetime(ps["cutoff"]).dt.strftime("%Y-%m-%d")
        loaded[model] = ps
    rows = []
    for model, ps in loaded.items():
        total = ps["y_sum"].sum()
        for kind in KINDS:
            g = ps.groupby(kind)
            s = g[["abs_err", "y_sum", "yhat_sum"]].sum().assign(n_series=g["unique_id"].nunique())
            s = s[s["y_sum"] > 0]
            for name in _order(kind, s.index):
                r = s.loc[name]
                bias = 100 * (r["yhat_sum"] / r["y_sum"] - 1)
                pw = pb = np.nan
                if n_boot and model != BASE and BASE in loaded:
                    a, b = (_series_level(d[d[kind] == name]) for d in (ps, loaded[BASE]))
                    pw, pb = segment_p(a, b, n_boot)
                rows.append(dict(segment_kind=kind, segment=name, n_series=int(r["n_series"]), vol_share=r["y_sum"] / total,
                                 model=model, wape=100 * r["abs_err"] / r["y_sum"], bias=bias, abs_bias=abs(bias),
                                 p_wape=pw, p_abs_bias=pb))
    return pd.DataFrame(rows)


def segment_p(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> tuple:
    """Paired bootstrap p of (WMAPE, |bias|) of a minus b over series: the same resampling and p as
    scoring.paired_bootstrap_p (same seed, same draws), vectorized for both statistics at once."""
    a, b = a.set_index("unique_id"), b.set_index("unique_id")
    ids = a.index.intersection(b.index)
    A = a.loc[ids, ["abs_err", "y_sum", "yhat_sum"]].to_numpy(float)
    B = b.loc[ids, ["abs_err", "y_sum", "yhat_sum"]].to_numpy(float)
    rng = np.random.default_rng(seed)
    dw, db = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(ids), len(ids))
        sa, sb = A[idx].sum(0), B[idx].sum(0)
        dw[i] = 100 * sa[0] / sa[1] - 100 * sb[0] / sb[1]
        db[i] = 100 * abs(sa[2] / sa[1] - 1) - 100 * abs(sb[2] / sb[1] - 1)
    p = lambda d: float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))  # noqa: E731
    return p(dw), p(db)


def _wape(d):
    return 100 * d["abs_err"].sum() / d["y_sum"].sum()


def _abs_bias(d):
    return 100 * abs(d["yhat_sum"].sum() / d["y_sum"].sum() - 1)


def overall_gap(results_dir: str, arm_a: str, arm_b: str, n_boot: int = 2000) -> dict:
    """arm_a minus arm_b on WMAPE and |bias| (points), each with a paired bootstrap p over series; signed biases too."""
    a, b = (_series_level(results.load_run(x, results_dir)["ps"]) for x in (arm_a, arm_b))
    bias = lambda d: 100 * (d["yhat_sum"].sum() / d["y_sum"].sum() - 1)  # noqa: E731
    return dict(wape_a=_wape(a), wape_b=_wape(b), gap=_wape(a) - _wape(b), p=paired_bootstrap_p(_wape, a, b, n_boot=n_boot),
                bias_a=bias(a), bias_b=bias(b), abs_bias_gap=_abs_bias(a) - _abs_bias(b),
                p_abs_bias=paired_bootstrap_p(_abs_bias, a, b, n_boot=n_boot))


def p_text(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def claim_text(g: dict) -> str:
    """Claim 4 ('deep learning can match gradient-boosted trees'), judged on N-HiTS Huber (f3_fix) vs LightGBM:
    holds unless N-HiTS Huber's WMAPE is worse by more than seed noise."""
    holds = g["gap"] <= SEED_NOISE["wape"]
    return (f"Claim 4 ('deep learning can match gradient-boosted trees'), judged on N-HiTS Huber (`f3_fix`, the pre-set recipe, "
            f"fit once) vs LightGBM (refit at every origin): WMAPE {g['wape_a']:.2f} vs {g['wape_b']:.2f} "
            f"(gap {g['gap']:+.2f}, {p_text(g['p'])}; seed noise ±{SEED_NOISE['wape']}), |bias| {abs(g['bias_a']):.1f}% vs "
            f"{abs(g['bias_b']):.1f}% ({p_text(g['p_abs_bias'])}). Verdict: " + ("holds." if holds else "does not hold."))


def gap_rows(results_dir: str = paths.RESULTS, models: dict = None, n_boot: int = 2000) -> pd.DataFrame:
    """One row per model: WMAPE, signed bias, |bias| and, for the deep models, the gaps vs LightGBM with bootstrap p."""
    models = models or MODELS
    rows = []
    for model, arm in models.items():
        if model == BASE:
            s = scoring.summarize(results.load_run(arm, results_dir)["ps"])
            g = dict(wape_a=s["wape"], bias_a=s["bias"], gap=np.nan, p=np.nan, abs_bias_gap=np.nan, p_abs_bias=np.nan)
        else:
            g = overall_gap(results_dir, arm, models[BASE], n_boot=n_boot)
        rows.append(dict(model=model, arm=arm, wape=g["wape_a"], bias=g["bias_a"], abs_bias=abs(g["bias_a"]), gap=g["gap"],
                         p=g["p"], abs_bias_gap=g["abs_bias_gap"], p_abs_bias=g["p_abs_bias"]))
    return pd.DataFrame(rows)
