"""Pick the example series each figure shows, by a rule tied to the mechanism the section claims.

pair_table() lines up two arms series by series over the forecast weeks (both arms must have forecast the series).
Each pick_* rule returns a Pick with:
- the example (uid), the rule text and the example's rank among the series the rule ranked (n_candidates);
- n_pattern / pattern_share: how many series of the population show the rule's mechanism at all;
- n_helped / n_hurt / n_changed over the population (all series in the pair, or the named segment), counted among
  series selling more than HELP_MIN_Y units (n_counted), independent of the rule's filters;
- WMAPE and |bias| of both forecasts on the example series, for the figure.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from m5knobs import results

HELP_FRAC = 0.20     # helped/hurt: error changes by more than 20% of the mistaken arm's error ...
HELP_MIN_Y = 50      # ... counted only for series selling more than 50 units over the forecast weeks
CHANGE_FRAC = 0.10   # changed: forecasts differ by more than 10% of their mean volume
LAG = 52
LONG_GAP = 13   # a quarter or more without a sale in history: off the shelf or out of stock, not demand
SHORT_GAP = 8   # intermittent demand: zero weeks come in runs shorter than 8 weeks
STOCKOUT = 4    # a run of 4+ zero weeks in a series that otherwise sells: treated as a stock-out
FULL_HIST = 52  # "full history": the series sold for at least a year before the first origin (lag features exist)
LEVEL = (2 / 3, 1.5)  # "no level shift": forecast-week mean sales within these multiples of the history mean
CLOSE = 0.10    # "lands close": the second forecast's total within 10% of actual sales
LOW = 0.10      # "steady": no week below 10% of the series' median weekly sales in history (near-zero stock-out weeks)
FLATTER = 0.75  # "flatter": the forecast swings at most 3/4 as much (amp ratio) as the fix's
MOVES = 0.5     # a forecast "visibly follows" a shape when it swings at least half as much as actual sales


@dataclass
class Pick:
    uid: str
    rule: str
    rank: int
    n_candidates: int
    n_pattern: int
    pattern_share: float
    n_population: int
    n_counted: int
    n_helped: int
    n_hurt: int
    n_changed: int
    wmape_wrong: float
    wmape_right: float
    absbias_wrong: float
    absbias_right: float
    pattern_text: str = ""        # what "show the pattern" means, with {wrong} / {right} placeholders
    n_pattern_base: int = 0       # series the pattern share is over (default: the population)
    pattern_base_text: str = ""   # name of that base when it is not the population

    def text(self, s: str, labels) -> str:
        """s with {wrong} / {right} replaced by the two forecasts' labels."""
        return s.replace("{wrong}", labels[0]).replace("{right}", labels[1])


def history_features(hist: pd.DataFrame) -> pd.DataFrame:
    """unique_id, n_hist, mean_hist, zero_share_hist, max_zero_run_hist (longest run of consecutive zero weeks)."""
    h = hist.sort_values(["unique_id", "ds"])
    f = pd.DataFrame({"n_hist": h.groupby("unique_id").size(), "mean_hist": h.groupby("unique_id")["y"].mean(),
                      "zero_share_hist": h["y"].eq(0).groupby(h["unique_id"]).mean(),
                      "max_zero_run_hist": _max_zero_run(h)})
    f.index.name = "unique_id"
    return f.reset_index()


def _max_zero_run(h: pd.DataFrame) -> pd.Series:
    """Longest run of consecutive zero weeks per series; h sorted by unique_id, ds."""
    uid, zero = h["unique_id"], h["y"].eq(0)
    run = (zero.ne(zero.shift()) | uid.ne(uid.shift())).cumsum()
    return zero.groupby(run).sum().groupby(uid.groupby(run).first()).max()


def _corr(df: pd.DataFrame, a: str, b: str, min_n: int = 13) -> pd.Series:
    """Pearson correlation of columns a, b per series over rows where both exist; NaN under min_n rows or if flat."""
    d = df[df[a].notna() & df[b].notna()]
    g = d.groupby("unique_id")
    da, db = d[a] - g[a].transform("mean"), d[b] - g[b].transform("mean")
    num = (da * db).groupby(d["unique_id"]).sum()
    den = np.sqrt((da ** 2).groupby(d["unique_id"]).sum() * (db ** 2).groupby(d["unique_id"]).sum())
    return (num / den.where(den > 0)).where(g.size() >= min_n)


def _amp(df: pd.DataFrame, col: str) -> pd.Series:
    """std(forecast)/std(actual) per series over all its forecast weeks (population std, as scoring._amp)."""
    g = df.groupby("unique_id")
    sa = g["y"].std(ddof=0)
    return (g[col].std(ddof=0) / sa).where(sa > 0)


def pair_table(results_dir: str, arm_wrong: str, arm_right: str) -> pd.DataFrame:
    """One row per series both arms forecast, summed over every forecast week the two arms and actuals share.

    gain = err_wrong - err_right (> 0: the second arm is better). Columns beyond the sums:
    - yearly_repeat: correlation of forecast-week actuals with the same weeks a year earlier (taken from history;
      60 weeks of history alone hold only 8 lag-52 pairs). NaN under 53 history weeks, under 13 pairs, or flat sales.
    - corr52_wrong / corr52_right: correlation of each forecast with last year's sales on the same weeks.
    - zero_share_fc, max_zero_run_fc, zeros_y52: zero weeks in the forecast-week actuals / in last year's same weeks.
    - shape_gain: gain after rescaling each forecast to the actual total (the part not explained by level).
    - origins, origins_wrong_below_right, origins_wrong_below_y: per-origin totals compared.
    - low_weeks_hist, low_weeks_fc: weeks selling under LOW x the series' median weekly sales in history (near-zero weeks).
    Assumes origins do not overlap in ds (true for the runner's 13-week origins); weeks counts unique ds.
    """
    key = ["unique_id", "ds", "cutoff"]
    fw = results.load_run(arm_wrong, results_dir)["fc"][key + ["yhat"]].rename(columns={"yhat": "f_wrong"})
    fr = results.load_run(arm_right, results_dir)["fc"][key + ["yhat"]].rename(columns={"yhat": "f_right"})
    m = results.actuals(results_dir)[key + ["y"]].merge(fw, on=key).merge(fr, on=key)
    m[["y", "f_wrong", "f_right"]] = m[["y", "f_wrong", "f_right"]].astype(float)
    hist = results.history(results_dir)
    prev = hist[["unique_id", "ds", "y"]].rename(columns={"y": "y52"})
    prev["ds"] = prev["ds"] + pd.Timedelta(weeks=LAG)
    m = m.merge(prev.astype({"y52": float}), on=["unique_id", "ds"], how="left").sort_values(["unique_id", "ds"])
    m["err_wrong"] = (m["f_wrong"] - m["y"]).abs()
    m["err_right"] = (m["f_right"] - m["y"]).abs()
    m["diff"] = (m["f_wrong"] - m["f_right"]).abs()
    g = m.groupby("unique_id")
    t = g[["y", "f_wrong", "f_right", "err_wrong", "err_right", "diff"]].sum()
    t.insert(0, "weeks", g["ds"].nunique())
    t["gain"] = t["err_wrong"] - t["err_right"]
    t["rel_change"] = t.pop("diff") / np.maximum((t["f_wrong"] + t["f_right"]) / 2, 1)
    t["amp_wrong"] = _amp(m, "f_wrong")
    t["amp_right"] = _amp(m, "f_right")
    for c in ("wrong", "right"):
        scale = (t["y"] / t[f"f_{c}"].where(t[f"f_{c}"] > 0)).reindex(m["unique_id"]).to_numpy()
        m[f"se_{c}"] = (m[f"f_{c}"] * scale - m["y"]).abs()
    se = m.groupby("unique_id")[["se_wrong", "se_right"]].sum(min_count=1)
    t["shape_gain"] = se["se_wrong"] - se["se_right"]
    t["zero_share_fc"] = m["y"].eq(0).groupby(m["unique_id"]).mean()
    t["max_zero_run_fc"] = _max_zero_run(m)
    t["zeros_y52"] = m["y52"].eq(0).groupby(m["unique_id"]).sum()
    t["corr52_wrong"] = _corr(m, "f_wrong", "y52")
    t["corr52_right"] = _corr(m, "f_right", "y52")
    t["yearly_repeat"] = _corr(m, "y", "y52")
    o = m.groupby(["unique_id", "cutoff"])[["y", "f_wrong", "f_right"]].sum()
    og = o.groupby("unique_id")
    t["origins"] = og.size()
    t["origins_wrong_below_right"] = (o["f_wrong"] < o["f_right"]).groupby("unique_id").sum()
    t["origins_wrong_below_y"] = (o["f_wrong"] < o["y"]).groupby("unique_id").sum()
    hf = history_features(hist)
    med = hist.groupby("unique_id")["y"].median()
    low = lambda d: (d["y"] < LOW * d["unique_id"].map(med)).groupby(d["unique_id"]).sum()  # noqa: E731
    t["low_weeks_fc"] = low(m)
    hf["low_weeks_hist"] = hf["unique_id"].map(low(hist)).fillna(0).astype(int)
    t = t.reset_index().merge(hf, on="unique_id", how="left")
    t.loc[~(t["n_hist"] > LAG), "yearly_repeat"] = np.nan
    cols = ["unique_id", "weeks", "y", "f_wrong", "f_right", "err_wrong", "err_right", "gain", "rel_change",
            "n_hist", "mean_hist", "zero_share_hist", "max_zero_run_hist", "yearly_repeat", "amp_wrong", "amp_right",
            "shape_gain", "zero_share_fc", "max_zero_run_fc", "zeros_y52", "corr52_wrong", "corr52_right",
            "origins", "origins_wrong_below_right", "origins_wrong_below_y", "low_weeks_hist", "low_weeks_fc"]
    return t[cols]


REGULAR = (f"series selling at least the median weekly volume in history, with no run of {LONG_GAP}+ zero weeks "
           "(off the shelf or out of stock)")
STEADY = (f"series selling at least the median weekly volume in history, with no run of {STOCKOUT}+ zero weeks and no week "
          f"under {LOW:.0%} of the series' median weekly sales (history or forecast weeks), at least {FULL_HIST} weeks "
          f"of history, and forecast-week sales within {LEVEL[0]:.2f}-{LEVEL[1]:.1f}x the history level (no level shift)")
# Rule and pattern texts name the two forecasts as {wrong} and {right}; Pick.text() fills in the labels.


def _regular(t: pd.DataFrame) -> pd.Series:
    return (t["mean_hist"] >= t["mean_hist"].median()) & (t["max_zero_run_hist"] < LONG_GAP)


def _steady(t: pd.DataFrame) -> pd.Series:
    return ((t["mean_hist"] >= t["mean_hist"].median()) & (t["max_zero_run_hist"] < STOCKOUT)
            & (t["max_zero_run_fc"] < STOCKOUT) & (t["low_weeks_hist"] == 0) & (t["low_weeks_fc"] == 0)
            & (t["n_hist"] >= FULL_HIST)
            & ((t["y"] / t["weeks"]) / t["mean_hist"].where(t["mean_hist"] > 0)).between(*LEVEL))


def _intermittent(t: pd.DataFrame) -> pd.Series:
    return t["zero_share_hist"] >= 0.30


def _mask(t: pd.DataFrame, mask) -> pd.Series:
    """Boolean mask aligned to t: accepts a mask on t's index or one keyed by unique_id; refuses anything else."""
    if mask is None:
        return pd.Series(True, index=t.index)
    if mask.index.equals(t.index):
        return mask.fillna(False).astype(bool)
    if set(t["unique_id"]) <= set(mask.index):
        return pd.Series(t["unique_id"].map(mask).fillna(False).astype(bool).to_numpy(), index=t.index)
    raise ValueError("mask must be indexed like the pair table or keyed by unique_id")


def _pct(f, y):
    return float(100 * f / y) if y > 0 else float("nan")


def _make(t, row, rule, rank, n_cand, pattern, pop, pattern_text, pattern_base=None, base_text="") -> Pick:
    """Population counts: helped/hurt/changed over pop and y > HELP_MIN_Y; pattern share over pattern_base (default pop)."""
    p = t[pop]
    big = p[p["y"] > HELP_MIN_Y]
    base = pop if pattern_base is None else (_mask(t, pattern_base) & pop)
    base_text = base_text or ("" if pop.all() else "series in the segment")
    pat = int((pattern & base).fillna(False).sum())
    nb = int(base.sum())
    r = t.loc[row]
    return Pick(uid=r["unique_id"], rule=rule, rank=int(rank), n_candidates=int(n_cand), n_pattern=pat,
                pattern_share=pat / nb if nb else float("nan"), n_population=len(p), n_counted=len(big),
                n_helped=int((big["gain"] > HELP_FRAC * big["err_wrong"]).sum()),
                n_hurt=int((-big["gain"] > HELP_FRAC * big["err_wrong"]).sum()),
                n_changed=int((big["rel_change"] > CHANGE_FRAC).sum()),
                wmape_wrong=_pct(r["err_wrong"], r["y"]), wmape_right=_pct(r["err_right"], r["y"]),
                absbias_wrong=abs(_pct(r["f_wrong"], r["y"]) - 100), absbias_right=abs(_pct(r["f_right"], r["y"]) - 100),
                pattern_text=pattern_text, n_pattern_base=nb, pattern_base_text=base_text)


def _max(t, cand, score, rule, pattern, pattern_text, pop=None, tiebreak=None, pattern_base=None, base_text="") -> Pick:
    """Candidate with the largest score (then tiebreak, then unique_id); typical series if none has score > 0."""
    pop = _mask(t, pop)
    cand = cand.fillna(False).astype(bool) & pop
    c = t[cand].assign(_s=score[cand], _t=(tiebreak if tiebreak is not None else score)[cand])
    c = c[c["_s"].notna()].sort_values(["_s", "_t", "unique_id"], ascending=[False, False, True], kind="mergesort")
    if c.empty or not c["_s"].iloc[0] > 0:
        p = pick_typical(t, pop)
        p.rule = f"no series matched {rule}; typical series instead: {p.rule}"
        return p
    return _make(t, c.index[0], rule, 1, len(c), pattern, pop, pattern_text, pattern_base, base_text)


def pick_typical(t: pd.DataFrame, pop=None) -> Pick:
    """Series with the median forecast change between the two arms (rank = its place in that order).

    Pattern: forecasts change by at most CHANGE_FRAC (the "no clear effect" reading).
    """
    pop = _mask(t, pop)
    reg = _regular(t) & pop
    c = t[reg if reg.any() else pop].sort_values(["rel_change", "unique_id"], kind="mergesort")
    i = (len(c) - 1) // 2
    return _make(t, c.index[i], f"median forecast change between {{wrong}} and {{right}} among {REGULAR}", i + 1, len(c),
                 t["rel_change"] <= CHANGE_FRAC, pop,
                 f"have {{wrong}} and {{right}} forecasts within {CHANGE_FRAC:.0%} of each other (no clear change)")


def pick_overforecast_fix(t: pd.DataFrame) -> Pick:
    """Clipping: largest error reduction where the mistaken arm forecasts above both the fix and the actuals."""
    pattern = (t["f_wrong"] > t["f_right"]) & (t["f_wrong"] > t["y"])
    return _max(t, _regular(t) & pattern, t["gain"], "largest error reduction by {right} where {wrong} over-forecasts "
                f"(above {{right}} and above actual sales), among {REGULAR}", pattern,
                "have the {wrong} total above both the {right} total and actual sales")


def pick_seasonal_gain(t: pd.DataFrame) -> Pick:
    """Input window: the fix follows last year's shape and the mistaken setting does not.

    Candidates: top quartile of yearly_repeat, no stock-outs (<5% zero weeks in history, no zero week in the forecast
    weeks or in last year's same weeks), the fix lowers the error and its forecast visibly moves (amp_right >= MOVES).
    Score: corr52_right - corr52_wrong.
    """
    pattern = (t["yearly_repeat"] >= t["yearly_repeat"].quantile(0.75)) & (t["corr52_right"] > t["corr52_wrong"])
    cand = (_regular(t) & pattern & (t["zero_share_hist"] < 0.05) & (t["zero_share_fc"] == 0) & (t["zeros_y52"] == 0)
            & (t["gain"] > 0) & (t["amp_right"] >= MOVES))
    return _max(t, cand, t["corr52_right"] - t["corr52_wrong"],
                "largest gain in how closely the forecast follows last year's sales on the same weeks ({right} minus "
                "{wrong}), among series whose sales repeat last year's (top quartile), without stock-outs, "
                f"where {{right}} lowers the error and its forecast swings at least {MOVES:.0%} as much as actual sales, "
                f"among {REGULAR}", pattern,
                "repeat last year's sales (top quartile) and have the {right} forecast that follows last year's sales more "
                "closely than {wrong}")


def pick_flatter_wrong(t: pd.DataFrame) -> Pick:
    """Early stopping: the mistaken setting's forecast is clearly flatter, and the fix's shape (not level) is better.

    Score: shape_gain (error gain after rescaling both forecasts to the actual total); no stock-out runs of
    STOCKOUT+ weeks in history or the forecast weeks.
    """
    pattern = t["amp_wrong"] < FLATTER * t["amp_right"]
    cand = (_regular(t) & pattern & (t["max_zero_run_hist"] < STOCKOUT) & (t["max_zero_run_fc"] < STOCKOUT)
            & (t["gain"] > 0))
    return _max(t, cand, t["shape_gain"], "largest error reduction from the shape of the forecast (both forecasts "
                f"rescaled to actual sales) where {{wrong}} swings at most {FLATTER:.0%} as much as {{right}}, among "
                f"{REGULAR} and no run of {STOCKOUT}+ zero weeks", pattern,
                f"have the {{wrong}} forecast swinging at most {FLATTER:.0%} as much as the {{right}} forecast")


def pick_underforecast_wrong(t: pd.DataFrame) -> Pick:
    """Loss: on an intermittent, moderate-volume series the mistaken setting sits below actuals and below the fix at
    every origin. Score: share of the under-forecast closed by the fix, (min(f_right, y) - f_wrong) / (y - f_wrong);
    ties by the amount closed.
    """
    pattern = _intermittent(t) & (t["f_wrong"] < t["f_right"])
    cand = (_intermittent(t) & (t["zero_share_fc"] >= 0.30) & (t["max_zero_run_hist"] < SHORT_GAP)
            & (t["max_zero_run_fc"] < SHORT_GAP) & (t["y"] > HELP_MIN_Y)
            & (t["mean_hist"] < t["mean_hist"].quantile(0.75))
            & (t["origins_wrong_below_right"] == t["origins"]) & (t["origins_wrong_below_y"] == t["origins"]))
    closed = np.minimum(t["f_right"], t["y"]) - t["f_wrong"]
    return _max(t, cand, closed / (t["y"] - t["f_wrong"]),
                "largest share of the {wrong} under-forecast closed by {right}, among series with at least "
                f"30% zero weeks in history and in the forecast weeks (runs under {SHORT_GAP} weeks), selling over "
                f"{HELP_MIN_Y} units but below the top volume quartile, where {{wrong}} is below actual sales and below "
                "{right} at every origin", pattern,
                "are intermittent (at least 30% zero weeks) and have the {wrong} total below the {right} total",
                tiebreak=closed)


def pick_floor(t: pd.DataFrame) -> Pick:
    """Tweedie floor: mostly-zero series still on sale, forecast at about 1 a week; largest error cut by the fix.

    Pattern (over the mostly-zero series, >= 50% zero weeks in history): the mistaken setting forecasts 0.9 a week or
    more while actual sales average under 0.5 a week, i.e. the floor bites.
    """
    per_week_f, per_week_y = t["f_wrong"] / t["weeks"], t["y"] / t["weeks"]
    mostly_zero = t["zero_share_hist"] >= 0.5
    pattern = mostly_zero & (per_week_f >= 0.9) & (per_week_y < 0.5)
    cand = pattern & (t["y"] >= 1) & (per_week_f <= 1.2)
    return _max(t, cand, t["gain"], "largest error reduction by {right} among series with at least 50% zero weeks in "
                "history that sold under 0.5 a week (but sold) in the forecast weeks while {wrong} forecast 0.9-1.2 a "
                "week", pattern, "have the {wrong} forecast at 0.9 a week or more while actual sales average under 0.5 "
                "a week", pattern_base=mostly_zero, base_text="series with at least 50% zero weeks")


def pick_right_wins_segment(t: pd.DataFrame, mask) -> Pick:
    """Largest error advantage of the second arm inside the segment (population = the segment), among steady,
    long-history series."""
    wm = t["gain"] / t["y"].where(t["y"] > 0)
    return _max(t, _steady(t), wm, f"largest WMAPE advantage (points on the series) of {{right}} over {{wrong}} in the segment, "
                f"among {STEADY}", t["gain"] > 0, "have the lower error with {right}", pop=mask, tiebreak=t["gain"])


def pick_right_closer_total(t: pd.DataFrame, mask) -> Pick:
    """For a claim about bias inside a segment (population = the segment): the steady, long-history series where the second
    arm's total lands within CLOSE of actual sales and cuts the first arm's miss on total volume the most, in points of the
    series' actual total (|f_wrong - y| - |f_right - y|) / y.

    Steady: _steady (no stock-outs, full history, no level shift)."""
    cut = ((t["f_wrong"] - t["y"]).abs() - (t["f_right"] - t["y"]).abs()) / t["y"].where(t["y"] > 0)
    cand = _steady(t) & ((t["f_right"] - t["y"]).abs() <= CLOSE * t["y"])
    return _max(t, cand, cut, "largest cut in the miss on total volume (|forecast total - actual total| as a share of actual) "
                f"by {{right}} compared with {{wrong}} in the segment, where the {{right}} total is within {CLOSE:.0%} of "
                f"actual, among {STEADY}",
                cut > 0, "have the {right} total closer to actual sales than the {wrong} total", pop=mask)


def pick_wrong_wins_segment(t: pd.DataFrame, mask) -> Pick:
    """Largest error advantage of the first arm inside the segment (population = the segment), among steady,
    long-history series."""
    wm = -t["gain"] / t["y"].where(t["y"] > 0)
    return _max(t, _steady(t), wm, f"largest WMAPE advantage (points on the series) of {{wrong}} over {{right}} in the segment, "
                f"among {STEADY}", t["gain"] < 0, "have the lower error with {wrong}", pop=mask, tiebreak=-t["gain"])


def bias_summary(t: pd.DataFrame, mask=None) -> dict:
    """Bias-based typicality over a population (default: intermittent series, >= 30% zero weeks in history):
    n, share where the mistaken forecast is below the fix, and total bias (%) of each forecast."""
    p = t[_mask(t, _intermittent(t) if mask is None else mask)]
    y = p["y"].sum()
    return dict(n=len(p), share_wrong_below_right=float((p["f_wrong"] < p["f_right"]).mean()) if len(p) else float("nan"),
                bias_wrong=_pct(p["f_wrong"].sum(), y) - 100, bias_right=_pct(p["f_right"].sum(), y) - 100)
