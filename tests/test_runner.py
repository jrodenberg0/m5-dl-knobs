import os

import pandas as pd
import pytest

import arms
import run_experiments as rx

EXPECTED_IDS = ["f2_noclip", "f2_clip", "f3_fix", "f3_fix_noguard", "f3_patience250", "f3_input13", "f3_minmax", "f3_mae",
                "f4_lgbm", "f4_nhits", "f4_tcn"]


def _panel():
    ds = pd.date_range("2015-01-03", periods=80, freq="W-SAT")
    p = pd.concat([pd.DataFrame({"unique_id": f"s{i}", "ds": ds, "y": float(i + 1)}) for i in range(4)], ignore_index=True)
    p.loc[(p.unique_id == "s3") & (p.ds.dt.isocalendar().week % 3 != 0), "y"] = 0.0  # one intermittent series (~2/3 zeros)
    return p


@pytest.fixture
def fake(monkeypatch):
    calls = []
    panel = _panel()
    cutoffs = [panel.ds.max() - pd.Timedelta(weeks=13 * (3 - i)) for i in range(3)]

    def train_arm(arm, seed, panel, cutoffs):
        calls.append((arm.id, seed))
        fc = []
        for c in cutoffs:
            ds = pd.date_range(c + pd.Timedelta(weeks=1), periods=13, freq="W-SAT")
            fc.append(pd.DataFrame({"unique_id": "s0", "ds": ds, "cutoff": c, "yhat": 1.0}))
        trace = pd.DataFrame({"step": [1, 2], "grad_norm": [1.0, 2.0], "clip_thr": [1.0, 1.0], "lr": [1e-3, 1e-3],
                              "train_loss": [1.0, 0.9], "val_loss": [None, 0.8]})
        ins = pd.DataFrame({"unique_id": ["s0"], "abs_err": [1.0], "y_abs": [2.0]})
        return dict(forecasts=pd.concat(fc), trace=trace, insample=ins, fold_stats=[{"best_step": 2.0}])

    monkeypatch.setattr(rx, "train_arm", train_arm)
    monkeypatch.setattr(rx, "load_data", lambda: (panel, cutoffs))
    return calls


def test_skips_done_arm(tmp_path, fake):
    d = tmp_path / "runs" / "f2_clip__s1"
    d.mkdir(parents=True)
    (d / "DONE").write_text("")
    rx.run(["f2_clip"], seeds=[1], results_dir=str(tmp_path))
    assert fake == []


def test_redoes_partial_arm(tmp_path, fake):
    d = tmp_path / "runs" / "f2_clip__s1"
    d.mkdir(parents=True)
    (d / "summary.json").write_text("{}")
    rx.run(["f2_clip"], seeds=[1], results_dir=str(tmp_path))
    assert fake == [("f2_clip", 1)]
    for f in ["DONE", "summary.json", "trace.parquet", "forecasts.parquet", "per_series.parquet", "insample.parquet"]:
        assert (d / f).exists(), f
    assert (tmp_path / "actuals.parquet").exists() and (tmp_path / "examples.json").exists()


def test_error_is_recorded_and_run_continues(tmp_path, fake, monkeypatch):
    ok = rx.train_arm

    def boom(arm, seed, panel, cutoffs):
        if arm.id == "f2_noclip":
            raise RuntimeError("nan loss")
        return ok(arm, seed, panel, cutoffs)

    monkeypatch.setattr(rx, "train_arm", boom)
    rx.run(["f2_noclip", "f2_clip"], seeds=[1], results_dir=str(tmp_path))
    assert "nan loss" in (tmp_path / "runs" / "f2_noclip__s1" / "ERROR.txt").read_text()
    assert not (tmp_path / "runs" / "f2_noclip__s1" / "DONE").exists()
    assert (tmp_path / "runs" / "f2_clip__s1" / "DONE").exists()


def test_arm_table():
    assert [a.id for a in arms.ARMS if a.group in ("f2", "f3", "f4")] == EXPECTED_IDS
    by = {a.id: a for a in arms.ARMS}
    assert by["f3_patience250"].overrides["min_steps"] == 0
    assert by["f3_patience250"].overrides["early_stop_patience_steps"] * by["f3_patience250"].overrides["val_check_steps"] == 250
    assert by["f3_fix_noguard"].overrides["min_steps"] == 0
    assert {a.id for a in arms.ARMS if "min_steps" in a.overrides and a.group != "es"} == {"f3_patience250", "f3_fix_noguard"}
    assert by["f2_noclip"].overrides["clip_percentile"] is None
    assert by["f3_mae"].overrides == {"loss": "mae"}
    assert by["f3_minmax"].overrides["scaler_type"] == "minmax"
    assert by["f4_tcn"].arch == "TCN" and by["f4_lgbm"].arch == "LGBM"
    assert all(a.refit_each_fold == (a.group == "f4") for a in arms.ARMS)


def test_overrides_route_to_knobs(monkeypatch):
    import common
    import knobs
    seen = {}
    monkeypatch.setattr(knobs, "Knobs", lambda **kw: seen.update(kw))
    common.make_nf("NHITS", 1, log_model=False, **arms.by_id("f3_input13").overrides)
    assert seen["input_size"] == 13 and seen["loss"] == "huber" and seen["huber_delta"] == 5.0 and "log_model" not in seen
    seen.clear()
    common.make_nf("NHITS", 1, **arms.by_id("f3_patience250").overrides)
    assert seen["min_steps"] == 0 and seen["early_stop_patience_steps"] == 5 and seen["val_check_steps"] == 50



def test_partial_final_week_dropped_from_actuals():
    p = _panel()
    last = p.ds.max()
    p.loc[p.ds == last, "y"] *= 0.1  # one day of sales in the final week
    cutoffs = [last - pd.Timedelta(weeks=13 * (3 - i)) for i in range(3)]
    act = rx.actuals(p, cutoffs)
    assert last not in set(act.ds)
    assert act.ds.max() == last - pd.Timedelta(weeks=1)


def test_full_final_week_kept():
    p = _panel()
    cutoffs = [p.ds.max() - pd.Timedelta(weeks=13 * (3 - i)) for i in range(3)]
    assert p.ds.max() in set(rx.actuals(p, cutoffs).ds)


def test_lgbm_is_direct(monkeypatch):
    import lgbm_baseline
    seen = {}

    class FakeMLF:
        def fit(self, df, **kw):
            seen.update(kw)

        def predict(self, h):
            return pd.DataFrame({"unique_id": ["s0"], "ds": [pd.Timestamp("2016-01-09")], "yhat": [1.0]})

    monkeypatch.setattr(lgbm_baseline, "make_lgbm", lambda seed, freq: FakeMLF())
    p = _panel()
    rx._train_lgbm(arms.by_id("f4_lgbm"), 1, p, [p.ds.max() - pd.Timedelta(weeks=13)])
    assert seen["max_horizon"] == 13


def test_rescore_uses_new_actuals(tmp_path, fake):
    import json
    rx.run(["f2_clip"], seeds=[1], results_dir=str(tmp_path))
    act = pd.read_parquet(tmp_path / "actuals.parquet")
    act = act[act.ds < act.ds.max()]
    act.to_parquet(tmp_path / "actuals.parquet")
    rx.rescore(str(tmp_path))
    ps = pd.read_parquet(tmp_path / "runs" / "f2_clip__s1" / "per_series.parquet")
    s = json.load(open(tmp_path / "runs" / "f2_clip__s1" / "summary.json"))
    assert ps.y_sum.sum() == pytest.approx(act.y.sum())
    assert "fold_stats" in s and "insample_mae" in s


def test_stress_and_early_stopping_arms():
    by = {a.id: a for a in arms.ARMS}
    stress = [a for a in arms.ARMS if a.group == "stress"]
    assert len(stress) == 6 and all(a.overrides.get("loss", "tweedie") == "tweedie" for a in stress)
    for s in ("lr1e2", "lr3e2", "nofilter"):
        off, on = by[f"st_{s}_noclip"], by[f"st_{s}_clip"]
        assert off.overrides["clip_percentile"] is None and "clip_percentile" not in on.overrides
        assert {k: v for k, v in off.overrides.items() if k != "clip_percentile"} == on.overrides
    assert by["st_lr1e2_clip"].overrides["lr_max"] == 1e-2 and by["st_lr3e2_clip"].overrides["lr_max"] == 3e-2
    assert by["st_nofilter_clip"].overrides["availability_threshold"] == [0.0, 0.0]
    es = {a.id: a.overrides for a in arms.ARMS if a.group == "es"}
    assert all(o["max_steps"] == 10000 and o["min_steps"] == 0 and o["loss"] == "huber" for o in es.values())
    assert {k: o["early_stop_patience_steps"] * o.get("val_check_steps", 50) for k, o in es.items() if k.startswith("es_p")} == \
        {"es_p250": 250, "es_p500": 500, "es_p1000": 1000, "es_p2000": 2000, "es_p4000": 4000}
    assert es["es_val26"]["val_size"] == 26 and es["es_val52"]["val_size"] == 52
    assert es["es_check25"]["early_stop_patience_steps"] * es["es_check25"]["val_check_steps"] == 1000
    assert es["es_check100"]["early_stop_patience_steps"] * es["es_check100"]["val_check_steps"] == 1000
    assert es["es_norestore"]["restore_best"] is False
    assert len(es) == 10


def test_failed_fit_saves_partial_trace(tmp_path, fake, monkeypatch):
    def boom(arm, seed, panel, cutoffs):
        e = RuntimeError("Loss is NaN")
        e.partial_trace = pd.DataFrame({"step": [1, 2], "grad_norm": [1.0, float("inf")], "clip_thr": [None, None], "lr": [1e-2, 1e-2],
                                        "train_loss": [1.0, None], "val_loss": [None, None]})
        raise e
    monkeypatch.setattr(rx, "train_arm", boom)
    rx.run(["st_lr3e2_noclip"], seeds=[1], results_dir=str(tmp_path))
    d = tmp_path / "runs" / "st_lr3e2_noclip__s1"
    assert "Loss is NaN" in (d / "ERROR.txt").read_text() and (d / "trace.parquet").exists() and not (d / "DONE").exists()


def test_es_arms_skip_insample_and_others_keep_it():
    assert all(not a.insample for a in arms.ARMS if a.group == "es")
    assert all(a.insample for a in arms.ARMS if a.group != "es")


def test_summary_records_accelerator(tmp_path, fake, monkeypatch):
    import json
    monkeypatch.setenv("ACCEL", "cpu")
    rx.run(["f2_clip"], seeds=[1], results_dir=str(tmp_path))
    assert json.load(open(tmp_path / "runs" / "f2_clip__s1" / "summary.json"))["accelerator"] == "cpu"


def test_existing_actuals_not_rewritten(tmp_path, fake):
    import os
    rx.run(["f2_clip"], seeds=[1], results_dir=str(tmp_path))
    p = tmp_path / "actuals.parquet"
    m = os.path.getmtime(p)
    rx.run(["f2_noclip"], seeds=[1], results_dir=str(tmp_path))
    assert os.path.getmtime(p) == m


def test_cli_accepts_every_arm_group(monkeypatch):
    seen = {}
    monkeypatch.setattr(rx, "run", lambda ids, seeds, results_dir: seen.update(ids=ids))
    for g in ("stress", "es"):
        monkeypatch.setattr("sys.argv", ["run_experiments.py", "--group", g])
        rx.main()
        assert seen["ids"] and all(arms.by_id(i).group == g for i in seen["ids"])


def test_manifest_mismatch_refuses(tmp_path, fake, monkeypatch):
    rx.run(["f2_clip"], seeds=[1], results_dir=str(tmp_path))
    monkeypatch.setattr(rx.common, "N_SERIES", 123)
    with pytest.raises(SystemExit, match="manifest"):
        rx.run(["f2_noclip"], seeds=[1], results_dir=str(tmp_path))


def test_cli_accepts_comma_separated_groups(monkeypatch):
    seen = {}
    monkeypatch.setattr(rx, "run", lambda ids, seeds, results_dir: seen.update(ids=ids))
    monkeypatch.setattr("sys.argv", ["run_experiments.py", "--group", "f2,f3,f4"])
    rx.main()
    assert {arms.by_id(i).group for i in seen["ids"]} == {"f2", "f3", "f4"}


def test_default_panel_is_full_m5():
    import common
    assert common.N_SERIES == 30490 or "N_SERIES" in __import__("os").environ
