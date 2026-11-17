from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import knobs


class _Mod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(2))


def _step(cb, mod, grad):
    mod.w.grad = torch.tensor(grad, dtype=torch.float32)
    cb.on_before_optimizer_step(None, mod, None)
    return float(mod.w.grad.norm())


def test_autoclip_threshold_is_percentile_of_history_including_current():
    cb, m = knobs.AutoClip(10.0), _Mod()
    cb.on_train_start(None, m)
    assert _step(cb, m, [3.0, 4.0]) == pytest.approx(5.0)  # history [5] -> thr 5, not clipped
    after = _step(cb, m, [6.0, 8.0])  # history [5, 10] -> idx round(0.1*1)=0 -> thr 5
    assert after == pytest.approx(5.0)
    assert cb.norms == pytest.approx([5.0, 10.0]) and cb.thresholds == pytest.approx([5.0, 5.0]) and cb.n_clipped == 1


def test_autoclip_nonfinite_recorded_not_inserted():
    cb, m = knobs.AutoClip(10.0), _Mod()
    cb.on_train_start(None, m)
    _step(cb, m, [3.0, 4.0])
    _step(cb, m, [float("nan"), 0.0])
    assert np.isnan(cb.norms[1]) and len(cb.thresholds) == 1 and cb._hist == [5.0]


def test_autoclip_off_only_records():
    cb, m = knobs.AutoClip(None), _Mod()
    cb.on_train_start(None, m)
    assert _step(cb, m, [30.0, 40.0]) == pytest.approx(50.0)
    assert cb.thresholds == [] and cb.norms == pytest.approx([50.0])


def test_best_weights_restores_best_snapshot():
    cb, m = knobs.BestWeights(), _Mod()
    tr = SimpleNamespace(sanity_checking=False, callback_metrics={})
    cb.on_train_start(tr, m)
    for w, v in [([1.0, 1.0], 3.0), ([2.0, 2.0], 1.0), ([9.0, 9.0], 2.0)]:
        m.w.data = torch.tensor(w)
        tr.callback_metrics = {"ptl/val_loss": torch.tensor(v)}
        cb.on_validation_end(tr, m)
    cb.on_train_end(tr, m)
    assert m.w.data.tolist() == [2.0, 2.0]


def test_best_weights_ignores_sanity_check_and_nan():
    cb, m = knobs.BestWeights(), _Mod()
    tr = SimpleNamespace(sanity_checking=True, callback_metrics={"ptl/val_loss": torch.tensor(0.1)})
    cb.on_train_start(tr, m)
    cb.on_validation_end(tr, m)
    tr.sanity_checking, tr.callback_metrics = False, {"ptl/val_loss": torch.tensor(float("nan"))}
    cb.on_validation_end(tr, m)
    assert cb.state is None


def test_tweedie_decouple_finite_at_zero_loc():
    raw = torch.tensor([[100.0, -100.0, 0.0]])
    mu, rho = knobs.tweedie_decouple((raw, torch.tensor(1.5)), loc=torch.zeros(1, 3), scale=torch.ones(1, 3))
    assert torch.isfinite(mu).all()
    assert mu[0, 0] == pytest.approx(6 + np.log(1e-2), abs=1e-5) and mu[0, 1] == pytest.approx(-6 + np.log(1e-2), abs=1e-5)
    mu2, _ = knobs.tweedie_decouple((raw, torch.tensor(1.5)))
    assert mu2.tolist() == [[6.0, -6.0, 0.0]]


def test_pad_short_pads_to_val_size_plus_one():
    d = pd.DataFrame({"unique_id": ["a"] * 3 + ["b"] * 20, "ds": list(pd.date_range("2016-01-02", periods=3, freq="W-SAT")) +
                      list(pd.date_range("2015-01-03", periods=20, freq="W-SAT")), "y": 1.0})
    p = knobs.pad_short(d, 13, "W-SAT")
    a = p[p.unique_id == "a"].sort_values("ds")
    assert len(a) == 14 and (a.y.iloc[:11] == 0).all() and a.ds.diff().dropna().eq(pd.Timedelta(weeks=1)).all()
    assert len(p[p.unique_id == "b"]) == 20


def test_validation_loader_keeps_last_partial_batch():
    import neuralforecast.common._base_model as nfbm
    assert nfbm.TimeSeriesDataModule is knobs.KeepValBatchDataModule
    seen = {}

    class Base:
        def val_dataloader(self):
            seen["val"] = self.drop_last

        def train_dataloader(self):
            seen["train"] = self.drop_last

    class T(knobs._ValFix, Base):
        pass

    dm = T()
    dm.drop_last, dm.batch_size, dm.dataset = True, 128, list(range(300))
    dm.val_dataloader()
    dm.train_dataloader()
    assert seen == {"val": False, "train": True} and dm.drop_last is True
    dm.dataset = list(range(50))
    dm.train_dataloader()
    assert seen["train"] is False


def test_built_model_matches_inventory():
    k = knobs.Knobs(arch="NHITS", h=13, input_size=52, loss="tweedie", seed=1, accelerator="cpu", max_steps=5000)
    m = k.nf.models[0]
    assert m.hparams.scaler_type == "revin" and m.hparams.start_padding_enabled
    assert list(m.hparams.training_data_availability_threshold) == [0.05, 0.0] and m.hparams.drop_last_loader
    assert m.hparams.optimizer is torch.optim.NAdam
    assert m.hparams.optimizer_kwargs == {"weight_decay": 0.01, "decoupled_weight_decay": True}
    assert m.hparams.lr_scheduler is torch.optim.lr_scheduler.OneCycleLR
    assert m.hparams.lr_scheduler_kwargs == dict(max_lr=1e-3, total_steps=5001, pct_start=0.3, div_factor=25.0, final_div_factor=1e3,
                                                 anneal_strategy="cos", cycle_momentum=False)
    assert m.trainer_kwargs["min_steps"] == 1500
    assert m.early_stop_patience_steps == 20 and m.val_check_steps == 50
    assert m.loss.scale_decouple is knobs.tweedie_decouple
    assert {type(c).__name__ for c in m.trainer_kwargs["callbacks"]} >= {"AutoClip", "BestWeights"}


def test_overrides_reach_model():
    k = knobs.Knobs(arch="NHITS", h=13, input_size=13, loss="huber", huber_delta=5.0, seed=1, accelerator="cpu", min_steps=0,
                    early_stop_patience_steps=5, scaler_type="minmax", clip_percentile=None)
    m = k.nf.models[0]
    assert m.hparams.input_size == 13 and m.trainer_kwargs["min_steps"] == 0 and m.early_stop_patience_steps == 5
    assert m.hparams.scaler_type == "minmax" and m.loss.delta == 5.0 and k._gn.clip_percentile is None


def test_fold_stats_from_trajectories():
    st = knobs.fit_stats(valid=[(0, 9.0), (50, 3.0), (100, 2.0), (150, 2.5)], norms=[1.0, 2.0, 3.0], thresholds=[1.0, 1.0],
                         n_clipped=1, max_steps=150, early_stop=True, fit_seconds=1.0)
    assert st["best_step"] == 100 and st["stop_step"] == 150 and st["hit_cap"] == 1.0 and st["grad_clip_frac"] == 0.5
    assert st["grad_norm_max"] == 3.0


def test_availability_threshold_overridable():
    k = knobs.Knobs(arch="NHITS", h=13, loss="tweedie", seed=1, accelerator="cpu", availability_threshold=[0.0, 0.0])
    assert list(k.nf.models[0].hparams.training_data_availability_threshold) == [0.0, 0.0]


def test_val_size_reaches_fit_padding():
    k = knobs.Knobs(arch="NHITS", h=13, loss="huber", seed=1, accelerator="cpu", val_size=26)
    assert k.val_size == 26
