"""Stock NeuralForecast plus the few training settings the article's experiments depend on.

Everything here is either a plain NeuralForecast/Lightning argument set explicitly, or one of four small additions:
  * AutoClip: clip each update's gradient norm to a percentile of the norms seen so far in this fit (Seetharaman et al. 2020).
  * BestWeights: restore the weights from the best validation check when training ends (Lightning's EarlyStopping only stops).
  * tweedie_decouple: a finite, symmetric Tweedie output link (stock NF takes log of the window mean, -inf on all-zero windows).
  * KeepValBatchDataModule: stock NF drops the last partial batch of series in *validation* when drop_last_loader=True; this keeps
    every series in every validation check (and keeps the training batch when there are fewer series than batch_size).
Plus pad_short, which left-pads series too short for the validation split.
"""
import bisect
import os
import time

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Tweedie predictions sample torch.poisson, which MPS lacks

import neuralforecast.common._base_model as _nfbm  # noqa: E402
import neuralforecast.models as _nfm  # noqa: E402
from neuralforecast import NeuralForecast  # noqa: E402
from neuralforecast.losses.pytorch import MAE, DistributionLoss, HuberLoss  # noqa: E402
from neuralforecast.tsdataset import TimeSeriesDataModule  # noqa: E402


class _ValFix:
    def train_dataloader(self):
        keep, self.drop_last = self.drop_last, self.drop_last and len(self.dataset) >= self.batch_size
        try:
            return super().train_dataloader()
        finally:
            self.drop_last = keep

    def val_dataloader(self):
        keep, self.drop_last = self.drop_last, False
        try:
            return super().val_dataloader()
        finally:
            self.drop_last = keep


class KeepValBatchDataModule(_ValFix, TimeSeriesDataModule):
    pass


_nfbm.TimeSeriesDataModule = KeepValBatchDataModule  # the name BaseModel._fit builds its data module from


class AutoClip(pl.Callback):
    """Records the global gradient L2 norm every optimizer step (Lightning calls this hook before any clipping of its own).
    With clip_percentile=p, clips to the p-th percentile of the finite norms seen so far in this fit, current step included."""

    def __init__(self, clip_percentile=None):
        self.clip_percentile = clip_percentile
        self.on_train_start(None, None)

    def on_train_start(self, trainer, pl_module):
        self.norms, self.thresholds, self.n_clipped, self._hist = [], [], 0, []

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        params = [p for p in pl_module.parameters() if p.grad is not None]
        if not params:
            return
        n = float(torch.stack([p.grad.detach().float().norm() for p in params]).norm())
        self.norms.append(n)
        if self.clip_percentile is None or not np.isfinite(n):
            return
        bisect.insort(self._hist, n)
        thr = self._hist[int(round(self.clip_percentile / 100 * (len(self._hist) - 1)))]
        self.thresholds.append(thr)
        if n > thr > 0:
            self.n_clipped += 1
            torch.nn.utils.clip_grad_norm_(params, thr)


class BestWeights(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        self.best, self.state = float("inf"), None

    def on_validation_end(self, trainer, pl_module):
        v = trainer.callback_metrics.get("ptl/val_loss")
        if trainer.sanity_checking or v is None or not torch.isfinite(v) or float(v) >= self.best:
            return
        self.best, self.state = float(v), {k: t.detach().clone() for k, t in pl_module.state_dict().items()}

    def on_train_end(self, trainer, pl_module):
        if self.state is not None:
            pl_module.load_state_dict(self.state)


class LRCapture(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        self.lr = []

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        self.lr.append(float(optimizer.param_groups[0]["lr"]))


def tweedie_decouple(output, loc=None, scale=None):
    """log(mu) = clamp(raw, -6, 6) + log(max(window mean, 1e-2)): finite for all-zero windows, no built-in floor at the mean."""
    raw, rho = output
    if loc is None or scale is None:
        return (torch.clamp(raw, -6, 6), rho)
    return (torch.clamp(raw, -6, 6) + torch.log(loc.clamp_min(1e-2)), rho)


def make_loss(kind: str, huber_delta: float = 5.0, tweedie_rho: float = 1.5):
    if kind == "huber":
        return HuberLoss(delta=huber_delta)
    if kind == "mae":
        return MAE()
    if kind == "tweedie":
        loss = DistributionLoss(distribution="Tweedie", rho=tweedie_rho)
        loss.scale_decouple = tweedie_decouple
        return loss
    raise ValueError(f"loss must be huber, mae or tweedie, got {kind!r}")


def pad_short(d: pd.DataFrame, val_size: int, freq: str) -> pd.DataFrame:
    """NeuralForecast needs more than val_size rows per series; left-pad shorter series with zero rows at earlier dates."""
    need = val_size + 1
    n = d.groupby("unique_id")["ds"].transform("size")
    pads = []
    for uid, g in d[n < need].groupby("unique_id"):
        k = need - len(g)
        pad = pd.DataFrame(0, index=range(k), columns=d.columns)
        pad["unique_id"], pad["ds"] = uid, pd.date_range(end=g["ds"].min(), periods=k + 1, freq=freq)[:-1]
        pads.append(pad)
    return pd.concat([*pads, d], ignore_index=True).sort_values(["unique_id", "ds"]) if pads else d


def fit_stats(valid, norms, thresholds, n_clipped, max_steps, early_stop, fit_seconds) -> dict:
    v = [(int(s), float(x)) for s, x in valid if s > 0]  # step 0 = Lightning's sanity check
    out = {"fit_seconds": fit_seconds}
    if v:
        best = min(v, key=lambda t: t[1])
        out.update(best_step=float(best[0]), stop_step=float(v[-1][0]), val_best=best[1], val_last=v[-1][1], n_checks=float(len(v)))
        if early_stop and v[-1][0] >= max_steps:
            out["hit_cap"] = 1.0
    g = np.asarray(norms, dtype=float)
    g = g[np.isfinite(g)]
    if g.size:
        out.update(grad_norm_median=float(np.median(g)), grad_norm_p95=float(np.percentile(g, 95)), grad_norm_max=float(g.max()))
    if thresholds:
        out.update(clip_threshold_median=float(np.median(thresholds)), grad_clip_frac=n_clipped / len(thresholds))
    return out


class Knobs:
    """One NeuralForecast model configured as in the article. Interface used by the experiment runner: fit, predict (optionally
    from newer history with the fitted weights), .nf, ._gn (AutoClip record), ._lr (LR record), .fold_stats."""

    def __init__(self, arch="NHITS", h=13, input_size=52, loss="tweedie", seed=1, accelerator="mps", freq="W-SAT",
                 clip_percentile=10.0, restore_best=True, lr_max=1e-3, max_steps=5000, val_check_steps=50,
                 early_stop_patience_steps=20, min_steps=None, scaler_type="revin", huber_delta=5.0, tweedie_rho=1.5,
                 batch_size=128, windows_batch_size=1024, val_size=13, availability_threshold=(0.05, 0.0)):
        self.freq, self.val_size, self.max_steps, self.fold_stats = freq, val_size, max_steps, []
        self._gn, self._lr = AutoClip(clip_percentile), LRCapture()
        callbacks = [self._gn, self._lr] + ([BestWeights()] if restore_best else [])
        sch = torch.optim.lr_scheduler
        model = getattr(_nfm, arch)(
            h=h, input_size=input_size, loss=make_loss(loss, huber_delta, tweedie_rho), scaler_type=scaler_type,
            max_steps=max_steps, val_check_steps=val_check_steps, early_stop_patience_steps=early_stop_patience_steps,
            batch_size=batch_size, windows_batch_size=windows_batch_size, random_seed=seed,
            start_padding_enabled=True, training_data_availability_threshold=list(availability_threshold), drop_last_loader=True,
            optimizer=torch.optim.NAdam, optimizer_kwargs={"weight_decay": 0.01, "decoupled_weight_decay": True},
            lr_scheduler=sch.OneCycleLR, lr_scheduler_kwargs=dict(max_lr=lr_max, total_steps=max_steps + 1, pct_start=0.3, div_factor=25.0,
                                                                  final_div_factor=1e3, anneal_strategy="cos", cycle_momentum=False),
            min_steps=int(0.3 * max_steps) if min_steps is None else min_steps,  # no stopping before the one-cycle LR peak
            accelerator=accelerator, devices=1, enable_progress_bar=False, enable_checkpointing=False, logger=False,
            callbacks=callbacks, alias="yhat")
        self.nf = NeuralForecast(models=[model], freq=freq)
        cbs = self.nf.models[0].trainer_kwargs["callbacks"]  # NeuralForecast deep-copies the model; read back the copies that train
        self._gn = next(c for c in cbs if isinstance(c, AutoClip))
        self._lr = next(c for c in cbs if isinstance(c, LRCapture))
        self.early_stop = early_stop_patience_steps > 0

    def fit(self, df: pd.DataFrame):
        m = self.nf.models[0]
        m.valid_trajectories.clear()
        m.train_trajectories.clear()
        t0 = time.time()
        self.nf.fit(df=pad_short(df[["unique_id", "ds", "y"]], self.val_size, self.freq), val_size=self.val_size)
        self.fold_stats.append(fit_stats(m.valid_trajectories, self._gn.norms, self._gn.thresholds, self._gn.n_clipped,
                                         self.max_steps, self.early_stop, time.time() - t0))
        return self

    def predict(self, h=None, predict_df=None) -> pd.DataFrame:
        kw = {} if predict_df is None else {"df": pad_short(predict_df[["unique_id", "ds", "y"]], self.val_size, self.freq)}
        return self.nf.predict(**kw)[["unique_id", "ds", "yhat"]]
