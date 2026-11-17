# The knobs that make deep learning fit supply-chain data

Code for the article *Making Deep Learning Sing; In Supply Chain Forecasting*. It reproduces every number and figure in the article
from the public M5 data, using stock [NeuralForecast](https://github.com/Nixtla/neuralforecast) plus a few small additions in
`knobs.py`.

## What's tested

All 30,490 M5 item-store series, weekly, 13-week horizon, 3 rolling origins, single seed (1).

**Setting cards** (`fig_cards.py`): one N-HiTS setting changed at a time.

| Card (NeuralForecast argument) | Without | With | Loss in both arms |
|---|---|---|---|
| Gradient clipping (`gradient_clip_val`, AutoClip on top) | off | AutoClip, 10th percentile | Tweedie (the recipe) |
| Input window (`input_size`) | 13 weeks | 52 weeks | Huber δ5 |
| Early stopping (`early_stop_patience_steps` × `val_check_steps`) | 5 × 50 = 250 steps | 20 × 50 = 1,000 steps | Huber δ5 |
| Loss (`loss`) | `MAE()` | `HuberLoss(delta=5)` | (the setting itself) |
| Scaler (`scaler_type`) | `'minmax'` | `'revin'` | Huber δ5 |

Four cards use Huber δ5 instead of the recipe's Tweedie loss; the "with" arm is otherwise the recipe. The early-stopping card turns
off the minimum-steps guard in both arms, so patience is the only difference (its "with" arm gives the same results as the recipe
with the guard on).

**Clipping stress test** (`fig_stress.py`): Tweedie N-HiTS, clipping off vs AutoClip, under three stressors fixed before any run:
learning rate 1e-2 and 3e-2 (recipe: 1e-3), and the short-history window filter off (`training_data_availability_threshold=[0, 0]`).

**Early-stopping study** (`fig_es_sweep.py`): Huber δ5, every arm on one schedule (one-cycle over 10,000 steps, no minimum-steps
guard) so only the stopping rule differs: patience window 250 to 4,000 steps; `val_size` 13 / 26 / 52 weeks; `val_check_steps`
25 / 50 / 100 at a fixed 1,000-step window; `restore_best` on / off.

**Architecture table** (`fig_f4_table.py`): LightGBM (direct, Tweedie) vs N-HiTS vs TCN, refit at every origin, Tweedie loss.

**The recipe** (`knobs.Knobs` defaults): Tweedie loss (rho 1.5), RevIN scaling, NAdam (weight decay 0.01, decoupled), one-cycle LR
peaking at 1e-3, at most 5,000 steps, no early stop before 30% of them, patience 20 checks × 50 steps with the best weights
restored, AutoClip at the 10th percentile, training windows need ≥ 5% real history.

## What `knobs.py` adds to stock NeuralForecast

- **AutoClip** (Seetharaman et al. 2020): each step, clip the gradient norm to the 10th percentile of the norms seen so far.
- **Best-weights restore**: Lightning's early stopping only stops; this loads the best validation checkpoint's weights back.
- **Tweedie zero-window guard**: the Tweedie loss and output link are NeuralForecast's own (`softplus(raw) + log(window mean)`,
  clamped to [1e-9, 37]), with one change: the window mean is floored at 0.01 before the log. Without it, an all-zero input
  window (common on M5) takes `log(0)` and the gradient into RevIN's learnable parameters is NaN on the first step, so every
  Tweedie run fails immediately. Everywhere else the link is identical to stock, including its floor: forecasts can't go below
  the input window's mean, nor below about 1.
- **Validation keeps every series**: with `drop_last_loader=True`, stock NeuralForecast 3.1.2 also drops the last partial batch
  of series from every validation check (26 of 30,490 here). `knobs.py` replaces NeuralForecast's data-module class *for the whole
  Python process* as soon as it is imported, so validation keeps all series.
- **Short-series padding**: series with ≤ 13 weeks are left-padded with zeros so the validation split exists.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate          # Python 3.10–3.12
pip install -r requirements.txt
python3 prepare_data.py                    # downloads M5 (~450 MB) via datasetsforecast -> data/m5_weekly.parquet
python3 -m pytest tests -m "not slow"      # ~15 s

# Experiments (one results dir; finished runs are skipped on rerun). Devices are the ones the published results used.
python3 run_experiments.py --group f2,f3,f4                # setting cards + architecture table, MPS: ~6.5 h on an Apple M4
ACCEL=cpu python3 run_experiments.py --group es            # early-stopping study, CPU: ~1 h
ACCEL=cpu python3 run_experiments.py --group stress        # clipping stress test, CPU: ~2 h
python3 toy_clip.py                                        # 2-weight toy fit (fixed 5,000-series sample)

python3 fig_cards.py && python3 fig_stress.py && python3 fig_es_sweep.py && python3 fig_f4_table.py && python3 fig_f1_toy.py
python3 findings.py                                        # -> FINDINGS.md, every number behind the figures
```

- **Data**: `prepare_data.py` uses `datasetsforecast`'s M5 loader (Nixtla's public mirror, including the evaluation-phase test
  days through 2016-06-19). The Kaggle download alone (`sales_train_evaluation.csv`) ends 4 weeks earlier, shifts every origin,
  and won't reproduce these numbers. The final weekly bucket holds one day and is excluded from scoring.
- **Devices**: `ACCEL` defaults to `mps`; use `cpu` or `gpu` elsewhere. For this small model the CPU was faster than MPS
  (≈45 vs ≈80 ms per step), which is why the two studies ran on CPU. Each run records its device in `summary.json`.
- **Determinism**: on the same machine and device, reruns are bit-identical (checked across Python 3.9 and 3.12). Across devices or
  machines expect small differences. In the author's earlier 3-seed M5 runs, seed-to-seed noise was about ±0.2 WAPE and ±1pt bias,
  the noise bar used in the figures. The high-learning-rate stress runs are chaotic and may differ more.
- **One results dir per configuration**: `run_experiments.py` writes `results/manifest.json` (panel size, smoke mode, recipe) and
  refuses to mix runs made with different settings. `N_SERIES` (default 30,490) and `SMOKE=1` (100-step test runs) are for quick
  checks; use a separate `--results-dir` for them.

## How results are judged

A "without" arm is labeled from per-series results against the "with" arm:
- **underfit**: worse fit to the training history, worse week-to-week tracking, or flatter forecasts (paired bootstrap over
  series, p < 0.05), *and* WAPE worse by more than 0.2;
- **biased**: total-volume bias worse (bootstrap p < 0.05) by more than 1pt;
- **no clear effect on M5** otherwise.

Two rules were set after seeing results, and are disclosed wherever they're used: the effect floor above (with thousands of
series the bootstrap alone flags differences far inside seed noise), and the stress test's divergence rule (validation loss ending
above 2× its best, i.e. not recovering).

## Files

`prepare_data.py` data · `knobs.py` model · `arms.py` experiments · `run_experiments.py` runner · `verdict.py` labeling rule ·
`fig_*.py` figures · `findings.py` report · `toy_clip.py` toy fit · `lgbm_baseline.py` LightGBM · `tests/`.

## License

MIT, see `LICENSE`. Copyright (c) 2026 Jack Rodenberg.
