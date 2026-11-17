# The knobs that make deep learning fit supply-chain data

Which training settings make a deep model (N-HiTS in stock [NeuralForecast](https://github.com/Nixtla/neuralforecast)) fit
intermittent weekly retail demand, and how much each one matters, tested on all 30,490 M5 item-store series. The notebooks walk
through the evidence for each setting; everything is reproducible from the public M5 data. This repo is self-contained; it started
as the code for the article *Making Deep Learning Sing; In Supply Chain Forecasting*.

## The recipe

`m5knobs.model.Knobs` defaults. The clipping card, the clipping stress test and the architecture table use it as is; the other
setting cards and the early-stopping sweep differ from it as described in notebook 03 (for example, they use Huber δ5 loss).

- Tweedie loss (rho 1.5), RevIN scaling, 52-week input window.
- NAdam (weight decay 0.01, decoupled), one-cycle LR peaking at 1e-3, at most 5,000 steps.
- No early stop before 30% of the steps; patience 20 checks × 50 steps, best weights restored.
- AutoClip at the 10th percentile.
- Training windows need ≥ 5% real history.

Setup: weekly data, 13-week horizon, 3 rolling origins, single seed (1).

## Read the story

The notebooks put four claims that practitioners often make to the test:

1. A handful of training settings decide whether deep learning works on retail demand.
2. Getting those settings wrong mostly makes the model underfit.
3. Gradient clipping is non-negotiable for likelihood losses such as Tweedie.
4. With the right settings, deep learning can match gradient-boosted trees.

They read finished results from `results/` and never train. Read them in order:

1. `notebooks/01_the_problem.ipynb`: how sparse weekly M5 demand is, the forecast setup, and how results are judged (WAPE,
   bias, the verdict rule, the seed-noise bar, the two rules set after seeing results).
2. `notebooks/02_the_recipe.ipynb`: every training setting, why it is there, the exact NeuralForecast call, and what
   `m5knobs/model.py` adds, including a check of the Tweedie zero-window guard.
3. `notebooks/03_one_setting_at_a_time.ipynb`: claims 1 and 2. Five setting cards, each changing one setting away from the
   recipe, plus a sweep of the early-stopping rule.
4. `notebooks/04_gradient_clipping.ipynb`: claim 3. Clipping on a two-weight toy fit, at the recipe's settings, and under three
   stressors.
5. `notebooks/05_dl_vs_lightgbm.ipynb`: claim 4. N-HiTS and TCN against a LightGBM baseline, same series, origins and loss.
6. `notebooks/06_summary.ipynb`: every verdict in one table, the four claims with their verdicts, run status and caveats; writes
   `FINDINGS.md`.

## What `m5knobs/model.py` adds to stock NeuralForecast

- **AutoClip** (Seetharaman et al. 2020): each step, clip the gradient norm to the 10th percentile of the norms seen so far.
- **Best-weights restore**: Lightning's early stopping only stops; this loads the best validation checkpoint's weights back.
- **Tweedie zero-window guard**: the Tweedie loss and output link are NeuralForecast's own (`softplus(raw) + log(window mean)`,
  clamped to [1e-9, 37]), with one change: the window mean is floored at 0.01 before the log. Without it, an all-zero input
  window (common on M5) takes `log(0)` and the gradient into RevIN's learnable parameters is NaN on the first step, so every
  Tweedie run fails immediately. Everywhere else the link is identical to stock, including its floor: forecasts can't go below
  the input window's mean, nor below about 1 unit a week. Both floors are in sales units, not scaled units; notebook 05 shows
  where forecasts on slow sellers land.
- **Validation keeps every series**: with `drop_last_loader=True`, stock NeuralForecast 3.1.2 also drops the last partial batch
  of series from every validation check (26 of 30,490 here). `m5knobs/model.py` replaces NeuralForecast's data-module class *for
  the whole Python process* as soon as it is imported, so validation keeps all series.
- **Short-series padding**: series with ≤ 13 weeks are left-padded with zeros so the validation split exists.

## Run the notebooks (no training)

```bash
git clone https://github.com/jrodenberg0/m5-dl-knobs.git && cd m5-dl-knobs
python3 -m venv .venv && source .venv/bin/activate          # Python 3.9–3.12
pip install -r requirements.txt && pip install -e . && pip install jupyterlab
python3 scripts/download_results.py        # finished runs (~250 MB) from the GitHub release -> results/
python3 scripts/prepare_data.py            # optional: M5 (~450 MB) -> data/, for the panel statistics in notebooks 01-02
jupyter lab notebooks/                     # read 01 -> 06 in order
```

`scripts/download_results.py` checks the archive's SHA-256 and won't overwrite an existing `results/` (use `--force`).
Without `data/`, notebooks 01 and 02 skip the panel-wide statistics and say so; everything else runs from `results/`.
Headless, in order: `jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb`.
If Jupyter's `python3` kernel points at another environment, run `python3 -m ipykernel install --user --name python3` from
this one.

Notebooks 03 to 05 write derived files to `results/` (`card_verdicts.json`, `es_sweep.json`, `stress_outcomes.json`,
`f4_table.csv`) and notebook 06 reads them, so run 06 last. Notebook 06 checks they are newer than the newest finished run,
then rewrites `FINDINGS.md`.

## Retrain from scratch

```bash
python3 scripts/prepare_data.py            # required for training
python3 -m pytest tests -m "not slow"      # ~20 s
# One results dir; finished runs are skipped on rerun.
python3 scripts/run_experiments.py --group f2,f3,f4          # setting cards + architecture table (ACCEL defaults to mps)
ACCEL=cpu python3 scripts/run_experiments.py --group es      # early-stopping study, CPU: ~1 h
ACCEL=cpu python3 scripts/run_experiments.py --group stress  # clipping stress test
python3 scripts/toy_clip.py                                  # 2-weight toy fit (fixed 5,000-series sample)
```

Tweedie runs take tens of minutes each on an Apple M4 (training plus in-sample scoring and prediction); the full set takes
several hours. Each run's `fit_seconds` is in its `summary.json` (under `fold_stats`), also the "fit s" column of `FINDINGS.md`.
The results in this repo: setting cards, early-stopping study and architecture table on an Apple M4 (MPS, the study on CPU);
the Tweedie stress test on a Databricks GPU node (A10G). `python3 -m pytest tests -m slow` executes every notebook against
`results/` (skipped when `results/` is missing). Maintainers: `python3 scripts/download_results.py --pack results.tar.gz`
builds the release archive and prints the SHA-256 to put in `m5knobs/download.py`.

- **Locations**: `RESULTS_DIR` sets the results location for both training and the notebooks and figures; `FIG_DIR` sets where
  figures go; `FINDINGS_PATH` sets where notebook 06 writes `FINDINGS.md` (the slow test points it and `FIG_DIR` at a temp
  dir). Set them before importing `m5knobs` (they are read once, in `m5knobs/paths.py`).
- **Data**: `scripts/prepare_data.py` uses `datasetsforecast`'s M5 loader (Nixtla's public mirror, including the
  evaluation-phase test days through 2016-06-19). The Kaggle download alone (`sales_train_evaluation.csv`) ends 4 weeks earlier,
  shifts every origin, and won't reproduce these numbers. The final weekly bucket holds one day and is excluded from scoring.
- **Devices**: `ACCEL` defaults to `mps`; use `cpu` or `gpu` elsewhere. For this small model the CPU was faster per step than MPS,
  which is why the early-stopping study ran on CPU. Runs made by the current runner record their device in `summary.json`; the Huber
  setting-card runs (including the early-stopping card) were made before that and show "not recorded". LightGBM runs on CPU.
- **Determinism**: on the same machine and device, reruns are bit-identical (checked across Python 3.9 and 3.12). Across devices
  or machines expect small differences. The seed-noise bar used in the figures comes from the author's earlier 3-seed runs of the
  same model family on M5 (not part of this repo, so treat ±0.2 WAPE / ±1pt bias as an approximate bar). The high-learning-rate stress runs are chaotic and may differ more.
- **One results dir per configuration**: `scripts/run_experiments.py` writes `results/manifest.json` (panel size, smoke mode,
  recipe) and refuses to mix runs made with different settings. `N_SERIES` (default 30,490) and `SMOKE=1` (100-step test runs)
  are for quick checks; use a separate `--results-dir` for them.

How results are judged (WAPE, bias, the verdict rule and the two rules set after seeing results) is in
`notebooks/01_the_problem.ipynb`.

## Repo layout

```
README.md
FINDINGS.md                 every number behind the figures (written by notebook 06)
notebooks/                  the story, 01 to 06, committed with outputs
scripts/
  prepare_data.py           download M5 -> data/m5_weekly.parquet
  run_experiments.py        train the arms (--group, --arms, --seeds, --results-dir, --rescore)
  toy_clip.py               the 2-weight toy fit -> results/toy_*
m5knobs/
  paths.py                  repo locations; RESULTS_DIR, FIG_DIR, FINDINGS_PATH, M5_WEEKLY override them
  data.py                   the weekly panel, forecast cutoffs, example series
  model.py                  Knobs (the recipe) and the additions to stock NeuralForecast
  experiments.py            every arm and the runner that trains them
  scoring.py                per-series errors, training traces, the verdict rule
  results.py                readers for results/
  baselines.py              LightGBM baseline
  toy.py                    the 2-weight Tweedie toy fit
  report.py                 writes FINDINGS.md
  figures/                  drawing code for each figure (style, cards, stress, early_stopping, architectures, toy)
tests/                      fast suite; -m slow runs the notebooks
results/  data/             training output and the M5 panel (gitignored)
figures/                    PNG and SVG figures
```

## License

MIT, see `LICENSE`. Copyright (c) 2026 Jack Rodenberg.
