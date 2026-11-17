"""Fit the 2-weight Tweedie toy model and write results/toy_paths.parquet and results/toy_surface.npz (see m5knobs.toy)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from m5knobs.toy import main  # noqa: E402

p, _ = main()
print(p.groupby("run")[["grad_norm", "loss"]].describe().T)
