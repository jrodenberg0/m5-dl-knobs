"""Train experiment arms and save raw results under results/runs/ (see m5knobs.experiments for flags)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from m5knobs.experiments import main  # noqa: E402

main()
