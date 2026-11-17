"""Download M5 and build data/m5_weekly.parquet (see m5knobs.data)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from m5knobs.data import main  # noqa: E402

main()
