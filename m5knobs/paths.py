"""Repo locations, resolved from the package (not the working directory); RESULTS_DIR, FIG_DIR, FINDINGS_PATH and M5_WEEKLY override them."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
FIGURES = os.environ.get("FIG_DIR", os.path.join(ROOT, "figures"))
FINDINGS = os.environ.get("FINDINGS_PATH", os.path.join(ROOT, "FINDINGS.md"))
M5_WEEKLY = os.environ.get("M5_WEEKLY", os.path.join(DATA, "m5_weekly.parquet"))
