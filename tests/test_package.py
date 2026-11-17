import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_light_import_does_not_load_torch():
    code = ("import m5knobs.data, m5knobs.scoring, m5knobs.experiments, m5knobs.results, m5knobs.report, m5knobs.figures.style, "
            "m5knobs.figures.cards, m5knobs.figures.stress, m5knobs.figures.early_stopping, m5knobs.figures.architectures, "
            "m5knobs.figures.compare, m5knobs.figures.segments, m5knobs.examples, m5knobs.segments, m5knobs.story, sys; print('torch' in sys.modules, 'neuralforecast' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "False"]


def test_paths_resolve_from_package_not_cwd(tmp_path):
    code = "from m5knobs import paths; print(paths.RESULTS)"
    env = {**os.environ, "PYTHONPATH": ROOT}
    env.pop("RESULTS_DIR", None)
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True, check=True).stdout.strip()
    assert out == os.path.join(ROOT, "results")


def test_scripts_cli_flags_unchanged():
    out = subprocess.run([sys.executable, "scripts/run_experiments.py", "--help"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    for flag in ("--group", "--arms", "--seeds", "--results-dir", "--rescore"):
        assert flag in out


def test_top_level_is_simple():
    py = sorted(f for f in os.listdir(ROOT) if f.endswith(".py"))
    assert py == []
