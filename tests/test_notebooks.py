import glob
import os

import nbformat
import pytest
from nbclient import NotebookClient

from m5knobs import paths

NB = sorted(glob.glob(os.path.join(paths.ROOT, "notebooks", "*.ipynb")))


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(os.path.join(paths.RESULTS, "manifest.json")), reason="needs results/")
@pytest.mark.parametrize("path", NB, ids=os.path.basename)  # sorted 01..06: 06 reads what 03-05 write to RESULTS_DIR
def test_notebook_executes(path, tmp_path, monkeypatch):
    # the kernel inherits this env: figures and FINDINGS.md go to tmp_path, not the repo
    monkeypatch.setenv("FIG_DIR", str(tmp_path))
    monkeypatch.setenv("FINDINGS_PATH", str(tmp_path / "FINDINGS.md"))
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": os.path.dirname(path)}}).execute()


def test_six_notebooks_and_no_local_paths():
    assert [os.path.basename(p)[:2] for p in NB] == ["01", "02", "03", "04", "05", "06"]
    for p in NB:
        assert "/Users/" not in open(p).read()
