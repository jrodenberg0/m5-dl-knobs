import os

import nbformat
import pytest
from nbclient import NotebookClient

from m5knobs import paths

NB = os.path.join(paths.ROOT, "notebooks", "m5_deep_learning_story.ipynb")


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(os.path.join(paths.RESULTS, "manifest.json")), reason="needs results/")
def test_notebook_executes(tmp_path, monkeypatch):
    # the kernel inherits this env: figures and FINDINGS.md go to tmp_path, not the repo
    monkeypatch.setenv("FIG_DIR", str(tmp_path))
    monkeypatch.setenv("FINDINGS_PATH", str(tmp_path / "FINDINGS.md"))
    nb = nbformat.read(NB, as_version=4)
    NotebookClient(nb, timeout=1200, kernel_name="python3", resources={"metadata": {"path": os.path.dirname(NB)}}).execute()


def test_one_story_notebook_without_local_paths_or_errors():
    assert sorted(f for f in os.listdir(os.path.dirname(NB)) if f.endswith(".ipynb")) == [os.path.basename(NB)]
    text = open(NB).read()
    assert "/Users/" not in text
    nb = nbformat.reads(text, as_version=4)
    outs = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])]
    assert outs, "commit the notebook with its outputs"
    assert not [o for o in outs if o.output_type == "error" or (o.output_type == "stream" and o.name == "stderr")]
