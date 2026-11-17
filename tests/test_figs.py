import json
import os

import pytest
from PIL import Image

from tests.fixtures.make_fake_results import make


@pytest.fixture
def res(tmp_path):
    return make(str(tmp_path / "results"))


def _png_ok(path, min_w=1200):
    assert os.path.exists(path), path
    assert os.path.exists(path.replace(".png", ".svg"))
    assert Image.open(path).size[0] >= min_w


def test_f4_renders(res, tmp_path):
    import pandas as pd
    import fig_f4_table
    fig_f4_table.main(results_dir=res, out_dir=str(tmp_path / "fig"), n_boot=50)
    _png_ok(str(tmp_path / "fig" / "f4_table.png"), min_w=1000)
    t = pd.read_csv(os.path.join(res, "f4_table.csv"))
    assert list(t["model"]) == ["LGBM (baseline)", "N-HiTS", "TCN"]
    assert t["p_vs_lgbm"].dropna().between(0, 1).all()


def _toy_data(n=400, seed=0):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    x = np.log1p(rng.poisson(2.0, n) * (rng.random(n) < 0.5))
    y = rng.poisson(np.exp(-0.5 + 0.8 * x)).astype(float)
    y[rng.random(n) < 0.02] *= 30  # a few spikes, like promo weeks
    return pd.DataFrame({"x": x, "y": y})


def test_toy_clip_paths(tmp_path):
    import numpy as np
    import toy_clip
    paths, surf = toy_clip.main(steps=30, data=_toy_data(), results_dir=str(tmp_path))
    assert set(paths["run"]) == {"unclipped", "clipped"}
    assert (paths.groupby("run").size() == 31).all()
    step_len = paths.sort_values(["run", "step"]).groupby("run")[["a", "b"]].diff().pow(2).sum(axis=1).pow(0.5)
    mx = step_len.groupby(paths.sort_values(["run", "step"])["run"]).max()
    assert mx["clipped"] <= mx["unclipped"] + 1e-12
    assert np.isfinite(surf["loss"]).all()
    assert os.path.exists(os.path.join(str(tmp_path), "toy_paths.parquet"))


def test_f1_renders(tmp_path):
    import fig_f1_toy
    import toy_clip
    toy_clip.main(steps=30, data=_toy_data(), results_dir=str(tmp_path))
    fig_f1_toy.main(results_dir=str(tmp_path), out_dir=str(tmp_path / "fig"))
    _png_ok(str(tmp_path / "fig" / "f1_toy.png"), min_w=1000)
