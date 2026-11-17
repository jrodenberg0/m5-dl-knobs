# Parity: knobs.py vs the original private wrapper

5,000-series public panel (identical to the original panel), seed 1, Apple M4 / MPS, each path in its own process. Original results: the article's earlier wrapper runs.

| arm       |   wape_a |   wape_b |   d_wape |   bias_a |   bias_b |   d_bias |    best_a |    best_b |    stop_a |    stop_b |   max_abs_dfc |   corr_fc | pass_   |
|:----------|---------:|---------:|---------:|---------:|---------:|---------:|----------:|----------:|----------:|----------:|--------------:|----------:|:--------|
| f2_clip   |  46.8500 |  46.8500 |   0.0000 |   0.1439 |   0.1439 |   0.0000 | 2100.0000 | 2100.0000 | 3100.0000 | 3100.0000 |        0.0000 |    1.0000 | True    |
| f2_noclip |  48.0459 |  48.0459 |   0.0000 |   4.7505 |   4.7505 |   0.0000 | 2100.0000 | 2100.0000 | 3100.0000 | 3100.0000 |        0.0000 |    1.0000 | True    |
| f3_fix    |  45.6845 |  45.6845 |   0.0000 |  -3.4305 |  -3.4305 |   0.0000 | 2350.0000 | 2350.0000 | 3350.0000 | 3350.0000 |        0.0000 |    1.0000 | True    |
PARITY PASS

Every forecast is identical (max |Δ| = 0), as are best and stop steps.
