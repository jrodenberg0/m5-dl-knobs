"""Every run behind the figures (DESIGN.md). F3 row pairs are in F3_ROWS."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Arm:
    id: str
    group: str
    arch: str  # NeuralForecast class name, or "LGBM"
    overrides: dict = field(default_factory=dict)
    refit_each_fold: bool = False
    insample: bool = True  # in-sample error is the slowest step (~10 min at 30k); only the verdicts use it


HUBER = dict(loss="huber", huber_delta=5.0)

ARMS = [
    Arm("f2_noclip", "f2", "NHITS", dict(clip_percentile=None)),
    Arm("f2_clip", "f2", "NHITS", {}),
    Arm("f3_fix", "f3", "NHITS", dict(HUBER)),
    Arm("f3_fix_noguard", "f3", "NHITS", dict(HUBER, min_steps=0)),
    Arm("f3_patience250", "f3", "NHITS", dict(HUBER, min_steps=0, early_stop_patience_steps=5, val_check_steps=50)),
    Arm("f3_input13", "f3", "NHITS", dict(HUBER, input_size=13)),
    Arm("f3_minmax", "f3", "NHITS", dict(HUBER, scaler_type="minmax")),
    Arm("f3_mae", "f3", "NHITS", dict(loss="mae")),
    Arm("f4_lgbm", "f4", "LGBM", {}, True),
    Arm("f4_nhits", "f4", "NHITS", {}, True),
    Arm("f4_tcn", "f4", "TCN", {}, True),
]

# Clipping stress test (Tweedie): stressors fixed before any result. Each pair differs only in clipping.
STRESSORS = {"lr1e2": dict(lr_max=1e-2), "lr3e2": dict(lr_max=3e-2), "nofilter": dict(availability_threshold=[0.0, 0.0])}
for _k, _o in STRESSORS.items():
    ARMS += [Arm(f"st_{_k}_noclip", "stress", "NHITS", dict(_o, clip_percentile=None)), Arm(f"st_{_k}_clip", "stress", "NHITS", dict(_o))]

# Early-stopping study (Huber d5): one schedule for every arm (one-cycle over 10,000 steps, no minimum-steps guard), so only
# the stopping rule differs. Base = 1,000-step window (20 checks x 50 steps), 13-week validation tail, best weights restored.
ES_BASE = dict(HUBER, max_steps=10000, min_steps=0)
ARMS += [Arm(f"es_p{w}", "es", "NHITS", dict(ES_BASE, early_stop_patience_steps=w // 50, val_check_steps=50), insample=False)
         for w in (250, 500, 1000, 2000, 4000)]
ARMS += [Arm("es_val26", "es", "NHITS", dict(ES_BASE, early_stop_patience_steps=20, val_size=26), insample=False),
         Arm("es_val52", "es", "NHITS", dict(ES_BASE, early_stop_patience_steps=20, val_size=52), insample=False),
         Arm("es_check25", "es", "NHITS", dict(ES_BASE, early_stop_patience_steps=40, val_check_steps=25), insample=False),
         Arm("es_check100", "es", "NHITS", dict(ES_BASE, early_stop_patience_steps=10, val_check_steps=100), insample=False),
         Arm("es_norestore", "es", "NHITS", dict(ES_BASE, early_stop_patience_steps=20, restore_best=False), insample=False)]

# (row label, loss label, mistake arm, fix arm, training losses on the same scale in both arms)
F3_ROWS = [
    ("No gradient clipping", "Tweedie", "f2_noclip", "f2_clip", True),
    ("Early-stop window 250 steps", "Huber δ5", "f3_patience250", "f3_fix_noguard", True),
    ("Input window 13 weeks", "Huber δ5", "f3_input13", "f3_fix", False),  # RevIN stats over 13 vs 52 weeks
    ("MinMax scaler", "Huber δ5 (≈ MSE on MinMax scale)", "f3_minmax", "f3_fix", False),  # δ5 never caps errors in [0, 1] units
    ("MAE loss", "MAE vs Huber δ5", "f3_mae", "f3_fix", False),
]


def by_id(arm_id: str) -> Arm:
    return next(a for a in ARMS if a.id == arm_id)
