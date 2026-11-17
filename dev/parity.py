"""Dev tool, needs the private NFGenericWrapper: train the parity arms with either code path, ONE PATH PER PROCESS (exec-ing
the wrapper patches NeuralForecast's data module globally, which would leak into a standalone run in the same process).

python3 dev/parity.py --path standalone --out scratch/parity_standalone
python3 dev/parity.py --path wrapper --out scratch/parity_wrapper --wrapper /path/to/private/model_wrapper.py
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARMS = ["f2_clip", "f2_noclip", "f3_fix"]

ap = argparse.ArgumentParser()
ap.add_argument("--path", choices=["standalone", "wrapper"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--wrapper", default=None)
a = ap.parse_args()
os.environ.setdefault("N_SERIES", "5000")
import common, run_experiments as rx  # noqa: E402

if a.path == "wrapper":
    ns = {"__name__": "nf_generic_wrapper"}
    exec(compile(open(os.path.expanduser(a.wrapper)).read(), a.wrapper, "exec"), ns)

    def make_nf(arch, seed, **ov):
        import knobs
        kw = {**dict(input_size=52, max_steps=5000, lr_schedule="onecycle", lr_max=1e-3, clip_percentile=10.0, optimizer="nadam",
                     batch_size=128, windows_batch_size=1024, restore_best=True, loss="tweedie", tweedie_rho=1.5), **ov}
        w = ns["NFGenericWrapper"](horizon=common.H, freq=common.FREQ, name="yhat", arch=arch, season_length=kw.pop("input_size"),
                                   random_seed=seed, accelerator=os.environ.get("ACCEL", "mps"), devices=1, log_diagnostics=True, **kw)
        w._lr = knobs.LRCapture()
        w.nf.models[0].trainer_kwargs["callbacks"].append(w._lr)
        return w
    common.make_nf = make_nf
rx.run(ARMS, seeds=[1], results_dir=a.out)
