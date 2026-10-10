"""
verify.py
Two quick checks of the code, run on three small datasets (a few minutes). From the project root:

    python src/verify.py

1. Reproducibility: the same tasks are run again from scratch. Every selected subset,
   inner score and test score must match results/runs/ exactly.
2. Leakage: the tasks are run again with every TEST row replaced by random numbers.
   If the search never looks at test rows, it must select exactly the same subsets
   with exactly the same inner scores. (Only the test scores may change.)

Nothing in results/runs/ is changed. The reruns go to results/verify/.
"""

import copy
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiments as R  # noqa: E402
from utils import RESULTS_DIR, load_config, outer_split  # noqa: E402

QUICK = "--quick" in sys.argv
DATASETS = ["sonar", "ionosphere", "wine_red"]
METHODS = ["ga", "pso", "random", "sfs"]
REPEAT = 0


def get_cfg():
    cfg = load_config()
    if QUICK:
        cfg = copy.deepcopy(cfg)
        cfg.update(outer_repeats=2, max_evaluations=200, checkpoints=[50, 100, 200])
    return cfg


def worker(dataset, method, mode, out_dir):
    cfg = get_cfg()
    if mode == "leakage":
        original = R.load_clean

        def corrupted(name, folder=None):
            X, y, cols = original(name, folder)
            _, test_idx = outer_split(X, y, REPEAT, cfg)      # same split: it depends on y only
            X = X.copy()
            X[test_idx] = np.random.default_rng(123).normal(0, 1000, size=(len(test_idx), X.shape[1]))
            return X, y, cols

        R.load_clean = corrupted                              # only inside this worker process
    return R.run_task(dataset, REPEAT, method, cfg, out_dir)


def main():
    runs = RESULTS_DIR / ("runs_quick" if QUICK else "runs")
    base = RESULTS_DIR / "verify"
    if base.exists():
        shutil.rmtree(base)
    dirs = {m: base / m for m in ["reproducibility", "leakage"]}
    for d in dirs.values():
        d.mkdir(parents=True)

    jobs = [(d, m, mode) for mode in dirs for d in DATASETS for m in METHODS]
    print(f"running {len(jobs)} tasks ...")
    Parallel(n_jobs=-1, verbose=5)(delayed(worker)(d, m, mode, dirs[mode]) for d, m, mode in jobs)

    rows = []
    for d in DATASETS:
        for m in METHODS:
            name = f"{d}__{m}__r{REPEAT}.csv"
            orig = pd.read_csv(runs / name).sort_values("checkpoint").reset_index(drop=True)
            rep = pd.read_csv(dirs["reproducibility"] / name).sort_values("checkpoint").reset_index(drop=True)
            leak = pd.read_csv(dirs["leakage"] / name).sort_values("checkpoint").reset_index(drop=True)
            rows.append({
                "dataset": d, "method": m,
                "repro_same_subsets": bool((orig["selected"] == rep["selected"]).all()),
                "repro_same_scores": bool(np.allclose(orig[["inner_ba", "outer_ba"]], rep[["inner_ba", "outer_ba"]], atol=1e-12)),
                "leak_same_subsets": bool((orig["selected"] == leak["selected"]).all()),
                "leak_same_inner": bool(np.allclose(orig["inner_ba"], leak["inner_ba"], atol=1e-12)),
                "leak_test_changed": bool(not np.allclose(orig["outer_ba"], leak["outer_ba"])),
            })
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n" + out.to_string(index=False))
    ok = out[["repro_same_subsets", "repro_same_scores", "leak_same_subsets", "leak_same_inner"]].all().all()
    print("\nREPRODUCIBILITY:", "PASSED" if out[["repro_same_subsets", "repro_same_scores"]].all().all() else "FAILED")
    print("LEAKAGE CHECK:  ", "PASSED" if out[["leak_same_subsets", "leak_same_inner"]].all().all() else "FAILED")
    print("(leak_test_changed = True only shows the test rows really were replaced)")
    out.to_csv(base / "verify_summary.csv", index=False)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()