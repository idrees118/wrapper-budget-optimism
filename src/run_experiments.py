"""
run_experiments.py
Step 2 of the study. From the project root:

    python src/run_experiments.py --quick      # smoke test (a few minutes), results go to results/runs_quick/
    python src/run_experiments.py --classifier logreg   # robustness check (logistic regression)
    python src/run_experiments.py              # the full study (about 13 CPU-core hours in total;
                                               # roughly 2-3 hours on an 8-core laptop)

Revision checks (added after the main study; they do not touch results/runs of the main methods):
    python src/run_experiments.py --methods random_bern     # random search with p = 0.5 per feature
    python src/run_experiments.py --with_duplicates         # 3 datasets with duplicate rows kept

One "task" = one dataset x one outer repeat x one method. Each task writes its own
CSV into results/runs/. If you stop the run and start it again, finished tasks are
skipped, so nothing is lost.

How one task works (this is the whole protocol):
  1. split the data into outer-train (70%) and outer-test (30%), stratified
  2. build 5 stratified inner folds from outer-train ONLY
  3. the search method scores subsets with those inner folds (it never sees outer-test)
  4. at every checkpoint (100, 250, ... evaluations) we keep the best-so-far subset
  5. AFTER the search is over, each checkpoint subset is refitted on the whole
     outer-train part and scored once on outer-test
Step 5 happens only after step 3 has finished, so the test score can never steer the search.
"""

import argparse
import copy
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import run_baseline  # noqa: E402
from search_methods import SEARCH_METHODS, Evaluator  # noqa: E402
from utils import (DUPS_DIR, RESULTS_DIR, ScaledFolds, inner_folds,  # noqa: E402
                   load_clean, load_config, make_classifier, outer_split, stable_seed)


def noise_columns(dataset, columns, cfg):
    """Boolean flag per column: True if it is a known pure-noise column."""
    rng = cfg.get("known_noise", {}).get(dataset)
    if rng is None:
        return None
    lo, hi = rng
    return np.array([lo <= int(c[1:]) <= hi for c in columns])


def outer_score(cols, X_tr, y_tr, X_te, y_te, cfg):
    """Refit scaler + classifier on outer-train (chosen columns only), score on outer-test."""
    scaler = StandardScaler().fit(X_tr[:, cols])
    model = make_classifier(cfg, len(y_tr)).fit(scaler.transform(X_tr[:, cols]), y_tr)
    return balanced_accuracy_score(y_te, model.predict(scaler.transform(X_te[:, cols])))


def run_task(dataset, repeat, method, cfg, out_dir, data_folder=None):
    out_file = out_dir / f"{dataset}__{method}__r{repeat}.csv"
    if out_file.exists():
        return "skipped"

    X, y, columns = load_clean(dataset, data_folder)
    train_idx, test_idx = outer_split(X, y, repeat, cfg)
    assert len(np.intersect1d(train_idx, test_idx)) == 0, "train and test overlap"
    X_tr, y_tr, X_te, y_te = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
    folds = inner_folds(y_tr, repeat, cfg)            # indices into X_tr only
    seed = stable_seed(cfg["seed"], dataset, repeat, method)
    noise = noise_columns(dataset, columns, cfg)

    base = {"dataset": dataset, "repeat": repeat, "method": method,
            "n_train": len(y_tr), "n_test": len(y_te), "p": X.shape[1]}
    rows = []

    if method == "baselines":
        for name in cfg["baselines"]:
            t0 = time.perf_counter()
            r = run_baseline(name, X_tr, y_tr, X_te, y_te, folds, cfg, seed)
            rows.append({**base, "method": name, "checkpoint": np.nan, "evals_used": np.nan,
                         "n_features": r["n_features"], "n_noise": np.nan,
                         "inner_ba": r["inner_ba"], "outer_ba": r["outer_ba"],
                         "fitness": np.nan, "seconds": time.perf_counter() - t0,
                         "selected": "", "setting": r["setting"]})
    else:
        ev = Evaluator(ScaledFolds(X_tr, y_tr, folds), X.shape[1], cfg)
        SEARCH_METHODS[method](ev, np.random.default_rng(seed), cfg)
        snapshots = ev.finish()
        # ---- the search is finished; only now do we touch the test rows ----
        for s in snapshots:
            cols = np.flatnonzero(s["mask"])
            rows.append({**base, "checkpoint": s["checkpoint"], "evals_used": s["evals_used"],
                         "n_features": int(cols.size),
                         "n_noise": int(noise[cols].sum()) if noise is not None else np.nan,
                         "inner_ba": s["inner_ba"],
                         "outer_ba": outer_score(cols, X_tr, y_tr, X_te, y_te, cfg),
                         "fitness": s["fitness"], "seconds": s["seconds"],
                         "selected": ";".join(columns[c] for c in cols), "setting": ""})

    df = pd.DataFrame(rows)
    df["gap"] = df["inner_ba"] - df["outer_ba"]       # positive = the inner score was optimistic
    tmp = out_file.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, out_file)                         # write-then-rename: no half-written files
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny settings to check the pipeline works")
    ap.add_argument("--datasets", nargs="*", help="run only these datasets")
    ap.add_argument("--n_jobs", type=int, help="override n_jobs from the config")
    ap.add_argument("--classifier", choices=["knn", "logreg"],
                    help="logreg = robustness check on the datasets listed in the config")
    ap.add_argument("--methods", nargs="*", help="run only these methods, e.g. random_bern")
    ap.add_argument("--with_duplicates", action="store_true",
                    help="revision check: Red Wine, Spambase and Libras with duplicate rows kept")
    args = ap.parse_args()

    cfg = load_config()
    out_dir = RESULTS_DIR / "runs"
    default_datasets = cfg["datasets"]
    if args.classifier == "logreg":
        cfg = copy.deepcopy(cfg)
        cfg["classifier"] = "logreg"
        default_datasets = cfg["robustness"]["datasets"]
        out_dir = RESULTS_DIR / "runs_logreg"
    if args.quick:
        cfg = copy.deepcopy(cfg)
        cfg.update(outer_repeats=2, max_evaluations=200, checkpoints=[50, 100, 200])
        out_dir = out_dir.parent / (out_dir.name + "_quick")
    out_dir.mkdir(parents=True, exist_ok=True)

    data_folder = None
    if args.with_duplicates:
        default_datasets = ["wine_red", "spambase", "libras"]
        data_folder = DUPS_DIR
        out_dir = out_dir.parent / (out_dir.name + "_dups")
        out_dir.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets or default_datasets
    methods = args.methods or (list(cfg["search_methods"]) + ["baselines"])
    tasks = [(d, r, m) for d in datasets for r in range(cfg["outer_repeats"]) for m in methods]
    n_jobs = args.n_jobs if args.n_jobs is not None else cfg["n_jobs"]
    print(f"{len(tasks)} tasks, writing to {out_dir}")

    # Logistic regression may warn about convergence on tiny subsets; the warnings
    # do not change results and would flood the log, so workers silence them.
    os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning,ignore::RuntimeWarning")
    t0 = time.perf_counter()
    status = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_task)(d, r, m, cfg, out_dir, data_folder) for d, r, m in tasks
    )
    print(f"finished: {status.count('done')} new, {status.count('skipped')} skipped, "
          f"{(time.perf_counter() - t0) / 60:.1f} minutes")


if __name__ == "__main__":
    main()