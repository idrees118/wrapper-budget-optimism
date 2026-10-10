"""
revision_analysis.py
Revision checks. Run AFTER the two extra runs, from the project root:

    python src/run_experiments.py --methods random_bern
    python src/run_experiments.py --with_duplicates
    python src/revision_analysis.py

Part A  random search with the same starting distribution as GA and BPSO
        (each feature on with probability 0.5), 12 datasets, k-NN.
Part B  Red Wine, Spambase and Libras with duplicate rows kept, compared with the
        main study (duplicates removed), k-NN.
Reads saved results only. Output: results/stats/revision_*.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import holm, wilcoxon  # noqa: E402
from extra_analysis import tost  # noqa: E402
from utils import CLEAN_DIR, DUPS_DIR, RESULTS_DIR, STATS_DIR  # noqa: E402

QUICK = "--quick" in sys.argv
FIRST, MID, FINAL = (50, 100, 200) if QUICK else (100, 1000, 4000)
RUNS = RESULTS_DIR / ("runs_quick" if QUICK else "runs")
DUPS = RESULTS_DIR / ("runs_quick_dups" if QUICK else "runs_dups")
pd.set_option("display.width", 220)


def load_runs(folder, methods=None):
    files = sorted(Path(folder).glob("*.csv"))
    if methods:
        files = [f for f in files if f.name.split("__")[1] in methods]
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def dataset_means(runs):
    """Average the outer repeats of each dataset (the dataset is the unit of analysis)."""
    s = runs[runs["checkpoint"].notna()]
    return s.groupby(["dataset", "method", "checkpoint"], as_index=False)[
        ["inner_ba", "outer_ba", "gap", "n_features"]].mean()


def wide(dm, metric, cp, method):
    return dm[(dm.method == method) & (dm.checkpoint == cp)].set_index("dataset")[metric]


# ---------------- Part A ----------------
main_runs = load_runs(RUNS, ["ga", "pso", "random", "random_bern"])
if "random_bern" not in set(main_runs.get("method", [])):
    print("Part A skipped: run  python src/run_experiments.py --methods random_bern  first.")
else:
    dm = dataset_means(main_runs)
    n_done = dm[dm.method == "random_bern"]["dataset"].nunique()
    print(f"===== Part A: random search, p = 0.5 per feature ({n_done} datasets) =====")
    rows = []
    for m in ["ga", "pso", "random", "random_bern"]:
        rows.append({"method": m,
                     "gap_100": wide(dm, "gap", FIRST, m).mean(),
                     "gap_4000": wide(dm, "gap", FINAL, m).mean(),
                     "test_1000": wide(dm, "outer_ba", MID, m).mean(),
                     "test_4000": wide(dm, "outer_ba", FINAL, m).mean(),
                     "size_4000": wide(dm, "n_features", FINAL, m).mean()})
    summary = pd.DataFrame(rows)
    print(summary.round(4).to_string(index=False))

    b = "random_bern"
    tests = []
    p, r, md = wilcoxon(wide(dm, "gap", FINAL, b), wide(dm, "gap", FIRST, b))
    tests.append({"test": "gap growth 4000 vs 100 (random_bern)", "median_diff": md, "r": r, "p": p})
    for other in ["random", "ga", "pso"]:
        a_, o_ = wide(dm, "outer_ba", FINAL, b), wide(dm, "outer_ba", FINAL, other)
        p, r, md = wilcoxon(a_, o_.reindex(a_.index))
        tests.append({"test": f"test BA at 4000: random_bern vs {other}", "median_diff": md, "r": r, "p": p})
        a_, o_ = wide(dm, "gap", FINAL, b), wide(dm, "gap", FINAL, other)
        p, r, md = wilcoxon(a_, o_.reindex(a_.index))
        tests.append({"test": f"gap at 4000: random_bern vs {other}", "median_diff": md, "r": r, "p": p})
    tests = pd.DataFrame(tests)
    tests["p_holm"] = holm(tests["p"])
    print("\n" + tests.round(4).to_string(index=False))

    e = tost(wide(dm, "outer_ba", FINAL, b), wide(dm, "outer_ba", MID, b))
    print("\nEquivalence, random_bern test BA at 4000 vs 1000 (margin 0.01): "
          f"HL {e['hodges_lehmann']:.4f}, 90% CI [{e['hl90_low']:.4f}, {e['hl90_high']:.4f}], "
          f"TOST p {e['p_tost']:.4f}")
    summary.round(5).to_csv(STATS_DIR / "revision_random_bern_summary.csv", index=False)
    tests.round(5).to_csv(STATS_DIR / "revision_random_bern_tests.csv", index=False)

# ---------------- Part B ----------------
dup_runs = load_runs(DUPS)
if dup_runs.empty:
    print("\nPart B skipped: run  python src/run_experiments.py --with_duplicates  first.")
else:
    print("\n===== Part B: duplicates kept vs removed (k-NN) =====")
    sets = sorted(dup_runs["dataset"].unique())
    for d in sets:
        a = pd.read_csv(CLEAN_DIR / f"{d}.csv")["target"].value_counts(normalize=True)
        k = pd.read_csv(DUPS_DIR / f"{d}.csv")["target"].value_counts(normalize=True)
        print(f"{d}: smallest class {100 * a.min():.1f}% after removal, {100 * k.min():.1f}% with duplicates")
    clean_runs = load_runs(RUNS)
    clean_runs = clean_runs[clean_runs["dataset"].isin(sets)]
    out = []
    for label, runs in [("removed", clean_runs), ("kept", dup_runs)]:
        dm = dataset_means(runs)
        for (d, m), g in dm.groupby(["dataset", "method"]):
            if m == "random_bern":
                continue
            g = g.set_index("checkpoint")
            out.append({"dataset": d, "method": m, "duplicates": label,
                        "gap_100": g.loc[FIRST, "gap"], "gap_4000": g.loc[FINAL, "gap"],
                        "test_1000": g.loc[MID, "outer_ba"], "test_4000": g.loc[FINAL, "outer_ba"]})
        base = runs[runs["checkpoint"].isna()].groupby(["dataset", "method"], as_index=False)["outer_ba"].mean()
        for _, row in base.iterrows():
            out.append({"dataset": row["dataset"], "method": row["method"], "duplicates": label,
                        "test_4000": row["outer_ba"]})
    out = pd.DataFrame(out).sort_values(["dataset", "method", "duplicates"])
    print(out.round(3).to_string(index=False))
    out.round(5).to_csv(STATS_DIR / "revision_duplicates.csv", index=False)