"""
gap_comparison.py
Revision check. Reads saved results only (a few seconds). From the project root:

    python src/gap_comparison.py

Compares the optimism gap BETWEEN search methods, which the within-method tests in the
paper do not do:
  family 1: gap at 4,000 evaluations, every pair of methods (6 tests)
  family 2: growth of the gap from 100 to 4,000 evaluations, every pair (6 tests)
Two-sided Wilcoxon signed-rank tests on dataset means, rank-biserial r, Holm within each
family. Done for k-NN and for logistic regression. Output: results/stats/gap_comparison.csv
"""

import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import holm, wilcoxon  # noqa: E402
from utils import STATS_DIR, TABLES_DIR  # noqa: E402

FIRST, FINAL = 100, 4000
METHODS = ["ga", "pso", "random", "sfs"]
rows = []
for tag, name in [("", "kNN"), ("_logreg", "LR")]:
    path = TABLES_DIR / f"dataset_means_search{tag}.csv"
    if not path.exists():
        continue
    ds = pd.read_csv(path)
    gap = ds.pivot_table(index="dataset", columns=["method", "checkpoint"], values="gap")
    final = {m: gap[(m, FINAL)] for m in METHODS}
    growth = {m: gap[(m, FINAL)] - gap[(m, FIRST)] for m in METHODS}
    for family, data in [("gap at 4000", final), ("gap growth 100 to 4000", growth)]:
        fam = []
        for a, b in combinations(METHODS, 2):
            p, r, md = wilcoxon(data[a], data[b])
            fam.append({"classifier": name, "family": family, "comparison": f"{a} vs {b}",
                        "mean_a": data[a].mean(), "mean_b": data[b].mean(),
                        "median_difference": md, "rank_biserial": r, "p_raw": p})
        fam = pd.DataFrame(fam)
        fam["p_holm"] = holm(fam["p_raw"])
        rows.append(fam)

out = pd.concat(rows, ignore_index=True).round(4)
out.to_csv(STATS_DIR / "gap_comparison.csv", index=False)
pd.set_option("display.width", 220)
print(out.to_string(index=False))