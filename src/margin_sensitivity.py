"""
margin_sensitivity.py
Revision check. Reads saved results only; nothing is re-run. From the project root:

    python src/margin_sensitivity.py

For each classifier (kNN, logistic regression) and each search method it repeats the
equivalence test of test balanced accuracy at 4,000 versus 1,000 evaluations with
margins of 0.005, 0.01 and 0.02. It also reports the Hodges-Lehmann estimate (the
location that the Wilcoxon-based TOST tests) next to the mean, and how many evaluations
SFS really used. Output: results/stats/margin_sensitivity.csv and
results/stats/sfs_evals_used.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import holm  # noqa: E402
from extra_analysis import tost  # noqa: E402
from utils import STATS_DIR, TABLES_DIR  # noqa: E402

MARGINS = [0.005, 0.01, 0.02]
QUICK = "--quick" in sys.argv
FINAL, PLATEAU = (200, 100) if QUICK else (4000, 1000)


def hodges_lehmann(d):
    d = np.asarray(d, float)
    walsh = [(d[i] + d[j]) / 2 for i in range(len(d)) for j in range(i, len(d))]
    return float(np.median(walsh))


rows, sfs_rows = [], []
for tag, name in [("", "kNN"), ("_logreg", "LR")]:
    path = TABLES_DIR / f"dataset_means_search{tag}{'_quick' if QUICK else ''}.csv"
    if not path.exists():
        print(f"skipping {name}: {path.name} not found")
        continue
    ds = pd.read_csv(path)
    for margin in MARGINS:
        family = []
        for m in sorted(ds["method"].unique()):
            a = ds[(ds.method == m) & (ds.checkpoint == FINAL)].set_index("dataset")["outer_ba"]
            b = ds[(ds.method == m) & (ds.checkpoint == PLATEAU)].set_index("dataset")["outer_ba"]
            b = b.reindex(a.index)
            r = tost(a, b, margin=margin)
            family.append({"classifier": name, "margin": margin, "method": m,
                           "n_datasets": len(a), **r})
        p_adj = holm(pd.Series([f["p_tost"] for f in family]))
        for f, p in zip(family, p_adj):
            f["p_tost_holm"] = p
            f["equivalent"] = bool(p < 0.05 and f["equivalent_by_ci"])
        rows += family
    s = ds[(ds.method == "sfs") & (ds.checkpoint == FINAL)][["dataset", "evals_used"]]
    sfs_rows += [{"classifier": name, **r} for r in s.to_dict("records")]

out = pd.DataFrame(rows).round(4)
out.to_csv(STATS_DIR / "margin_sensitivity.csv", index=False)
pd.DataFrame(sfs_rows).round(1).to_csv(STATS_DIR / "sfs_evals_used.csv", index=False)
pd.set_option("display.width", 200)
print(out[["classifier", "margin", "method", "mean_difference", "hodges_lehmann",
           "hl90_low", "hl90_high", "p_tost_holm", "equivalent"]].to_string(index=False))
print()
print(pd.DataFrame(sfs_rows).to_string(index=False))