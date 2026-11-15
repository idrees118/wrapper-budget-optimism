"""
revision_summary.py - reads saved results only. From the project root:
    python src/revision_summary.py
Part 1: sensitivity grid (4 datasets, 5 splits) against the main-study baseline cell.
Part 2: grey wolf (gwo) and whale (woa) against GA, BPSO, random and SFS on 12 datasets.
Part 3: share of repeated subsets (cache hits) at 4,000 evaluations.
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

R = Path("results")
KEEP = ["ga", "pso", "random", "sfs", "gwo", "woa"]
SENS = ["k1", "k11", "a90", "a95", "pop20", "pop50"]
DS4 = ["sonar", "ionosphere", "german_credit", "qsar_biodeg"]


def load(folder, repeats=None):
    files = sorted(glob.glob(str(R / folder / "*.csv")))
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df = df[df["method"].isin(KEEP)].copy()
    df["checkpoint"] = df["checkpoint"].astype(int)
    if repeats is not None:
        df = df[df["repeat"] < repeats]
    return df


def dataset_means(df):
    return (df.groupby(["dataset", "method", "checkpoint"])[["inner_ba", "outer_ba", "gap"]]
              .mean().reset_index())


def wide(dm, m):
    return dm[dm["method"] == m].pivot_table(index="dataset", columns="checkpoint",
                                             values=["inner_ba", "outer_ba", "gap"])


# ---------------- Part 1: sensitivity ----------------
base = load("runs", repeats=5)
base = base[base["dataset"].isin(DS4) & base["method"].isin(["ga", "pso", "random"])]
frames = {"base k5 a.99 pop30": base}
for v in SENS:
    frames[v] = load(f"runs_sens_{v}")

rows = []
for name, df in frames.items():
    dm = dataset_means(df)
    for m in ["ga", "pso", "random"]:
        w = wide(dm, m)
        g, o, i = w["gap"], w["outer_ba"], w["inner_ba"]
        rows.append({"setting": name, "method": m, "n_datasets": len(g),
                     "gap_100": g[100].mean(), "gap_4000": g[4000].mean(),
                     "gap_grew_on": f"{int((g[4000] > g[100]).sum())}/{len(g)}",
                     "inner_4000_minus_1000": (i[4000] - i[1000]).mean(),
                     "test_4000_minus_1000": (o[4000] - o[1000]).mean(),
                     "test_4000": o[4000].mean()})
sens = pd.DataFrame(rows).round(4)
sens.to_csv(R / "stats" / "revision_sensitivity.csv", index=False)
pd.set_option("display.width", 250)
print("\n=== PART 1: sensitivity (kNN, 4 datasets, 5 splits) ===")
print(sens.to_string(index=False))
order = sens.pivot(index="setting", columns="method", values="gap_4000")
print("\nGA gap > BPSO gap > random gap after 4,000 evaluations, by setting:")
print((order["ga"] > order["pso"]).rename("GA>BPSO").to_frame().join(
      (order["pso"] > order["random"]).rename("BPSO>random")).to_string())

# ---------------- Part 2: newer optimisers ----------------
main = load("runs")
new = load("runs_newopt")
dm = dataset_means(pd.concat([main, new], ignore_index=True))
rows = []
for m in ["ga", "pso", "gwo", "woa", "random", "sfs"]:
    w = wide(dm, m)
    g, o, i = w["gap"], w["outer_ba"], w["inner_ba"]
    p = wilcoxon(g[4000], g[100]).pvalue
    d = o[4000] - o[1000]
    rows.append({"method": m, "n_datasets": len(g),
                 "inner_100": i[100].mean(), "inner_4000": i[4000].mean(),
                 "test_100": o[100].mean(), "test_1000": o[1000].mean(), "test_4000": o[4000].mean(),
                 "gap_100": g[100].mean(), "gap_4000": g[4000].mean(),
                 "gap_grew_on": f"{int((g[4000] > g[100]).sum())}/{len(g)}",
                 "p_gap_growth": p,
                 "test_4000_minus_1000": d.mean(),
                 "datasets_within_0.01": f"{int((d.abs() < 0.01).sum())}/{len(d)}"})
nw = pd.DataFrame(rows)
nw.to_csv(R / "stats" / "revision_newopt.csv", index=False)
print("\n=== PART 2: all methods, kNN, 12 datasets (raw p, exploratory) ===")
print(nw.round(4).to_string(index=False))

# ---------------- Part 3: repeated subsets ----------------
print("\n=== PART 3: share of evaluations that were repeats, at 4,000 ===")
parts = [new[new["method"].isin(["gwo", "woa"])]]
for v in SENS:
    parts.append(frames[v])
c = pd.concat(parts, ignore_index=True)
c = c[c["checkpoint"] == 4000].copy()
c["repeat_share"] = c["cache_hits"] / c["evals_used"]
print(c.groupby("method")["repeat_share"].agg(["mean", "min", "max", "count"]).round(3).to_string())
print("(gwo/woa: 12 datasets from runs_newopt; ga/pso/random: the 4-dataset sensitivity runs, mixed settings)")
