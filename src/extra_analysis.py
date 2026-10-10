"""
extra_analysis.py
Step 4 of the study. Run AFTER analysis.py, from the project root:

    python src/extra_analysis.py            # full results
    python src/extra_analysis.py --quick    # smoke-test results
    python src/extra_analysis.py --tag logreg   # logistic-regression robustness run

It only reads saved results; nothing is re-run. It adds what reviewers usually ask for:

  1. Equivalence tests (TOST). "Not significantly different" is not the same as
     "the same". TOST checks whether a difference is SMALLER than a practical margin
     (here 0.01 balanced accuracy, i.e. one percentage point). We use two one-sided
     Wilcoxon signed-rank tests on dataset-level differences. The matching estimate is the
     Hodges-Lehmann estimate with its exact 90% interval (equivalent if inside +-0.01).
  2. Sensitivity: the key tests repeated without Hill-Valley (kNN is at chance there)
     and without Madelon (population methods start from ~250 of 500 features).
  3. Budget efficiency: what the first 1,000 evaluations buy versus the last 3,000,
     in inner score, test score, optimism gap and seconds.
  4. Peak analysis: on how many datasets the test score peaked BEFORE the full budget.
  5. Paper-ready figures (PDF + PNG, colour-blind-safe) and the dataset table.

As in analysis.py, the dataset is the unit of analysis (repeats averaged first).
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import LABELS, bootstrap_ci, holm, wilcoxon  # noqa: E402
from utils import FIGURES_DIR, STATS_DIR, TABLES_DIR, load_config  # noqa: E402

MARGIN = 0.01                      # practical-equivalence margin (balanced accuracy)
EXAMPLES = ["sonar", "madelon", "waveform", "ionosphere"]
# Okabe-Ito colour-blind-safe palette
COLORS = {"ga": "#0072B2", "pso": "#D55E00", "random": "#7F7F7F", "sfs": "#009E73"}
DOMAINS = {"wine_red": "Food chemistry", "vehicle": "Image shape", "german_credit": "Finance",
           "ionosphere": "Radar signals", "waveform": "Synthetic signals",
           "qsar_biodeg": "Environmental chemistry", "spambase": "E-mail text",
           "sonar": "Sonar signals", "libras": "Hand-movement video", "hill_valley": "Synthetic curves",
           "semeion": "Handwritten digits", "madelon": "Synthetic (NIPS 2003)"}


# ---------------- equivalence test ----------------

def walsh_ci(d, level=0.90):
    """Hodges-Lehmann estimate and the matching distribution-free confidence interval.
    The interval comes from inverting the Wilcoxon signed-rank test (Walsh averages, exact
    null distribution), so it answers the same question as the Wilcoxon-based TOST below:
    where is the centre (pseudo-median) of the dataset-level differences?"""
    d = np.sort(np.asarray(d, float))
    n = d.size
    walsh = np.sort([(d[i] + d[j]) / 2 for i in range(n) for j in range(i, n)])
    # exact distribution of the signed-rank statistic T+ under H0 (counts of each value)
    M = n * (n + 1) // 2
    counts = np.zeros(M + 1)
    counts[0] = 1
    for k in range(1, n + 1):
        counts[k:] = counts[k:] + counts[:-k].copy()
    cdf = np.cumsum(counts) / 2 ** n
    alpha = (1 - level) / 2
    c = int(np.searchsorted(cdf, alpha, side="right"))   # P(T+ <= c-1) <= alpha
    c = max(c, 1)
    return float(np.median(walsh)), float(walsh[c - 1]), float(walsh[M - c])


def tost(a, b, margin=MARGIN, seed=0):
    """Paired equivalence test of a vs b on dataset-level values.
    Estimand: the pseudo-median (centre of symmetry) of the differences, which is what the
    Wilcoxon signed-rank test is about. We report its Hodges-Lehmann estimate and the
    matching 90% interval; at the 5% level, the two one-sided Wilcoxon tests reject exactly
    when this interval lies inside the margin. The mean and its bootstrap interval are kept
    as descriptive numbers only."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    lo, hi = np.nan, np.nan
    if d.size >= 2:
        rng = np.random.default_rng(seed)
        means = rng.choice(d, size=(10_000, d.size), replace=True).mean(axis=1)
        lo, hi = np.percentile(means, [5, 95])
    hl, hl_lo, hl_hi = walsh_ci(d)
    # H0a: difference <= -margin  vs  H1a: difference > -margin
    # H0b: difference >= +margin  vs  H1b: difference < +margin
    pa = stats.wilcoxon(d + margin, alternative="greater").pvalue
    pb = stats.wilcoxon(d - margin, alternative="less").pvalue
    p = max(pa, pb)
    return {"mean_difference": d.mean(), "ci90_low": lo, "ci90_high": hi,
            "hodges_lehmann": hl, "hl90_low": hl_lo, "hl90_high": hl_hi,
            "p_tost": p, "equivalent_by_ci": bool(-margin < hl_lo and hl_hi < margin)}


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="", help="e.g. logreg, to analyse the robustness run")
    args = ap.parse_args()
    cfg = load_config()
    suffix = (f"_{args.tag}" if args.tag else "") + ("_quick" if args.quick else "")
    cps = [50, 100, 200] if args.quick else list(cfg["checkpoints"])
    final = cps[-1]
    plateau = 1000 if 1000 in cps else cps[-2]   # where the test curve flattens in the full run

    path_s = TABLES_DIR / f"dataset_means_search{suffix}.csv"
    if not path_s.exists():
        sys.exit("Run analysis.py first.")
    ds = pd.read_csv(path_s)
    base = pd.read_csv(TABLES_DIR / f"dataset_means_baselines{suffix}.csv")
    methods = [m for m in cfg["search_methods"] if m in ds["method"].unique()]
    datasets = sorted(ds["dataset"].unique())

    def wide(metric, cp, frame=ds, keep=None):
        t = frame[frame["checkpoint"] == cp].pivot(index="dataset", columns="method", values=metric)
        t = t.reindex(columns=methods)
        return t.loc[[d for d in t.index if keep is None or d in keep]]

    # ---- 1. Equivalence tests ----
    eq = []
    for m in methods:
        r = tost(wide("outer_ba", final)[m], wide("outer_ba", plateau)[m])
        eq.append({"comparison": f"{LABELS[m]}: test BA at {final} vs {plateau} evals", **r})
    for m in [x for x in methods if x != "random"]:
        r = tost(wide("outer_ba", final)[m], wide("outer_ba", final)["random"])
        eq.append({"comparison": f"{LABELS[m]} vs Random search at {final} evals", **r})
    mi = base[base["method"] == "mi_filter"].set_index("dataset")["outer_ba"]
    for m in methods:
        r = tost(wide("outer_ba", final)[m], mi.reindex(wide("outer_ba", final).index))
        eq.append({"comparison": f"{LABELS[m]} ({final} evals) vs MI filter", **r})
    eq = pd.DataFrame(eq)
    eq["p_tost_holm"] = holm(eq["p_tost"])
    eq["equivalent_holm"] = eq["p_tost_holm"] < 0.05
    eq.round(5).to_csv(STATS_DIR / f"equivalence_tost{suffix}.csv", index=False)

    # ---- 2. Sensitivity analysis ----
    candidates = {f"all {len(datasets)} datasets": datasets,
                  "without hill_valley": [d for d in datasets if d != "hill_valley"],
                  "without madelon": [d for d in datasets if d != "madelon"],
                  "without both": [d for d in datasets if d not in ("hill_valley", "madelon")]}
    # keep a subset only if it actually differs from the ones already kept
    subsets = {}
    for label, keep in candidates.items():
        if keep not in subsets.values():
            subsets[label] = keep
    sens = []
    for label, keep in subsets.items():
        for m in methods:
            p, r, md = wilcoxon(wide("gap", final, keep=keep)[m], wide("gap", cps[0], keep=keep)[m])
            sens.append({"subset": label, "n_datasets": len(keep), "method": LABELS[m],
                         "test": f"gap {final} vs {cps[0]}", "median_difference": md,
                         "rank_biserial": r, "p_raw": p})
            p, r, md = wilcoxon(wide("outer_ba", final, keep=keep)[m], wide("outer_ba", plateau, keep=keep)[m])
            sens.append({"subset": label, "n_datasets": len(keep), "method": LABELS[m],
                         "test": f"test BA {final} vs {plateau}", "median_difference": md,
                         "rank_biserial": r, "p_raw": p})
        w = wide("outer_ba", final, keep=keep).dropna()
        chi2, p = stats.friedmanchisquare(*[w[m] for m in methods])
        sens.append({"subset": label, "n_datasets": len(keep), "method": "all four",
                     "test": f"Friedman at {final}", "median_difference": np.nan,
                     "rank_biserial": chi2 / (len(w) * (len(methods) - 1)),  # Kendall's W in this row
                     "p_raw": p})
    sens = pd.DataFrame(sens)
    sens["p_holm_within_subset"] = np.nan
    for label in subsets:
        idx = sens["subset"] == label
        sens.loc[idx, "p_holm_within_subset"] = holm(sens.loc[idx, "p_raw"])
    sens.round(5).to_csv(STATS_DIR / f"sensitivity{suffix}.csv", index=False)

    # ---- 3. Budget efficiency: first part of the budget vs the rest ----
    eff = []
    for m in methods:
        for a, b in [(cps[0], plateau), (plateau, final)]:
            row = {"method": LABELS[m], "from": a, "to": b}
            for metric in ["inner_ba", "outer_ba", "gap", "seconds"]:
                diff = wide(metric, b)[m] - wide(metric, a)[m]
                row[f"delta_{metric}"] = diff.mean()
                if metric == "outer_ba":
                    row["delta_outer_ci_low"], row["delta_outer_ci_high"] = bootstrap_ci(diff)
            eff.append(row)
    pd.DataFrame(eff).round(4).to_csv(TABLES_DIR / f"table5_budget_efficiency{suffix}.csv", index=False)

    # ---- 4. Peak analysis ----
    peak = []
    for (d, m), sub in ds.groupby(["dataset", "method"]):
        sub = sub.sort_values("checkpoint")
        best = sub.loc[sub["outer_ba"].idxmax()]
        peak.append({"dataset": d, "method": m, "peak_budget": int(best["checkpoint"]),
                     "peak_test_ba": best["outer_ba"],
                     "final_test_ba": sub.loc[sub["checkpoint"] == final, "outer_ba"].iloc[0]})
    peak = pd.DataFrame(peak)
    peak["drop_from_peak"] = peak["peak_test_ba"] - peak["final_test_ba"]
    peak.round(4).to_csv(TABLES_DIR / f"table6_peaks{suffix}.csv", index=False)
    peak_summary = peak.groupby("method").agg(
        # counted only when the final test score is actually LOWER than the peak
        # (a method that stopped early, like SFS, has a flat curve and is not counted)
        datasets_lower_at_final=("drop_from_peak", lambda s: int((s > 1e-9).sum())),
        mean_drop_from_peak=("drop_from_peak", "mean"),
        max_drop_from_peak=("drop_from_peak", "max")).reset_index()
    peak_summary["method"] = peak_summary["method"].map(LABELS)
    peak_summary.round(4).to_csv(TABLES_DIR / f"table6_peak_summary{suffix}.csv", index=False)

    # ---- 5a. Dataset table for the paper ----
    audit_path = TABLES_DIR / "data_audit.csv"
    if audit_path.exists():
        a = pd.read_csv(audit_path)
        t = pd.DataFrame({"Dataset": a["dataset"], "Domain": a["dataset"].map(DOMAINS),
                          "Instances (raw)": a["rows_raw"], "Duplicates removed": a["exact_duplicates_removed"],
                          "Instances (used)": a["rows_clean"], "Features": a["features_clean"],
                          "Classes": a["classes"], "Minority class (%)": a["minority_class_%"]})
        t["n/p (training)"] = (0.7 * t["Instances (used)"] / t["Features"]).round(1)
        t.sort_values("Features").to_csv(TABLES_DIR / "table0_datasets.csv", index=False)

    # ---- 5b. Paper figures ----
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    def budget_axis(ax):
        """Log x-axis showing only the checkpoint values (no stray minor labels)."""
        ax.set_xscale("log")
        ax.set_xticks(cps, [str(c) for c in cps])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.tick_params(axis="x", which="minor", length=0)

    def save(fig, name):
        for ext in ("pdf", "png"):
            fig.savefig(FIGURES_DIR / f"{name}{suffix}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Figure: optimism gap with CIs
    t2 = pd.read_csv(TABLES_DIR / f"table2_budget_curves{suffix}.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for m in methods:
        s = t2[t2["method"] == LABELS[m]]
        ax.plot(s["budget"], s["gap"], "-o", ms=3.5, color=COLORS[m], label=LABELS[m])
        ax.fill_between(s["budget"], s["gap_ci_low"], s["gap_ci_high"], color=COLORS[m], alpha=0.12, lw=0)
    ax.axhline(0, color="black", lw=0.7)
    budget_axis(ax)
    ax.set_xlabel("Fitness evaluations (log scale)")
    ax.set_ylabel("Inner-CV minus test balanced accuracy")
    ax.legend(frameon=False)
    save(fig, "paper_fig_gap")

    # Figure: four example datasets, test (solid) and inner CV (dashed)
    ex = [d for d in EXAMPLES if d in datasets]
    if ex:
        fig, axes = plt.subplots(1, len(ex), figsize=(3.0 * len(ex), 2.8), squeeze=False)
        for ax, d in zip(axes[0], ex):
            for m in methods:
                s = ds[(ds["dataset"] == d) & (ds["method"] == m)].sort_values("checkpoint")
                ax.plot(s["checkpoint"], s["outer_ba"], "-o", ms=3, color=COLORS[m], label=f"{LABELS[m]}")
                ax.plot(s["checkpoint"], s["inner_ba"], "--", lw=1, color=COLORS[m], alpha=0.7)
            budget_axis(ax)
            ax.tick_params(axis="x", labelsize=7, rotation=45)
            ax.set_title(d.replace("_", " ").title())
            ax.set_xlabel("Fitness evaluations")
        axes[0][0].set_ylabel("Balanced accuracy")
        handles, labels = axes[0][0].get_legend_handles_labels()
        handles.append(plt.Line2D([], [], color="black", ls="--", lw=1))
        labels.append("Inner CV (dashed)")
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
                   bbox_to_anchor=(0.5, -0.08))
        fig.tight_layout()
        save(fig, "paper_fig_examples")

    # Figure: noise share on waveform
    noise_path = TABLES_DIR / f"table4_noise_selection{suffix}.csv"
    if noise_path.exists():
        nz = pd.read_csv(noise_path)
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        for m in methods:
            s = nz[nz["method"] == m].sort_values("checkpoint")
            ax.plot(s["checkpoint"], 100 * s["noise_share"], "-o", ms=3.5, color=COLORS[m], label=LABELS[m])
        ax.axhline(100 * 19 / 40, color="black", ls=":", lw=0.8)
        ax.text(cps[0], 100 * 19 / 40 + 1, "all 40 features (47.5% noise)", fontsize=7)
        budget_axis(ax)
        ax.set_ylim(-2, 52)
        ax.set_xlabel("Fitness evaluations (log scale)")
        ax.set_ylabel("Selected features that are noise (%)")
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
        save(fig, "paper_fig_noise")

    pd.set_option("display.width", 220)
    print("\n===== EQUIVALENCE (TOST, margin 0.01) =====")
    print(eq.round(4).to_string(index=False))
    print("\n===== SENSITIVITY =====")
    print(sens.round(4).to_string(index=False))
    print("\n===== BUDGET EFFICIENCY =====")
    print(pd.DataFrame(eff).round(4).to_string(index=False))
    print("\n===== PEAKS =====")
    print(peak_summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()