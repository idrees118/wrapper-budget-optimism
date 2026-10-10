"""
analysis.py
Step 3 of the study. From the project root:

    python src/analysis.py            # analyses results/runs/
    python src/analysis.py --quick    # analyses results/runs_quick/ (smoke test)
    python src/analysis.py --tag logreg   # analyses results/runs_logreg/ (robustness check)

Statistical rules used throughout (so the numbers hold up in review):
  * The unit of analysis is the DATASET. The outer repeats of one dataset are
    averaged first, because repeats of the same dataset are not independent
    (Demsar, 2006, JMLR). With 12 datasets every test has n = 12.
  * Two related samples: Wilcoxon signed-rank test, effect size = matched-pairs
    rank-biserial correlation (from -1 to +1).
  * More than two methods: Friedman test with Kendall's W, followed by pairwise
    Wilcoxon tests. Following Benavoli et al. (2016, JMLR) we do not use the
    Nemenyi test, whose result depends on which other methods are in the comparison.
  * Every family of tests is corrected with Holm's method; both raw and adjusted
    p-values are saved.
  * Means across datasets come with 95% bootstrap confidence intervals
    (10,000 resamples of datasets, fixed seed).
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import FIGURES_DIR, RESULTS_DIR, STATS_DIR, TABLES_DIR, load_config  # noqa: E402

SATURATION_TOLERANCE = 0.01   # "stopped helping" = within 1 balanced-accuracy point of the best checkpoint
LABELS = {"ga": "GA", "pso": "BPSO", "random": "Random search", "sfs": "SFS",
          "all_features": "All features", "pca": "PCA", "mi_filter": "MI filter"}


# ---------------- small statistical helpers ----------------

def wilcoxon(a, b):
    """Paired two-sided Wilcoxon test of a vs b. Returns (p, rank-biserial r, median difference)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    nz = d[d != 0]
    if nz.size == 0:
        return np.nan, 0.0, 0.0
    ranks = stats.rankdata(np.abs(nz))
    r = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()
    p = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue
    return float(p), float(r), float(np.median(d))


def holm(pvals):
    """Holm step-down adjustment. NaN p-values are left as NaN and not counted."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    order = np.argsort(p[ok])
    m = ok.sum()
    adj = np.maximum.accumulate((m - np.arange(m)) * p[ok][order])
    tmp = np.empty(m)
    tmp[order] = np.minimum(adj, 1.0)
    out[ok] = tmp
    return out


def bootstrap_ci(values, n_boot=10_000, seed=0):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if v.size < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------- loading ----------------

def load_runs(run_dir, cfg, quick):
    files = sorted(run_dir.glob("*.csv"))
    if not files:
        sys.exit(f"No result files in {run_dir}. Run run_experiments.py first.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    repeats = 2 if quick else cfg["outer_repeats"]
    expected = len(df["dataset"].unique()) * repeats * (len(cfg["search_methods"]) + 1)
    if len(files) < expected:
        print(f"WARNING: only {len(files)} of {expected} task files found; results are incomplete.")
    return df


# ---------------- main analysis ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="", help="e.g. logreg -> reads results/runs_logreg/")
    args = ap.parse_args()
    cfg = load_config()
    if args.quick:
        cfg["checkpoints"] = [50, 100, 200]
    suffix = (f"_{args.tag}" if args.tag else "") + ("_quick" if args.quick else "")
    run_dir = RESULTS_DIR / f"runs{suffix}"
    for d in (TABLES_DIR, STATS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    df = load_runs(run_dir, cfg, args.quick)
    search = df[df["method"].isin(cfg["search_methods"])].copy()
    base = df[df["method"].isin(cfg["baselines"])].copy()
    cps = list(cfg["checkpoints"])
    final_cp = cps[-1]
    methods = [m for m in cfg["search_methods"] if m in search["method"].unique()]
    datasets = sorted(df["dataset"].unique())
    n_ds = len(datasets)
    if n_ds < 6:
        print(f"NOTE: only {n_ds} datasets; p-values with so few datasets are not meaningful.")

    # Step 1: average the repeats inside each dataset (the dataset is the unit of analysis)
    keys = ["dataset", "method", "checkpoint"]
    ds_mean = search.groupby(keys, as_index=False)[
        ["inner_ba", "outer_ba", "gap", "n_features", "n_noise", "seconds", "evals_used"]].mean()
    base_mean = base.groupby(["dataset", "method"], as_index=False)[
        ["inner_ba", "outer_ba", "gap", "n_features", "seconds"]].mean()
    meta = df.groupby("dataset")[["n_train", "p"]].first()

    ds_mean.to_csv(TABLES_DIR / f"dataset_means_search{suffix}.csv", index=False)
    base_mean.to_csv(TABLES_DIR / f"dataset_means_baselines{suffix}.csv", index=False)

    def wide(metric, cp):
        """datasets x methods table of one metric at one checkpoint."""
        t = ds_mean[ds_mean["checkpoint"] == cp].pivot(index="dataset", columns="method", values=metric)
        return t.reindex(index=datasets, columns=methods)

    # ---- Table 1: final outer balanced accuracy, mean +- sd over repeats ----
    fin = search[search["checkpoint"] == final_cp]
    both = pd.concat([fin, base])
    t1 = both.groupby(["dataset", "method"])["outer_ba"].agg(["mean", "std"]).reset_index()
    t1["value"] = t1.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
    t1 = t1.pivot(index="dataset", columns="method", values="value")
    t1 = t1[[c for c in methods + cfg["baselines"] if c in t1.columns]].rename(columns=LABELS)
    t1.to_csv(TABLES_DIR / f"table1_final_outer_ba{suffix}.csv")

    # ---- Table 2: inner score, outer score and gap at each budget (mean over datasets + CI) ----
    rows = []
    for m in methods:
        for cp in cps:
            sub = ds_mean[(ds_mean["method"] == m) & (ds_mean["checkpoint"] == cp)]
            lo, hi = bootstrap_ci(sub["gap"])
            rows.append({"method": LABELS[m], "budget": cp,
                         "inner_ba": sub["inner_ba"].mean(), "outer_ba": sub["outer_ba"].mean(),
                         "gap": sub["gap"].mean(), "gap_ci_low": lo, "gap_ci_high": hi,
                         "n_features": sub["n_features"].mean(), "seconds": sub["seconds"].mean()})
    pd.DataFrame(rows).round(4).to_csv(TABLES_DIR / f"table2_budget_curves{suffix}.csv", index=False)

    # ---- RQ1: does more search make the inner score more optimistic, and does outer score improve? ----
    rq1 = []
    for m in methods:
        g_first, g_last = wide("gap", cps[0])[m], wide("gap", final_cp)[m]
        p, r, md = wilcoxon(g_last, g_first)
        rq1.append({"method": LABELS[m], "test": f"gap at {final_cp} vs gap at {cps[0]}",
                    "median_difference": md, "rank_biserial": r, "p_raw": p})
        for cp in cps[:-1]:
            p, r, md = wilcoxon(wide("outer_ba", final_cp)[m], wide("outer_ba", cp)[m])
            rq1.append({"method": LABELS[m], "test": f"outer BA at {final_cp} vs {cp}",
                        "median_difference": md, "rank_biserial": r, "p_raw": p})
    rq1 = pd.DataFrame(rq1)
    rq1["p_holm"] = holm(rq1["p_raw"])
    rq1.round(5).to_csv(STATS_DIR / f"rq1_overfitting{suffix}.csv", index=False)

    # ---- RQ2: do the search methods differ at the same budget? and against the baselines? ----
    rq2, fried = [], []
    for cp in cps:
        w = wide("outer_ba", cp).dropna()
        if len(methods) >= 3 and len(w) >= 2:
            chi2, p = stats.friedmanchisquare(*[w[m] for m in methods])
            kendall_w = chi2 / (len(w) * (len(methods) - 1))
            fried.append({"budget": cp, "n_datasets": len(w), "chi2": chi2,
                          "kendall_W": kendall_w, "p_raw": p,
                          **{f"mean_rank_{LABELS[m]}": v for m, v in
                             w.rank(axis=1, ascending=False).mean().items()}})
        for i, a in enumerate(methods):
            for b in methods[i + 1:]:
                p, r, md = wilcoxon(w[a], w[b])
                rq2.append({"budget": cp, "comparison": f"{LABELS[a]} vs {LABELS[b]}",
                            "median_difference": md, "rank_biserial": r, "p_raw": p})
    fried = pd.DataFrame(fried)
    if len(fried):
        fried["p_holm"] = holm(fried["p_raw"])
    fried.round(5).to_csv(STATS_DIR / f"rq2_friedman{suffix}.csv", index=False)
    rq2 = pd.DataFrame(rq2)
    rq2["p_holm"] = np.nan
    for cp in cps:                                     # Holm within each budget
        idx = rq2["budget"] == cp
        rq2.loc[idx, "p_holm"] = holm(rq2.loc[idx, "p_raw"])
    rq2.round(5).to_csv(STATS_DIR / f"rq2_pairwise{suffix}.csv", index=False)

    bw = base_mean.pivot(index="dataset", columns="method", values="outer_ba").reindex(datasets)
    vs = []
    for m in methods:
        for b in cfg["baselines"]:
            if b in bw:
                p, r, md = wilcoxon(wide("outer_ba", final_cp)[m], bw[b])
                vs.append({"comparison": f"{LABELS[m]} ({final_cp} evals) vs {LABELS[b]}",
                           "median_difference": md, "rank_biserial": r, "p_raw": p})
    vs = pd.DataFrame(vs)
    vs["p_holm"] = holm(vs["p_raw"])
    vs.round(5).to_csv(STATS_DIR / f"rq2_search_vs_baselines{suffix}.csv", index=False)

    # ---- RQ3: when does extra search stop helping, and does that depend on n/p? ----
    sat = []
    for (d, m), sub in ds_mean.groupby(["dataset", "method"]):
        sub = sub.sort_values("checkpoint")
        best = sub["outer_ba"].max()
        first_ok = sub.loc[sub["outer_ba"] >= best - SATURATION_TOLERANCE, "checkpoint"].min()
        sat.append({"dataset": d, "method": m, "saturation_budget": first_ok,
                    "final_gap": sub.loc[sub["checkpoint"] == final_cp, "gap"].iloc[0],
                    "log_n_over_p": np.log(meta.loc[d, "n_train"] / meta.loc[d, "p"])})
    sat = pd.DataFrame(sat)
    sat.to_csv(TABLES_DIR / f"table3_saturation{suffix}.csv", index=False)
    corr = []
    for m in methods:
        s = sat[sat["method"] == m]
        for target in ["saturation_budget", "final_gap"]:
            rho, p = stats.spearmanr(s["log_n_over_p"], s[target]) if len(s) >= 3 else (np.nan, np.nan)
            corr.append({"method": LABELS[m], "variable": target, "spearman_rho_vs_log_n_over_p": rho, "p_raw": p})
    corr = pd.DataFrame(corr)
    corr["p_holm"] = holm(corr["p_raw"])
    corr.round(5).to_csv(STATS_DIR / f"rq3_correlations{suffix}.csv", index=False)

    # ---- Known-noise analysis (waveform) ----
    noise = ds_mean.dropna(subset=["n_noise"]).copy()
    if len(noise):
        noise["noise_share"] = noise["n_noise"] / noise["n_features"]
        noise.to_csv(TABLES_DIR / f"table4_noise_selection{suffix}.csv", index=False)

    # ---- Figures ----
    colors = dict(zip(methods, ["#1f77b4", "#d62728", "#7f7f7f", "#2ca02c"]))
    ncol = 4
    nrow = int(np.ceil(n_ds / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow), squeeze=False)
    for ax, d in zip(axes.flat, datasets):
        for m in methods:
            s = ds_mean[(ds_mean["dataset"] == d) & (ds_mean["method"] == m)].sort_values("checkpoint")
            ax.plot(s["checkpoint"], s["outer_ba"], "-o", ms=3, color=colors[m], label=f"{LABELS[m]} (test)")
            ax.plot(s["checkpoint"], s["inner_ba"], "--", color=colors[m], alpha=0.6, label=f"{LABELS[m]} (inner CV)")
        ax.set_xscale("log")
        ax.set_title(d, fontsize=10)
        ax.set_xlabel("fitness evaluations")
        ax.set_ylabel("balanced accuracy")
    for ax in axes.flat[n_ds:]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIGURES_DIR / f"fig1_curves_per_dataset{suffix}.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    t2 = pd.read_csv(TABLES_DIR / f"table2_budget_curves{suffix}.csv")
    for m in methods:
        s = t2[t2["method"] == LABELS[m]]
        ax.plot(s["budget"], s["gap"], "-o", ms=3, color=colors[m], label=LABELS[m])
        ax.fill_between(s["budget"], s["gap_ci_low"], s["gap_ci_high"], color=colors[m], alpha=0.15)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("fitness evaluations")
    ax.set_ylabel("inner CV − test balanced accuracy")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"fig2_optimism_gap{suffix}.png", dpi=200)
    plt.close(fig)

    if len(noise):
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        for m in methods:
            s = noise[noise["method"] == m].sort_values("checkpoint")
            ax.plot(s["checkpoint"], s["noise_share"], "-o", ms=3, color=colors[m], label=LABELS[m])
        ax.set_xscale("log")
        ax.set_xlabel("fitness evaluations")
        ax.set_ylabel("share of selected features that are noise")
        ax.set_title("Waveform (19 of 40 features are pure noise)", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"fig3_noise_selection{suffix}.png", dpi=200)
        plt.close(fig)

    print(f"Done. {n_ds} datasets, {len(methods)} search methods.")
    print(f"Tables -> {TABLES_DIR}\nStatistics -> {STATS_DIR}\nFigures -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()