import argparse
import copy
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from utils import (RESULTS_DIR, inner_folds, load_clean, load_config,
                   make_classifier, outer_split)


def fit_score(cfg, Xa, ya, Xb, yb):
    sc = StandardScaler().fit(Xa)
    m = make_classifier(cfg, len(ya)).fit(sc.transform(Xa), ya)
    return balanced_accuracy_score(yb, m.predict(sc.transform(Xb)))


def one_file(path, cfg, out_dir):
    out = out_dir / path.name
    if out.exists():
        return "skipped"
    dataset, _, rep = path.stem.split("__")
    rep = int(rep[1:])
    df = pd.read_csv(path)
    if df["selected"].isna().all():          # baselines have no subset
        return "no subset"
    X, y, cols = load_clean(dataset)
    tr, te = outer_split(X, y, rep, cfg)
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    folds = inner_folds(ytr, rep, cfg)
    pos = {c: i for i, c in enumerate(cols)}
    cache, rows = {}, []
    for _, r in df.iterrows():
        idx = np.array(sorted(pos[c] for c in str(r["selected"]).split(";")))
        k = tuple(idx)
        if k not in cache:
            inner = np.mean([fit_score(cfg, Xtr[a][:, idx], ytr[a],
                                       Xtr[b][:, idx], ytr[b]) for a, b in folds])
            full = fit_score(cfg, Xtr[:, idx], ytr, Xte[:, idx], yte)
            # same 80% training sets the search used, scored on the test rows
            matched = np.mean([fit_score(cfg, Xtr[a][:, idx], ytr[a],
                                         Xte[:, idx], yte) for a, _ in folds])
            cache[k] = (inner, full, matched)
        inner, full, matched = cache[k]

        rows.append(dict(dataset=dataset, repeat=rep, method=r["method"],
            checkpoint=r["checkpoint"], inner=inner, test_full=full,
                test_matched=matched,
                d_inner=inner - r["inner_ba"], d_full=full - r["outer_ba"]))


    res = pd.DataFrame(rows)
    tmp = out.with_suffix(".tmp")
    res.to_csv(tmp, index=False)
    tmp.replace(out)
    return "done"


def run(a):
    cfg = load_config(a.config) if a.config else load_config()
    if a.classifier == "logreg":
        cfg = copy.deepcopy(cfg)
        cfg["classifier"] = "logreg"
    src = RESULTS_DIR / a.runs
    out_dir = RESULTS_DIR / "matched" / a.runs
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.glob("*.csv") if "__baselines__" not in p.name)
    if a.datasets:
        files = [p for p in files if p.name.split("__")[0] in a.datasets]
    res = Parallel(n_jobs=a.n_jobs)(delayed(one_file)(p, cfg, out_dir) for p in files)
    print({s: res.count(s) for s in set(res)})


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj, run = np.empty_like(p), 0.0
    for rank, i in enumerate(o):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def summary(a):
    d = RESULTS_DIR / "matched" / a.runs
    df = pd.concat(pd.read_csv(f) for f in d.glob("*.csv"))
    df["inner"] = df.inner - df.d_inner          # the search's own stored score
    df["gap_total"] = df.inner - df.test_full
    df["gap_matched"] = df.inner - df.test_matched
    df["size_effect"] = df.test_full - df.test_matched
    ds = (df.groupby(["method", "checkpoint", "dataset"])
            [["gap_total", "gap_matched", "size_effect"]].mean().reset_index())
    print(ds.groupby(["method", "checkpoint"])
            [["gap_total", "gap_matched", "size_effect"]].mean().round(3))
    names, ps = [], []
    for m, g in ds.groupby("method"):
        hi = g[g.checkpoint == 4000].set_index("dataset").gap_matched
        lo = g[g.checkpoint == 100].set_index("dataset").gap_matched
        diff = (hi - lo).dropna()
        names.append(m)
        ps.append(wilcoxon(diff).pvalue)
        print(f"{m}: matched gap growth median {diff.median():.3f}, "
              f"grew on {(diff > 0).sum()}/{len(diff)}, raw p {ps[-1]:.4f}")
    print("Holm:", dict(zip(names, holm(ps).round(4))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs", help="folder name under results/")
    ap.add_argument("--classifier", choices=["knn", "logreg"], default="knn")
    ap.add_argument("--config")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--n_jobs", type=int, default=4)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    summary(a) if a.summary else run(a)