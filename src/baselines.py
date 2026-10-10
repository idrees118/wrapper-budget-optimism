"""
baselines.py
Three reference methods that do NOT search over subsets:

  all_features : kNN on every feature
  pca          : PCA, keeping enough components to explain a share of the variance
  mi_filter    : keep the top-k features ranked by mutual information with the label

PCA and the MI filter have one setting each. It is chosen with the SAME inner folds
the search methods use, through a scikit-learn Pipeline, so the scaler, PCA and the
MI ranking are always fitted on inner-training rows only. The chosen setting is then
refitted on the whole outer-training part and scored once on the outer-test part.
"""

from functools import partial

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _pipeline(step, cfg):
    steps = [("scale", StandardScaler())]
    if step is not None:
        steps.append(step)
    steps.append(("knn", KNeighborsClassifier(n_neighbors=cfg["knn_neighbors"])))
    return Pipeline(steps)


def run_baseline(name, X_tr, y_tr, X_te, y_te, folds, cfg, seed):
    """Returns one result dict for one baseline on one outer split."""
    D = X_tr.shape[1]

    if name == "all_features":
        model = _pipeline(None, cfg)
        grid = {}
    elif name == "pca":
        model = _pipeline(("pca", PCA(svd_solver="full")), cfg)
        grid = {"pca__n_components": list(cfg["pca_variance_grid"])}
    elif name == "mi_filter":
        score_fn = partial(mutual_info_classif, random_state=seed % (2**31))
        model = _pipeline(("select", SelectKBest(score_func=score_fn)), cfg)
        ks = sorted({max(1, int(round(f * D))) for f in cfg["mi_fraction_grid"]})
        grid = {"select__k": ks}
    else:
        raise ValueError(f"unknown baseline {name}")

    # GridSearchCV with an empty grid is just plain cross-validation, which is what we want
    # for all_features: it gives the inner score on the same folds as everything else.
    search = GridSearchCV(model, param_grid=grid or [{}], cv=folds,
                          scoring="balanced_accuracy", refit=True, n_jobs=1)
    search.fit(X_tr, y_tr)
    outer_ba = balanced_accuracy_score(y_te, search.predict(X_te))

    best = search.best_estimator_
    if name == "pca":
        n_used = int(best.named_steps["pca"].n_components_)
    elif name == "mi_filter":
        n_used = int(best.named_steps["select"].k)
    else:
        n_used = D

    return {
        "inner_ba": float(search.best_score_),
        "outer_ba": float(outer_ba),
        "n_features": n_used,
        "setting": str(search.best_params_) if grid else "",
    }