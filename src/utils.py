"""
utils.py
Small shared helpers: paths, config, seeds, data loading and the inner folds.

Nothing in this file looks at test data. The only data functions here either
load the cleaned CSV or work on the TRAINING part of an outer split.
"""

import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Project folders (this file lives in src/, so the project root is one level up)
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "datasets"
CLEAN_DIR = ROOT / "datasets" / "clean"
DUPS_DIR = ROOT / "datasets" / "with_duplicates"   # revision check: same cleaning, duplicates kept
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
STATS_DIR = RESULTS_DIR / "stats"
CONFIG_PATH = ROOT / "configs" / "experiment.yaml"


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Basic sanity checks so a typo in the config cannot silently change the study
    assert cfg["checkpoints"] == sorted(cfg["checkpoints"]), "checkpoints must be increasing"
    assert cfg["checkpoints"][-1] == cfg["max_evaluations"], "last checkpoint must equal max_evaluations"
    assert 0 < cfg["test_size"] < 1
    assert 0 < cfg["alpha"] <= 1
    return cfg


def make_classifier(cfg, n_train=None):
    """The classifier that scores every subset. Same settings everywhere in a run."""
    name = cfg.get("classifier", "knn")
    if name == "knn":
        k = cfg["knn_neighbors"] if n_train is None else min(cfg["knn_neighbors"], n_train)
        return KNeighborsClassifier(n_neighbors=k)
    if name == "logreg":
        return LogisticRegression(C=1.0, max_iter=2000)
    raise ValueError(f"unknown classifier {name}")


def stable_seed(*parts):
    """Turn any mix of strings and numbers into a fixed 32-bit seed.
    Python's built-in hash() changes between runs, so we use crc32 instead."""
    text = "|".join(str(p) for p in parts)
    return zlib.crc32(text.encode("utf-8"))


def load_clean(name, folder=None):
    """Load a cleaned dataset. Returns X (float array), y (int array), column names.
    folder=DUPS_DIR loads the version that keeps duplicate rows (revision check)."""
    df = pd.read_csv((folder or CLEAN_DIR) / f"{name}.csv")
    y = df["target"].to_numpy(dtype=int)
    X_df = df.drop(columns="target")
    return X_df.to_numpy(dtype=float), y, list(X_df.columns)


def outer_split(X, y, repeat, cfg):
    """One stratified train/test split. The same repeat number always gives the
    same split, and every method sees exactly the same split."""
    seed = stable_seed(cfg["seed"], "outer", repeat)
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=cfg["test_size"], stratify=y, random_state=seed % (2**31)
    )
    return np.sort(train_idx), np.sort(test_idx)


def inner_folds(y_train, repeat, cfg):
    """Stratified K folds built from the TRAINING labels only.
    All methods reuse the same folds for the same outer split (a fair comparison)."""
    seed = stable_seed(cfg["seed"], "inner", repeat)
    skf = StratifiedKFold(n_splits=cfg["inner_folds"], shuffle=True, random_state=seed % (2**31))
    return list(skf.split(np.zeros(len(y_train)), y_train))


class ScaledFolds:
    """Pre-scaled copies of every inner fold.

    For each fold the scaler is fitted on that fold's training rows only, then
    applied to its validation rows. Standard scaling treats each column on its
    own, so scaling all columns once and then picking a subset of columns gives
    exactly the same numbers as scaling the subset directly. Doing it once here
    saves a lot of time and keeps every fold leak-free.
    """

    def __init__(self, X_train, y_train, folds):
        self.parts = []
        for tr, va in folds:
            scaler = StandardScaler().fit(X_train[tr])
            self.parts.append((
                scaler.transform(X_train[tr]), y_train[tr],
                scaler.transform(X_train[va]), y_train[va],
            ))