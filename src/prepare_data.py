"""
prepare_data.py
Step 1 of the study. Run once, from the project root:

    python src/prepare_data.py

What it does, for each of the 12 datasets:
  1. reads the raw files from datasets/ (sonar and waveform are downloaded from OpenML)
  2. audits the raw data (duplicates, conflicting labels, constant columns, missing values)
  3. removes exact duplicate rows (same features AND same label), keeping the first copy.
     A duplicate that lands in both the training and the test part would let the model
     "remember" test rows, which inflates the held-out score.
  4. removes columns that have one single value in the whole file (they carry no
     information and break scaling). This uses no labels, so it cannot leak anything.
  5. saves datasets/clean/<name>.csv and results/tables/data_audit.csv

Columns are named by their ORIGINAL position (f0, f1, ...), even after a constant
column is dropped, so the known noise columns of waveform keep their numbers.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import CLEAN_DIR, DUPS_DIR, RAW_DIR, TABLES_DIR  # noqa: E402


# ---------- raw loaders: each returns (features DataFrame, labels Series) ----------

def raw_hill_valley():
    a = pd.read_csv(RAW_DIR / "hill+valley/Hill_Valley_with_noise_Training.data")
    b = pd.read_csv(RAW_DIR / "hill+valley/Hill_Valley_with_noise_Testing.data")
    df = pd.concat([a, b], ignore_index=True)
    return df.iloc[:, :-1], df.iloc[:, -1]


def raw_ionosphere():
    df = pd.read_csv(RAW_DIR / "ionosphere/ionosphere.data", header=None)
    return df.iloc[:, :-1], df.iloc[:, -1]


def raw_libras():
    df = pd.read_csv(RAW_DIR / "libras+movement/movement_libras.data", header=None)
    return df.iloc[:, :-1], df.iloc[:, -1]


def raw_madelon():
    d = RAW_DIR / "madelon/MADELON"
    Xa = pd.read_csv(d / "madelon_train.data", sep=r"\s+", header=None)
    Xb = pd.read_csv(d / "madelon_valid.data", sep=r"\s+", header=None)
    ya = pd.read_csv(d / "madelon_train.labels", header=None)[0]
    yb = pd.read_csv(d / "madelon_valid.labels", header=None)[0]
    assert len(Xa) == len(ya) and len(Xb) == len(yb), "madelon rows and labels do not match"
    return pd.concat([Xa, Xb], ignore_index=True), pd.concat([ya, yb], ignore_index=True)


def raw_qsar():
    df = pd.read_csv(RAW_DIR / "qsar+biodegradation/biodeg.csv", sep=";", header=None)
    return df.iloc[:, :-1], df.iloc[:, -1]


def raw_semeion():
    df = pd.read_csv(RAW_DIR / "semeion+handwritten+digit/semeion.data", sep=r"\s+", header=None)
    X, onehot = df.iloc[:, :256], df.iloc[:, 256:266]
    assert (onehot.sum(axis=1) == 1).all(), "semeion: each row must have exactly one digit label"
    return X, pd.Series(onehot.to_numpy().argmax(axis=1))


def raw_spambase():
    df = pd.read_csv(RAW_DIR / "spambase/spambase.data", header=None)
    return df.iloc[:, :-1], df.iloc[:, -1]


def raw_german():
    df = pd.read_csv(RAW_DIR / "statlog+german+credit+data/german.data-numeric", sep=r"\s+", header=None)
    return df.iloc[:, :-1], df.iloc[:, -1]


def raw_vehicle():
    df = pd.read_csv(RAW_DIR / "statlog+vehicle+silhouettes/vehicle.dat", sep=r"\s+", header=None)
    return df.iloc[:, :-1], df.iloc[:, -1]


def raw_wine_red():
    df = pd.read_csv(RAW_DIR / "wine+quality/winequality-red.csv", sep=";")
    # Binary task used widely in the literature: "good" wine means quality >= 7
    return df.drop(columns="quality"), (df["quality"] >= 7).astype(int)


def raw_openml(name, folder):
    """Download once from OpenML and keep a raw copy in datasets/<folder>/."""
    path = RAW_DIR / folder / f"{name}.csv"
    if not path.exists():
        from sklearn.datasets import fetch_openml
        bunch = fetch_openml(name=name, version=1, as_frame=True, parser="auto")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = bunch.data.copy()
        frame["class"] = bunch.target
        frame.to_csv(path, index=False)
    df = pd.read_csv(path)
    return df.drop(columns="class"), df["class"]


LOADERS = {
    "wine_red": raw_wine_red,
    "vehicle": raw_vehicle,
    "german_credit": raw_german,
    "ionosphere": raw_ionosphere,
    "waveform": lambda: raw_openml("waveform-5000", "waveform"),
    "qsar_biodeg": raw_qsar,
    "spambase": raw_spambase,
    "sonar": lambda: raw_openml("sonar", "sonar"),
    "libras": raw_libras,
    "hill_valley": raw_hill_valley,
    "semeion": raw_semeion,
    "madelon": raw_madelon,
}


# ---------- audit and cleaning ----------

def clean_one(name):
    X, y = LOADERS[name]()
    X = X.apply(pd.to_numeric, errors="raise").reset_index(drop=True).astype(float)
    X.columns = [f"f{i}" for i in range(X.shape[1])]           # original positions
    # Labels -> 0..K-1 (sorted, so the mapping is reproducible)
    y = pd.Series(pd.factorize(y.astype(str).str.strip(), sort=True)[0], name="target")

    df = pd.concat([X, y], axis=1)
    n_raw = len(df)

    # Audit on the RAW data
    missing = int(X.isna().sum().sum())
    exact_dup = int(df.duplicated().sum())
    same_x = X.duplicated(keep=False).to_numpy()
    conflict_groups = 0
    if same_x.any():
        g = df.loc[same_x].groupby(list(X.columns))["target"].nunique()
        conflict_groups = int((g > 1).sum())
    constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]

    if missing:
        raise ValueError(f"{name}: {missing} missing values found; decide how to handle them before running")

    # Revision check: the same cleaning WITHOUT removing duplicates, saved separately.
    # The main study never reads this folder.
    DUPS_DIR.mkdir(parents=True, exist_ok=True)
    df.drop(columns=constant_cols).to_csv(DUPS_DIR / f"{name}.csv", index=False)

    # Cleaning
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    df = df.drop(columns=constant_cols)

    counts = df["target"].value_counts()
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DIR / f"{name}.csv", index=False)

    return {
        "dataset": name,
        "rows_raw": n_raw,
        "exact_duplicates_removed": exact_dup,
        "duplicate_%": round(100 * exact_dup / n_raw, 2),
        "conflicting_label_groups": conflict_groups,
        "constant_columns_removed": len(constant_cols),
        "rows_clean": len(df),
        "features_clean": df.shape[1] - 1,
        "classes": int(counts.size),
        "minority_class_%": round(100 * counts.min() / len(df), 1),
    }


def main():
    rows = []
    for name in LOADERS:
        try:
            info = clean_one(name)
        except Exception as e:  # report and keep going so one bad file does not hide the rest
            print(f"[FAILED] {name}: {e}")
            continue
        rows.append(info)
        print(f"[OK] {name}: {info['rows_clean']} rows x {info['features_clean']} features "
              f"({info['exact_duplicates_removed']} duplicates removed)")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame(rows)
    audit.to_csv(TABLES_DIR / "data_audit.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n" + audit.to_string(index=False))
    if len(rows) != len(LOADERS):
        sys.exit("Some datasets failed. Fix them before running the experiments.")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()