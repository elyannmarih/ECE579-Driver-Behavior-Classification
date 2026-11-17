from scipy.stats import randint

import numpy as np
import pandas as pd
import ast
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, RandomizedSearchCV

# configuration
DATA_DIR = Path("data")
TRAIN_DRIVERS = ["D1", "D2", "D3", "D4", "D5"]
CLASSES = ["NORMAL", "AGGRESSIVE", "DROWSY"]

SIGNALS = ["ACC_X", "ACC_Y", "ACC_Z", 
           "ACC_X_KF", "ACC_Y_KF", "ACC_Z_KF",
           "ROLL", "PITCH", "YAW"]

STAT_NAMES = ["MEAN", "STD", "MIN", "MAX", "PEAK_ABS"]
FEATURE_NAMES = [F"{signal}_{stat}" for stat in STAT_NAMES for signal in SIGNALS]

# finds driver files
def find_files(driver, root=DATA_DIR):
    hits = sorted(Path(root).rglob(f"{driver}traindata50.csv"))
    return hits[0]

# takes window samples
def parse_window(cell):
    lines = ast.literal_eval(cell)
    return np.array([ln.split() for ln in lines], dtype=float)

# loads driver csv files
def load_driver(path, driver):
    df = pd.read_csv(path)
    W = np.stack([parse_window(cell) for cell in df["data"]])
    labels = df["label"].str.strip().str.upper().to_numpy()
    drivers = np.full(len(labels), driver)
    return W, labels, drivers

def extract_features(w):
    feats = [
        w.mean(axis=1),
        w.std(axis=1),
        w.min(axis=1),
        w.max(axis=1),
        np.abs(w).max(axis=1) 
    ]
    return pd.DataFrame(np.concatenate(feats, axis=1), columns=FEATURE_NAMES)

def lodo_eval(X, y, groups, params):
    rows = []
    total_cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for test_driver in TRAIN_DRIVERS:
        train_mask = groups != test_driver
        test_mask = groups == test_driver

        model = RandomForestClassifier(random_state=42, n_jobs=1, **params)
        model.fit(X[train_mask], y[train_mask])
        pred = model.predict(X[test_mask])
    
        cm = confusion_matrix(y[test_mask], pred, labels=CLASSES)
        total_cm += cm
        print(f"\nConfusion Matrix - test driver {test_driver}")
        print(pd.DataFrame(cm, index=[f"true_{c}" for c in CLASSES], columns=[f"pred_{c}" for c in CLASSES]))
        rows.append({
            "test_driver": test_driver,
            "accuracy": accuracy_score(y[test_mask], pred),
            "precision": precision_score(y[test_mask], pred, average="macro", labels=CLASSES, zero_division=0),
            "recall": recall_score(y[test_mask], pred, average="macro", labels=CLASSES, zero_division=0),
            "macro-f1": f1_score(y[test_mask], pred, average="macro", labels=CLASSES, zero_division=0)
        })
        
    return pd.DataFrame(rows), total_cm

def hyperparameter_tuning(X, y, groups) -> None:
    param_dict = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 15),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2', 0.5],
        'class_weight': [None, 'balanced']
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=1)

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dict,
        n_iter=30,
        cv=LeaveOneGroupOut(),
        scoring='f1_macro',
        n_jobs=-1, random_state=42
    )
    search.fit(X, y, groups=groups)
    print('Best hyperparameters:', search.best_params_)
    print('Best mean LODO macro-f1:', round(search.best_score_, 4))
    return search.best_params_

def main():
    #
    FILES = {d: find_files(d) for d in TRAIN_DRIVERS}
    print()
    print("========================= FILES FOUND =========================")
    print(FILES)
    print()

    # loading drivers to W, CLASSES, and drivers
    df = [load_driver(path, driver) for driver, path in FILES.items()]
    W_all = np.concatenate([p[0] for p in df])
    y_all = np.concatenate([p[1] for p in df])
    driver_all = np.concatenate([p[2] for p in df])

    print("windows array:", W_all.shape, "| NaNs:", int(np.isnan(W_all).sum()))
    print(pd.crosstab(driver_all, y_all, margins=True,
                      rownames=["driver"], colnames=["label"])[CLASSES + ["All"]])

    # Feature Extraction
    X_all = extract_features(W_all)
    print("feature matrix:", X_all.shape)
    print(X_all.head())

    # Load one Drive out
    params = {
        'n_estimators': 479, 
        'class_weight': "balanced",
        'max_depth': 12,
        'max_features': 'sqrt',
        'min_samples_leaf': 4,
        'min_samples_split': 5,
    }
    results, total_cm = lodo_eval(X_all, y_all, driver_all, params)

    print(f"\nConfusion Matrix - combined drivers")
    print(pd.DataFrame(total_cm, index=[f"true_{c}" for c in CLASSES], columns=[f"pred_{c}" for c in CLASSES]))

    print(results.round(4))
    print("\nmean\n", results[["accuracy", "precision", "recall", "macro-f1"]].mean().round(4))
    print("std\n", results[["accuracy", "precision", "recall", "macro-f1"]].std().round(4))

    best_parameters = hyperparameter_tuning(X_all, y_all, driver_all)
    print(best_parameters)
main()
