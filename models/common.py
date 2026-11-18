"""Shared pieces for training, tuning and inference: data loading, model list, grouped CV."""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_processing.paths import work_root
from evaluation.metrics import point_metrics, top1_predictions

SEED = 42


def load_partition(partition):
    """Labelled candidate rows for one partition, only points with a reliable reference."""
    table = pd.read_parquet(work_root() / "candidates" / partition)
    table = table[table["ref_arc"] >= 0].reset_index(drop=True)
    return table


def load_references(partition):
    return pd.read_parquet(work_root() / "references" / partition)


def feature_columns(table):
    """Only f_ columns are inputs. label and ref_arc are never used as features."""
    return [c for c in table.columns if c.startswith("f_")]


def sample_points(table, frac, seed=SEED):
    """Keep a fraction of GPS points, with all their candidates, so groups stay whole."""
    if frac >= 1.0:
        return table
    points = table[["trip_id", "obs_idx"]].drop_duplicates()
    keep = points.sample(frac=frac, random_state=seed)
    return table.merge(keep, on=["trip_id", "obs_idx"]).reset_index(drop=True)


# name: (family, builder, params). Scaling is added for models that need it.
MODELS = {
    "logreg": ("Linear", lambda p: make_pipeline(StandardScaler(), LogisticRegression(**p)),
               {"C": 1.0, "max_iter": 1000}),
    "knn": ("Instance-based", lambda p: make_pipeline(StandardScaler(), KNeighborsClassifier(**p)),
            {"n_neighbors": 25, "weights": "distance", "n_jobs": -1}),
    "random_forest": ("Ensemble (bagging)", lambda p: RandomForestClassifier(**p),
                      {"n_estimators": 150, "min_samples_leaf": 5, "max_features": 0.2, "max_depth": 40,
                       "n_jobs": -1, "random_state": SEED}),
    "hist_gb": ("Boosting", lambda p: HistGradientBoostingClassifier(**p),
                {"learning_rate": 0.1, "max_iter": 300, "max_leaf_nodes": 31,
                 "min_samples_leaf": 20, "l2_regularization": 0.0,
                 "early_stopping": False, "random_state": SEED}),
    "mlp": ("Neural network", lambda p: make_pipeline(StandardScaler(), MLPClassifier(**p)),
            {"hidden_layer_sizes": (64, 32), "alpha": 1e-4, "learning_rate_init": 1e-3,
             "batch_size": 512, "max_iter": 40, "early_stopping": True,
             "random_state": SEED}),
}


def build_model(name, params=None):
    family, builder, defaults = MODELS[name]
    return builder({**defaults, **(params or {})})


def score_table(model, table, features):
    out = table.copy()
    out["p_match"] = model.predict_proba(table[features].to_numpy())[:, 1]
    return out


def evaluate(model, table, features, references=None):
    """Score a labelled table and return metrics plus inference time."""
    start = time.perf_counter()
    scored = score_table(model, table, features)
    seconds = time.perf_counter() - start
    metrics = point_metrics(top1_predictions(scored), scored, references)
    metrics["inference_ms_per_1k_rows"] = 1000 * seconds / len(table) * 1000
    return metrics, scored


def grouped_cv(name, params, table, features, folds=5):
    """Grouped K-fold by trip_id, so one trip is never in both train and validation."""
    rows = []
    splitter = GroupKFold(n_splits=folds)
    for fold, (tr, va) in enumerate(splitter.split(table, groups=table["trip_id"])):
        model = build_model(name, params)
        start = time.perf_counter()
        model.fit(table.iloc[tr][features].to_numpy(), table.iloc[tr]["label"].to_numpy())
        train_s = time.perf_counter() - start
        m, _ = evaluate(model, table.iloc[va], features)
        rows.append({"model": name, "fold": fold, "train_time_s": train_s,
                     "top1_accuracy": m["top1_accuracy"], "macro_f1": m["macro_f1"],
                     "log_loss": m["log_loss"], "precision": m["precision"],
                     "recall": m["recall"]})
    return pd.DataFrame(rows)
