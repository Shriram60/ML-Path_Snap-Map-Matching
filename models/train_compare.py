"""Train and compare the candidate models.

Run from the repo root:
    python -m models.train_compare
    python -m models.train_compare --models hist_gb mlp --sample-frac 0.2

For each model: grouped CV on the training trips, then a fit on the training
sample and one evaluation on the 10 student-test trips. Each model's result is
saved on its own, so models can be rerun one at a time.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_processing.loader import load_trip
from data_processing.paths import trip_dirs
from evaluation.metrics import route_continuity, top1_predictions
from models.common import (MODELS, SEED, build_model, evaluate, feature_columns, grouped_cv,
                           load_partition, load_references, sample_points)

RESULTS = Path("results")
PER_MODEL = RESULTS / "per_model"
PLOTS = RESULTS / "plots"


def test_route_continuity(scored, trips):
    pred = top1_predictions(scored)
    values = [route_continuity(pred[pred["trip_id"] == tid], trip) for tid, trip in trips.items()]
    return float(np.nanmean(values))


def run_model(name, train, test, refs, features, trips, folds):
    family, _, defaults = MODELS[name]
    print(f"\n=== {name} ({family})")
    cv = grouped_cv(name, None, train, features, folds)
    print(cv[["fold", "top1_accuracy", "macro_f1", "log_loss", "train_time_s"]].round(4).to_string(index=False))

    model = build_model(name)
    start = time.perf_counter()
    model.fit(train[features].to_numpy(), train["label"].to_numpy())
    train_s = time.perf_counter() - start
    m, scored = evaluate(model, test, features, refs)
    continuity = test_route_continuity(scored, trips)

    # Training-set score, to compare with CV and test when checking for overfitting.
    # A 10% subset of the training points is enough and keeps KNN fast.
    m_train, _ = evaluate(model, sample_points(train, 0.1), features)

    result = {
        "model": name, "family": family,
        "config": json.dumps({k: v for k, v in defaults.items() if k != "n_jobs"}, default=str),
        "train_top1": m_train["top1_accuracy"], "train_macro_f1": m_train["macro_f1"],
        "cv_top1_mean": cv["top1_accuracy"].mean(), "cv_top1_std": cv["top1_accuracy"].std(),
        "cv_macro_f1_mean": cv["macro_f1"].mean(), "cv_macro_f1_std": cv["macro_f1"].std(),
        "cv_log_loss_mean": cv["log_loss"].mean(),
        "test_top1": m["top1_accuracy"], "test_macro_f1": m["macro_f1"],
        "test_precision": m["precision"], "test_recall": m["recall"],
        "test_accuracy": m["accuracy"], "test_log_loss": m["log_loss"],
        "test_route_continuity": continuity, "test_candidate_miss_rate": m["candidate_miss_rate"],
        "train_time_s": train_s, "inference_ms_per_1k_rows": m["inference_ms_per_1k_rows"],
        "confusion_matrix": json.dumps(m["confusion_matrix"]),
    }
    PER_MODEL.mkdir(parents=True, exist_ok=True)
    cv.to_csv(PER_MODEL / f"{name}_cv_folds.csv", index=False)
    (PER_MODEL / f"{name}.json").write_text(json.dumps(result, indent=2))

    # Epoch-wise loss only exists for the MLP.
    est = model[-1] if hasattr(model, "steps") else model
    if hasattr(est, "loss_curve_"):
        hist = pd.DataFrame({"epoch": np.arange(1, len(est.loss_curve_) + 1),
                             "train_loss": est.loss_curve_,
                             "val_accuracy": est.validation_scores_})
        hist.to_csv(PER_MODEL / f"{name}_epochs.csv", index=False)
    print(f"test top1={m['top1_accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
          f"logloss={m['log_loss']:.4f} continuity={continuity:.4f} fit={train_s:.1f}s")
    return result


def make_plots():
    PLOTS.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(p.read_text()) for p in sorted(PER_MODEL.glob("*.json"))]
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "model_comparison.csv", index=False)
    names = table["model"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, table["cv_macro_f1_mean"], yerr=table["cv_macro_f1_std"], capsize=4)
    ax.set_title("Grouped CV Macro F1 (train partition)")
    ax.set_ylabel("Macro F1")
    fig.tight_layout(); fig.savefig(PLOTS / "cv_macro_f1.png", dpi=150); plt.close(fig)

    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, table["test_top1"], 0.4, label="Top-1 accuracy")
    ax.bar(x + 0.2, table["test_macro_f1"], 0.4, label="Macro F1")
    ax.set_xticks(x, names); ax.set_ylim(0, 1); ax.legend()
    ax.set_title("Student-test Top-1 accuracy and Macro F1 (student_test partition)")
    fig.tight_layout(); fig.savefig(PLOTS / "test_top1_macro_f1.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, table["cv_log_loss_mean"], 0.4, label="CV (train)")
    ax.bar(x + 0.2, table["test_log_loss"], 0.4, label="student_test")
    ax.set_xticks(x, names); ax.set_title("Log loss (lower is better)"); ax.legend()
    fig.tight_layout(); fig.savefig(PLOTS / "log_loss.png", dpi=150); plt.close(fig)

    for _, r in table.iterrows():
        cm = np.array(json.loads(r["confusion_matrix"]))
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
        ax.set_xticks([0, 1], ["pred 0", "pred 1"]); ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(f"{r['model']} (student_test)")
        fig.tight_layout(); fig.savefig(PLOTS / f"confusion_{r['model']}.png", dpi=150); plt.close(fig)

    for path in PER_MODEL.glob("*_epochs.csv"):
        hist = pd.read_csv(path)
        fig, ax1 = plt.subplots(figsize=(6, 3.5))
        ax1.plot(hist["epoch"], hist["train_loss"], label="training loss")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("training log loss")
        ax2 = ax1.twinx()
        ax2.plot(hist["epoch"], hist["val_accuracy"], color="tab:orange", label="validation accuracy")
        ax2.set_ylabel("validation accuracy")
        ax1.set_title(f"{path.stem.replace('_epochs', '')} training history (train partition)")
        fig.legend(loc="center right"); fig.tight_layout()
        fig.savefig(PLOTS / f"{path.stem}.png", dpi=150); plt.close(fig)

    cols = ["model", "cv_top1_mean", "cv_macro_f1_mean", "cv_log_loss_mean", "test_top1",
            "test_macro_f1", "test_log_loss", "test_route_continuity", "train_time_s",
            "inference_ms_per_1k_rows"]
    print("\n" + table[cols].round(4).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--sample-frac", type=float, default=0.2,
                        help="share of training GPS points used, to keep training time practical")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    train_all = load_partition("train")
    test = load_partition("student_test")
    refs = load_references("student_test")
    features = feature_columns(train_all)

    balance = train_all["label"].value_counts().sort_index()
    print(f"Class balance (train): {balance.to_dict()}  positive share {balance[1] / balance.sum():.4f}")
    print(f"Candidates per point (train): {len(train_all) / train_all[['trip_id', 'obs_idx']].drop_duplicates().shape[0]:.2f}")

    train = sample_points(train_all, args.sample_frac)
    trips = {d.name: load_trip(d) for d in trip_dirs("student_test")}
    print(f"Training rows used: {len(train):,} of {len(train_all):,}, trips {train['trip_id'].nunique()}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "run_config.json").write_text(json.dumps({
        "seed": SEED, "sample_frac": args.sample_frac, "folds": args.folds,
        "train_rows_used": len(train), "train_trips": int(train["trip_id"].nunique()),
        "positive_share_train": float(balance[1] / balance.sum()),
        "features": features}, indent=2))

    for name in args.models:
        run_model(name, train, test, refs, features, trips, args.folds)
    make_plots()


if __name__ == "__main__":
    main()
