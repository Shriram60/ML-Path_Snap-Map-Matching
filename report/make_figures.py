
"""
PathSnap report figure generator.

Reads final result CSVs directly and generates report charts.
Run from the project root:

    python report/make_figures.py

Output:
    report/figures/
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "report" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


FILES = {
    "models": ROOT / "results" / "model_comparison.csv",
    "tuning": ROOT / "results" / "tuning" /
              "random_forest_tuned_vs_untuned.csv",
    "hmm": ROOT / "hmm" / "outputs" / "hmm_comparison.csv",
}


def load_results():
    """Load result CSVs and verify required columns."""
    models = pd.read_csv(FILES["models"])
    tuning = pd.read_csv(FILES["tuning"])
    hmm = pd.read_csv(FILES["hmm"])

    required = {
        "models": {
            "model", "cv_macro_f1_mean", "test_top1",
            "test_macro_f1", "cv_log_loss_mean", "test_log_loss"
        },
        "tuning": {
            "version", "cv_macro_f1", "test_top1",
            "test_macro_f1", "cv_log_loss", "test_log_loss"
        },
        "hmm": {
            "dataset", "method", "top1_accuracy",
            "route_continuity", "route_f1"
        },
    }

    data = {
        "models": models,
        "tuning": tuning,
        "hmm": hmm,
    }

    for name, columns in required.items():
        missing = columns - set(data[name].columns)
        if missing:
            raise ValueError(
                f"{name} CSV is missing columns: {sorted(missing)}"
            )

    return models, tuning, hmm


def save_chart(filename):
    """Save the current Matplotlib chart."""
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_cv_macro_f1(models):
    """Compare cross-validation Macro F1 across models."""
    data = models.sort_values("cv_macro_f1_mean", ascending=False)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        data["model"],
        data["cv_macro_f1_mean"]
    )

    plt.title("Cross-Validation Macro F1 Across Models")
    plt.xlabel("Model")
    plt.ylabel("Mean CV Macro F1")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)

    for bar, value in zip(bars, data["cv_macro_f1_mean"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{value:.4f}",
            ha="center",
            fontsize=9
        )

    save_chart("01_cv_macro_f1.png")


def plot_student_test_metrics(models):
    """Compare student-test Top-1 and Macro F1."""
    x = np.arange(len(models))
    width = 0.36

    plt.figure(figsize=(11, 6))

    plt.bar(
        x - width / 2,
        models["test_top1"],
        width,
        label="Top-1 Accuracy"
    )

    plt.bar(
        x + width / 2,
        models["test_macro_f1"],
        width,
        label="Macro F1"
    )

    plt.xticks(x, models["model"], rotation=20)
    plt.ylabel("Score")
    plt.title("Student-Test Performance Across Models")
    plt.ylim(0, 1)
    plt.legend()

    save_chart("02_student_test_metrics.png")


def plot_log_loss(models):
    """Compare CV and student-test log loss."""
    x = np.arange(len(models))
    width = 0.36

    plt.figure(figsize=(11, 6))

    plt.bar(
        x - width / 2,
        models["cv_log_loss_mean"],
        width,
        label="CV Log Loss"
    )

    plt.bar(
        x + width / 2,
        models["test_log_loss"],
        width,
        label="Student-Test Log Loss"
    )

    plt.xticks(x, models["model"], rotation=20)
    plt.ylabel("Log Loss (lower is better)")
    plt.title("Log Loss Across Models")
    plt.legend()

    save_chart("03_log_loss.png")


def plot_tuning_comparison(tuning):
    """Compare untuned and tuned Random Forest."""
    metrics = [
        ("cv_macro_f1", "CV Macro F1"),
        ("test_top1", "Test Top-1"),
        ("test_macro_f1", "Test Macro F1"),
    ]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(9, 6))

    for index, version in enumerate(["untuned", "tuned"]):
        row = tuning.loc[tuning["version"] == version]

        if len(row) != 1:
            raise ValueError(
                f"Expected one row for tuning version: {version}"
            )

        values = [float(row.iloc[0][column]) for column, _ in metrics]

        plt.bar(
            x + (index - 0.5) * width,
            values,
            width,
            label=version.capitalize()
        )

    plt.xticks(x, [label for _, label in metrics])
    plt.ylabel("Score")
    plt.title("Random Forest: Untuned vs Tuned")
    plt.ylim(0, 1)
    plt.legend()

    save_chart("04_random_forest_tuning.png")


def plot_hmm_comparison(hmm):
    """Compare classifier-only and HMM on student-test data."""
    data = hmm[hmm["dataset"] == "student_test"].copy()

    if set(data["method"]) != {"classifier_only", "hmm"}:
        raise ValueError(
            "Expected classifier_only and hmm student-test results"
        )

    metrics = [
        ("top1_accuracy", "Top-1 Accuracy"),
        ("route_continuity", "Route Continuity"),
        ("route_f1", "Route F1"),
    ]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(9, 6))

    for index, method in enumerate(["classifier_only", "hmm"]):
        row = data.loc[data["method"] == method]

        if len(row) != 1:
            raise ValueError(
                f"Expected one student-test row for {method}"
            )

        values = [float(row.iloc[0][column]) for column, _ in metrics]

        label = (
            "Classifier Only"
            if method == "classifier_only"
            else "Classifier + HMM"
        )

        plt.bar(
            x + (index - 0.5) * width,
            values,
            width,
            label=label
        )

    plt.xticks(x, [label for _, label in metrics])
    plt.ylabel("Score")
    plt.title("Effect of HMM on Student-Test Performance")
    plt.ylim(0, 1)
    plt.legend()

    save_chart("05_classifier_vs_hmm.png")


def main():
    models, tuning, hmm = load_results()

    plot_cv_macro_f1(models)
    plot_student_test_metrics(models)
    plot_log_loss(models)
    plot_tuning_comparison(tuning)
    plot_hmm_comparison(hmm)

    print("\nAll five report figures generated successfully.")


if __name__ == "__main__":
    main()
