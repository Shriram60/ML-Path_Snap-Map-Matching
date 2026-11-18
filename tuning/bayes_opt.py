"""Bayesian hyperparameter optimisation of the selected model with Optuna (TPE sampler).

Run from the repo root, after choosing the model from results/model_comparison.csv:
    python -m tuning.bayes_opt --model hist_gb --trials 40 --timeout 900

Objective: maximise mean grouped-CV Top-1 accuracy on the training trips.
Trial 0 is the untuned default configuration, so tuned and untuned are scored
on exactly the same folds. Only training trips are used for the search.
Student-test trips are used once at the end to compare tuned against untuned.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from data_processing.loader import load_trip
from data_processing.paths import trip_dirs
from models.common import (MODELS, SEED, build_model, evaluate, feature_columns, grouped_cv,
                           load_partition, load_references, sample_points)
from models.train_compare import test_route_continuity

OUT = Path("results") / "tuning"
ARTIFACTS = Path("models") / "artifacts"

# Search spaces. Each range contains the untuned default so trial 0 is valid.
SPACES = {
    "hist_gb": {
        "learning_rate": ("float_log", 0.01, 0.3),
        "max_iter": ("int", 100, 800),
        "max_leaf_nodes": ("int", 15, 127),
        "min_samples_leaf": ("int", 5, 300),
        "l2_regularization": ("float", 0.0, 5.0),
    },
    "random_forest": {
        "n_estimators": ("int", 50, 300),
        "min_samples_leaf": ("int", 1, 50),
        "max_features": ("float", 0.1, 0.8),
        "max_depth": ("int", 8, 40),
    },
    "logreg": {"C": ("float_log", 1e-3, 100.0)},
    "mlp": {
        "alpha": ("float_log", 1e-6, 1e-2),
        "learning_rate_init": ("float_log", 1e-4, 1e-2),
        "width": ("int", 16, 128),
        "depth": ("int", 1, 3),
    },
    "knn": {"n_neighbors": ("int", 5, 100), "p": ("int", 1, 2)},
}
# Defaults that are not in the model's own parameters, used for trial 0.
EXTRA_DEFAULTS = {
    "mlp": {"width": 64, "depth": 2},
    "knn": {"p": 2},
}


def suggest(trial, space):
    params = {}
    for name, (kind, low, high) in space.items():
        if kind == "int":
            params[name] = trial.suggest_int(name, low, high)
        elif kind == "float_log":
            params[name] = trial.suggest_float(name, low, high, log=True)
        else:
            params[name] = trial.suggest_float(name, low, high)
    return params


def to_model_params(name, params):
    params = dict(params)
    if name == "mlp":
        width, depth = params.pop("width"), params.pop("depth")
        params["hidden_layer_sizes"] = tuple(max(width // (2 ** i), 8) for i in range(depth))
    return params


def untuned_params(name):
    defaults = {**MODELS[name][2], **EXTRA_DEFAULTS.get(name, {})}
    return {k: defaults[k] for k in SPACES[name]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hist_gb", choices=list(SPACES))
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=900, help="seconds, the search stops at whichever comes first")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--search-frac", type=float, default=0.1, help="share of training points used in the search")
    parser.add_argument("--final-frac", type=float, default=0.2, help="share of training points for the final refit")
    args = parser.parse_args()
    name, space = args.model, SPACES[args.model]

    train_all = load_partition("train")
    features = feature_columns(train_all)
    search = sample_points(train_all, args.search_frac)
    print(f"Search rows: {len(search):,}, trips {search['trip_id'].nunique()}")

    def objective(trial):
        params = to_model_params(name, suggest(trial, space))
        cv = grouped_cv(name, params, search, features, args.folds)
        trial.set_user_attr("cv_macro_f1", float(cv["macro_f1"].mean()))
        trial.set_user_attr("cv_log_loss", float(cv["log_loss"].mean()))
        trial.set_user_attr("cv_top1_std", float(cv["top1_accuracy"].std()))
        trial.set_user_attr("fit_seconds", float(cv["train_time_s"].sum()))
        return float(cv["top1_accuracy"].mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial(untuned_params(name))  # trial 0 = untuned baseline
    start = time.perf_counter()
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)
    search_seconds = time.perf_counter() - start

    OUT.mkdir(parents=True, exist_ok=True)
    trials = study.trials_dataframe()
    trials.to_csv(OUT / f"{name}_trials.csv", index=False)
    best = to_model_params(name, study.best_params)
    untuned = to_model_params(name, untuned_params(name))

    # Refit both on the same training data and compare on student test.
    final = sample_points(train_all, args.final_frac)
    test = load_partition("student_test")
    refs = load_references("student_test")
    trips = {d.name: load_trip(d) for d in trip_dirs("student_test")}
    rows, models = [], {}
    for label, params, trial in [("untuned", untuned, study.trials[0]), ("tuned", best, study.best_trial)]:
        model = build_model(name, params)
        t0 = time.perf_counter()
        model.fit(final[features].to_numpy(), final["label"].to_numpy())
        fit_s = time.perf_counter() - t0
        m, scored = evaluate(model, test, features, refs)
        rows.append({
            "version": label, "params": json.dumps(params, default=str),
            "cv_top1": trial.value, "cv_macro_f1": trial.user_attrs["cv_macro_f1"],
            "cv_log_loss": trial.user_attrs["cv_log_loss"],
            "test_top1": m["top1_accuracy"], "test_macro_f1": m["macro_f1"],
            "test_log_loss": m["log_loss"],
            "test_route_continuity": test_route_continuity(scored, trips),
            "fit_time_s": fit_s, "inference_ms_per_1k_rows": m["inference_ms_per_1k_rows"],
        })
        models[label] = model
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / f"{name}_tuned_vs_untuned.csv", index=False)

    (OUT / f"{name}_search.json").write_text(json.dumps({
        "model": name, "sampler": "TPE", "seed": SEED,
        "objective": f"maximise mean {args.folds}-fold grouped CV Top-1 accuracy (groups = trip_id)",
        "search_space": {k: list(v) for k, v in space.items()},
        "trials_run": len(study.trials), "search_rows": len(search),
        "search_seconds": search_seconds, "best_trial": study.best_trial.number,
        "best_params": best, "final_fit_rows": len(final)}, indent=2, default=str))

    fig, ax = plt.subplots(figsize=(7, 4))
    values = trials["value"].to_numpy()
    ax.plot(trials["number"], values, "o", alpha=0.6, label="trial")
    ax.plot(trials["number"], np.maximum.accumulate(values), label="best so far")
    ax.axhline(values[0], color="grey", ls="--", label="untuned (trial 0)")
    ax.set_xlabel("trial"); ax.set_ylabel("CV Top-1 accuracy"); ax.legend()
    ax.set_title(f"Bayesian optimisation of {name} (train partition)")
    fig.tight_layout(); fig.savefig(OUT / f"{name}_optimisation_history.png", dpi=150); plt.close(fig)

    # Export the tuned model for the HMM and the API.
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(models["tuned"], ARTIFACTS / "final_model.joblib", compress=3)
    (ARTIFACTS / "model_config.json").write_text(json.dumps({
        "model": name, "feature_columns": features, "params": best,
        "output_column": "p_match", "trained_rows": len(final), "seed": SEED}, indent=2, default=str))

    print(f"\nTrials: {len(study.trials)}  search time: {search_seconds:.0f}s")
    print("Best params:", best)
    print(comparison.drop(columns="params").round(4).to_string(index=False))
    print(f"Saved {ARTIFACTS / 'final_model.joblib'}")


if __name__ == "__main__":
    main()
