"""
run_evaluation.py
===========================================================================
Step 3 of 4. Nested cross-validation evaluation: Athena vs. NIAK, three
classifiers each (Chapter 5.4-5.5 of the seminar).

For each pipeline (Athena, NIAK) and each classifier (SVM, RandomForest, MLP):
  - Outer stratified 5-fold CV estimates generalization performance
  - Inner stratified 2-fold CV (within each outer training fold) selects
    hyperparameters, so no test data ever influences model selection
  - Feature standardization is fit on the outer training fold only (no leakage)
  - Metrics: accuracy, sensitivity, specificity, AUC-ROC, vs. majority baseline

Requires output/multisite_data_final.npz (see build_dataset.py).
Results are saved incrementally to output/evaluation_results.csv, so partial
progress survives even if the run is interrupted.

NIAK's ~378,000 features made unrestricted GridSearchCV parallelism
unreliable on a 16GB machine (each parallel worker needs its own copy of
the training fold); see the n_jobs settings below and the n_jobs_override
used for NIAK specifically in main().

Set QUICK_TEST = True to sanity-check the whole pipeline in seconds before
committing to the full run, which took roughly 1.5 hours on the hardware
used for this seminar (NIAK's feature count makes it the dominant cost).
QUICK_TEST is left as False here, reflecting the run that produced this
seminar's reported results.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.dummy import DummyClassifier

# ── CONFIG ───────────────────────────────────────────────────────────────
OUTPUT_ROOT = Path("output")
NPZ_PATH = OUTPUT_ROOT / "multisite_data_final.npz"
RESULTS_CSV = OUTPUT_ROOT / "evaluation_results.csv"

QUICK_TEST = False   # <-- set to True first to sanity-check before a full run
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 2
RANDOM_STATE = 42

CLASSIFIER_GRIDS = {
    # NIAK's ~378,000-feature space means every parallel GridSearchCV worker
    # needs its own multi-GB copy of the training fold, regardless of which
    # classifier is being fit. n_jobs=-1 at the search level exhausted memory
    # for MLP first, then SVM, on a full-size fold. Cap all three at n_jobs=2
    # so at most 2 candidate fits (each with its own data copy) run at once.
    "SVM": {
        "estimator": LinearSVC(max_iter=5000),
        "param_grid": {"C": [0.01, 0.1, 1, 10]},
        "score_fn": "decision",  # use decision_function for AUC
        "search_n_jobs": 2,
    },
    "RandomForest": {
        # n_jobs=1 on the estimator: GridSearchCV already parallelizes across
        # candidates/folds (search_n_jobs below), so parallelizing here too
        # would oversubscribe CPUs (nested parallelism).
        "estimator": RandomForestClassifier(n_jobs=1, random_state=RANDOM_STATE),
        "param_grid": {"n_estimators": [100, 300], "max_depth": [None, 10, 20]},
        "score_fn": "proba",
        "search_n_jobs": 2,
    },
    "MLP": {
        "estimator": MLPClassifier(
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=10, max_iter=300, random_state=RANDOM_STATE,
        ),
        "param_grid": [
            {"hidden_layer_sizes": [(64,)], "alpha": [0.0001]},
            {"hidden_layer_sizes": [(128,)], "alpha": [0.0001]},
            {"hidden_layer_sizes": [(128, 64)], "alpha": [0.001]},
        ],
        "score_fn": "proba",
        "search_n_jobs": 2,
    },
}
# ─────────────────────────────────────────────────────────────────────────


def get_scores(model, score_fn, X):
    """Returns a continuous score per sample, used for AUC-ROC."""
    if score_fn == "decision":
        return model.decision_function(X)
    else:
        return model.predict_proba(X)[:, 1]


def append_results(fold_metrics, csv_path):
    """Append one classifier's fold results to the results CSV, so a crash
    partway through the run doesn't lose already-completed work."""
    df = pd.DataFrame(fold_metrics)
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=write_header, index=False)


def evaluate_pipeline(pipeline_name, X, y, quick=False, save_incremental=False, n_jobs_override=None):
    print(f"\n{'=' * 70}\n{pipeline_name}\n{'=' * 70}")

    if quick:
        # Smoke-test mode: tiny subsample, tiny folds, just check it runs
        rng = np.random.RandomState(0)
        idx = rng.choice(len(y), size=min(60, len(y)), replace=False)
        X, y = X[idx], y[idx]
        n_outer, n_inner = 2, 2
    else:
        n_outer, n_inner = N_OUTER_FOLDS, N_INNER_FOLDS

    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=RANDOM_STATE)
    all_results = []

    for clf_name, spec in CLASSIFIER_GRIDS.items():
        print(f"\n  --- {clf_name} ---")
        fold_metrics = []

        for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            t0 = time.time()
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Standardize using training-fold statistics only (no leakage,
            # see Chapter 5.3 / 3.6)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            search_n_jobs = spec["search_n_jobs"] if n_jobs_override is None else n_jobs_override
            inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=RANDOM_STATE)
            search = GridSearchCV(
                spec["estimator"], spec["param_grid"],
                cv=inner_cv, scoring="roc_auc", n_jobs=search_n_jobs,
            )
            search.fit(X_train_s, y_train)
            best_model = search.best_estimator_

            y_pred = best_model.predict(X_test_s)
            y_score = get_scores(best_model, spec["score_fn"], X_test_s)

            acc = accuracy_score(y_test, y_pred)
            sens = recall_score(y_test, y_pred, pos_label=1)  # ADHD recall
            spec_ = recall_score(y_test, y_pred, pos_label=0)  # control recall
            auc = roc_auc_score(y_test, y_score)

            # Majority-class baseline on the SAME fold, for direct comparison
            dummy = DummyClassifier(strategy="most_frequent")
            dummy.fit(X_train_s, y_train)
            baseline_acc = accuracy_score(y_test, dummy.predict(X_test_s))

            elapsed = time.time() - t0
            print(f"    fold {fold_i+1}/{n_outer}: acc={acc:.3f} sens={sens:.3f} "
                  f"spec={spec_:.3f} auc={auc:.3f}  (baseline={baseline_acc:.3f}, {elapsed:.0f}s, "
                  f"best_params={search.best_params_})")

            fold_metrics.append({
                "pipeline": pipeline_name, "classifier": clf_name, "fold": fold_i,
                "accuracy": acc, "sensitivity": sens, "specificity": spec_,
                "auc_roc": auc, "baseline_accuracy": baseline_acc,
                "best_params": str(search.best_params_),
            })

        df = pd.DataFrame(fold_metrics)
        all_results.extend(fold_metrics)
        means = df[["accuracy", "sensitivity", "specificity", "auc_roc"]].mean()
        stds = df[["accuracy", "sensitivity", "specificity", "auc_roc"]].std()
        print(f"  {clf_name} MEAN: acc={means['accuracy']:.3f}\u00b1{stds['accuracy']:.3f}  "
              f"sens={means['sensitivity']:.3f}\u00b1{stds['sensitivity']:.3f}  "
              f"spec={means['specificity']:.3f}\u00b1{stds['specificity']:.3f}  "
              f"auc={means['auc_roc']:.3f}\u00b1{stds['auc_roc']:.3f}")

        if save_incremental:
            append_results(fold_metrics, RESULTS_CSV)
            print(f"  (appended {pipeline_name}/{clf_name} results to {RESULTS_CSV})")

    return all_results


def main():
    OUTPUT_ROOT.mkdir(exist_ok=True)

    data = np.load(NPZ_PATH)
    X_athena, y_athena = data["X_athena"], data["y_athena"]
    X_niak, y_niak = data["X_niak"], data["y_niak"]

    # Collapse to binary ADHD (1) vs control (0)
    y_athena_bin = (y_athena != 0).astype(int)
    y_niak_bin = (y_niak != 0).astype(int)

    if QUICK_TEST:
        print("### QUICK_TEST MODE: tiny subsample, just checking everything runs ###")

    save_incremental = not QUICK_TEST
    if save_incremental and RESULTS_CSV.exists():
        # Fresh full run: start the CSV clean rather than appending onto a
        # stale/partial file left over from an earlier crashed run.
        RESULTS_CSV.unlink()

    t_start = time.time()
    results = []
    results += evaluate_pipeline("Athena", X_athena, y_athena_bin, quick=QUICK_TEST, save_incremental=save_incremental)
    # NIAK's ~378,000 features make process-based GridSearchCV parallelism
    # unreliable regardless of n_jobs (see the module docstring above) --
    # force fully serial fitting here. Athena is unaffected and keeps the
    # per-classifier search_n_jobs configured above.
    results += evaluate_pipeline("NIAK", X_niak, y_niak_bin, quick=QUICK_TEST, save_incremental=save_incremental, n_jobs_override=1)
    total_time = time.time() - t_start

    if save_incremental:
        print(f"\nFull results saved incrementally to {RESULTS_CSV}")

    print(f"\nTotal runtime: {total_time/60:.1f} min")

    if QUICK_TEST:
        print("\n### Smoke test complete. If everything above looks sane "
              "(no errors, reasonable-looking numbers), set QUICK_TEST = False "
              "and run the full version. ###")


if __name__ == "__main__":
    main()