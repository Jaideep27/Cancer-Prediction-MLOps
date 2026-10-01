"""The training workflow, end to end.

1. Load + validate + stratified split (test set is locked away until step 5).
2. Tune each base model with RandomizedSearchCV (stratified k-fold on TRAIN only).
3. Build two ensembles from the tuned models: soft voting and stacking.
4. Pick the best candidate by cross-validated ROC-AUC (NOT by test score).
5. Pick the decision threshold on out-of-fold TRAIN predictions (target recall).
6. Evaluate once on the held-out test set, with bootstrap confidence intervals.
7. Quality gate -> register in MLflow -> if passed: alias "champion" + export bundle.

Every candidate is an MLflow run (params, CV metrics, test metrics, model file),
nested under one parent run per training session.
"""

import json
import os
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.model_selection import (
    ParameterGrid,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

from src.config import PROJECT_ROOT, data_config, model_config
from src.data.preprocess import load_dataset
from src.data.schema import FEATURE_COLUMNS
from src.models.build import (
    build_base_models,
    build_stacking_ensemble,
    build_voting_ensemble,
    normalize_search_space,
)
from src.models.bundle import save_bundle
from src.models.evaluate import METRIC_NAMES, bootstrap_ci, choose_threshold, compute_metrics

DEFAULT_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
# Exact list reported by skops.io.get_untrusted_types() for our five models (all sklearn internals).
SKOPS_TRUSTED_TYPES = [
    "sklearn.tree._tree.Tree",  # GradientBoosting tree storage
    "sklearn.neural_network._stochastic_optimizers.AdamOptimizer",  # MLP optimizer state
    "sklearn.utils._bunch.Bunch",  # ensembles' named_estimators_
    "sklearn.model_selection._split.StratifiedKFold",  # stacking's cv object
]


@dataclass
class Candidate:
    name: str
    model: Any
    cv: dict[str, float]
    test: dict[str, float]
    model_uri: str
    run_id: str


def setup_mlflow(experiment_name: str) -> str:
    """Use $MLFLOW_TRACKING_URI if set (e.g. the Compose server), else a local SQLite file."""
    uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    if uri.startswith("sqlite") and mlflow.get_experiment_by_name(experiment_name) is None:
        # Local mode: keep model files next to the project in ./mlruns
        mlflow.create_experiment(
            experiment_name, artifact_location=(PROJECT_ROOT / "mlruns").as_uri()
        )
    mlflow.set_experiment(experiment_name)
    return uri


def tune(
    pipeline: Any,
    space: dict[str, list[Any]],
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    tcfg: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Random search over the space; refits the best combination on all of X."""
    space = normalize_search_space(space)
    n_iter = min(tcfg["n_iter"], len(ParameterGrid(space)))  # small spaces: try everything
    search = RandomizedSearchCV(
        pipeline,
        space,
        n_iter=n_iter,
        scoring=tcfg["tuning_metric"],
        cv=cv,
        random_state=tcfg["random_state"],
        n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def cv_scores(model: Any, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> dict[str, float]:
    """Mean and std of every metric across the k folds."""
    res = cross_validate(clone(model), X, y, cv=cv, scoring=METRIC_NAMES, n_jobs=-1)
    out: dict[str, float] = {}
    for m in METRIC_NAMES:
        scores = res[f"test_{m}"]
        out[f"cv_{m}_mean"] = float(np.mean(scores))
        out[f"cv_{m}_std"] = float(np.std(scores))
    return out


def evaluate_and_log(
    name: str,
    model: Any,
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: StratifiedKFold,
) -> Candidate:
    """One nested MLflow run per candidate model."""
    with mlflow.start_run(run_name=name, nested=True) as run:
        cv_metrics = cv_scores(model, X_train, y_train, cv)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, proba, threshold=0.5)

        mlflow.set_tag("model_type", name)
        mlflow.log_params({k: str(v) for k, v in params.items()} or {"params": "defaults"})
        mlflow.log_metrics(cv_metrics)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(X_train.head(5), model.predict(X_train.head(5))),
            input_example=X_train.head(2),
            # MLflow saves with skops (safer than pickle), which refuses unknown types.
            # We trained these models ourselves, so we explicitly trust the few sklearn
            # internal types they contain -- and nothing else.
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
    return Candidate(name, model, cv_metrics, test_metrics, info.model_uri, run.info.run_id)


def run_training() -> dict[str, Any]:
    dcfg, mcfg = data_config(), model_config()
    tcfg = mcfg["training"]
    rs = tcfg["random_state"]

    X_train, X_test, y_train, y_test = load_dataset(dcfg)
    cv = StratifiedKFold(n_splits=tcfg["cv_folds"], shuffle=True, random_state=rs)
    tracking_uri = setup_mlflow(mcfg["mlflow"]["experiment_name"])
    started = datetime.now(timezone.utc)

    with mlflow.start_run(run_name=f"training-{started:%Y%m%d-%H%M%S}") as parent:
        mlflow.log_params(
            {
                "n_train": len(X_train),
                "n_test": len(X_test),
                "train_malignant_ratio": round(float(y_train.mean()), 4),
                "test_malignant_ratio": round(float(y_test.mean()), 4),
                "cv_folds": tcfg["cv_folds"],
                "n_iter": tcfg["n_iter"],
                "tuning_metric": tcfg["tuning_metric"],
            }
        )

        # --- Steps 2-3: tune base models, then build ensembles from them ---
        tuned: dict[str, Any] = {}
        candidates: list[Candidate] = []
        for name, pipeline in build_base_models(rs).items():
            print(f"Tuning {name} ...", flush=True)
            best, params = tune(pipeline, mcfg["search_spaces"][name], X_train, y_train, cv, tcfg)
            tuned[name] = best
            candidates.append(
                evaluate_and_log(name, best, params, X_train, y_train, X_test, y_test, cv)
            )

        ensemble_params = {
            f"{base}.{k}": v
            for base, model in tuned.items()
            for k, v in model.get_params().items()
            if k.startswith("clf__") and k.split("__")[1] in _INTERESTING_PARAMS
        }
        print("Evaluating voting ensemble ...", flush=True)
        candidates.append(
            evaluate_and_log(
                "voting_ensemble",
                build_voting_ensemble(tuned),
                {"voting": "soft", **ensemble_params},
                X_train,
                y_train,
                X_test,
                y_test,
                cv,
            )
        )
        print("Evaluating stacking ensemble ...", flush=True)
        candidates.append(
            evaluate_and_log(
                "stacking_ensemble",
                build_stacking_ensemble(tuned, cv),
                {"final_estimator": "logistic_regression", **ensemble_params},
                X_train,
                y_train,
                X_test,
                y_test,
                cv,
            )
        )

        # --- Step 4: choose by cross-validation, never by the test set ---
        best = max(candidates, key=lambda c: c.cv["cv_roc_auc_mean"])

        # --- Step 5: decision threshold from out-of-fold TRAIN predictions ---
        oof_proba = cross_val_predict(
            clone(best.model), X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]
        threshold = choose_threshold(y_train, oof_proba, mcfg["threshold"]["target_recall"])

        # --- Step 6: one honest evaluation on the held-out test set ---
        test_proba = best.model.predict_proba(X_test)[:, 1]
        final = compute_metrics(y_test, test_proba, threshold)
        ci = bootstrap_ci(y_test, test_proba, threshold)

        # --- Step 7: quality gate, registry, export ---
        gate = mcfg["promotion"]
        gate_passed = (
            final["recall"] >= gate["min_recall"] and final["roc_auc"] >= gate["min_roc_auc"]
        )

        mlflow.set_tags({"selected_model": best.name, "gate_passed": str(gate_passed)})
        mlflow.log_metric("threshold", threshold)
        mlflow.log_metrics({f"final_test_{k}": v for k, v in final.items()})
        mlflow.log_metrics({f"final_test_{k}_ci_low": v[0] for k, v in ci.items()})
        mlflow.log_metrics({f"final_test_{k}_ci_high": v[1] for k, v in ci.items()})

        reg = mcfg["mlflow"]
        version = mlflow.register_model(
            best.model_uri,
            reg["registered_model_name"],
            tags={
                "model_type": best.name,
                "threshold": str(threshold),
                "gate_passed": str(gate_passed),
                "test_recall": f"{final['recall']:.4f}",
                "test_roc_auc": f"{final['roc_auc']:.4f}",
            },
        )

        exported_to = None
        if gate_passed:
            # "champion" is a movable pointer. Rolling back = pointing it at an older version.
            MlflowClient().set_registered_model_alias(
                reg["registered_model_name"], reg["champion_alias"], version.version
            )
            metadata = {
                "model_name": reg["registered_model_name"],
                "model_version": str(version.version),
                "model_type": best.name,
                "threshold": threshold,
                "feature_columns": FEATURE_COLUMNS,
                "test_metrics": final,
                "test_metrics_95ci": ci,
                "cv_metrics": best.cv,
                "mlflow_run_id": best.run_id,
                "trained_at": started.isoformat(),
                "git_sha": os.getenv("GIT_SHA", "local"),
                "sklearn_version": sklearn.__version__,
                "python_version": platform.python_version(),
            }
            exported_to = str(
                save_bundle(
                    PROJECT_ROOT / mcfg["export"]["model_dir"], best.model, metadata, X_train
                )
            )

    summary = {
        "parent_run_id": parent.info.run_id,
        "tracking_uri": tracking_uri,
        "candidates": {
            c.name: {
                **{k: round(v, 4) for k, v in c.cv.items()},
                **{f"test_{k}": round(c.test[k], 4) for k in METRIC_NAMES},
            }
            for c in candidates
        },
        "selected_model": best.name,
        "threshold": round(threshold, 4),
        "final_test_metrics": final,
        "final_test_metrics_95ci": ci,
        "registered_version": str(version.version),
        "gate_passed": gate_passed,
        "exported_to": exported_to,
    }
    out = PROJECT_ROOT / "artifacts" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    (out / "latest_run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# Hyperparameters worth recording for the ensembles (the tuned values of each base model).
_INTERESTING_PARAMS = {
    "C",
    "n_estimators",
    "learning_rate",
    "max_depth",
    "subsample",
    "min_samples_leaf",
    "hidden_layer_sizes",
    "alpha",
    "learning_rate_init",
}
