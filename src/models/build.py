"""Model definitions: three base models and two ways of combining them.

Every model is a sklearn Pipeline. A Pipeline glues preprocessing (the scaler)
to the classifier, so whenever the model is fitted -- on the full train set or
on one cross-validation fold -- the scaler only ever sees that training data.
"""

from typing import Any

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_MODEL_NAMES = ["logistic_regression", "gradient_boosting", "neural_network"]


def build_base_models(random_state: int = 42) -> dict[str, Pipeline]:
    """Untuned base models. Hyperparameters are filled in later by the search."""
    return {
        # Linear model: needs scaled inputs (features range from 0.05 to 2500).
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000, random_state=random_state)),
            ]
        ),
        # Trees split on thresholds, so scaling changes nothing -> no scaler.
        "gradient_boosting": Pipeline(
            [("clf", GradientBoostingClassifier(random_state=random_state))]
        ),
        # Neural net: very sensitive to input scale -> scaler is essential.
        "neural_network": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        max_iter=2000,
                        # Stop when an internal 10% validation split stops improving.
                        early_stopping=True,
                        n_iter_no_change=20,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def build_voting_ensemble(tuned: dict[str, Any]) -> VotingClassifier:
    """Soft voting: average the three models' predicted probabilities."""
    return VotingClassifier(
        estimators=[(name, clone(model)) for name, model in tuned.items()],
        voting="soft",
    )


def build_stacking_ensemble(tuned: dict[str, Any], cv: StratifiedKFold) -> StackingClassifier:
    """Stacking: a small logistic regression LEARNS how much to trust each base model.

    The meta-model is trained on out-of-fold predictions (cv=...), so it never
    sees a base model's prediction on data that base model was trained on.
    """
    return StackingClassifier(
        estimators=[(name, clone(model)) for name, model in tuned.items()],
        final_estimator=LogisticRegression(max_iter=5000),
        cv=cv,
        stack_method="predict_proba",
    )


def normalize_search_space(space: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """YAML has no tuples; MLP hidden_layer_sizes must be tuples like (64, 32)."""
    return {
        key: [tuple(v) if isinstance(v, list) else v for v in values]
        for key, values in space.items()
    }
