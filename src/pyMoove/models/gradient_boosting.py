"""Histogram-based gradient-boosting classifier."""

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def make_hist_gradient_boosting_model(
    *,
    learning_rate: float = 0.1,
    max_iter: int = 200,
    max_leaf_nodes: int = 31,
    l2_regularization: float = 1.0,
    class_weight: str | dict[int, float] | None = "balanced",
    random_state: int = 0,
) -> Pipeline:
    """Create a median-imputation + histogram gradient-boosting pipeline.

    Scaling is not needed for this tree-based estimator. The default class
    weighting is a useful starting point when replacement events are less
    common than non-replacements.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=learning_rate,
                    max_iter=max_iter,
                    max_leaf_nodes=max_leaf_nodes,
                    l2_regularization=l2_regularization,
                    class_weight=class_weight,
                    random_state=random_state,
                ),
            ),
        ]
    )
