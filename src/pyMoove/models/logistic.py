"""Logistic-regression baseline for binary gos-event classification."""

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logistic_regression_model(
    *,
    c: float = 1.0,
    class_weight: str | dict[int, float] | None = "balanced",
    random_state: int = 0,
) -> Pipeline:
    """Create a preprocessing + logistic-regression classification pipeline.

    The pipeline imputes missing feature values, scales features using training
    data, and fits a regularized logistic regression. ``class_weight="balanced"`
    is a useful starting point when replacement and non-replacement examples
    are unevenly represented.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c,
                    class_weight=class_weight,
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )
