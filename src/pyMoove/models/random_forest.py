"""Random-forest baseline for binary gos-event classification."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def make_random_forest_model(
    *,
    n_estimators: int = 200,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    class_weight: str | dict[int, float] | None = "balanced",
    random_state: int = 0,
) -> Pipeline:
    """Create a median-imputation + random-forest classification pipeline.

    Unlike logistic regression, a random forest does not need feature scaling.
    ``class_weight="balanced"`` is a useful starting point when replacement
    and non-replacement examples are unevenly represented.

    Parameters
    ----------
    n_estimators:
        Number of decision trees in the forest.
    max_depth:
        Maximum depth of each tree. ``None`` lets each tree grow until its
        leaves are pure or contain fewer than ``min_samples_split`` samples.
    min_samples_leaf:
        Minimum number of training examples required in a leaf. Increasing
        this can reduce overfitting, especially with a small event dataset.
    class_weight:
        Optional class weights. The default balances the classes using the
        training labels.
    random_state:
        Seed for reproducible forest construction.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline that fills missing numeric features with their training-set
        median, then fits a ``RandomForestClassifier``.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    class_weight=class_weight,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
