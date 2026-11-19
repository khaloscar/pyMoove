"""Small multilayer-perceptron classifier."""

from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_mlp_model(
    *,
    hidden_layer_sizes: tuple[int, ...] = (32, 16),
    alpha: float = 0.001,
    max_iter: int = 1000,
    early_stopping: bool = True,
    random_state: int = 0,
) -> Pipeline:
    """Create an imputation, scaling, and small MLP classification pipeline.

    To balance classes during fitting, compute sample weights from the
    training labels and pass them as ``classifier__sample_weight`` to
    ``Pipeline.fit``. For example::

        from sklearn.utils.class_weight import compute_sample_weight

        weights = compute_sample_weight("balanced", y_train)
        model.fit(X_train, y_train, classifier__sample_weight=weights)

    Parameters
    ----------
    hidden_layer_sizes:
        Number of neurons in each hidden layer.
    alpha:
        L2 regularization strength.
    max_iter:
        Maximum training iterations.
    early_stopping:
        Stop training when the internal validation score stops improving.
    random_state:
        Seed for reproducible initialization and validation splitting.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    alpha=alpha,
                    max_iter=max_iter,
                    early_stopping=early_stopping,
                    random_state=random_state,
                ),
            ),
        ]
    )
