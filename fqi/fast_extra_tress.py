import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils.validation import check_is_fitted


def single_thread_predict(tree, input_full):
    input_flat = np.reshape(input_full, (-1, input_full.shape[-1]))
    try:
        check_is_fitted(tree)
    except NotFittedError:
        return np.zeros(input_full.shape[:-1])

    input_flat = tree._validate_X_predict(input_flat)

    # avoid storing the output of every estimator by summing them here
    if tree.n_outputs_ > 1:
        y_hat = np.zeros((input_flat.shape[0], tree.n_outputs_), dtype=np.float64)
    else:
        y_hat = np.zeros((input_flat.shape[0]), dtype=np.float64)

    for e in tree.estimators_:
        prediction = e.predict(input_flat, check_input=False)
        y_hat += prediction
    y_hat /= len(tree.estimators_)

    return np.reshape(y_hat, input_full.shape[:-1])


class FastExtraTreesActionRegressor:

    def __init__(self, *args, **kwargs):
        updated_kwargs = kwargs.copy()
        if "output_shape" in kwargs:
            self.n_actions = kwargs["output_shape"][0]
            del updated_kwargs["output_shape"]

        if "input_shape" in kwargs:
            del updated_kwargs["input_shape"]
        self.sklearn_tree = ExtraTreesRegressor(*args, **updated_kwargs)

    def fit(self, states, qs, **kwargs):
        self.sklearn_tree.fit(states, qs, **kwargs)

    def predict(self, state, **kwargs):
        return single_thread_predict(self.sklearn_tree, state)
