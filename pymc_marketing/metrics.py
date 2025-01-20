#   Copyright 2022 - 2025 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Evaluation Metrics."""

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)


def per_observation_crps(y_true: npt.NDArray, y_pred: npt.NDArray) -> npt.NDArray:
    """Compute the continuous ranked probability score (CRPS) for each observation.

    The CRPS — Continuous Ranked Probability Score — is a score function that compares a
    single ground truth value to a Cumulative Distribution Function.

    Parameters
    ----------
    y_true : array-like
        The ground truth values.
    y_pred : array-like
        The predicted values. It is expected that y_pred has one extra sample
        dimension on the left.

    Returns
    -------
    array-like
        The CRPS for each observation.

    Examples
    --------
    .. code-block:: python

        import numpy as np

        from pymc_marketing.metrics import per_observation_crps

        # y_true shape is (3,)
        y_true = np.array([1, 1, 1])
        # y_pred shape is (10, 3). The extra dimension on the left is the number of samples.
        y_pred = np.repeat(np.array([[0, 1, 0]]), 10, axis=0)

        # The result has shape (3,), one value per observation.
        per_observation_crps(y_true, y_pred)

        >> array([1., 0., 1.])

    References
    ----------
    - This implementation is a minimal adaptation from the one in the Pyro project: https://docs.pyro.ai/en/dev/_modules/pyro/ops/stats.html#crps_empirical
    - For an introduction to CRPS, see https://towardsdatascience.com/crps-a-scoring-function-for-bayesian-machine-learning-models-dd55a7a337a8
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_pred.shape[1:] != (1,) * (y_pred.ndim - y_true.ndim - 1) + y_true.shape:
        raise ValueError(
            f"""Expected y_pred to have one extra sample dim on left.
                Actual shapes: {y_pred.shape} versus {y_true.shape}"""
        )

    absolute_error = np.mean(np.abs(y_pred - y_true), axis=0)

    num_samples = y_pred.shape[0]
    if num_samples == 1:
        return absolute_error

    y_pred = np.sort(y_pred, axis=0)
    diff = y_pred[1:] - y_pred[:-1]
    weight = np.arange(1, num_samples) * np.arange(num_samples - 1, 0, -1)
    weight = weight.reshape(weight.shape + (1,) * (diff.ndim - 1))

    return absolute_error - np.sum(diff * weight, axis=0) / num_samples**2


def crps(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
    sample_weight: npt.NDArray | None = None,
) -> float:
    """Compute the (possibly weighted) average of the continuous ranked probability score.

    Parameters
    ----------
    y_true : array-like
        The ground truth values.
    y_pred : array-like
        The predicted values. It is expected that y_pred has one extra sample
        dimension on the left.
    sample_weight : array-like, optional
        The sample weights.

    Returns
    -------
    float
        The CRPS value as a (possibly weighted) average of the per-observation CRPS values.

    Examples
    --------
    .. code-block:: python

        import numpy as np

        from pymc_marketing.metrics import crps

        # y_true shape is (3,)
        y_true = np.array([1, 1, 1])
        # y_pred shape is (10, 3). The extra dimension on the left is the number of samples.
        y_pred = np.repeat(np.array([[0, 1, 0]]), 10, axis=0)

        # The result is a scalar.
        crps(y_true, y_pred)

        >> 0.666

    References
    ----------
    - This implementation is a minimal adaptation from the one in the Pyro project: https://docs.pyro.ai/en/dev/_modules/pyro/ops/stats.html#crps_empirical
    - For an introduction to CRPS, see https://towardsdatascience.com/crps-a-scoring-function-for-bayesian-machine-learning-models-dd55a7a337a8
    """
    return np.average(per_observation_crps(y_true, y_pred), weights=sample_weight)


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Normalized Root Mean Square Error (NRMSE).

    Normalization allows for comparison across different data sets and methodologies.
    NRMSE is one of the key metrics used in Robyn MMMs.

    Parameters
    ----------
    y_true : np.ndarray
        True values for target metric
    y_pred : np.ndarray
        Predicted values for target metric

    Returns
    -------
    float
        Normalized root mean square error.
    """
    return root_mean_squared_error(y_true, y_pred) / (y_true.max() - y_true.min())


def nmae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Normalized Mean Absolute Error (NMAE).

    Normalization allows for comparison across different data sets and methodologies.

    Parameters
    ----------
    y_true : np.ndarray
        True values for target metric
    y_pred : np.ndarray
        Predicted values for target metric

    Returns
    -------
    float
        Normalized mean absolute error.
    """
    return mean_absolute_error(y_true, y_pred) / (y_true.max() - y_true.min())
