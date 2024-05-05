#   Copyright 2024 The PyMC Labs Developers
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
"""Lift test functions for the MMM."""

from collections.abc import Callable
from functools import partial
from typing import Union

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from numpy import typing as npt

from pymc_marketing.mmm.transformers import logistic_saturation
from pymc_marketing.mmm.utils import michaelis_menten


class MissingLiftTestError(Exception):
    def __init__(self, missing_values: npt.NDArray[np.int_]) -> None:
        self.missing_values = missing_values
        super().__init__(
            f"Some lift test values are not in the model: {missing_values}"
        )


Index = npt.NDArray[np.int_]
Indices = dict[str, Index]
Values = Union[npt.NDArray[np.int_], npt.NDArray[np.float_], npt.NDArray[np.str_]]  # noqa: UP007


def _lift_test_index(lift_values: Values, model_values: Values) -> Index:
    same_value = lift_values[:, None] == model_values
    if not (same_value.sum(axis=1) == 1).all():
        missing_values = np.argwhere(same_value.sum(axis=1) == 0).flatten()
        raise MissingLiftTestError(missing_values)

    return np.argmax(same_value, axis=1)


def lift_test_indices(df_lift_test: pd.DataFrame, model: pm.Model) -> Indices:
    """Get the indices of the lift test results in the model.

    Assumes any column in the DataFrame is a coordinate in the model with the
    same name.

    Parameters
    ----------
    df_lift_test : pd.DataFrame
        DataFrame with lift test results.
    model : pm.Model
        PyMC model with all the coordinates in the DataFrame.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary of indices for the lift test results in the model.

    Raises
    ------
    MissingLiftTestError
        If some lift test values are not in the model.

    """

    columns = df_lift_test.columns.tolist()

    return {
        col: _lift_test_index(
            df_lift_test[col].to_numpy(),
            # Coords in the model become tuples
            # Reference: https://github.com/pymc-devs/pymc/blob/04b6881efa9f69711d604d2234c5645304f63d28/pymc/model/core.py#L998
            # which become pd.Timestamp if from pandas objects
            # Convert to Series stores them as np.datetime64
            pd.Series(model.coords[col]).to_numpy(),
        )
        for col in columns
    }


def calculate_lift_measurements_from_curve(
    x_before: npt.NDArray[np.float_],
    x_after: npt.NDArray[np.float_],
    saturation_curve: Callable[[npt.NDArray[np.float_]], npt.NDArray[np.float_]],
    pt=pt,
) -> npt.NDArray[np.float_]:
    """Calculate the lift measurements at two spends.

    Parameters
    ----------
    x_before : npt.NDArray[float]
        Array of x before the change.
    x_after : npt.NDArray[float]
        Array of x after the change.
    saturation_curve : Callable[[npt.NDArray[float]], npt.NDArray[float]]
        Function that takes spend and returns saturation.
    pt : tensor module, optional. Default is pytensor.tensor.

    Returns
    -------
    npt.NDArray[float]
        Array of lift measurements based on a given saturation curve

    """
    return pt.diff(
        saturation_curve(pt.stack([x_before, x_after])),
        axis=0,
    ).flatten()


def required_dims_from_named_vars_to_dims(
    named_vars_to_dims: dict[str, tuple[str, ...]],
) -> list[str]:
    """Get the required dimensions from a named_vars_to_dims dictionary.

    Parameters
    ----------
    named_vars_to_dims : dict[str, tuple[str, ...]]
        Dictionary of variable names to dimensions.

    Returns
    -------
    list[str]
        List of required dimensions.

    """
    required_dims = set()
    for dims in named_vars_to_dims.values():
        for dim in dims:
            required_dims.add(dim)

    return list(required_dims)


def indices_from_lift_tests(
    df_lift_test: pd.DataFrame,
    model: pm.Model,
    var_names: list[str],
) -> Indices:
    """Get the indices of the lift test results in the model.

    These are the mapping from the lift test result to the index of the
    corresponding variable in the model.

    Parameters
    ----------
    df_lift_test : pd.DataFrame
        DataFrame with lift test results with at least the following columns:
            * `x`: x axis value of the lift test.
            * `delta_x`: change in x axis value of the lift test.
            * `delta_y`: change in y axis value of the lift test.
            * `sigma`: standard deviation of the lift test.
        Any additional columns are assumed to be coordinates in the model.
    model : pm.Model
        PyMC model with arbitrary number of coordinates.
    var_names : list[str]
        List of variable names in the model.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary of indices for the lift test results in the model.
    """

    named_vars_to_dims = {
        name: dims
        for name, dims in model.named_vars_to_dims.items()
        if name in var_names
    }

    required_dims = required_dims_from_named_vars_to_dims(named_vars_to_dims)

    for col in required_dims:
        if col not in df_lift_test.columns:
            raise KeyError(f"The required coordinates are {required_dims}")

    return lift_test_indices(df_lift_test[required_dims], model)


def index_variable(
    var_dims: tuple[str, ...],
    var: pt.TensorVariable,
    indices: Indices,
) -> pt.TensorVariable:
    """Index the TensorVariable based on the required lift test indices."""
    idx = tuple([indices[dim] for dim in var_dims])
    return var.__getitem__(idx)


class NonMonotonicLiftError(Exception):
    """Raised when the lift test results do not satisfy the increasing assumption."""


def check_increasing_assumption(df_lift_tests: pd.DataFrame) -> None:
    """Checks if the lift test results satisfy the increasing assumption.

    If delta_x is positive, delta_y must be positive, and vice versa.
    """
    increasing = df_lift_tests["delta_x"] * df_lift_tests["delta_y"] >= 0

    if not increasing.all():
        raise NonMonotonicLiftError(
            "The lift test results do not satisfy the increasing assumption."
        )


def add_lift_measurements_to_likelihood(
    df_lift_test: pd.DataFrame,
    variable_mapping,
    saturation_function,
    model: pm.Model | None = None,
    dist=pm.Gamma,
    name: str = "lift_measurements",
) -> None:
    """Add lift measurements to the likelihood of the model.

    General function to add lift measurements to the likelihood of the model.

    Parameters
    ----------
    df_lift_test : pd.DataFrame
        DataFrame with lift test results with at least the following columns:
            * `x`: x axis value of the lift test.
            * `delta_x`: change in x axis value of the lift test.
            * `delta_y`: change in y axis value of the lift test.
            * `sigma`: standard deviation of the lift test.
        Any additional columns are assumed to be coordinates in the model.
    variable_mapping : dict[str, str]
        Dictionary of variable names to dimensions.
    saturation_function : Callable[[np.ndarray], np.ndarray]
        Function that takes spend and returns saturation.
    model : Optional[pm.Model], optional
        PyMC model with arbitrary number of coordinates, by default None
    dist : pm.Distribution, optional
        PyMC distribution to use for the likelihood, by default pm.Gamma
    name : str, optional
        Name of the likelihood, by default "lift_measurements"

    Examples
    --------
    Add an arbitrary lift test to a model:

    .. code-block:: python

        import pymc as pm
        import pandas as pd
        from pymc_marketing.mmm.lift_test import add_lift_measurements_to_likelihood

        df_base_lift_test = pd.DataFrame({
            "x": [1, 2, 3],
            "delta_x": [1, 2, 3],
            "delta_y": [1, 2, 3],
            "sigma": [0.1, 0.2, 0.3],
        })

        def saturation_function(x, alpha, lam):
            return alpha * x / (x + lam)

        df_lift_test = df_base_lift_test.assign(
            channel="channel_1",
            date=["2019-01-01", "2019-01-02", "2019-01-03"],
        )

        coords = {
            "channel": ["channel_1", "channel_2"],
            "date": ["2019-01-01", "2019-01-02", "2019-01-03", "2019-01-04"],
        }
        with pm.Model(coords=coords) as model:
            alpha = pm.HalfNormal("alpha_in_model", dims=("channel", "date"))
            lam = pm.HalfNormal("lam_in_model", dims="channel")

            add_lift_measurements_to_likelihood(
                df_lift_test,
                {"alpha": "alpha_in_model", "lam": "lam_in_model"},
                saturation_function,
                model=model,
            )

    """
    required_columns = ["x", "delta_x", "delta_y", "sigma"]

    missing_cols = set(required_columns).difference(df_lift_test.columns)
    if missing_cols:
        raise KeyError(f"Missing from DataFrame: {list(missing_cols)}")

    check_increasing_assumption(df_lift_test)

    model = pm.modelcontext(model)

    var_names = list(variable_mapping.values())
    indices = indices_from_lift_tests(df_lift_test, model, var_names)

    x_before = df_lift_test["x"].to_numpy()
    x_after = x_before + df_lift_test["delta_x"].to_numpy()

    kwargs = {
        name: index_variable(
            var_dims=model.named_vars_to_dims[var_name],
            var=model[var_name],
            indices=indices,
        )
        for name, var_name in variable_mapping.items()
    }

    partial_saturation_function = partial(saturation_function, **kwargs)
    model_estimated_lift = calculate_lift_measurements_from_curve(
        x_before, x_after, partial_saturation_function
    )

    dist(
        name=name,
        mu=pt.abs(model_estimated_lift),
        sigma=df_lift_test["sigma"].to_numpy(),
        observed=np.abs(df_lift_test["delta_y"].to_numpy()),
    )


def add_menten_empirical_lift_measurements_to_likelihood(
    df_lift_test: pd.DataFrame,
    alpha_name: str,
    lam_name: str,
    dist=pm.Gamma,
    model: pm.Model | None = None,
    name: str = "lift_measurements",
) -> None:
    """Add empirical lift measurements to the likelihood of the model.

    Specific implementation of the add_lift_measurements_to_likelihood function
    for the Michaelis-Menten saturation function.

    Parameters
    ----------
    df_lift_test : pd.DataFrame
        DataFrame with lift test results with at least the following columns:
            * `x`: x axis value of the lift test.
            * `delta_x`: change in x axis value of the lift test.
            * `delta_y`: change in y axis value of the lift test.
            * `sigma`: standard deviation of the lift test.
        Any additional columns are assumed to be coordinates in the model.
    alpha_name : str
        Name of the alpha parameter in the model.
    lam_name : str
        Name of the lambda parameter in the model.
    dist : pm.Distribution, optional
        PyMC distribution to use for the likelihood, by default pm.Gamma
    model : Optional[pm.Model], optional
        PyMC model with date and channel coordinates, by default None
    name : str, optional
        Name of the likelihood, by default "lift_measurements"
    """
    variable_mapping = {
        "alpha": alpha_name,
        "lam": lam_name,
    }

    add_lift_measurements_to_likelihood(
        df_lift_test,
        variable_mapping,
        saturation_function=michaelis_menten,
        model=model,
        dist=dist,
        name=name,
    )


def add_logistic_empirical_lift_measurements_to_likelihood(
    df_lift_test: pd.DataFrame,
    lam_name: str,
    beta_name: str,
    dist: pm.Distribution = pm.Gamma,
    model: pm.Model | None = None,
    name: str = "lift_measurements",
) -> None:
    """Add empirical lift measurements to the likelihood of the model.

    Specific implementation of add_lift_measurements_to_likelihood for the
    logistic saturation function.

    Parameters
    ----------
    df_lift_test : pd.DataFrame
        DataFrame with lift test results with at least the following columns:
            * `x`: x axis value of the lift test.
            * `delta_x`: change in x axis value of the lift test.
            * `delta_y`: change in y axis value of the lift test.
            * `sigma`: standard deviation of the lift test.
        Any additional columns are assumed to be coordinates in the model.
    lam_name : str
        Name of the lambda parameter in the model.
    beta_name : str
        Name of the beta parameter in the model.
    dist : pm.Distribution, optional
        PyMC distribution to use for the likelihood, by default pm.Gamma
    model : Optional[pm.Model], optional
        PyMC model with date and channel coordinates, by default None
    name : str, optional
        Name of the likelihood, by default "lift_measurements"
    """
    variable_mapping = {
        "lam": lam_name,
        "beta": beta_name,
    }

    def saturation_function(x, beta, lam):
        return beta * logistic_saturation(x, lam)

    add_lift_measurements_to_likelihood(
        df_lift_test,
        variable_mapping,
        saturation_function=saturation_function,
        model=model,
        dist=dist,
        name=name,
    )


def _swap_columns_and_last_index_level(df: pd.DataFrame) -> pd.DataFrame:
    """Take a DataFrame with a MultiIndex and swap the columns and the last index level."""
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("Index must be a MultiIndex")

    return df.stack().unstack(level=-2)  # type: ignore


def scale_channel_lift_measurements(
    df_lift_test: pd.DataFrame,
    channel_col: str,
    channel_columns: list[str],
    transform: Callable[[np.ndarray], np.ndarray],
) -> pd.DataFrame:
    """Scale the lift measurements for a specific channel.

    Parameters
    ----------
    df_lift_test : pd.DataFrame
        DataFrame with lift test results with the following columns:
            * `x`: x axis value of the lift test.
            * `delta_x`: change in x axis value of the lift test.
            * `channel_col`: channel to scale.
    channel_col : str
        Name of the channel to scale.
    channel_columns : list[str]
        List of channel values in the model. All lift tests results will be
        a subset of these values.
    transform : Callable[[np.ndarray], np.ndarray]
        Function to scale the lift measurements.

    Returns
    -------
    pd.DataFrame
        DataFrame with the scaled lift measurements.

    """

    # DataFrame with MultiIndex (RangeIndex, channel_col)
    # columns: x, delta_x
    df_original = df_lift_test.loc[:, [channel_col, "x", "delta_x"]].set_index(
        channel_col, append=True
    )

    # DataFrame with MultiIndex (RangeIndex, (x, delta_x))
    # columns: channel_columns values
    df_to_rescale = (
        df_original.pipe(_swap_columns_and_last_index_level)
        .reindex(channel_columns, axis=1)
        .fillna(0)
    )

    df_rescaled = pd.DataFrame(
        transform(df_to_rescale.to_numpy()),
        index=df_to_rescale.index,
        columns=df_to_rescale.columns,
    )

    return (
        df_rescaled.pipe(_swap_columns_and_last_index_level)
        .loc[df_original.index, :]
        .reset_index(channel_col)
    )


def scale_target_for_lift_measurements(
    target: pd.Series,
    transform: Callable[[np.ndarray], np.ndarray],
) -> pd.Series:
    """Scale the target for the lift measurements.

    Parameters
    ----------
    target : pd.Series
        Series with the target variable.
    transform : Callable[[np.ndarray], np.ndarray]
        Function to scale the target.

    Returns
    -------
    pd.Series
        Series with the scaled target.

    """
    target_to_scale = target.to_numpy().reshape(-1, 1)

    return pd.Series(
        transform(target_to_scale).flatten(), index=target.index, name=target.name
    )


def scale_lift_measurements(
    df_lift_test: pd.DataFrame,
    channel_col: str,
    channel_columns: list[str],
    channel_transform: Callable[[np.ndarray], np.ndarray],
    target_transform: Callable[[np.ndarray], np.ndarray],
) -> pd.DataFrame:
    """Scale the DataFrame with lift test results to be used in the model.

    Parameters
    ----------
    df_lift_test : pd.DataFrame
        DataFrame with lift test results with at least the following columns:
            * `x`: x axis value of the lift test.
            * `delta_x`: change in x axis value of the lift test.
            * `delta_y`: change in y axis value of the lift test.
            * `sigma`: standard deviation of the lift test.
    channel_col : str
        Name of the channel to scale.
    channel_columns : list[str]
        List of channel names.
    channel_transform : Callable[[np.ndarray], np.ndarray]
        Function to scale the lift measurements.
    target_transform : Callable[[np.ndarray], np.ndarray]
        Function to scale the target.

    Returns
    -------
    pd.DataFrame
        DataFrame with the scaled lift measurements. Will be same columns and
        index as the input DataFrame, but with the values scaled.

    """
    df_lift_test_channel_scaled = scale_channel_lift_measurements(
        df_lift_test.copy(),
        # Based on the model coords
        channel_col=channel_col,
        channel_columns=channel_columns,  # type: ignore
        transform=channel_transform,
    )
    df_target_scaled = scale_target_for_lift_measurements(
        df_lift_test["delta_y"],
        target_transform,
    )
    df_sigma_scaled = scale_target_for_lift_measurements(
        df_lift_test["sigma"],
        target_transform,
    )

    return pd.concat(
        [df_lift_test_channel_scaled, df_target_scaled, df_sigma_scaled],
        axis=1,
    )
