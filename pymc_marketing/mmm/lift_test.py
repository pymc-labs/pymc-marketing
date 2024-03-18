"""Lift test functions for the MMM."""

from functools import partial
from typing import Callable, Optional

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


def _lift_test_index(
    lift_values: np.ndarray, model_values: np.ndarray
) -> npt.NDArray[np.int_]:
    same_value = lift_values[:, None] == model_values
    if not (same_value.sum(axis=1) == 1).all():
        missing_values = np.argwhere(same_value.sum(axis=1) == 0).flatten()
        raise MissingLiftTestError(missing_values)

    return np.argmax(same_value, axis=1)


Index = npt.NDArray[np.int_]
Indices = dict[str, Index]


def lift_test_indices(df_lift_test: pd.DataFrame, model: pm.Model) -> Indices:
    """Get the indices of the lift test results in the model.

    Assumes any column in the DataFrame is a coordinate in the model with the
    same name.

    Args:
        df_lift_test: DataFrame with lift test results.
        model: PyMC model with date and channel coordinates.

    Returns:
        Dictionary of indices for the lift test results in the model.

    """
    columns = df_lift_test.columns.tolist()
    return {
        col: _lift_test_index(df_lift_test[col].to_numpy(), np.array(model.coords[col]))
        for col in columns
    }


def empirical_lift_measurements(x_before, x_after, saturation_curve, pt=pt):
    """Calculate the empirical lift measurements at two spends.

    Args:
        x_before: Array of x before the change.
        x_after: Array of x after the change.
        saturation_curve: Function that takes spend and returns saturation.
        pt: PyTensor module.

    Returns:
        Array of empirical lift measurements.

    """
    return pt.diff(
        saturation_curve(pt.stack([x_before, x_after])),
        axis=0,
    ).flatten()


def required_dims_from_named_vars_to_dims(
    named_vars_to_dims: dict[str, tuple[str, ...]],
) -> list[str]:
    """Get the required dimensions from a named_vars_to_dims dictionary.

    Args:
        named_vars_to_dims: Dictionary of variable names to dimensions.

    Returns:
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

    Args:
        df_lift_test: DataFrame with lift test results.
        model: PyMC model with arbitrary number of coordinates.
        alpha_name: Name of the alpha parameter in the model.
        lam_name: Name of the lambda parameter in the model.

    Returns:
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


def add_lift_measurements_to_likelihood(
    df_lift_test: pd.DataFrame,
    variable_mapping,
    saturation_function,
    model: Optional[pm.Model] = None,
    dist=pm.Gamma,
    name: str = "lift_measurements",
) -> None:
    """Add lift measurements to the likelihood of the model.

    Args:
        df_lift_test: DataFrame with lift test results.
        variable_mapping: Dictionary of variable names to dimensions.
        saturation_function: Function that takes spend and returns saturation.
        model: PyMC model with arbitrary number of coordinates.
        dist: PyMC distribution to use for the likelihood.
        name: Name of the likelihood.

    Returns:
        None

    """
    required_columns = ["x", "delta_x", "delta_y", "sigma"]

    missing_cols = set(required_columns).difference(df_lift_test.columns)
    if missing_cols:
        raise KeyError(f"Missing from DataFrame: {list(missing_cols)}")

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
    empirical_lift = empirical_lift_measurements(
        x_before, x_after, partial_saturation_function
    )

    dist(
        name=name,
        mu=empirical_lift,
        sigma=df_lift_test["sigma"].to_numpy(),
        observed=df_lift_test["delta_y"].to_numpy(),
    )


def add_menten_empirical_lift_measurements_to_likelihood(
    df_lift_test: pd.DataFrame,
    alpha_name: str,
    lam_name: str,
    dist=pm.Gamma,
    model: Optional[pm.Model] = None,
    name: str = "lift_measurements",
) -> None:
    """Add empirical lift measurements to the likelihood of the model.

    Args:
        df_lift_test: DataFrame with lift test results.
        alpha_name: Name of the alpha parameter in the model.
        lam_name: Name of the lambda parameter in the model.
        dist: PyMC distribution to use for the likelihood.
        model: PyMC model with date and channel coordinates.
        name: Name of the likelihood.

    Returns:
        None

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
    dist=pm.Gamma,
    model: Optional[pm.Model] = None,
    name: str = "lift_measurements",
) -> None:
    """Add empirical lift measurements to the likelihood of the model.

    Args:
        df_lift_test: DataFrame with lift test results.
        lam_name: Name of the lambda parameter in the model.
        beta_name: Name of the beta parameter in the model.
        dist: PyMC distribution to use for the likelihood.
        model: PyMC model with date and channel coordinates.
        name: Name of the likelihood.

    Returns:
        None

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


def scale_channel_lift_measurements(
    df_lift_test: pd.DataFrame,
    channel_col: str,
    channel_columns: list[str],
    transform: Callable[[np.ndarray], np.ndarray],
) -> pd.DataFrame:
    """Scale the lift measurements for a specific channel.

    Args:
        df_lift_test: DataFrame with lift test results.
        channel_col: Name of the channel to scale.
        channel_columns: List of channel names.
        transform: Function to scale the lift measurements.

    Returns:
        DataFrame with the scaled lift measurements.

    """
    df_original = df_lift_test.loc[:, [channel_col, "x", "delta_x"]].set_index(
        channel_col, append=True
    )
    df_to_rescale = (
        df_original.stack()
        .unstack(channel_col)
        .reindex(channel_columns, axis=1)  # type: ignore
        .fillna(0)
    )

    df_rescaled = pd.DataFrame(
        transform(df_to_rescale.to_numpy()),
        index=df_to_rescale.index,
        columns=df_to_rescale.columns,
    )

    return (
        df_rescaled.stack()
        .unstack(1)
        .loc[df_original.index, :]
        .reset_index(channel_col)
    )


def scale_target_for_lift_measurements(
    target: pd.Series,
    transform: Callable[[np.ndarray], np.ndarray],
) -> pd.Series:
    """Scale the target for the lift measurements.

    Args:
        target: Series with the target variable.
        transform: Function to scale the target.

    Returns:
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
