"""Lift test functions for the MMM."""
from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from numpy import typing as npt

from pymc_marketing.mmm.transformers import hill_saturation, michaelis_menten

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


def empirical_lift_measurements(spend_before, spend_after, saturation_curve, pt=pt):
    """Calculate the empirical lift measurements at two spends.

    Args:
        spend_before: Array of spend before the change.
        spend_after: Array of spend after the change.
        saturation_curve: Function that takes spend and returns saturation.
        pt: PyTensor module.

    Returns:
        Array of empirical lift measurements.

    """
    return pt.diff(
        saturation_curve(pt.stack([spend_before, spend_after])),
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
) -> None:
    required_columns = ["spend", "delta_spend", "delta_return", "confidence"]
    for col in required_columns:
        if col not in df_lift_test.columns:
            raise KeyError(f"The required columns are {required_columns}")

    model = pm.modelcontext(model)

    var_names = list(variable_mapping.values())
    indices = indices_from_lift_tests(df_lift_test, model, var_names)

    spend_before = df_lift_test["spend"].to_numpy()
    spend_after = spend_before + df_lift_test["delta_spend"].to_numpy()

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
        spend_before, spend_after, partial_saturation_function
    )

    dist(
        "lift_measurements",
        mu=empirical_lift,
        sigma=df_lift_test["confidence"].to_numpy(),
        observed=df_lift_test["delta_return"].to_numpy(),
    )


def add_menten_empirical_lift_measurements_to_likelihood(
    df_lift_test: pd.DataFrame,
    model: Optional[pm.Model] = None,
    alpha_name: str = "saturation_alpha_channel",
    lam_name: str = "saturation_lambda_channel",
    dist=pm.Gamma,
) -> None:
    """Add the empirical lift measurements to the model.

    Args:
        df_lift_test: DataFrame with lift test results.
        model: PyMC model with channel coordinates
            and alpha_name and lam_name parameters.
        alpha_name: Name of the alpha parameter in the model.
        lam_name: Name of the lambda parameter in the model.

    """

    variable_mapping = {
        "alpha": alpha_name,
        "lam": lam_name,
    }

    return add_lift_measurements_to_likelihood(
        df_lift_test,
        variable_mapping,
        saturation_function=michaelis_menten,
        model=model,
        dist=dist,
    )


def add_hill_empirical_lift_measurements_to_likelihood(
    df_lift_test: pd.DataFrame,
    model: Optional[pm.Model] = None,
    sigma_name: str = "saturation_sigma_channel",
    beta_name: str = "saturation_beta_channel",
    lam_name: str = "saturation_lambda_channel",
    dist=pm.Gamma,
) -> None:
    variable_mapping = {
        "sigma": sigma_name,
        "beta": beta_name,
        "lam": lam_name,
    }

    return add_lift_measurements_to_likelihood(
        df_lift_test,
        variable_mapping,
        saturation_function=hill_saturation,
        model=model,
        dist=dist,
    )