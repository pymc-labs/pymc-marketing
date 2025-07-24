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
"""Adding lift tests as observations of saturation function.

This provides the inner workings of `MMM.add_lift_test_measurements` method. Use that
method directly while working with the `MMM` class.

"""

from collections.abc import Callable, Sequence
from typing import Concatenate, ParamSpec

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from numpy import typing as npt
from pytensor.tensor.variable import TensorVariable

from pymc_marketing.mmm.components.saturation import SaturationTransformation

Index = Sequence[int]
Indices = dict[str, Index]
Values = npt.NDArray[np.int_] | npt.NDArray | npt.NDArray[np.str_]


def _find_unaligned_values(same_value: npt.NDArray[np.int_]) -> list[int]:
    return np.argwhere(same_value.sum(axis=1) == 0).flatten().tolist()


class UnalignedValuesError(Exception):
    """Raised when some values are not aligned."""

    def __init__(self, unaligned_values: dict[str, list[int]]) -> None:
        self.unaligned_values = unaligned_values

        combined: set[int] = set()
        for values in unaligned_values.values():
            combined = combined.union(values)
        self.unaligned_rows = list(combined)

        msg = (
            "The following rows of the DataFrame "
            f"are not aligned: {self.unaligned_rows}"
        )
        super().__init__(msg)


def exact_row_indices(df: pd.DataFrame, model: pm.Model) -> Indices:
    """Get indices in the model for each row in the DataFrame.

    Assumes any column in the DataFrame is a coordinate in the model with the
    same name.

    If the DataFrame has columns that are not in the model, it will raise an
    error.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with coordinates combinations.
    model : pm.Model
        PyMC model with all the coordinates in the DataFrame.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary of indices for the lift test results in the model.

    Raises
    ------
    UnalignedValuesError
        If some values are not aligned. This means that some values in the
        DataFrame are not in the model.
    KeyError
        If some coordinates in the DataFrame are not in the model.

    Examples
    --------
    Get the indices from a DataFrame and model:

    .. code-block:: python

        import pymc as pm
        import pandas as pd

        from pymc_marketing.mmm.lift_test import exact_row_indices

        df_lift_test = pd.DataFrame(
            {
                "channel": [0, 1, 0],
                "geo": ["A", "B", "B"],
            }
        )

        coords = {"channel": [0, 1, 2], "geo": ["A", "B", "C"]}
        model = pm.Model(coords=coords)

        indices = exact_row_indices(df_lift_test, model)
        # {'channel': array([0, 1, 0]), 'geo': array([0, 1, 1])}

    """
    columns = df.columns.tolist()

    unaligned_values: dict[str, list[int]] = {}
    missing_coords: list[str] = []
    indices: Indices = {}
    for col in columns:
        lift_values = df[col].to_numpy()

        if col not in model.coords:
            missing_coords.append(col)
            continue

        # Coords in the model become tuples
        # Reference: https://github.com/pymc-devs/pymc/blob/04b6881efa9f69711d604d2234c5645304f63d28/pymc/model/core.py#L998
        # which become pd.Timestamp if from pandas objects
        # Convert to Series stores them as np.datetime64
        model_values = pd.Series(model.coords[col]).to_numpy()
        same_value = lift_values[:, None] == model_values
        if not (same_value.sum(axis=1) == 1).all():
            missing_values = _find_unaligned_values(same_value)
            unaligned_values[col] = missing_values

        indices[col] = np.argmax(same_value, axis=1)

    if unaligned_values:
        raise UnalignedValuesError(unaligned_values)

    if missing_coords:
        coord, be = ("coords", "are") if len(missing_coords) > 1 else ("coord", "is")
        raise KeyError(f"The {coord} {missing_coords} {be} not in the model")

    return indices


VariableIndexer = Callable[[str], TensorVariable]


def create_variable_indexer(
    model: pm.Model,
    indices: Indices,
) -> VariableIndexer:
    """Create a function to index variables in the model.

    Parameters
    ----------
    model : pm.Model
        PyMC model
    indices : dict[str, np.ndarray]
        Dictionary of indices for the indices in the model.

    Returns
    -------
    Callable[[str], TensorVariable]
        Function to index variables in the model.

    Raises
    ------
    KeyError
        If the variable is not in the model.

    Examples
    --------
    Create a variable indexer:

    .. code-block:: python

        import numpy as np
        import pymc as pm

        from pymc_marketing.mmm.lift_test import create_variable_indexer

        coords = {"channel": [0, 1, 2], "geo": ["A", "B", "C"]}
        with pm.Model(coords=coords) as model:
            pm.Normal("alpha", dims=("channel", "geo"))
            pm.Normal("beta", dims="channel")

        # Usually from exact_row_indices
        indices = {"channel": [0, 1], "geo": [1, 0]}

        variable_indexer = create_variable_indexer(model, indices)

    Get alpha at indices:

    .. code-block:: python

        alpha_at_indices = variable_indexer("alpha")

    Get beta at indices:

    .. code-block:: python

        beta_at_indices = variable_indexer("beta")

    """
    named_vars_to_dims = model.named_vars_to_dims

    def variable_indexer(name: str) -> TensorVariable:
        if name not in named_vars_to_dims:
            raise KeyError(f"The variable {name!r} is not in the model")

        idx: tuple[Index] = tuple([indices[dim] for dim in named_vars_to_dims[name]])  # type: ignore
        return model[name][idx]

    return variable_indexer


class MissingValueError(KeyError):
    """Error when values are missing from a required set."""

    def __init__(self, missing_values: list[str], required_values: list[str]) -> None:
        self.missing_values = missing_values
        self.required_values = required_values

        value, be = ("values", "are") if len(missing_values) > 1 else ("value", "is")

        super().__init__(
            f"The {value} {missing_values} {be} missing of the required {required_values}"
        )


def assert_is_subset(required: set[str], available: set[str]) -> None:
    """Check if the available set is a subset of the required set.

    Parameters
    ----------
    required : set[str]
        Required values.
    available : set[str]
        Available values.

    Raises
    ------
    MissingValueError
        If the available set is not a subset of the required set.

    """
    missing = required - available
    if missing:
        raise MissingValueError(list(missing), list(required))


class NonMonotonicError(ValueError):
    """Data is not monotonic."""


def assert_monotonic(delta_x: pd.Series, delta_y: pd.Series) -> None:
    """
    Check if the lift test results satisfy the increasing assumption.

    The increasing assumption states that if delta_x is positive, delta_y must be positive, and vice versa.

    Parameters
    ----------
    delta_x : pd.Series
        Series with the change in x axis value of the lift test.
    delta_y : pd.Series
        Series with the change in y axis value of the lift test.

    Raises
    ------
    NonMonotonicError
        If the lift test results do not satisfy the increasing assumption.

    """
    if not (delta_x * delta_y >= 0).all():
        raise NonMonotonicError("The data is not monotonic.")


P = ParamSpec("P")
SaturationFunc = Callable[Concatenate[TensorVariable, P], TensorVariable]
VariableMapping = dict[str, str]


def add_saturation_observations(
    df_lift_test: pd.DataFrame,
    variable_mapping: VariableMapping,
    saturation_function: SaturationFunc,
    model: pm.Model | None = None,
    dist: type[pm.Distribution] = pm.Gamma,
    name: str = "lift_measurements",
    get_indices: Callable[[pd.DataFrame, pm.Model], Indices] = exact_row_indices,
    variable_indexer_factory: Callable[
        [pm.Model, Indices], VariableIndexer
    ] = create_variable_indexer,
) -> None:
    """Add saturation observations to the likelihood of the model.

    General function to add lift measurements to the likelihood of the model.

    Not to be used directly for general use. Use :func:`MMM.add_lift_test_measurements`
    or :func:`add_lift_measurements_to_likelihood_from_saturation` instead.

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
    dist : pm.Distribution class, optional
        PyMC distribution to use for the likelihood, by default pm.Gamma
    name : str, optional
        Name of the likelihood, by default "lift_measurements"
    get_indices : Callable[[pd.DataFrame, pm.Model], Indices], optional
        Function to get the indices of the DataFrame in the model, by default exact_row_indices
        which assumes that the columns map exactly to the model coordinates.
    variable_indexer_factory : Callable[[pm.Model, Indices], Callable[[str], TensorVariable]], optional
        Function to create a variable indexer, by default create_variable_indexer
        which creates a function to index variables in the model. This is used determine
        the values of the parameters to evaluate the saturation function.

    Examples
    --------
    Add lift tests for time-varying saturation to a model:

    .. code-block:: python

        import pymc as pm
        import pandas as pd
        from pymc_marketing.mmm.lift_test import add_saturation_observations

        df_base_lift_test = pd.DataFrame(
            {
                "x": [1, 2, 3],
                "delta_x": [1, 2, 3],
                "delta_y": [1, 2, 3],
                "sigma": [0.1, 0.2, 0.3],
            }
        )


        def saturation_function(x, alpha, lam):
            return alpha * x / (x + lam)


        # These are required since alpha and lam
        # have both channel and date dimensions
        df_lift_test = df_base_lift_test.assign(
            channel="channel_1",
            date=["2019-01-01", "2019-01-02", "2019-01-03"],
        )

        coords = {
            "channel": ["channel_1", "channel_2"],
            "date": ["2019-01-01", "2019-01-02", "2019-01-03", "2019-01-04"],
        }
        with pm.Model(coords=coords) as model:
            # Usually defined in a larger model.
            # Distributions dont matter here, just the shape
            alpha = pm.HalfNormal("alpha_in_model", dims=("channel", "date"))
            lam = pm.HalfNormal("lam_in_model", dims="channel")

            add_saturation_observations(
                df_lift_test,
                variable_mapping={
                    "alpha": "alpha_in_model",
                    "lam": "lam_in_model",
                },
                saturation_function=saturation_function,
            )

    Use the saturation classes to add lift tests to a model. NOTE: This is what
    happens internally of :class:`MMM`.

    .. code-block:: python

        import pymc as pm
        import pandas as pd

        from pymc_marketing.mmm import LogisticSaturation
        from pymc_marketing.mmm.lift_test import add_saturation_observations

        saturation = LogisticSaturation()

        df_base_lift_test = pd.DataFrame(
            {
                "x": [1, 2, 3],
                "delta_x": [1, 2, 3],
                "delta_y": [1, 2, 3],
                "sigma": [0.1, 0.2, 0.3],
            }
        )

        df_lift_test = df_base_lift_test.assign(
            channel="channel_1",
        )

        coords = {
            "channel": ["channel_1", "channel_2"],
        }
        with pm.Model(coords=coords) as model:
            # Usually defined in a larger model.
            # Distributions dont matter here, just the shape
            lam = pm.HalfNormal("saturation_lam", dims="channel")
            beta = pm.HalfNormal("saturation_beta", dims="channel")

            add_saturation_observations(
                df_lift_test,
                variable_mapping=saturation.variable_mapping,
                saturation_function=saturation.function,
            )

    Add lift tests for channel, geo saturation functions.

    .. code-block:: python

        import pymc as pm
        import pandas as pd

        from pymc_marketing.mmm import LogisticSaturation
        from pymc_marketing.mmm.lift_test import add_saturation_observations

        saturation = LogisticSaturation()

        df_base_lift_test = pd.DataFrame(
            {
                "x": [1, 2, 3],
                "delta_x": [1, 2, 3],
                "delta_y": [1, 2, 3],
                "sigma": [0.1, 0.2, 0.3],
            }
        )

        df_lift_test = df_base_lift_test.assign(
            channel="channel_1",
            geo=["G1", "G2", "G2"],
        )

        coords = {
            "channel": ["channel_1", "channel_2"],
            "geo": ["G1", "G2", "G3"],
        }
        with pm.Model(coords=coords) as model:
            # Usually defined in a larger model.
            # Distributions dont matter here, just the shape
            lam = pm.HalfNormal("saturation_lam", dims=("channel", "geo"))
            beta = pm.HalfNormal("saturation_beta", dims=("channel", "geo"))

            add_saturation_observations(
                df_lift_test,
                variable_mapping=saturation.variable_mapping,
                saturation_function=saturation.function,
            )

    """
    required_columns = ["x", "delta_x", "delta_y", "sigma"]
    assert_is_subset(set(required_columns), set(df_lift_test.columns))
    assert_monotonic(df_lift_test["delta_x"], df_lift_test["delta_y"])

    current_model: pm.Model = pm.modelcontext(model)

    var_names = list(variable_mapping.values())

    required_dims: list[str] = list(
        {
            dim
            for name, dims in current_model.named_vars_to_dims.items()
            if name in var_names
            for dim in dims
        }
    )

    assert_is_subset(set(required_dims), set(df_lift_test.columns))
    indices = get_indices(df_lift_test[required_dims], current_model)

    x_before = pt.as_tensor_variable(df_lift_test["x"].to_numpy())
    x_after = x_before + pt.as_tensor_variable(df_lift_test["delta_x"].to_numpy())

    variable_indexer = variable_indexer_factory(
        current_model,
        indices,
    )

    def saturation_curve(x):
        return saturation_function(
            x,
            **{
                parameter_name: variable_indexer(variable_name)
                for parameter_name, variable_name in variable_mapping.items()
            },
        )

    model_estimated_lift = saturation_curve(x_after) - saturation_curve(x_before)

    with current_model:
        dist(
            name=name,
            mu=pt.abs(model_estimated_lift),
            sigma=df_lift_test["sigma"].to_numpy(),
            observed=np.abs(df_lift_test["delta_y"].to_numpy()),
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
    dim_cols: list[str] | None = None,
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
    dim_cols : list[str], optional
        Column names for model dimensions.

    Returns
    -------
    pd.DataFrame
        DataFrame with the scaled lift measurements.

    """
    # either [*dim_cols , channel_col], or [channel_col]
    index_cols: list[str] = (dim_cols if dim_cols else []) + [channel_col]
    # DataFrame with MultiIndex (RangeIndex, index_cols),
    # where dim_cols  is optional.
    # columns: x, delta_x
    df_original = df_lift_test.loc[:, [*index_cols, "x", "delta_x"]].set_index(
        index_cols, append=True
    )

    # DataFrame with MultiIndex (RangeIndex, (x, *dim_cols , delta_x))
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
        .reset_index(index_cols)
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
        transform(target_to_scale).flatten(),
        index=target.index,
        name=target.name,
    )


def scale_lift_measurements(
    df_lift_test: pd.DataFrame,
    channel_col: str,
    channel_columns: list[str | int],
    channel_transform: Callable[[np.ndarray], np.ndarray],
    target_transform: Callable[[np.ndarray], np.ndarray],
    dim_cols: list[str] | None = None,
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
    dim_cols : list[str], optional
        Names of the columns for channel dimensions

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
        dim_cols=dim_cols,
    )
    df_target_scaled = scale_target_for_lift_measurements(
        df_lift_test["delta_y"],
        target_transform,
    )
    df_sigma_scaled = scale_target_for_lift_measurements(
        df_lift_test["sigma"],
        target_transform,
    )

    if "date" in df_lift_test.columns:
        return pd.concat(
            [
                df_lift_test_channel_scaled,
                df_target_scaled,
                df_sigma_scaled,
                pd.Series(df_lift_test["date"]),
            ],
            axis=1,
        )

    return pd.concat(
        [df_lift_test_channel_scaled, df_target_scaled, df_sigma_scaled], axis=1
    )


def create_time_varying_saturation(
    saturation: SaturationTransformation,
    time_varying_var_name: str,
) -> tuple[SaturationFunc, VariableMapping]:
    """Return function and variable mapping that use a time-varying variable.

    Parameters
    ----------
    saturation : SaturationTransformation
        Any SaturationTransformation instance.
    time_varying_var_name : str, optional
        Name of the time-varying variable in model.

    Returns
    -------
    tuple[SaturationFunc, VariableMapping]
        Tuple of function and variable mapping to be used in
        add_saturation_observations function.

    """

    def function(x, time_varying: TensorVariable, **kwargs):
        return time_varying * saturation.function(x, **kwargs)

    variable_mapping = {
        **saturation.variable_mapping,
        "time_varying": time_varying_var_name,
    }

    return function, variable_mapping


def add_lift_measurements_to_likelihood_from_saturation(
    df_lift_test: pd.DataFrame,
    saturation: SaturationTransformation,
    time_varying_var_name: str | None = None,
    model: pm.Model | None = None,
    dist: type[pm.Distribution] = pm.Gamma,
    name: str = "lift_measurements",
    get_indices: Callable[[pd.DataFrame, pm.Model], Indices] = exact_row_indices,
    variable_indexer_factory: Callable[
        [pm.Model, Indices], Callable[[str], TensorVariable]
    ] = create_variable_indexer,
) -> None:
    """
    Add lift measurements to the likelihood from a saturation transformation.

    Wrapper around :func:`add_saturation_observations` to work with
    SaturationTransformation instances and time-varying variables.

    Used internally of the :class:`MMM` class.

    Parameters
    ----------
    df_lift_test : pd.DataFrame
        DataFrame with lift test results with at least the following columns:
            * `x`: x axis value of the lift test.
            * `delta_x`: change in x axis value of the lift test.
            * `delta_y`: change in y axis value of the lift test.
            * `sigma`: standard deviation of the lift test.
    saturation : SaturationTransformation
        Any SaturationTransformation instance.
    time_varying_var_name : str, optional
        Name of the time-varying variable in model.
    model : Optional[pm.Model], optional
        PyMC model with arbitrary number of coordinates, by default None
    dist : pm.Distribution class, optional
        PyMC distribution to use for the likelihood, by default pm.Gamma
    name : str, optional
        Name of the likelihood, by default "lift_measurements"
    get_indices : Callable[[pd.DataFrame, pm.Model], Indices], optional
        Function to get the indices of the DataFrame in the model, by default exact_row_indices
        which assumes that the columns map exactly to the model coordinates.
    variable_indexer_factory : Callable[[pm.Model, Indices], Callable[[str], TensorVariable]], optional
        Function to create a variable indexer, by default create_variable_indexer
        which creates a function to index variables in the model. This is used determine
        the values of the parameters to evaluate the saturation function.

    """
    if time_varying_var_name:
        saturation_function, variable_mapping = create_time_varying_saturation(
            saturation=saturation,
            # This is coupled with the name of the
            # latent process Deterministic
            time_varying_var_name=time_varying_var_name,
        )
    else:
        saturation_function = saturation.function
        variable_mapping = saturation.variable_mapping

    add_saturation_observations(
        df_lift_test=df_lift_test,
        variable_mapping=variable_mapping,
        saturation_function=saturation_function,
        dist=dist,
        name=name,
        model=model,
        get_indices=get_indices,
        variable_indexer_factory=variable_indexer_factory,
    )
