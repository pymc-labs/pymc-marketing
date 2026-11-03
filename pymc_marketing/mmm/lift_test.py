#   Copyright 2022 - 2026 The PyMC Labs Developers
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

This provides the inner workings of `MMM.add_lift_test_measurements` method.
Other methods can be MMM.add_cost_per_target_calibration.
Use any of these methods directly while working with the `MMM` class.
"""

import warnings
from collections.abc import Callable, Collection, Sequence
from typing import Any, Concatenate, ParamSpec

import numpy as np
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import pytensor.xtensor as ptx
from numpy import typing as npt
from pymc import modelcontext
from pymc_extras.prior import Prior
from pytensor.graph.basic import Variable
from pytensor.graph.traversal import ancestors
from pytensor.xtensor import as_xtensor
from pytensor.xtensor.type import XTensorVariable

from pymc_marketing.mmm.components.saturation import SaturationTransformation

Index = Sequence[int] | npt.NDArray[np.integer]
Indices = dict[str, Index]
Values = npt.NDArray[np.int_] | npt.NDArray | npt.NDArray[np.str_]

_POSITIVE_SUPPORT_DISTRIBUTIONS = {
    "Gamma",
}
_REAL_LINE_LIKELIHOODS = {
    "Normal",
    "StudentT",
}
_LIFT_LIKELIHOODS = _POSITIVE_SUPPORT_DISTRIBUTIONS | _REAL_LINE_LIKELIHOODS
_COST_PER_TARGET_LIKELIHOODS = {"Normal", "StudentT", "LogNormal"}
_RATIO_DENOMINATOR_FLOOR = 1e-12
"""Floor of a cost-per-target ratio's denominator, which keeps it finite."""
_POSITIVE_SUPPORT_RV_OPS = {
    "beta",
    "exponential",
    "gamma",
    "halfcauchy",
    "halfnormal",
    "halft",
    "invgamma",
    "lognormal",
    "pareto",
    "weibull",
}


def _validate_measurement_values(values: npt.NDArray, column: str) -> None:
    """Check normalized measurement values before adding calibration variables."""
    if not np.isfinite(values).all():
        raise ValueError(
            f"Calibration column '{column}' must contain finite real numeric values."
        )
    if column == "sigma" and (values <= 0).any():
        raise ValueError("Calibration column 'sigma' must be strictly positive.")


def _validate_measurement_data(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Reject undefined observations and invalid measurement uncertainties."""
    for column in columns:
        dtype = df[column].dtype
        if (
            not pd.api.types.is_numeric_dtype(dtype)
            or pd.api.types.is_complex_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
        ):
            raise ValueError(
                f"Calibration column '{column}' must contain finite real numeric values."
            )
        try:
            values = df[column].to_numpy(dtype=float, na_value=np.nan)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Calibration column '{column}' must contain finite real numeric values."
            ) from error
        _validate_measurement_values(values, column)


_NESTED_PRIOR_ATTRIBUTES = ("distribution", "dist", "lower", "upper", "factor")
"""Attributes of prior wrappers (``Censored``, ``Scaled``) that hold nested values."""


def _has_extent(value: Any) -> bool:
    """Whether *value*, or anything nested in it, has dims or is not a scalar.

    Walks a prior's parameters and the inner distribution and bounds of a
    wrapper such as ``Censored`` or ``Scaled``; a constant counts when it has
    dims (a ``DataArray``) or at least one axis.
    """
    if getattr(value, "dims", None):
        return True
    if hasattr(value, "create_variable"):
        parameters = getattr(value, "parameters", None)
        nested = [
            *(parameters.values() if isinstance(parameters, dict) else ()),
            *(
                getattr(value, name)
                for name in _NESTED_PRIOR_ATTRIBUTES
                if hasattr(value, name)
            ),
        ]
        return any(_has_extent(item) for item in nested)
    ndim = getattr(value, "ndim", None)
    return bool(np.ndim(value) if ndim is None else ndim)


def _validate_likelihood_prior(likelihood: Prior, supported: Collection[str]) -> None:
    """Refuse a calibration likelihood whose contents the calibration rows decide.

    ``sigma`` comes from the ``sigma`` column, ``mu`` from the model and the
    dims from the rows, so the only parameter a caller may set is
    ``StudentT``'s ``nu``, which ``StudentT`` requires; the family must be
    one of *supported*. Everything is checked here because a parameter that
    slips through fails only once the calibration's coordinate is in the
    model.
    """
    if "sigma" in likelihood.parameters:
        raise ValueError(
            "The calibration `sigma` is taken from the `sigma` column. Do not pass "
            "`sigma` in `likelihood`; use a supported likelihood parameter such "
            "as `nu` for StudentT."
        )
    if "mu" in likelihood.parameters:
        raise ValueError(
            "The calibration `mu` is the model-implied value. Do not pass `mu` in "
            "`likelihood`."
        )
    if likelihood.dims is not None:
        raise ValueError(
            "The calibration dimensions are determined by the calibration rows. "
            "Do not pass `dims` in `likelihood`."
        )
    if likelihood.transform is not None:
        raise ValueError(
            "A calibration likelihood is observed, so it cannot be transformed. "
            "Do not pass `transform` in `likelihood`."
        )
    if not likelihood.centered:
        raise ValueError(
            "A non-centered calibration likelihood observes its standardized "
            "offset, so the model-implied value would not enter it. Do not pass "
            "`centered=False` in `likelihood`."
        )
    if likelihood.distribution == "StudentT" and "nu" not in likelihood.parameters:
        raise ValueError(
            "The StudentT calibration likelihood requires a `nu` parameter, for "
            "example `Prior('StudentT', nu=4)`."
        )
    if likelihood.distribution not in supported:
        raise ValueError(
            f"The {likelihood.distribution} distribution is not supported as a "
            "calibration likelihood. Supported distributions are "
            f"{', '.join(sorted(supported))}."
        )
    allowed = {"nu"} if likelihood.distribution == "StudentT" else set()
    if extra := sorted(set(likelihood.parameters) - allowed):
        raise ValueError(
            "The calibration scale comes from the `sigma` column and its location "
            "from the model, so StudentT's `nu` is the only parameter a "
            f"calibration likelihood takes. Do not pass "
            f"{', '.join(f'`{name}`' for name in extra)} in `likelihood`."
        )
    nu = likelihood.parameters.get("nu")
    if _has_extent(nu):
        raise ValueError(
            "The calibration dimensions are determined by the calibration rows, "
            "so `nu` must be a scalar. Do not pass `dims` or arrays in `nu` or in "
            "the priors it is built from."
        )
    if not (nu is None or hasattr(nu, "create_variable") or isinstance(nu, Variable)):
        value = np.asarray(nu)
        if not (np.isfinite(value) and value > 0):
            raise ValueError(
                "StudentT's `nu` must be a positive finite number or a prior, "
                f"got {nu!r}."
            )


def _validate_lift_likelihood_data(
    df_lift_test: pd.DataFrame, likelihood: Prior
) -> None:
    """Validate observations against known likelihood support constraints."""
    _validate_measurement_data(df_lift_test, ["x", "delta_x", "delta_y", "sigma"])
    _validate_likelihood_prior(likelihood, _LIFT_LIKELIHOODS)
    if likelihood.distribution in _POSITIVE_SUPPORT_DISTRIBUTIONS:
        if (df_lift_test["delta_y"] <= 0).any():
            raise ValueError(
                f"{likelihood.distribution} lift likelihood requires positive "
                "observed lift values; use a real-valued likelihood such as "
                "Prior('Normal') for signed estimates."
            )
        if (df_lift_test["delta_x"] <= 0).any():
            raise ValueError(
                f"{likelihood.distribution} lift likelihood is only valid when "
                "the spend changes and model-implied lifts are positive."
            )


def _validate_positive_model_lift(
    model: pm.Model, model_estimated_lift: XTensorVariable, likelihood: Prior
) -> None:
    """Ensure positive likelihoods have a valid model lift at initialization."""
    if likelihood.distribution not in _POSITIVE_SUPPORT_DISTRIBUTIONS:
        return

    free_rvs = set(model.free_RVs)
    parameter_rvs = [rv for rv in ancestors([model_estimated_lift]) if rv in free_rvs]
    unsupported_parameters = [
        rv.name
        for rv in parameter_rvs
        if rv.owner.op.name.lower() not in _POSITIVE_SUPPORT_RV_OPS
    ]
    if unsupported_parameters:
        raise ValueError(
            f"{likelihood.distribution} lift likelihood requires a model-implied "
            "lift that stays positive, but positivity could not be established "
            "for upstream random variables. Hierarchical or time-varying "
            "saturation terms may include real-line effects. Unsupported "
            f"upstream variables: {unsupported_parameters}."
        )

    replaced_lift = model.replace_rvs_by_values([model_estimated_lift])[0]
    initial_lift = model.compile_fn(
        replaced_lift,
        inputs=model.value_vars,
        point_fn=True,
        on_unused_input="ignore",
    )(model.initial_point())
    if (np.asarray(initial_lift) <= 0).any():
        raise ValueError(
            f"{likelihood.distribution} lift likelihood requires positive "
            "model-implied lifts at the model's initial point."
        )


def _resolve_likelihood(
    likelihood: Prior | type[pmd.DimDistribution] | None,
    dist: type[pmd.DimDistribution] | None = None,
) -> Prior:
    """Resolve the serializable likelihood, preserving the old ``dist`` alias."""
    if dist is not None:
        if likelihood is not None:
            raise ValueError("Specify only one of `likelihood` and `dist`.")
        warnings.warn(
            "The `dist` argument is deprecated; pass a serializable Prior "
            "using `likelihood` instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        likelihood = Prior(dist.__name__)
    elif likelihood is None:
        likelihood = Prior("Normal")
    elif isinstance(likelihood, type) and issubclass(likelihood, pmd.DimDistribution):
        warnings.warn(
            "Passing a distribution class as `likelihood` is deprecated; "
            "pass a serializable Prior instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        likelihood = Prior(likelihood.__name__)
    if not isinstance(likelihood, Prior):
        raise TypeError("`likelihood` must be a pymc_extras.prior.Prior.")
    return likelihood


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
    """Deprecated exception for the removed increasing-assumption check."""


def assert_monotonic(delta_x: pd.Series, delta_y: pd.Series) -> None:
    """Check monotonic lift measurements (deprecated).

    Lift measurements may have signed estimates, so the MMM no longer uses this
    check. This function remains temporarily for callers that imported it.
    """
    warnings.warn(
        "assert_monotonic is deprecated; signed lift estimates are supported.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not (delta_x * delta_y >= 0).all():
        raise NonMonotonicError("The data is not monotonic.")


P = ParamSpec("P")
SaturationFunc = Callable[Concatenate[XTensorVariable, P], XTensorVariable]
VariableMapping = dict[str, str]


def add_saturation_observations(
    df_lift_test: pd.DataFrame,
    variable_mapping: VariableMapping,
    saturation_function: SaturationFunc,
    model: pm.Model | None = None,
    likelihood: Prior | type[pmd.DimDistribution] | None = None,
    name: str = "lift_measurements",
    get_indices: Callable[[pd.DataFrame, pm.Model], Indices] = exact_row_indices,
    *,
    dist: type[pmd.DimDistribution] | None = None,
) -> None:
    """Add saturation observations to the likelihood of the model.

    General function to add lift measurements to the likelihood of the model.

    Not to be used directly for general use. Use :func:`MMM.add_lift_test_measurements`
    or :func:`add_lift_measurements_to_likelihood_from_saturation` instead.

    Parameters
    ----------
    df_lift_test : pd.DataFrame
        DataFrame with lift test results with at least the following columns:

        * ``x``: x axis value of the lift test.
        * ``delta_x``: change in x axis value of the lift test.
        * ``delta_y``: change in y axis value of the lift test.
        * ``sigma``: standard deviation of the lift test.

        Measurement columns must contain finite real numeric values, and
        ``sigma`` must be strictly positive. Invalid measurements raise
        ``ValueError`` before calibration variables are added.
        Any additional columns are assumed to be coordinates in the model.
    variable_mapping : dict[str, str]
        Dictionary of variable names to dimensions.
    saturation_function : Callable[[np.ndarray], np.ndarray]
        Function that takes spend and returns saturation.
    model : Optional[Model], optional
        PyMC model with arbitrary number of coordinates, by default None
    likelihood : Prior, optional
        Serializable likelihood prior, by default ``Prior("Normal")``. Its
        ``sigma`` parameter is set from the lift-test standard errors. Choose a
        distribution that matches the estimator's sampling model. The supported
        distributions are ``Normal``, ``StudentT``, and ``Gamma``. ``StudentT``'s
        ``nu`` is the only parameter accepted, since the data and the model
        decide the others; ``dims``, a ``transform`` and ``centered=False`` are
        refused too.
    name : str, optional
        Name of the likelihood, by default "lift_measurements"
    get_indices : Callable[[pd.DataFrame, pm.Model], Indices], optional
        Function to get the indices of the DataFrame in the model, by default exact_row_indices
        which assumes that the columns map exactly to the model coordinates.

    Examples
    --------
    Add lift tests for time-varying saturation to a model:

    .. code-block:: python

        import pymc as pm
        import pymc.dims as pmd
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
            # Distributions don't matter here, just the shape
            alpha = pmd.HalfNormal("alpha_in_model", dims=("channel", "date"))
            lam = pmd.HalfNormal("lam_in_model", dims="channel")

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
        import pymc.dims as pmd
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
            lam = pmd.HalfNormal("saturation_lam", dims="channel")
            beta = pmd.HalfNormal("saturation_beta", dims="channel")

            add_saturation_observations(
                df_lift_test,
                variable_mapping=saturation.variable_mapping,
                saturation_function=saturation.function,
            )

    Add lift tests for channel, geo saturation functions.

    .. code-block:: python

        import pymc as pm
        import pymc.dims as pmd
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
            lam = pmd.HalfNormal("saturation_lam", dims=("channel", "geo"))
            beta = pmd.HalfNormal("saturation_beta", dims=("channel", "geo"))

            add_saturation_observations(
                df_lift_test,
                variable_mapping=saturation.variable_mapping,
                saturation_function=saturation.function,
            )

    """
    required_columns = ["x", "delta_x", "delta_y", "sigma"]
    assert_is_subset(set(required_columns), set(df_lift_test.columns))
    likelihood = _resolve_likelihood(likelihood, dist)
    _validate_lift_likelihood_data(df_lift_test, likelihood)

    current_model: pm.Model = modelcontext(model)

    var_names = list(variable_mapping.values())

    required_dims: list[str] = list(
        {
            dim
            for name, dims in current_model.named_vars_to_dims.items()
            if name in var_names
            for dim in dims
        }
    )

    lift_dim = f"_{name}_dim"
    assert_is_subset(set(required_dims), set(df_lift_test.columns))
    indices = get_indices(df_lift_test[required_dims], current_model)
    indices_xr = {k: as_xtensor(v, dims=(lift_dim,)) for k, v in indices.items()}

    x_before = as_xtensor(df_lift_test["x"].to_numpy(), dims=(lift_dim,))
    x_after = x_before + as_xtensor(df_lift_test["delta_x"], dims=(lift_dim,))

    def saturation_curve(x):
        return saturation_function(
            x,
            **{
                parameter_name: current_model[variable_name].isel(
                    indices_xr, missing_dims="ignore"
                )
                for parameter_name, variable_name in variable_mapping.items()
            },
        )

    model_estimated_lift = saturation_curve(x_after) - saturation_curve(x_before)

    with current_model:
        current_model.add_coord(lift_dim, length=len(df_lift_test))
        _validate_positive_model_lift(current_model, model_estimated_lift, likelihood)
        model_estimated_lift = pmd.Deterministic(
            f"{name}_model_estimated_lift", model_estimated_lift
        )
        likelihood = likelihood.deepcopy()
        likelihood.parameters["sigma"] = as_xtensor(
            df_lift_test["sigma"].to_numpy(), dims=(lift_dim,)
        )
        likelihood.create_likelihood_variable(
            name=name,
            mu=model_estimated_lift,
            observed=as_xtensor(df_lift_test["delta_y"].to_numpy(), dims=(lift_dim,)),
            xdist=True,
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

    def function(x, time_varying: XTensorVariable, **kwargs):
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
    likelihood: Prior | type[pmd.DimDistribution] | None = None,
    name: str = "lift_measurements",
    get_indices: Callable[[pd.DataFrame, pm.Model], Indices] = exact_row_indices,
    *,
    dist: type[pmd.DimDistribution] | None = None,
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

        Measurement columns must contain finite real numeric values, and
        ``sigma`` must be strictly positive. Invalid measurements raise
        ``ValueError`` before calibration variables are added.
    saturation : SaturationTransformation
        Any SaturationTransformation instance.
    time_varying_var_name : str, optional
        Name of the time-varying variable in model.
    model : Optional[Model], optional
        PyMC model with arbitrary number of coordinates, by default None
    likelihood : Prior, optional
        Serializable likelihood prior, by default ``Prior("Normal")``. The
        standard errors are supplied as its ``sigma`` parameter.
    name : str, optional
        Name of the likelihood, by default "lift_measurements"
    get_indices : Callable[[pd.DataFrame, pm.Model], Indices], optional
        Function to get the indices of the DataFrame in the model, by default exact_row_indices
        which assumes that the columns map exactly to the model coordinates.

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
        likelihood=likelihood,
        dist=dist,
        name=name,
        model=model,
        get_indices=get_indices,
    )


def validate_cost_per_target_rows(
    calibration_df: pd.DataFrame,
    model: pm.Model,
    *,
    target_column: str = "cost_per_target",
    get_indices: Callable[[pd.DataFrame, pm.Model], Indices] = exact_row_indices,
    likelihood: Prior | None = None,
) -> tuple[Indices, np.ndarray, np.ndarray, Prior]:
    """Check a cost-per-target table and likelihood against *model* without touching the graph.

    Shared by :func:`add_cost_per_target_observations` and
    ``MMM.add_cost_per_target_calibration`` so that a table is refused by one
    set of checks, before either of them changes the model.

    Parameters
    ----------
    calibration_df : pd.DataFrame
        One row per calibration value, with a ``channel`` column, one column
        per non-date dim of ``channel_data``, *target_column* and ``sigma``.
    model : pm.Model
        The model whose coordinates the rows are mapped to.
    target_column : str, default ``"cost_per_target"``
        Column holding the calibration values.
    get_indices : callable, default :func:`exact_row_indices`
        Maps the dim columns of *calibration_df* to model coordinate indices.
    likelihood : Prior, optional
        Calibration likelihood; ``None`` means ``Prior("Normal")``. See
        :func:`add_cost_per_target_observations` for the supported families.

    Returns
    -------
    tuple[Indices, np.ndarray, np.ndarray, Prior]
        The row indices per dim, the calibration values and their ``sigma``
        as float arrays, and the resolved likelihood.

    Raises
    ------
    TypeError
        If *likelihood* is not a :class:`~pymc_extras.prior.Prior`.
    KeyError
        If a required column or dim column is missing.
    UnalignedValuesError
        If a ``channel`` or dim label is not a model coordinate (with the
        default *get_indices*).
    ValueError
        If *likelihood* sets any parameter other than ``StudentT``'s ``nu``,
        ``dims``, a ``transform`` or ``centered=False``, is a ``StudentT``
        without ``nu`` or with a ``nu`` that carries dims or is not a positive
        number, or is not ``Normal``, ``StudentT`` or ``LogNormal``; if
        *target_column* or ``sigma`` cannot be converted to finite float
        values; if ``sigma`` is not strictly positive; or if a calibration
        value is not strictly positive under ``LogNormal``.
    """
    likelihood = Prior("Normal") if likelihood is None else likelihood
    if not isinstance(likelihood, Prior):
        raise TypeError("`likelihood` must be a pymc_extras.prior.Prior.")
    _validate_likelihood_prior(likelihood, _COST_PER_TARGET_LIKELIHOODS)

    required_cols = {"channel", target_column, "sigma"}
    missing = required_cols - set(calibration_df.columns)
    if missing:
        raise KeyError(f"Missing required columns in calibration_df: {sorted(missing)}")

    cpt_dims = tuple(model.named_vars_to_dims["channel_data"])
    non_date_dims = [d for d in cpt_dims if d != "date"]

    missing_dims = [d for d in non_date_dims if d not in calibration_df.columns]
    if missing_dims:
        raise KeyError(
            f"Calibration data missing dimension columns: {missing_dims}. Required dims: {non_date_dims}"
        )

    indices = get_indices(calibration_df[non_date_dims], model)
    targets = calibration_df[target_column].to_numpy(dtype=float)
    sigmas = calibration_df["sigma"].to_numpy(dtype=float)
    _validate_measurement_values(targets, target_column)
    _validate_measurement_values(sigmas, "sigma")
    if likelihood.distribution == "LogNormal" and (targets <= 0).any():
        raise ValueError(
            "LogNormal calibration requires positive calibration values in "
            f"column '{target_column}'; use Prior('Normal') or Prior('StudentT', "
            "nu=...) for a value that can be zero or negative."
        )
    return indices, targets, sigmas, likelihood


def _validate_positive_cost_per_target_means(
    model: pm.Model, cost_rows: XTensorVariable, target_rows: XTensorVariable
) -> None:
    """Refuse a LogNormal calibration whose rows do not have a positive ratio to log.

    Both the mean spend and the mean contribution of every calibrated row
    must exceed :data:`_RATIO_DENOMINATOR_FLOOR` at the model's initial point,
    in either direction of the ratio. A zero numerator puts ``log(0)`` in the
    likelihood and stops the sampler with an opaque error; a denominator at
    the floor turns the ratio into an artifact of the clip, with no gradient
    through it. Both are named here, before the model changes.
    """
    cost_rows, target_rows = model.replace_rvs_by_values([cost_rows, target_rows])
    initial_cost, initial_target = (
        np.asarray(value)
        for value in model.compile_fn(
            [cost_rows, target_rows],
            inputs=model.value_vars,
            point_fn=True,
            on_unused_input="ignore",
        )(model.initial_point())
    )
    positive = (initial_cost > _RATIO_DENOMINATOR_FLOOR) & (
        initial_target > _RATIO_DENOMINATOR_FLOOR
    )
    if bad_rows := np.flatnonzero(~positive).tolist():
        raise ValueError(
            "LogNormal calibration requires a positive model-implied ratio at the "
            "model's initial point: a positive mean spend and a positive mean "
            f"contribution on every calibration row. Rows {bad_rows} have mean "
            f"spend {initial_cost[bad_rows].tolist()} and mean contribution "
            f"{initial_target[bad_rows].tolist()}. A channel with no spend over "
            "the model's dates, or whose contribution is zero or negative at the "
            "initial point, has no positive ratio; drop the row or use "
            "Prior('Normal')."
        )


def add_cost_per_target_observations(
    calibration_df: pd.DataFrame,
    *,
    model: pm.Model | None = None,
    cost_value: XTensorVariable,
    target_value: XTensorVariable,
    target_column: str = "cost_per_target",
    name_prefix: str = "cpt_calibration",
    target_per_cost: bool = False,
    get_indices: Callable[[pd.DataFrame, pm.Model], Indices] = exact_row_indices,
    likelihood: Prior | None = None,
) -> None:
    """Add an observed likelihood that calibrates a cost-per-target ratio.

    By default the ratio is ``mean(cost) / mean(target)`` (cost-per-target).
    Set ``target_per_cost=True`` to flip it to ``mean(target) / mean(cost)``
    (target-per-cost, e.g. conversions per dollar).

    One observed likelihood term is added per calibration row, centred on
    the model ratio ``r`` and matched to the table's ``sigma``, which is the
    standard error of the calibration value ``t`` in the units the table
    states it in, under every family:

    ==============  ==============  ================  ============  =================================
    ``likelihood``  ``mu``          ``sigma``         ``observed``  penalty as the ratio goes to zero
    ==============  ==============  ================  ============  =================================
    ``Normal``      ``r``           ``sigma``         ``t``         capped at ``t**2 / (2 sigma**2)``
    ``StudentT``    ``r``           ``sigma``         ``t``         capped lower; needs ``nu``
    ``LogNormal``   ``log(r)``      ``sigma / t``     ``t``         unbounded, symmetric in ``r / t``
    ==============  ==============  ================  ============  =================================

    A ``Normal`` on a ratio is bounded on one side: its negative log density
    tends to ``t**2 / (2 sigma**2)`` as the ratio goes to zero (8 nats at
    ``sigma = t / 4``), so under cost-per-target the calibration is a floor
    on ROAS but not a ceiling, and under ``target_per_cost=True`` the cap
    moves to the other side. ``StudentT`` caps lower on purpose, which is the
    choice when the experiment itself may be wrong. ``LogNormal`` measures
    the error on the log scale: a 2x miss costs the same in either direction
    and the cost keeps growing as ``log(r / t)**2``, so ``target_per_cost``
    only decides how the table is written.

    Near the target the families charge similar amounts: at
    ``sigma = t / 4`` a 25% miss costs 0.50 nats under ``Normal``, 0.56 under
    ``StudentT(nu=4)`` and 0.40 (over) or 0.66 (under) under ``LogNormal``.
    At the same width the ``Normal`` charges more than ``LogNormal`` for
    every ratio above the target and less for every ratio below it, since
    ``log(x) <= x - 1``.

    Pick the family by the error the experiment reports. A ROAS read off a
    lift test, an incremental outcome with a Normal standard error over a
    spend the test knows, has an additive error: the ``Normal`` on a ROAS
    table is then the sampling model, it pulls harder than ``LogNormal``
    against an overstated channel, and its cap as the ROAS goes to zero is
    what the experiment says about zero. ``LogNormal`` is the model for a
    value reported with a relative, asymmetric or log-scale interval, and
    for a CPA table, whose ``Normal`` is capped where an overstated channel
    goes. Its width ``sigma / t`` is the first-order translation of the
    table's standard error into a relative one, accurate while ``sigma / t``
    is about 0.3 or less: at 0.5 the ratios within 1.96 widths of the target
    run from 0.38 t to 2.66 t, against 0.02 t to 1.98 t under the ``Normal``.
    A log-scale standard error ``s`` is entered exactly as ``sigma = s * t``.

    Using the mean of numerator and denominator separately avoids
    ``mean(cost / target)`` which is numerically unstable when spend is spiky
    and the channel has slow or delayed adstock decay.

    Parameters
    ----------
    calibration_df : pd.DataFrame
        Must include columns ``channel``, ``sigma``, and a target column. By
        default the target column is assumed to be ``cost_per_target``. The
        DataFrame must also include one column per model dimension found in the
        CPT variable (excluding ``date``). ``sigma`` must be positive; under
        ``LogNormal`` the target column must be positive too.
    model : pm.Model, optional
        Model containing the cost-per-target tensor. If None, uses the current model context.
    cost_value : XTensorVariable
        XTensor representing cost (spend) values over the model coordinates,
        including a ``date`` dimension.
    target_value : XTensorVariable
        XTensor representing target (contribution) values over the model
        coordinates, including a ``date`` dimension.
    target_column : str
        Column in ``calibration_df`` containing the calibration targets.
    name_prefix : str
        Name for the observed likelihood variable.
    target_per_cost : bool
        If ``False`` (default), computes ``mean(cost) / mean(target)``.
        If ``True``, computes ``mean(target) / mean(cost)``.
    get_indices : Callable[[pd.DataFrame, pm.Model], Indices]
        Alignment function mapping rows to model coordinate indices.
    likelihood : Prior, optional
        Family of the calibration term, by default ``Prior("Normal")``;
        ``Prior("StudentT", nu=...)`` and ``Prior("LogNormal")`` are the
        alternatives. ``nu`` is the only parameter it takes: the others come
        from the model and the table, and they, ``dims``, a ``transform`` and
        ``centered=False`` are refused. A Prior-valued ``nu`` (without dims)
        adds a free variable named ``{name_prefix}_nu``.

    Raises
    ------
    TypeError
        If *likelihood* is not a :class:`~pymc_extras.prior.Prior`.
    KeyError
        If a required column or dim column is missing.
    UnalignedValuesError
        If a ``channel`` or dim label is not a model coordinate.
    ValueError
        As :func:`validate_cost_per_target_rows` (likelihood contents, values
        that are not finite, non-positive ``sigma``, non-positive values under
        ``LogNormal``), before any graph change.
    ValueError
        Under ``LogNormal``, if a calibrated row does not have a positive mean
        spend and a positive mean contribution at the model's initial point (a
        channel with no spend over the dates, or whose contribution is zero
        there); raised before the coordinate and the observed variable are
        added.

    Examples
    --------
    .. code-block:: python

        cost = as_xtensor(spend_array, dims=("date", "geo", "channel"))
        target = as_xtensor(contribution_array, dims=("date", "geo", "channel"))

        calibration_df = pd.DataFrame(
            {
                "channel": ["C1", "C2"],
                "geo": ["US", "US"],  # add dims as needed
                "cost_per_target": [30.0, 45.0],
                "sigma": [2.0, 3.0],
            }
        )

        add_cost_per_target_observations(
            calibration_df=calibration_df,
            model=mmm.model,
            cost_value=cost,
            target_value=target,
            name_prefix="cpt_calibration",
        )

    For a CPA table, measure the error on the log scale instead, so that the
    penalty keeps growing however far the channel's ROAS moves from the table:

    .. code-block:: python

        add_cost_per_target_observations(
            calibration_df=calibration_df,
            model=mmm.model,
            cost_value=cost,
            target_value=target,
            name_prefix="cpt_calibration",
            likelihood=Prior("LogNormal"),
        )
    """
    current_model: pm.Model = modelcontext(model)

    indices, target_values, sigma_values, likelihood = validate_cost_per_target_rows(
        calibration_df,
        current_model,
        target_column=target_column,
        get_indices=get_indices,
        likelihood=likelihood,
    )
    if likelihood.distribution == "LogNormal":
        _validate_positive_cost_per_target_means(
            current_model,
            *_gather_cost_per_target_means(
                cost_value, target_value, indices, dim=f"_{name_prefix}"
            ),
        )
    _add_cost_per_target_likelihood(
        current_model,
        cost_value,
        target_value,
        indices,
        target_values,
        sigma_values,
        likelihood,
        name_prefix=name_prefix,
        target_per_cost=target_per_cost,
    )


def _gather_cost_per_target_means(
    cost_value: XTensorVariable,
    target_value: XTensorVariable,
    indices: Indices,
    *,
    dim: str,
) -> tuple[XTensorVariable, XTensorVariable]:
    """Return ``mean(cost)`` and ``mean(target)`` over the dates at each calibration row.

    Both come out on *dim*, one entry per row of the calibration table, in
    the order of *indices*.
    """
    indices_xr = {k: as_xtensor(v, dims=(dim,)) for k, v in indices.items()}
    return (
        cost_value.mean(dim="date").isel(indices_xr, missing_dims="raise"),
        target_value.mean(dim="date").isel(indices_xr, missing_dims="raise"),
    )


def _add_cost_per_target_likelihood(
    model: pm.Model,
    cost_value: XTensorVariable,
    target_value: XTensorVariable,
    indices: Indices,
    target_values: np.ndarray,
    sigma_values: np.ndarray,
    likelihood: Prior,
    *,
    name_prefix: str,
    target_per_cost: bool,
) -> None:
    """Add the observed term of a validated calibration table to *model*.

    The ratio is ``mean(cost) / mean(target)`` over the dates
    (``mean(target) / mean(cost)`` with *target_per_cost*), its denominator
    clipped at :data:`_RATIO_DENOMINATOR_FLOOR`, taken at each row of
    *indices*; see :func:`add_cost_per_target_observations` for the family
    mapping.
    """
    dim = f"_{name_prefix}"
    indices_xr = {k: as_xtensor(v, dims=(dim,)) for k, v in indices.items()}
    targets = as_xtensor(target_values, dims=(dim,))
    sigmas = as_xtensor(sigma_values, dims=(dim,))
    cost_mean = cost_value.mean(dim="date")
    target_mean = target_value.mean(dim="date")
    numerator, denominator = (
        (target_mean, cost_mean) if target_per_cost else (cost_mean, target_mean)
    )
    # Divided before the rows are gathered, so the default Normal term is the
    # same computation as before the likelihood became configurable.
    ratio = (
        numerator / ptx.math.clip(denominator, _RATIO_DENOMINATOR_FLOOR, np.inf)
    ).isel(indices_xr, missing_dims="raise")

    if likelihood.distribution == "LogNormal":
        # First-order delta method: a standard error `sigma` on the table
        # value `t` is a width `sigma / t` on the log scale. The divisor is
        # the table's target, not the model ratio, so the penalty stays a
        # quadratic in log(ratio / target) however far the ratio moves.
        mu = ptx.math.log(ratio)
        sigma = sigmas / targets
    else:
        mu = ratio
        sigma = sigmas

    with model:
        model.add_coord(dim, length=len(target_values))
        likelihood = likelihood.deepcopy()
        likelihood.parameters["sigma"] = sigma
        likelihood.create_likelihood_variable(
            name=name_prefix, mu=mu, observed=targets, xdist=True
        )


def add_cost_per_target_potentials(
    calibration_df: pd.DataFrame,
    *,
    model: pm.Model | None = None,
    cost_value: XTensorVariable,
    target_value: XTensorVariable,
    target_column: str = "cost_per_target",
    name_prefix: str = "cpt_calibration",
    target_per_cost: bool = False,
    get_indices: Callable[[pd.DataFrame, pm.Model], Indices] = exact_row_indices,
    likelihood: Prior | None = None,
) -> None:
    """Call :func:`add_cost_per_target_observations` with a deprecation warning.

    .. deprecated:: 0.14.0
        Use :func:`add_cost_per_target_observations` instead.
    """
    warnings.warn(
        "add_cost_per_target_potentials is deprecated and will be removed in a "
        "future version. Use add_cost_per_target_observations instead.",
        FutureWarning,
        stacklevel=2,
    )
    add_cost_per_target_observations(
        calibration_df,
        model=model,
        cost_value=cost_value,
        target_value=target_value,
        target_column=target_column,
        name_prefix=name_prefix,
        target_per_cost=target_per_cost,
        get_indices=get_indices,
        likelihood=likelihood,
    )
