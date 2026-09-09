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
"""Construction, joint inference, and forward prediction for experimental MMMs."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from numbers import Integral
from types import MappingProxyType
from typing import Any

import pandas as pd
import pymc as pm
import pymc.dims as pmd
import xarray as xr
from pydantic import ConfigDict, PositiveInt, validate_call
from pymc_extras.prior import Prior

from pymc_marketing.mmm.components.adstock import GeometricAdstock
from pymc_marketing.mmm.components.base import Transformation
from pymc_marketing.mmm.components.saturation import LogisticSaturation
from pymc_marketing.mmm.experimental._data import compute_scales, normalize_data
from pymc_marketing.mmm.experimental._graph import (
    Binding,
    BuildContext,
    Equation,
    GraphTerm,
    copy_prior,
    specification_key,
    walk,
)
from pymc_marketing.mmm.experimental._media import Media
from pymc_marketing.mmm.fourier import YearlyFourier
from pymc_marketing.mmm.link import LinkFunction, get_link_spec
from pymc_marketing.mmm.scaling import Scaling
from pymc_marketing.terms import Intercept


class _Seasonality(GraphTerm):
    """Bind the existing Fourier component to the observation-date coordinate."""

    def __init__(self, order: int, prior: Prior) -> None:
        self.order = order
        self.prior = prior

    def dependencies(self) -> tuple[Any, ...]:
        return ()

    def _specification_state(self) -> Any:
        return self.order, self.prior

    def _build(self, context: BuildContext) -> Any:
        fourier = YearlyFourier(
            n_order=self.order,
            prefix="fourier_mode",
            variable_name="gamma_fourier",
            prior=copy_prior(self.prior),
        )
        days = pmd.Data(
            "_seasonality_dayofyear",
            context.ds["date"].dt.dayofyear.values,
            dims="date",
        )
        return fourier.apply(days)


class MMM:
    """Compose an outcome equation and fit its connected stochastic mechanisms.

    This experimental class is independent of the stable MMM's analysis and
    persistence interfaces. Construction only creates symbolic recipes.
    Replacing ``y`` replaces the entire outcome equation, including its scaling.

    Parameters
    ----------
    date_column : str, optional
        Date column, normalized to the internal ``date`` dimension.
    target_column : str, optional
        Fallback observation column for the outcome and name for the default equation.
    channel_columns : sequence of str, optional
        Ordered channel names. Omit when supplying a fully custom outcome.
    dims : tuple of str, optional
        Additional observation dimensions, such as ``("geo",)``.
    yearly_seasonality : int, optional
        Fourier order; omit to disable seasonality.
    media_transform : tuple of Transformation, optional
        One configured adstock and one saturation, executed left to right.
        Defaults to ``GeometricAdstock(l_max=8)`` then ``LogisticSaturation()``.
        Factories with read-only dimensions must declare dimensions on their underlying priors.
    model_config : dict, optional
        Priors for ``intercept``, ``likelihood``, and ``gamma_fourier``.
        Configure transformation priors on the transformation instances.
    scaling : Scaling or dict, optional
        Existing MMM scaling configuration. Exact-zero derived divisors become
        one; nonzero signed reductions retain their sign. A custom outcome is
        not automatically target-scaled. Media quantities themselves stay raw.
    sampler_config : dict, optional
        Defaults passed to ``pymc.sample``; per-fit arguments take precedence.

    Notes
    -----
    ``fit(X, y=None)`` accepts a DataFrame or xarray Dataset with embedded
    observations, or a separate outcome array. Observed intermediates use their
    own bindings. Missing observations are not implicitly imputed.

    Replacing ``y`` or a channel's equation invalidates the fitted model.
    In-place changes to a recipe or fitted scales require refitting before prediction.
    Create a new MMM to change constructor-only seasonality, default-prior entries, or dimension declarations.

    Custom temporal terms must expose ``required_history`` as a nonnegative integer.
    Each term declares its complete lookback in observation periods, including nested temporal operations.
    Prediction uses the largest reachable requirement and does not traverse mechanisms of conditioned equations.

    Examples
    --------
    .. code-block:: python

        from pymc_marketing.mmm import GeometricAdstock, MichaelisMentenSaturation
        from pymc_marketing.mmm.experimental import MMM

        mmm = MMM(
            target_column="sales",
            channel_columns=["tv", "search"],
            media_transform=(
                GeometricAdstock(l_max=8),
                MichaelisMentenSaturation(),
            ),
        )
        # mmm.fit(training_data)
        # mmm.sample_posterior_predictive(future_data)
    """

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        *,
        date_column: str = "date",
        target_column: str = "y",
        channel_columns: Sequence[str] | None = None,
        dims: tuple[str, ...] = (),
        yearly_seasonality: PositiveInt | None = None,
        media_transform: tuple[Transformation, ...] | None = None,
        model_config: dict[str, Prior] | None = None,
        scaling: Scaling | dict[str, Any] | None = None,
        sampler_config: dict[str, Any] | None = None,
    ) -> None:
        self.date_column = date_column
        self.target_column = target_column
        self.channel_columns = tuple(channel_columns or ())
        self.dims = dims
        self._yearly_seasonality = yearly_seasonality
        self.scaling = scaling
        self.sampler_config = dict(sampler_config or {})
        if not date_column or not target_column:
            raise ValueError("Date and target column names must not be empty.")
        if len(set(dims)) != len(dims) or {"date", "channel", date_column} & set(dims):
            raise ValueError("Panel dimensions must be distinct from date and channel.")
        self._geometry = self._geometry_key()
        transforms = (
            media_transform
            if media_transform is not None
            else (
                GeometricAdstock(l_max=8),
                LogisticSaturation(),
            )
        )
        self.media = Media(
            self.channel_columns, transforms, dims=dims, on_change=self._invalidate
        )
        defaults = self.default_model_config
        if model_config:
            if unknown := set(model_config) - set(defaults):
                raise ValueError(
                    f"Unknown model_config keys: {sorted(unknown)}. "
                    "Configure transformation priors on media_transform instances."
                )
            defaults.update(model_config)
        self._model_config = MappingProxyType(defaults)
        mean: Any = Intercept(prior=defaults["intercept"]) + self.media.total
        if yearly_seasonality is not None:
            mean = mean + _Seasonality(yearly_seasonality, defaults["gamma_fourier"])
        self._default_y = Equation(
            name=target_column,
            observed=target_column,
            mu=mean,
            likelihood=defaults["likelihood"],
        )
        self._y = self._default_y
        self.model: pm.Model | None = None
        self.idata: xr.DataTree | None = None
        self.scalers = xr.Dataset()
        self._training_data: xr.Dataset | None = None
        self._context: BuildContext | None = None
        self._fitted_key: Any = None
        self._parameter_names: tuple[str, ...] = ()

    @property
    def default_model_config(self) -> dict[str, Prior]:
        """Return fresh conventional intercept, likelihood, and Fourier priors."""
        link = get_link_spec(LinkFunction.IDENTITY)
        return {
            "intercept": link.default_intercept(self.dims),
            "likelihood": link.default_likelihood(self.dims),
            "gamma_fourier": Prior(
                "Laplace", mu=0, b=1, dims=(*self.dims, "fourier_mode")
            ),
        }

    @property
    def yearly_seasonality(self) -> int | None:
        """Return the constructor-only Fourier order."""
        return self._yearly_seasonality

    @property
    def model_config(self) -> Mapping[str, Prior]:
        """Return default-prior entries, which cannot be replaced after construction.

        Prior objects remain configurable in place; refit after changing them.
        """
        return self._model_config

    @property
    def media_transform(self) -> tuple[Transformation, ...]:
        """Return the configured transformations in execution order."""
        return self.media.transforms

    @property
    def y(self) -> Equation:
        """Return the complete symbolic outcome equation, not observed data."""
        return self._y

    @y.setter
    def y(self, equation: Equation) -> None:
        if not isinstance(equation, Equation):
            raise TypeError("y must be an Equation.")
        self._y = equation
        self._invalidate()

    def _invalidate(self) -> None:
        self.model = None
        self.idata = None
        self._context = None
        self._fitted_key = None
        self._parameter_names = ()

    def _geometry_key(self) -> tuple[Any, ...]:
        return (
            self.date_column,
            self.target_column,
            tuple(self.channel_columns),
            tuple(self.dims),
        )

    def _check_geometry(self) -> None:
        if self._geometry_key() != self._geometry:
            raise ValueError(
                "Create a new MMM to change date, channel, or dimension declarations."
            )

    def _key(self) -> Any:
        return specification_key(
            self.y,
            self._geometry_key(),
            self.scaling,
            self.scalers,
            tuple(getattr(node, "required_history", 0) for node in walk(self.y)),
        )

    def _outcome_column(self) -> str:
        return self.y.observed or self.target_column

    def _used_channels(self) -> tuple[str, ...]:
        identities = {id(node) for node in walk(self.y)}
        if id(self.media.total) in identities:
            return self.channel_columns
        return tuple(
            name
            for name in self.channel_columns
            if id(self.media[name].contribution) in identities
        )

    def _bindings(self, ds: xr.Dataset) -> dict[int, Binding]:
        bindings = (
            {
                identity: self._context.binding(equation)
                for identity, equation in self._context.equations.items()
            }
            if self._context is not None
            else {}
        )
        used = {id(node) for node in walk(self.y)}

        def add(equation: Equation, binding: Binding) -> None:
            previous = bindings.get(id(equation))
            if previous is not None and specification_key(
                previous
            ) != specification_key(binding):
                raise ValueError(
                    "One Equation cannot be bound to different observation slots."
                )
            bindings[id(equation)] = binding

        for channel, equation in self.media.equations.items():
            if id(equation) not in used:
                continue
            if equation.observed is not None and equation.observed != channel:
                raise ValueError(
                    f"Channel {channel!r} already binds its own observations; "
                    f"the equation instead requests {equation.observed!r}."
                )
            add(
                equation,
                Binding(
                    name=equation.name or channel,
                    observed=channel,
                    dims=equation.dims
                    if equation.dims is not None
                    else ("date", *self.dims),
                    scale=1.0,
                ),
            )
        target_scale: Any = (
            self.scalers.get("target_scale", 1.0) if self.y is self._default_y else 1.0
        )
        outcome_dims = self.y.dims
        if outcome_dims is None:
            if self._context is not None and id(self.y) in self._context.equations:
                outcome_dims = self._context.binding(self.y).dims
            elif self._outcome_column() in ds:
                outcome_dims = tuple(
                    str(dim) for dim in ds[self._outcome_column()].dims
                )
            else:
                outcome_dims = self.y.likelihood.dims or ("date", *self.dims)
        add(
            self.y,
            Binding(
                name=self.y.name or self.target_column,
                observed=self._outcome_column(),
                dims=outcome_dims,
                scale=target_scale,
            ),
        )
        return bindings

    def _new_model(
        self,
        ds: xr.Dataset,
        *,
        prediction: bool = False,
        condition_on: Sequence[str] = (),
        history_length: int = 0,
    ) -> tuple[pm.Model, BuildContext]:
        coords = (
            {
                dimension: labels
                for dimension, labels in self._context.model.coords.items()
                if dimension != "date" and labels is not None
            }
            if prediction and self._context is not None
            else {}
        )
        coords.update(
            {
                str(dim): values.values
                for dim, values in ds.coords.items()
                if dim in ds.dims
            }
        )
        media_nodes = {id(self.media.total)}
        for channel in self.media.values():
            media_nodes.update((id(channel.value), id(channel.contribution)))
        if any(id(node) in media_nodes for node in walk(self.y)):
            coords["channel"] = list(self.channel_columns)
        model = pm.Model(coords=coords)
        with model:
            context = BuildContext(
                ds,
                bindings=self._bindings(ds),
                prediction=prediction,
                condition_on=condition_on,
                history_length=history_length,
                default_dims=("date", *self.dims),
            )
            context.build(self.y)
        return model, context

    def build_model(self, X: pd.DataFrame | xr.Dataset, y: Any = None) -> pm.Model:
        """Build a fresh joint model from the supplied observations.

        Parameters
        ----------
        X : pandas.DataFrame or xarray.Dataset
            Inputs and intermediate observations, optionally including outcome observations.
        y : array-like, optional
            Separate outcome observations, bound to the outcome's declared column.

        Returns
        -------
        pymc.Model
            Joint probabilistic model. Any earlier posterior is invalidated,
            including when the replacement build fails.
        """
        self._invalidate()
        self._training_data = None
        self.scalers = xr.Dataset()
        self._check_geometry()
        ds = normalize_data(
            X,
            y,
            date_column=self.date_column,
            target_column=self._outcome_column(),
            dims=self.dims,
        )
        channels = self._used_channels()
        raw_media = any(
            getattr(stage, "requires_unscaled_input", False)
            for stage in self.media_transform
        )
        if channels and raw_media and self.scaling is not None:
            explicit_channel = (
                isinstance(self.scaling, Scaling) or "channel" in self.scaling
            )
            if explicit_channel:
                warnings.warn(
                    "This saturation requires raw input; channel scaling is not applied.",
                    UserWarning,
                    stacklevel=2,
                )
        self.scalers = compute_scales(
            ds,
            channels=() if raw_media else channels,
            target_column=self._outcome_column() if self.y is self._default_y else None,
            dims=self.dims,
            scaling=self.scaling,
        )
        scale = self.scalers.get("channel_scale")
        self.media.channel_scale = (
            scale.reindex(channel=list(self.channel_columns), fill_value=1.0)
            if scale is not None and "channel" in scale.dims
            else (scale if scale is not None else 1.0)
        )
        model, context = self._new_model(ds)
        self.model, self._context = model, context
        self._training_data = ds.copy(deep=True)
        return model

    def fit(
        self, X: pd.DataFrame | xr.Dataset, y: Any = None, **kwargs: Any
    ) -> xr.DataTree:
        """Sample the joint posterior using this call's data and newly fitted scales.

        Parameters
        ----------
        X : pandas.DataFrame or xarray.Dataset
            Training inputs and bound observations.
        y : array-like, optional
            Separate outcome observations.
        **kwargs : Any
            Arguments for ``pymc.sample``, overriding ``sampler_config``.

        Returns
        -------
        xarray.DataTree
            Posterior and sampler diagnostics. Stored as ``idata``.
        """
        model = self.build_model(X, y)
        options = {**self.sampler_config, **kwargs}
        if options.get("return_inferencedata") is False or "model" in options:
            raise ValueError(
                "fit manages the model and requires return_inferencedata=True."
            )
        options["return_inferencedata"] = True
        idata = pm.sample(model=model, **options)
        self.idata = idata
        self._fitted_key = self._key()
        self._parameter_names = tuple(str(rv.name) for rv in model.free_RVs)
        return idata

    def _check_fitted(self) -> None:
        self._check_geometry()
        if self.idata is None or self._context is None or self._training_data is None:
            raise RuntimeError("Call fit before posterior prediction.")
        if self._fitted_key != self._key():
            self._invalidate()
            raise RuntimeError(
                "The model specification changed; call fit again before prediction."
            )

    def _condition_names(self, condition_on: Sequence[str] | None) -> tuple[str, ...]:
        context = self._context
        if context is None:
            raise RuntimeError("The model has not been built.")
        observed = {
            identity: context.binding(equation)
            for identity, equation in context.equations.items()
            if identity != id(self.y) and context.binding(equation).observed is not None
        }
        if condition_on is None:
            return tuple(binding.name for binding in observed.values())
        if isinstance(condition_on, str):
            raise TypeError("condition_on must be a sequence of names, not one string.")
        resolved = []
        for selector in condition_on:
            candidates = {
                identity
                for identity, binding in observed.items()
                if selector == binding.name or selector == binding.observed
            }
            if len(candidates) != 1:
                raise ValueError(
                    f"Unknown or ambiguous intermediate observation: {selector!r}."
                )
            resolved.append(observed[candidates.pop()].name)
        return tuple(dict.fromkeys(resolved))

    def _prediction_data(self, X: pd.DataFrame | xr.Dataset) -> xr.Dataset:
        ds = normalize_data(
            X,
            date_column=self.date_column,
            target_column=self._outcome_column(),
            dims=self.dims,
        )
        training = self._training_data
        if training is None or self._context is None:
            raise RuntimeError("The model has not been fit.")
        for dimension, labels in self._context.model.coords.items():
            if dimension == "date" or labels is None or dimension not in ds.dims:
                continue
            old = (
                training.get_index(dimension)
                if dimension in training.dims
                else pd.Index(labels)
            )
            new = ds.get_index(dimension)
            if len(old) != len(new) or not old.isin(new).all():
                raise ValueError(
                    f"Prediction labels for {dimension!r} must match the training labels."
                )
            ds = ds.reindex({dimension: old})
        return ds

    def _required_history(self, condition_on: Sequence[str] = ()) -> int:
        stopped = (
            {
                identity
                for identity, name in self._context.equation_names.items()
                if name in condition_on
            }
            if self._context is not None
            else set()
        )
        contributions = {id(self.media.total)}
        contributions.update(id(self.media[name].contribution) for name in self.media)
        uses_media = False
        required = 0
        for node in walk(self.y, stop=lambda node: id(node) in stopped):
            if id(node) in stopped:
                continue
            uses_media |= id(node) in contributions
            lookback = getattr(node, "required_history", 0)
            if not isinstance(lookback, Integral) or lookback < 0:
                raise ValueError(
                    "Custom term required_history must be a nonnegative integer."
                )
            required = max(required, int(lookback))
        return max(required, self.media.required_history if uses_media else 0)

    def _with_history(
        self, ds: xr.Dataset, condition_on: Sequence[str] = ()
    ) -> tuple[xr.Dataset, int]:
        required = self._required_history(condition_on)
        if not required:
            return ds, 0
        training = self._training_data
        if training is None:
            raise RuntimeError("The model has not been fit.")
        old_dates = pd.DatetimeIndex(training["date"].values)
        new_dates = pd.DatetimeIndex(ds["date"].values)
        if len(old_dates) < 2:
            raise ValueError(
                "At least two training dates are needed to infer a prediction cadence."
            )
        frequency: Any = (
            pd.infer_freq(old_dates)
            if len(old_dates) >= 3
            else old_dates[1] - old_dates[0]
        )
        if frequency is None:
            raise ValueError(
                "Temporal history requires regularly spaced training dates."
            )
        expected = pd.date_range(
            old_dates[-1], periods=len(new_dates) + 1, freq=frequency
        )[1:]
        if not new_dates.equals(expected):
            raise ValueError(
                "With temporal history, prediction dates must immediately follow training at the fitted cadence. "
                "Use include_last_observations=False for independent or in-sample scenarios."
            )
        count = min(required, len(old_dates))
        history = training.isel(date=slice(-count, None))
        history = history.drop_vars(
            [
                name
                for name, data in history.data_vars.items()
                if "date" not in data.dims
            ]
        )
        combined = xr.concat(
            [history, ds],
            dim="date",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            join="exact",
        )
        return combined, count

    def sample_posterior_predictive(
        self,
        X: pd.DataFrame | xr.Dataset | None = None,
        *,
        condition_on: Sequence[str] | None = None,
        include_last_observations: bool = True,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Forward-simulate equations with fitted parameters and explicit intermediate inputs.

        Parameters
        ----------
        X : pandas.DataFrame or xarray.Dataset, optional
            Future inputs. Omit for in-sample posterior predictive draws.
        condition_on : sequence of str, optional
            Intermediate equation names or observation columns to hold fixed.
            The default holds all observed intermediates fixed; an empty tuple
            generates all of them. This policy does not update the parameter
            posterior using future intermediate observations.
        include_last_observations : bool, optional
            Prepend fitted observed history for adstock and custom temporal terms,
            then remove those rows from returned predictions.
            Enabled by default for supplied future X.
        **kwargs : Any
            Arguments for ``pymc.sample_posterior_predictive`` such as
            ``random_seed``, ``progressbar``, and ``var_names``. The workflow owns
            sampling/freeze policy and returns an xarray Dataset.

        Returns
        -------
        xarray.Dataset
            Predictive draws keyed by equation variable names. The default
            outcome is returned in original target units; custom outcomes and
            intermediate variables remain on their declared scales. The fitted
            model, observations, and posterior are not mutated.
        """
        self._check_fitted()
        condition = self._condition_names(condition_on)
        ds = self._training_data if X is None else self._prediction_data(X)
        if ds is None or self.idata is None:
            raise RuntimeError("The model has not been fit.")
        history_length = 0
        if X is not None and include_last_observations:
            ds, history_length = self._with_history(ds, condition)
        model, context = self._new_model(
            ds, prediction=True, condition_on=condition, history_length=history_length
        )
        managed = {
            "model",
            "sample_vars",
            "freeze_vars",
            "return_inferencedata",
            "extend_inferencedata",
            "predictions",
        }
        if conflicts := managed & kwargs.keys():
            raise ValueError(
                f"Prediction manages these arguments: {sorted(conflicts)}."
            )
        outputs = kwargs.pop("var_names", list(context.equation_names.values()))
        if isinstance(outputs, str):
            outputs = [outputs]
        if any(name not in model.named_vars for name in outputs):
            raise ValueError(
                "Every requested prediction variable must exist in the prediction model."
            )
        posterior = self.idata["posterior"].dataset
        resample = []
        for identity, equation in context.equations.items():
            binding = context.binding(equation)
            if binding.observed is not None or "date" in binding.dims:
                resample.append(context.equation_names[identity])
        keep = [
            name
            for name in self._parameter_names
            if name in posterior and name in model.named_vars and name not in resample
        ]
        for name in keep:
            if "date" in posterior[name].dims:
                dates = ds.get_index("date")
                fitted_dates = posterior.get_index("date")
                if not dates.isin(fitted_dates).all():
                    raise ValueError(
                        f"Parameter {name!r} needs an Equation to predict new dates."
                    )
                if not dates.equals(fitted_dates):
                    posterior = posterior.sel(date=dates)
                break
        result = pm.sample_posterior_predictive(
            posterior[keep],
            model=model,
            var_names=outputs,
            sample_vars=resample,
            freeze_vars=keep,
            predictions=True,
            **kwargs,
        )["predictions"].to_dataset()
        if history_length and "date" in result.dims:
            result = result.isel(date=slice(history_length, None))
        root_name = context.equation_names[id(self.y)]
        if self.y is self._default_y and root_name in result:
            result[root_name] = result[root_name] * self.scalers["target_scale"]
        return result
