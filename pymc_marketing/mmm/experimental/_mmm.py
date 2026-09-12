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
"""Joint inference and forward prediction for explicit observed equations."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import Any

import pandas as pd
import pymc as pm
import xarray as xr

from pymc_marketing.mmm.experimental._data import validate_dataset
from pymc_marketing.mmm.experimental._graph import (
    Binding,
    BuildContext,
    Equation,
    specification_key,
    walk,
)


class MMM:
    """Fit and forecast a graph of observed equations on one labeled dataset.

    Parameters
    ----------
    *equations : Equation
        Observed equations forming one joint model. Each names its observation
        variable through ``observed``; ``name`` defaults to that variable.
        Terms shared between equations are identified by object identity.

    Notes
    -----
    Data enter only through ``fit`` and ``sample_posterior_predictive`` as an
    ``xarray.Dataset`` whose every dimension carries unique labels. ``Data(name)``
    terms reference dataset variables; nothing is scaled, centered, or expanded.
    Each parameter and input keeps its own dimensions, and equation outputs take
    the dimensions of their observations.

    Prediction may change the ``date`` axis. Every other fitted coordinate must keep
    exactly the same labels, in any order; coordinates used only by omitted outputs
    are restored from fitting. Temporal terms declare ``required_history`` as a
    nonnegative integer, and forecasts prepend that many training rows when the
    future dates immediately follow training at the fitted cadence.
    Unconditioned observed equations and ``date``-indexed latent equations are
    regenerated; all other free parameters are frozen to posterior draws.

    Modifying the equations, their priors, the exposed model, or the posterior after
    fitting is unsupported; call ``fit`` again instead.

    Examples
    --------
    .. code-block:: python

        from pymc_extras.prior import Prior
        from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
        from pymc_marketing.mmm.experimental import (
            MMM,
            Data,
            Equation,
            MediaTransform,
        )
        from pymc_marketing.terms import Intercept, Transform

        response = MediaTransform(
            Data("spend"),
            GeometricAdstock(l_max=8),
            LogisticSaturation(priors={"beta": Prior("HalfNormal", dims="channel")}),
        )
        sales = Equation(
            observed="sales",
            mu=Intercept(prior=Prior("Normal", sigma=2))
            + Transform(response, lambda value: value.sum(dim="channel")),
            likelihood=Prior("Normal", sigma=Prior("HalfNormal", sigma=2)),
        )
        mmm = MMM(sales)
        # train: xr.Dataset with spend(date, channel) and sales(date)
        # mmm.fit(train)
        # mmm.sample_posterior_predictive(future)
    """

    def __init__(self, *equations: Equation) -> None:
        self._roots = tuple({id(root): root for root in equations}.values())
        self._validate_roots()
        self.model: pm.Model | None = None
        self.idata: xr.DataTree | None = None
        self._context: BuildContext | None = None
        self._training_data: xr.Dataset | None = None
        self._bindings: dict[int, Binding] = {}
        self._fitted_key: Any = None

    @property
    def equations(self) -> tuple[Equation, ...]:
        """Return the distinct observed equations in declaration order."""
        return self._roots

    def _validate_roots(self) -> None:
        if not self._roots or any(
            not isinstance(root, Equation) for root in self._roots
        ):
            raise TypeError("MMM requires one or more Equation instances.")
        if any(root.observed is None for root in self._roots):
            raise ValueError("Each equation given to MMM must name its observations.")
        names = [root.name or root.observed for root in self._roots]
        if len(set(names)) != len(names):
            raise ValueError("Equation names must be distinct.")

    def _key(self) -> Any:
        return specification_key(self._roots), tuple(
            id(node) for node in walk(self._roots)
        )

    def _new_model(
        self,
        data: xr.Dataset,
        *,
        prediction: bool = False,
        condition_on: Sequence[str] = (),
        history_length: int = 0,
    ) -> tuple[pm.Model, BuildContext]:
        model = pm.Model(coords={dim: data.coords[dim].values for dim in data.dims})
        with model:
            context = BuildContext(
                data,
                bindings=self._bindings if prediction else None,
                prediction=prediction,
                condition_on=condition_on,
                history_length=history_length,
            )
            for root in self._roots:
                context.build(root)
        return model, context

    def build_model(self, data: xr.Dataset) -> pm.Model:
        """Build a fresh joint model, invalidating any earlier fit even if building fails.

        Parameters
        ----------
        data : xarray.Dataset
            Inputs and observations for every equation, with labeled dimensions.

        Returns
        -------
        pymc.Model
            The joint model, also stored as ``model``.
        """
        self.model = self.idata = self._context = self._training_data = None
        self._bindings = {}
        self._fitted_key = None
        self._validate_roots()
        data = validate_dataset(data).copy(deep=True)
        for node in walk(self._roots):
            if (
                isinstance(node, Equation)
                and node.observed is not None
                and node.observed not in data
            ):
                raise ValueError(
                    f"Training data are missing observations {node.observed!r}."
                )
        model, context = self._new_model(data)
        self.model, self._context, self._training_data = model, context, data
        self._bindings = {
            identity: context.binding(equation)
            for identity, equation in context.equations.items()
        }
        return model

    def fit(self, data: xr.Dataset, **kwargs: Any) -> xr.DataTree:
        """Build from this call's data and sample the joint posterior.

        Parameters
        ----------
        data : xarray.Dataset
            Inputs and observations for every equation.
        **kwargs : Any
            Arguments for ``pymc.sample``; the model is managed here.

        Returns
        -------
        xarray.DataTree
            Posterior and sampler diagnostics, also stored as ``idata``.
        """
        model = self.build_model(data)
        if "model" in kwargs or kwargs.get("return_inferencedata") is False:
            raise ValueError(
                "fit manages the model and requires return_inferencedata=True."
            )
        kwargs["return_inferencedata"] = True
        key = self._key()
        idata = pm.sample(model=model, **kwargs)
        self.idata, self._fitted_key = idata, key
        return idata

    def _check_fitted(self) -> None:
        if self.idata is None or self._context is None or self._training_data is None:
            raise RuntimeError("Call fit before posterior prediction.")
        if self.model is not self._context.model or self._fitted_key != self._key():
            raise RuntimeError(
                "The fitted model specification changed; call fit again before prediction."
            )

    def _condition_names(self, condition_on: Sequence[str]) -> tuple[str, ...]:
        if isinstance(condition_on, str) or not isinstance(condition_on, Sequence):
            raise TypeError(
                "condition_on must be a sequence of equation or observation names."
            )
        observed = [
            binding
            for binding in self._bindings.values()
            if binding.observed is not None
        ]
        resolved = []
        for selector in condition_on:
            if not isinstance(selector, str):
                raise TypeError("condition_on entries must be strings.")
            matches = [
                binding.name
                for binding in observed
                if selector in (binding.name, binding.observed)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Unknown or ambiguous observed equation: {selector!r}."
                )
            resolved.append(matches[0])
        return tuple(dict.fromkeys(resolved))

    def _prediction_data(self, data: xr.Dataset) -> xr.Dataset:
        if not isinstance(data, xr.Dataset):
            raise TypeError("Data must be an xarray.Dataset with labeled variables.")
        if self._context is None:
            raise RuntimeError("Call fit before posterior prediction.")
        fitted = {
            dim: pd.Index(labels, name=dim)
            for dim, labels in self._context.model.coords.items()
            if dim != "date" and labels is not None
        }
        for dim, labels in fitted.items():
            if dim in data.coords and data.coords[dim].dims != (dim,):
                raise ValueError(
                    f"Prediction coordinate {dim!r} must label its own dimension."
                )
            if dim not in data.coords:
                if dim in data.sizes:
                    raise ValueError(
                        f"Prediction arrays using {dim!r} must provide coordinate labels."
                    )
                data = data.assign_coords({dim: labels})
        data = validate_dataset(data)
        for dim, old in fitted.items():
            new = data.get_index(dim)
            if len(old) != len(new) or not old.isin(new).all():
                raise ValueError(f"Prediction labels for {dim!r} must match training.")
            if not old.equals(new):
                data = data.reindex({dim: old})
        return data

    def _required_history(self, condition_on: Sequence[str]) -> int:
        stopped = {
            identity
            for identity, binding in self._bindings.items()
            if binding.name in condition_on
        }
        required = 0
        for node in walk(self._roots, stop=lambda node: id(node) in stopped):
            if id(node) in stopped:
                continue
            lookback = getattr(node, "required_history", 0)
            if (
                isinstance(lookback, bool)
                or not isinstance(lookback, Integral)
                or lookback < 0
            ):
                raise ValueError("Term required_history must be a nonnegative integer.")
            required = max(required, int(lookback))
        return required

    def _with_history(
        self, data: xr.Dataset, condition_on: Sequence[str]
    ) -> tuple[xr.Dataset, int]:
        required = self._required_history(condition_on)
        if not required:
            return data, 0
        training = self._training_data
        if training is None:
            raise RuntimeError("Call fit before posterior prediction.")
        if "date" not in training.dims or "date" not in data.dims:
            raise ValueError(
                "Temporal history requires a date dimension in training and prediction data."
            )
        old_dates = pd.DatetimeIndex(training["date"].values)
        new_dates = pd.DatetimeIndex(data["date"].values)
        if len(old_dates) < 2:
            raise ValueError(
                "At least two training dates are needed to infer a prediction cadence."
            )
        frequency = (
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
                for name, value in history.data_vars.items()
                if "date" not in value.dims
            ]
        )
        combined = xr.concat(
            [history, data],
            dim="date",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            join="exact",
        )
        return combined, count

    def sample_posterior_predictive(
        self,
        data: xr.Dataset,
        *,
        var_names: str | Sequence[str] | None = None,
        condition_on: Sequence[str] = (),
        include_last_observations: bool = True,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Forward-simulate the fitted equations on explicit inputs.

        Parameters
        ----------
        data : xarray.Dataset
            Prediction inputs. Observation variables of generated equations may be omitted.
        var_names : str or sequence of str, optional
            Model variables to return, including deterministics. Defaults to the equation names.
        condition_on : sequence of str, default ()
            Equation or observation names whose supplied observations are held fixed;
            every other observed equation is generated.
        include_last_observations : bool, default True
            Prepend the training history required by reachable temporal terms and
            drop those rows from the returned draws.
        **kwargs : Any
            Arguments for ``pymc.sample_posterior_predictive`` such as ``random_seed``.

        Returns
        -------
        xarray.Dataset
            Predictive draws in declared units for the supplied dates only.
            The fitted model, training data, and posterior are not mutated.
        """
        self._check_fitted()
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
        condition = self._condition_names(condition_on)
        data = self._prediction_data(data)
        history_length = 0
        if include_last_observations:
            data, history_length = self._with_history(data, condition)
        model, context = self._new_model(
            data, prediction=True, condition_on=condition, history_length=history_length
        )
        outputs = (
            [context.equation_names[id(root)] for root in self._roots]
            if var_names is None
            else [var_names]
            if isinstance(var_names, str)
            else list(var_names)
        )
        if any(name not in model.named_vars for name in outputs):
            raise ValueError(
                "Every requested prediction variable must exist in the prediction model."
            )
        resample = [
            context.equation_names[identity]
            for identity, equation in context.equations.items()
            if context.binding(equation).name not in condition
            and (
                context.binding(equation).observed is not None
                or "date" in context.binding(equation).dims
            )
        ]
        keep = [str(rv.name) for rv in model.free_RVs if rv.name not in resample]
        if self.idata is None:
            raise RuntimeError("Call fit before posterior prediction.")
        posterior = self.idata["posterior"].to_dataset()
        if missing := set(keep).difference(posterior.data_vars):
            raise ValueError(
                f"Fitted parameters are absent from the posterior: {sorted(missing)}. Refit including them."
            )
        for name in keep:
            if "date" in posterior[name].dims:
                dates = data.get_index("date")
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
        return result
