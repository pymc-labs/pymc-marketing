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
"""Joint inference, prediction, and persistence for explicit observed equations."""

from __future__ import annotations

import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import xarray as xr
import zarr

from pymc_marketing.mmm.experimental._data import validate_dataset
from pymc_marketing.mmm.experimental._graph import (
    Binding,
    BuildContext,
    Equation,
    specification_key,
    total_lookback,
    walk,
)
from pymc_marketing.mmm.experimental._serialize import spec_from_dict, spec_to_dict
from pymc_marketing.version import __version__

_LOGP_DRAWS = 3
_LOGP_RTOL = 1e-6


class GAM:
    """Fit, forecast, and save a Bayesian generalized additive model of observed equations.

    Each equation's mean is a sum of terms: parameters, linear effects, media
    response curves, seasonality, and any other ``ModelTerm``. Several equations
    with different likelihoods can share parameters in one joint model, and any
    distribution parameter, not only ``mu``, can be modeled. This makes the class a
    multi-response, distributional GAM.

    Parameters
    ----------
    *equations : Equation
        Observed equations forming one joint model. Each names its observation
        variable through ``observed``; ``name`` defaults to that variable.
        Terms shared between equations are identified by object identity.

    Notes
    -----
    Data enter only through ``fit``, ``sample_prior_predictive``, and
    ``sample_posterior_predictive`` as an ``xarray.Dataset`` whose every dimension
    carries unique labels. ``Data(name)`` terms reference dataset variables; nothing
    is scaled, centered, or expanded. Each parameter and input keeps its own
    dimensions, and equation outputs take the dimensions of their observations.

    Prediction may change the ``date`` axis. Every other fitted coordinate must keep
    exactly the same labels, in any order; coordinates used only by omitted outputs
    are restored from fitting. Temporal terms declare ``required_history`` as a
    nonnegative integer, and forecasts prepend the largest total along any
    dependency path in training rows when the future dates immediately follow
    training at the fitted cadence.
    Unconditioned observed equations and ``date``-indexed latent equations are
    regenerated; all other free parameters are frozen to posterior draws.

    Modifying the equations, their priors, the exposed model, or the posterior after
    fitting is unsupported; call ``fit`` again instead.

    Examples
    --------
    .. code-block:: python

        from pymc_extras.prior import Prior
        from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
        from pymc_marketing.mmm.experimental import GAM, Data, Equation
        from pymc_marketing.terms import Intercept

        response = (
            Data("spend")
            >> GeometricAdstock(l_max=8)
            >> LogisticSaturation(priors={"beta": Prior("HalfNormal", sigma=2)})
        )
        sales = Equation(
            observed="sales",
            mu=Intercept(prior=Prior("Normal", sigma=2)) + response.sum("channel"),
            likelihood=Prior("Normal", sigma=Prior("HalfNormal", sigma=2)),
        )
        gam = GAM(sales)
        # train: xr.Dataset with spend(date, channel) and sales(date)
        # prior = gam.sample_prior_predictive(train)
        # gam.fit(train)
        # gam.save("model.zarr")
        # gam = GAM.load("model.zarr")
        # gam.sample_posterior_predictive(future)
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
            raise TypeError("GAM requires one or more Equation instances.")
        if any(root.observed is None for root in self._roots):
            raise ValueError("Each equation given to GAM must name its observations.")
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

    def _training_model(
        self, data: xr.Dataset
    ) -> tuple[xr.Dataset, pm.Model, BuildContext]:
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
        return data, model, context

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
        data, model, context = self._training_model(data)
        self.model, self._context, self._training_data = model, context, data
        self._bindings = {
            identity: context.binding(equation)
            for identity, equation in context.equations.items()
        }
        return model

    def sample_prior_predictive(
        self,
        data: xr.Dataset,
        *,
        draws: int = 500,
        var_names: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> xr.DataTree:
        """Sample parameters and equations from the prior on training-shaped data.

        Parameters
        ----------
        data : xarray.Dataset
            Inputs and observations for every equation, as for ``fit``. The
            observations set each equation's dimensions and are returned for comparison.
        draws : int, default 500
            Number of prior draws.
        var_names : sequence of str, optional
            Variables to sample. Defaults to every parameter, named deterministic,
            and equation.
        **kwargs : Any
            Arguments for ``pymc.sample_prior_predictive`` such as ``random_seed``.

        Returns
        -------
        xarray.DataTree
            ``prior``, ``prior_predictive``, and ``observed_data`` groups. Nothing is
            stored on this object, and an existing fit is left untouched.
        """
        if conflicts := {"model", "return_inferencedata"} & kwargs.keys():
            raise ValueError(
                f"Prior predictive sampling manages these arguments: {sorted(conflicts)}."
            )
        _, model, _ = self._training_model(data)
        result = pm.sample_prior_predictive(
            draws=draws, model=model, var_names=var_names, **kwargs
        )
        if "constant_data" in result.children:
            result = result.drop_nodes("constant_data")
        return result

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

    def save(self, path: str | Path) -> None:
        """Save the fitted model as a Zarr store.

        Parameters
        ----------
        path : str or Path
            ``*.zarr`` for a directory store or ``*.zarr.zip`` for a single file.
            An existing store at ``path`` is replaced.

        Raises
        ------
        RuntimeError
            If the model is not fitted or its specification changed since fitting.
        ValueError
            If ``path`` does not end in ``.zarr`` or ``.zarr.zip``.
        SerializationError
            If a term cannot be saved, such as a ``Transform`` of a lambda. Register
            the function with ``pymc_extras.prior.register_tensor_transform`` instead.

        Notes
        -----
        The root ``zarr.json`` stores the equations under ``spec`` as a JSON node
        table in which shared terms appear once, the library ``versions``, and a
        ``logp_check`` of the joint log-density at a few posterior draws. The
        ``fit_data`` group holds the training dataset, followed by ``posterior``,
        ``sample_stats``, and ``observed_data``. The PyMC model itself is not
        stored: ``load`` rebuilds it from the equations and training data.
        """
        self._check_fitted()
        path, zipped = _store_path(path)
        idata = cast(xr.DataTree, self.idata)
        tree = xr.DataTree.from_dict(
            {
                "/fit_data": cast(xr.Dataset, self._training_data),
                **{
                    f"/{name}": node.to_dataset()
                    for name, node in idata.children.items()
                    if name != "constant_data"
                },
            }
        )
        tree.attrs = {
            "spec": spec_to_dict(self._roots),
            "versions": _versions(),
            "logp_check": _logp_check(
                cast(pm.Model, self.model), idata["posterior"].to_dataset()
            ),
        }
        if not zipped:
            tree.to_zarr(path, mode="w")
            return
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory) / "model.zarr"
            tree.to_zarr(store)
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
                for file in sorted(store.rglob("*")):
                    if file.is_file():
                        archive.write(file, file.relative_to(store).as_posix())

    @classmethod
    def load(cls, path: str | Path, *, check: bool = True) -> GAM:
        """Load a model written by ``save``, ready for prediction.

        Parameters
        ----------
        path : str or Path
            A ``*.zarr`` directory or ``*.zarr.zip`` file.
        check : bool, default True
            Recompute the joint log-density at the stored posterior draws and raise
            if the rebuilt model differs from the saved one, for example after a
            library upgrade changed a component.

        Returns
        -------
        GAM
            A fitted model. Posterior arrays load lazily when first used.

        Raises
        ------
        SerializationError
            If the file format is unknown or a saved term cannot be rebuilt.
        ValueError
            If ``check`` is true and the rebuilt model's log-density differs.
        """
        path, zipped = _store_path(path)
        tree = xr.open_datatree(
            zarr.storage.ZipStore(path, mode="r") if zipped else path, engine="zarr"
        )
        if "spec" not in tree.attrs or "fit_data" not in tree.children:
            raise ValueError(f"{str(path)!r} is not a model saved by GAM.save.")
        gam = cls(*spec_from_dict(tree.attrs["spec"]))
        model = gam.build_model(tree["fit_data"].to_dataset())
        gam.idata = xr.DataTree.from_dict(
            {
                f"/{name}": node.to_dataset()
                for name, node in tree.children.items()
                if name != "fit_data"
            }
        )
        gam._fitted_key = gam._key()
        if check:
            saved = tree.attrs["logp_check"]
            rebuilt = _joint_logp(
                model,
                gam.idata["posterior"].to_dataset(),
                saved["chain"],
                saved["draw"],
            )
            if not np.allclose(rebuilt, saved["logp"], rtol=_LOGP_RTOL, atol=0):
                raise ValueError(
                    "The rebuilt model does not reproduce the saved log-density "
                    f"(saved {saved['logp']}, rebuilt {rebuilt.tolist()}). Saved with "
                    f"{tree.attrs['versions']}, loading with {_versions()}. "
                    "Pass check=False to load anyway."
                )
        return gam

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
        return total_lookback(self._roots, stop=lambda node: id(node) in stopped)

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


def _store_path(path: str | Path) -> tuple[Path, bool]:
    path = Path(path)
    if path.name.endswith(".zarr.zip"):
        return path, True
    if path.suffix == ".zarr":
        return path, False
    raise ValueError("Use a '.zarr' directory or a '.zarr.zip' file.")


def _versions() -> dict[str, str]:
    return {
        "pymc_marketing": __version__,
        "pymc": pm.__version__,
        "pytensor": pytensor.__version__,
    }


def _joint_logp(
    model: pm.Model,
    posterior: xr.Dataset,
    chain: Sequence[int],
    draw: Sequence[int],
) -> np.ndarray:
    """Joint log-density, prior plus likelihood without Jacobians, at selected draws."""
    subset = (
        posterior.isel(
            chain=xr.DataArray(list(chain), dims="sample"),
            draw=xr.DataArray(list(draw), dims="sample"),
        )
        .drop_vars(["chain", "draw"], errors="ignore")
        .rename(sample="draw")
        .expand_dims(chain=[0])
        .assign_coords(draw=np.arange(len(draw)))
    )
    tree = xr.DataTree.from_dict({"posterior": subset})
    options: dict[str, Any] = {
        "model": model,
        "extend_inferencedata": False,
        "progressbar": False,
    }
    total: Any = 0
    for result in (
        pm.stats.compute_log_prior(tree, **options),
        pm.stats.compute_log_likelihood(tree, **options),
    ):
        for value in result.data_vars.values():
            total = total + value.sum(
                [dim for dim in value.dims if dim not in ("chain", "draw")]
            )
    return np.asarray(total.transpose("chain", "draw")).ravel()


def _logp_check(model: pm.Model, posterior: xr.Dataset) -> dict[str, list[Any]]:
    draws = posterior.sizes["draw"]
    total = posterior.sizes["chain"] * draws
    flat = np.unique(np.linspace(0, total - 1, num=_LOGP_DRAWS).round().astype(int))
    chain, draw = np.divmod(flat, draws)
    return {
        "chain": chain.tolist(),
        "draw": draw.tolist(),
        "logp": _joint_logp(model, posterior, chain.tolist(), draw.tolist()).tolist(),
    }
