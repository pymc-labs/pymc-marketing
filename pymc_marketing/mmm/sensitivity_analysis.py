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

"""Counterfactual sweeps for Marketing Mix Models (MMM).

Quick start
-----------

SensitivityAnalysis can work with any PyMC model whose response depends on an
input `pm.Data` tensor. The functional relationship between inputs and the
response can be arbitrary (e.g., adstock, saturation, splines, custom
PyTensor ops). You only need to provide:

- the name of the input data variable (`var_input`, e.g., `"channel_data"`),
- the name of the deterministic/response variable you want to analyze
  (`var_names`, e.g., `"channel_contribution"`),
- a grid of `sweep_values`, and the sweep type.

Example
-------
The example below uses an arbitrary saturation transformation, but any
PyTensor-compatible transformation will work the same way.

.. code-block:: python

    import numpy as np
    import pymc as pm
    import pymc.dims as pmd
    from pymc_marketing.mmm.sensitivity_analysis import SensitivityAnalysis


    def saturation(x, alpha, lam):
        return (alpha * x) / (x + lam)


    coords = {"date": np.arange(52), "channel": list("abcde")}
    with pm.Model(coords=coords) as m:
        X = pmd.Data(
            "channel_data",
            np.random.gamma(13, 12, size=(52, 5)),
            dims=("date", "channel"),
        )
        alpha = pmd.Gamma("alpha", 1, 1, dims="channel")
        lam = pmd.Gamma("lam", 1, 1, dims="channel")
        contrib = pmd.Deterministic(
            "channel_contribution", saturation(X, alpha, lam), dims=("date", "channel")
        )
        mu = pmd.Normal("intercept", 0.0, 1.0) + contrib.sum("channel")
        pmd.Normal("likelihood", mu=mu, sigma=1.0, dims="date")
        idata = pm.sample(1000, tune=500, chains=2, target_accept=0.95, random_seed=42)

    sa = SensitivityAnalysis(m, idata)
    sweeps = np.linspace(0.2, 3.0, 8)
    # Response summed over `date`; shape: (sample, sweep, channel)
    result = sa.run_sweep(
        sweep_values=sweeps,
        var_input="channel_data",
        var_names="channel_contribution",
        sweep_type="multiplicative",
    )
    # Marginal effects: the derivative of the response with respect to the sweep
    # grid.
    marginal_effects = sa.compute_marginal_effects(result)

Notes
-----
- Arbitrary models: As long as the response graph depends (directly or indirectly)
  on the `pm.Data` provided via `var_input`, and you pass the name of the
  deterministic/response via `var_names`, the class evaluates the response graph
  across the sweep grid automatically.
- Multi-dimensional responses: If the response has dims like
  `(date, country, channel)`, the output shape is
  `(sample, sweep, country, channel)`.
- Sweep types: `"multiplicative"`, `"additive"`, and `"absolute"` are supported.
- To persist results, pass `extend_idata=True` to `run_sweep` to store them under
  `idata.sensitivity_analysis`.  `compute_marginal_effects(..., extend_idata=True)`
  adds to that group, so it needs an earlier `run_sweep(..., extend_idata=True)`.
"""

import math
import warnings
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import pytensor.tensor as pt
import pytensor.xtensor as ptx
import xarray as xr
from pymc import Model
from pytensor import function
from pytensor.graph.basic import Variable
from pytensor.graph.traversal import ancestors
from pytensor.xtensor.type import XTensorVariable, as_xtensor
from pytensor.xtensor.vectorization import vectorize_graph
from xarray import DataArray

from pymc_marketing.names.mmm import Dim, Group, Var
from pymc_marketing.pytensor_utils import (
    SharedPosterior,
    _raise_for_missing_posterior_variables,
    extract_response_distribution,
    needed_posterior_names,
    reorder_to_model_coords,
)

_SWEEP_COMBINATIONS: dict[
    str, Callable[[XTensorVariable, XTensorVariable], XTensorVariable]
] = {
    "multiplicative": lambda sweep, base_value: sweep * base_value,
    "additive": lambda sweep, base_value: sweep + base_value,
    "absolute": lambda sweep, base_value: sweep.broadcast_like(base_value),
}
"""The swept input of each sweep type, from the sweep grid and the model input."""


def _compose_sweep(
    response: XTensorVariable,
    base_value: XTensorVariable,
    sweep_type: str,
    mask_shape: tuple[int, ...] | None,
) -> tuple[list[Variable], XTensorVariable]:
    """Compose the symbolic inputs and the graph of one sweep.

    The sweep grid, and the response mask when there is one, are inputs of the
    compiled function.

    Parameters
    ----------
    response : XTensorVariable
        The extracted response graph, with a leading ``sample`` dimension.
    base_value : XTensorVariable
        The model input the sweep replaces.
    sweep_type : str
        A key of ``_SWEEP_COMBINATIONS``; the caller checks it.
    mask_shape : tuple of int or None
        Shape of the response mask, or ``None`` for a sweep without one.

    Returns
    -------
    list of Variable
        The inputs of the compiled sweep: the sweep grid and, with a mask, the
        mask.
    XTensorVariable
        The response summed over ``date``, transposed to ``sample, sweep, ...``.

    Raises
    ------
    ValueError
        If the response graph does not read ``base_value``.
    """
    sweep = ptx.xtensor("sweep", dims=(Dim.SWEEP,))
    inputs: list[Variable] = [sweep]
    if mask_shape is not None:
        mask = pt.tensor("mask", shape=mask_shape, dtype=bool)
        # The mask is broadcast over the whole response, its sample batch
        # included, because a set_subtensor with slices cannot be vectorized.
        mask_bcast, _ = pt.broadcast_arrays(mask, response.values)
        response = as_xtensor(response.values[~mask_bcast].set(0), dims=response.dims)
        inputs.append(mask)

    sweep_combinations = _SWEEP_COMBINATIONS[sweep_type](sweep, base_value)

    # Vectorizing over an input the response graph does not read fails inside
    # PyTensor with a message about a missing dimension, so the case is named
    # here.  The graph is the conditioned one: a free variable or a frozen
    # deterministic is its draws there, and what it reads is no longer in it.
    if not any(variable is base_value for variable in ancestors([response])):
        raise ValueError(
            f"The response graph does not read {base_value.name!r} once the "
            "posterior draws are bound, so sweeping it changes nothing. var_input "
            "must be a pm.Data variable that the response reads outside its "
            "frozen deterministics."
        )
    sweep_response = vectorize_graph(response, replace={base_value: sweep_combinations})
    return inputs, sweep_response.sum(dim=Dim.DATE).transpose("sample", Dim.SWEEP, ...)


class SensitivityAnalysis:
    """Run counterfactual sweeps over a fitted model's response.

    Every call extracts the response conditioned on the posterior draws, composes
    the sweep graph and compiles its PyTensor function; nothing is kept between
    calls.  The draws are bound through a
    :class:`~pymc_marketing.pytensor_utils.SharedPosterior`, so they are shared
    variables of the compiled sweep, not constants welded into it.  That keeps a
    repeat call cheap: PyTensor's numba backend caches the compiled code of a
    graph that holds no large constants, so a sweep is JIT-compiled once and
    loaded afterwards, in later calls and in later processes.

    ``dims`` is accepted for backwards compatibility and ignored: the response's
    own dims decide the output axes.

    Parameters
    ----------
    pymc_model : Model
        The PyMC model whose response is swept.
    idata : xr.DataTree
        The inference data holding the model's posterior.
    dims : tuple of str, default ()
        Accepted for backwards compatibility and ignored.
    frozen_deterministics : sequence of str, default ()
        Deterministics taken from the posterior rather than recomputed, such as a
        latent path over the dates the model was fitted on. A sweep keeps those
        dates, so their posterior draws hold in every sweep.
        :attr:`MMM.frozen_deterministics` lists an MMM's.
    """

    def __init__(
        self,
        pymc_model: Model,
        idata: xr.DataTree,
        dims: tuple[str, ...] = (),
        frozen_deterministics: Sequence[str] = (),
    ) -> None:
        """Initialize the analysis; the parameters are documented on the class."""
        self.model = pymc_model
        self.idata = idata
        self.dims = dims
        self.frozen_deterministics = tuple(frozen_deterministics)

    def _draw_indices_for_percentage(
        self, posterior_sample_fraction: float
    ) -> slice | np.ndarray:
        if not (0 < posterior_sample_fraction <= 1):
            raise ValueError(
                "'posterior_sample_fraction' must be in the interval (0, 1]."
            )

        draws = int(self.idata.posterior.sizes.get("draw", 0))
        if draws <= 1:
            return slice(None)

        retained_fraction = posterior_sample_fraction**2
        if math.isclose(retained_fraction, 1.0, rel_tol=1e-9):
            return slice(None)

        target_draws = max(1, round(draws * retained_fraction))
        if target_draws >= draws:
            return slice(None)

        indices = np.linspace(0, draws - 1, target_draws, dtype=int)
        indices = np.unique(indices)
        if indices.size == 0:
            return np.array([0], dtype=int)
        indices[-1] = draws - 1
        return indices

    def _posterior_draws(
        self, response_variable: str, posterior_sample_fraction: float
    ) -> xr.Dataset:
        """Return the draws the response reads, thinned and in the model's order.

        Only the posterior variables the response reads are kept, so a call
        neither stacks nor loads the rest of the posterior, however many
        deterministics it carries.  The draws are realigned to the model's coords
        by label, so a label-permuted posterior gives the numbers the model's own
        label order gives; renamed, unlabeled or duplicated labels keep their
        order.  The labels are then dropped, because the draws are bound by
        position.

        Parameters
        ----------
        response_variable : str
            The response the sweep evaluates.
        posterior_sample_fraction : float
            The draw-thinning control of :meth:`run_sweep`.

        Returns
        -------
        xr.Dataset
            The retained draws of the variables the response reads.

        Raises
        ------
        ValueError
            If ``posterior_sample_fraction`` is outside ``(0, 1]``, or the
            response reads no posterior variable.
        KeyError
            If the posterior lacks a variable the response reads.
        """
        draw_selection = self._draw_indices_for_percentage(posterior_sample_fraction)
        read_names = needed_posterior_names(
            self.model, [self.model[response_variable]], self.frozen_deterministics
        )
        if not read_names:
            raise ValueError(
                f"{response_variable!r} reads no posterior variable, so there are "
                "no draws to sweep over."
            )
        posterior = self.idata.posterior.to_dataset()
        _raise_for_missing_posterior_variables(read_names, posterior)
        draws = reorder_to_model_coords(
            posterior[read_names].isel(draw=draw_selection), self.model.coords
        )
        # The per-call SharedPosterior is never rebound, so it has no use for the
        # labels, and without them a dim whose labels repeat stays usable.
        return draws.drop_vars(
            [name for name in draws.indexes if name not in ("chain", "draw")]
        )

    def _length_mismatches(self, draws: xr.Dataset) -> list[str]:
        """Describe each read variable whose length on a dim is not the model's.

        Parameters
        ----------
        draws : xr.Dataset
            The draws :meth:`_posterior_draws` returned.

        Returns
        -------
        list of str
            One description per posterior variable and dim whose length differs
            from the number of coordinates the model has for that dim; empty
            when every length fits.
        """
        mismatches = []
        for name, variable in draws.data_vars.items():
            for dim, length in variable.sizes.items():
                coords = self.model.coords.get(dim)
                if coords is not None and len(coords) != length:
                    mismatches.append(
                        f"{name!r} has length {length} on {dim!r}, the model has "
                        f"{len(coords)}"
                    )
        return mismatches

    def _prepare_response_mask(
        self,
        response_mask: xr.DataArray | np.ndarray,
        response_dims: tuple[str, ...],
        *,
        var_names: str,
    ) -> np.ndarray:
        """Validate and align a response mask with the deterministic dims."""
        if isinstance(response_mask, xr.DataArray):
            mask_xr = response_mask
            if response_dims:
                missing_dims = set(response_dims) - set(mask_xr.dims)
                if missing_dims:
                    raise ValueError(
                        "response_mask is missing required dims for "
                        f"{var_names!r}: {sorted(missing_dims)}"
                    )
                try:
                    mask_xr = mask_xr.transpose(*response_dims)
                except ValueError as err:
                    raise ValueError(
                        "response_mask dims must match the order of the "
                        f"{var_names!r} deterministic"
                    ) from err
            mask_values = mask_xr.to_numpy()
        else:
            mask_values = np.asarray(response_mask)

        if mask_values.dtype != bool:
            if np.issubdtype(mask_values.dtype, np.integer) or np.issubdtype(
                mask_values.dtype, np.floating
            ):
                mask_values = mask_values.astype(bool)
            else:
                raise TypeError("response_mask must be boolean or castable to boolean")

        if response_dims and mask_values.ndim != len(response_dims):
            raise ValueError(
                "response_mask must have the same number of dims as the "
                f"{var_names!r} deterministic (expected {len(response_dims)}, "
                f"got {mask_values.ndim})"
            )

        expected_shape: list[int] = []
        for axis, dim_name in enumerate(response_dims):
            if dim_name in self.model.coords:
                expected_shape.append(len(self.model.coords[dim_name]))
            else:
                expected_shape.append(mask_values.shape[axis])

        if expected_shape and tuple(expected_shape) != mask_values.shape:
            raise ValueError(
                "response_mask shape does not match deterministic dims: "
                f"expected {tuple(expected_shape)}, got {mask_values.shape}"
            )

        return mask_values

    def _add_to_idata(self, result: xr.DataArray) -> None:
        """Add the result to the idata.

        If the 'sensitivity_analysis' group already exists, emit a warning and only
        update/overwrite the 'x' variable to preserve any additional stored results
        (e.g., 'uplift_curve', 'marginal_effects').
        """
        dataset = xr.Dataset({"x": result})
        if hasattr(self.idata, Group.SENSITIVITY_ANALYSIS):
            warnings.warn(
                "'sensitivity_analysis' group already exists; updating variable 'x' with new results.",
                UserWarning,
                stacklevel=2,
            )
            existing = self.idata.sensitivity_analysis  # type: ignore[attr-defined]
            if isinstance(existing, xr.Dataset):
                existing["x"] = result
                self.idata.sensitivity_analysis = existing  # type: ignore[attr-defined]
            else:
                # Legacy case: replace DataArray with Dataset for consistency
                self.idata.sensitivity_analysis = dataset  # type: ignore[attr-defined]
        else:
            self.idata[f"/{Group.SENSITIVITY_ANALYSIS}"] = dataset

    def _sweep_dataarray(
        self,
        dims: tuple[str, ...],
        result: np.ndarray,
        sweep_values: np.ndarray,
    ) -> DataArray:
        """Label the compiled output with the sweep grid and the model coords."""
        coords = {}
        for dim, shape in zip(dims, result.shape, strict=True):
            if dim == Dim.SWEEP:
                dim_coords = np.asarray(sweep_values)
            else:
                dim_coords = self.model.coords.get(dim)
                if dim_coords is None:
                    dim_coords = np.arange(shape)
                else:
                    dim_coords = np.asarray(dim_coords)
            coords[dim] = dim_coords

        return DataArray(result, dims=dims, coords=coords)

    def run_sweep(
        self,
        var_input: str,
        sweep_values: np.ndarray,
        *,
        var_names: str = Var.CHANNEL_CONTRIBUTION,
        sweep_type: Literal[
            "multiplicative", "additive", "absolute"
        ] = "multiplicative",
        posterior_sample_fraction: float = 1.0,
        response_mask: xr.DataArray | np.ndarray | None = None,
        extend_idata: bool = False,
    ) -> xr.DataArray | None:
        """
        Run sweeps by forward-evaluating the response graph (no Jacobian).

        Every call extracts the response conditioned on the posterior draws,
        composes the sweep graph and compiles it, so ``sweep_values``,
        ``response_mask`` and the posterior are inputs of that call rather than
        state kept between calls.

        Parameters
        ----------
        var_input : str
            Name of the pm.Data variable (e.g., "channel_data").
            Expected shape: ``(date, *dims, arbitrary_dim)`` that match var_input dims.
        sweep_values : np.ndarray
            Values to sweep over.
        var_names : str
            The deterministic variable of interest.  It must be a ``pymc.dims``
            variable with a ``date`` dimension, for example a
            ``pmd.Deterministic`` built with ``dims=(...)``.
        sweep_type : {"multiplicative","additive","absolute"}
            Type of sweep to apply.
        posterior_sample_fraction : float
            Posterior sampling control in (0, 1]. The retained fraction of draws is
            roughly ``posterior_sample_fraction ** 2`` (e.g., 0.1 keeps about 1% of
            the draws, while 1.0 keeps the full posterior).
        response_mask : xr.DataArray | np.ndarray | None
            Optional boolean mask with the same non-sample dims as ``var_names``.
            When provided, the mask zeroes out the response prior to evaluating
            the sweeps. Useful for focusing on specific coordinates of the
            deterministic response (e.g., single channels).

        Returns
        -------
        xarray.DataArray | None
            If extend_idata is False, returns an xarray.DataArray with shape
            ``(sample, sweep, *response_dims)``, where ``response_dims`` are the
            non-date dims of ``var_names`` in model order.  The response is
            summed over the `date` axis. If extend_idata is True, stores the
            result under `idata.sensitivity_analysis` and returns None.

        Raises
        ------
        KeyError
            If ``var_names`` or ``var_input`` is not in the model, or ``idata``
            lacks a posterior variable the response reads.
        ValueError
            If ``posterior_sample_fraction`` is outside ``(0, 1]``.
            If ``var_names`` is not a ``pymc.dims`` variable, has no ``date``
            dimension or reads no posterior variable.
            If the response does not read ``var_input`` once the posterior draws
            are bound: ``var_input`` must be a ``pm.Data`` variable that the
            response reads outside its frozen deterministics.
            If ``sweep_type`` is not one of the three supported types.
            If ``response_mask`` does not match the response's dimensions.
            If the sweep cannot be evaluated and a posterior variable's length
            on a dim differs from the model's, for example one with another
            number of channels.  The error names the variable and the dim, for
            a dim the model has coordinates for.
        TypeError
            If ``response_mask`` cannot be cast to boolean.

        Notes
        -----
        The posterior in ``idata`` may be replaced between calls, for example by
        assigning ``mmm.idata``.  A posterior stored in another order of the
        model's own labels is realigned by label; renamed, unlabeled or
        duplicated labels are bound by position.  The number of draws may
        change, and values are cast to the dtype the model expects.
        """
        response = self.model[var_names]
        response_dims = getattr(response, "dims", None)
        if response_dims is None:
            raise ValueError(
                f"{var_names!r} is not a pymc.dims variable. Sensitivity sweeps "
                "require a response with named dimensions, for example one built "
                "with pmd.Deterministic(..., dims=(...))."
            )
        response_dims = tuple(response_dims)
        if Dim.DATE not in response_dims:
            raise ValueError(
                f"{var_names!r} has no 'date' dimension. Sensitivity sweeps "
                "require a response with a 'date' dimension, for example one "
                "built with pmd.Deterministic(..., dims=(...))."
            )
        # The cheap checks come before the posterior is read and the graph built.
        base_value = self.model[var_input]
        if not isinstance(sweep_type, str) or sweep_type not in _SWEEP_COMBINATIONS:
            raise ValueError(f"Unknown sweep_type {sweep_type!r}")

        posterior = self._posterior_draws(var_names, posterior_sample_fraction)
        mask_values = (
            None
            if response_mask is None
            else self._prepare_response_mask(
                response_mask, response_dims, var_names=var_names
            )
        )
        # The draws are bound as shared variables, not constants: numba cannot
        # cache a function with large constant arrays, so a graph with the draws
        # baked in is compiled again on every call.
        response_graph = extract_response_distribution(
            pymc_model=self.model,
            idata=posterior,
            response_variable=var_names,
            frozen_deterministics=list(self.frozen_deterministics),
            shared_posterior=SharedPosterior(),
        )
        inputs, graph = _compose_sweep(
            response=response_graph,
            base_value=base_value,
            sweep_type=sweep_type,
            mask_shape=None if mask_values is None else mask_values.shape,
        )
        compiled = function(inputs, graph)
        arguments = (
            (sweep_values,) if mask_values is None else (sweep_values, mask_values)
        )
        try:
            result = compiled(*arguments)
        except Exception as err:
            # A posterior of another size than the model fails here, with a
            # message about the shape of an input of the compiled function.  It
            # is not refused before compiling: a response that reduces such a
            # dim before it meets the data evaluates.  So the error reports the
            # lengths that differ, without claiming they are the cause.
            mismatches = self._length_mismatches(posterior)
            if not mismatches:
                raise
            raise ValueError(
                "The sweep could not be evaluated. These posterior variables do "
                "not have the model's length: " + "; ".join(mismatches) + "."
            ) from err
        xr_result = self._sweep_dataarray(graph.dims, result, sweep_values)

        if extend_idata:
            self._add_to_idata(xr_result)
            return None
        else:
            return xr_result

    def compute_uplift_curve_respect_to_base(
        self,
        results: xr.DataArray,
        ref: float | int | xr.DataArray,
        extend_idata: bool = False,
    ) -> xr.DataArray:
        """Return uplift curves referenced to a baseline.

        Parameters
        ----------
        results
            Output from :meth:`run_sweep`, typically with dims ``("sample", "sweep", *extra_dims)``.
        ref
            Baseline to subtract from ``results``. Can be a scalar (float or int) or an
            :class:`xarray.DataArray` sharing a subset of the ``results`` dimensions (e.g. control,
            country). The coordinates for overlapping dimensions must match exactly so that the
            subtraction is well defined.
        extend_idata
            When ``True`` the uplift is also persisted under
            ``idata.sensitivity_analysis["uplift_curve"]``.
        """
        if isinstance(ref, xr.DataArray):
            ref_da = ref
        else:
            try:
                ref_da = xr.DataArray(ref)
            except Exception as err:  # pragma: no cover - defensive
                raise TypeError(
                    "ref must be a scalar or an xarray.DataArray broadcastable to the sweep results"
                ) from err

        extra_dims = set(ref_da.dims).difference(results.dims)
        if extra_dims:
            raise ValueError(
                "Reference array includes dimensions absent from the sweep results: "
                f"{sorted(extra_dims)}"
            )

        try:
            results_aligned, ref_aligned = xr.align(
                results, ref_da, join="exact", copy=False
            )
        except ValueError as err:  # pragma: no cover - coordinate mismatch
            raise ValueError(
                "Reference array coordinates must match the sweep results for shared dimensions."
            ) from err

        xr_result = results_aligned - ref_aligned
        if extend_idata:
            if not hasattr(self.idata, Group.SENSITIVITY_ANALYSIS):
                raise ValueError(
                    "No sensitivity analysis results found in 'self.idata'. "
                    "Run 'mmm.sensitivity.run_sweep()' first."
                )
            self.idata.sensitivity_analysis["uplift_curve"] = xr_result
        return xr_result

    def compute_marginal_effects(
        self, results: xr.DataArray, extend_idata: bool = False
    ) -> xr.DataArray:
        """Return marginal effects from sweep results.

        Parameters
        ----------
        results : xr.DataArray
            The output from ``run_sweep``.
        extend_idata : bool, default False
            When ``True``, the effects are also stored under
            ``idata.sensitivity_analysis["marginal_effects"]``.

        Returns
        -------
        xr.DataArray
            The derivative of ``results`` with respect to the ``sweep`` grid.

        Raises
        ------
        ValueError
            If ``extend_idata`` is ``True`` and ``idata`` has no
            ``sensitivity_analysis`` group yet; ``run_sweep(...,
            extend_idata=True)`` creates it.
        """
        xr_result = results.differentiate(coord=Dim.SWEEP)
        if extend_idata:
            if not hasattr(self.idata, Group.SENSITIVITY_ANALYSIS):
                raise ValueError(
                    "No sensitivity analysis results found in 'self.idata'. "
                    "Run 'mmm.sensitivity.run_sweep()' first."
                )
            self.idata.sensitivity_analysis["marginal_effects"] = xr_result
        return xr_result
