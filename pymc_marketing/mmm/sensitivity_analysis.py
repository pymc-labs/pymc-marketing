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
from typing import Literal

import numpy as np
import xarray as xr
from pymc import Model
from pytensor import function
from xarray import DataArray

from pymc_marketing.mmm.leased_cache import Lease
from pymc_marketing.mmm.sweep_cache import SweepCache
from pymc_marketing.mmm.sweep_graph import (
    SWEEP_DIM,
    SweepCacheKey,
    SweepEntry,
    compose_sweep,
)
from pymc_marketing.pytensor_utils import (
    ModelCloneError,
    extract_response_distribution,
    needed_posterior_names,
    reorder_to_model_coords,
)


class SensitivityAnalysis:
    """Run counterfactual sweeps over a fitted model's response.

    The compiled-sweep cache belongs to the owning MMM when the instance comes
    from :attr:`~pymc_marketing.mmm.mmm.MMM.sensitivity`: every access shares
    one cache, one per model.
    A separately constructed instance keeps a private cache.  The cache key
    describes the sweep graph only; the source model is checked separately.
    Replacing a data buffer with ``pm.set_data`` rebuilds the affected entry;
    writing into a data buffer in place is not detected, the same as for the
    incrementality evaluator, and neither is a change to the source model's
    coords that leaves its buffers in place.  Rebuilding or refitting the model
    clears the cache an :class:`~pymc_marketing.mmm.mmm.MMM` owns; a private
    cache instead detects the new model and rebuilds its entries.

    ``dims`` is accepted for backwards compatibility and ignored: the response's
    own dims decide the output axes.
    """

    def __init__(
        self,
        pymc_model: Model,
        idata: xr.DataTree,
        dims: tuple[str, ...] = (),
        *,
        _cache: SweepCache | None = None,
    ):
        """Initialize the analysis.

        Parameters
        ----------
        pymc_model : Model
            The model whose response is swept.
        idata : xr.DataTree
            Fitted inference data holding the posterior draws.
        dims : tuple of str, default ()
            Accepted for backwards compatibility and ignored.  The response's
            own dims decide the output axes, so passing the model's extra dims
            here is harmless but unnecessary.
        _cache : SweepCache, optional
            Internal cache supplied by
            :attr:`~pymc_marketing.mmm.mmm.MMM.sensitivity`.  Without one
            the instance keeps a private cache, so repeated calls on the
            instance compile once.
        """
        self.model = pymc_model
        self.idata = idata
        self.dims = dims
        self._cache = SweepCache() if _cache is None else _cache

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

    def _extract_response_distribution(
        self, response_variable: str, idata: xr.DataTree
    ):
        """Extract the response graph conditioned on draws in the model's order.

        :class:`~pymc_marketing.pytensor_utils.ConditionedModel` realigns every
        posterior to the model's coords by label; the per-call path does the
        same here so a label-permuted posterior gives the cached path's
        numbers.  Renamed or unlabeled coords keep their given order and are
        read by position.
        """
        posterior = reorder_to_model_coords(
            idata["/posterior"].to_dataset(), self.model.coords
        )
        return extract_response_distribution(
            pymc_model=self.model,
            idata=posterior,
            response_variable=response_variable,
        )

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
        if hasattr(self.idata, "sensitivity_analysis"):
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
            self.idata["/sensitivity_analysis"] = dataset

    def _selected_idata(
        self, var_names: str, posterior_sample_fraction: float
    ) -> xr.DataTree:
        """Return the posterior the sweep reads, narrowed to the retained draws.

        A cache entry keeps the posterior it was built on, so handing it the
        whole tree would pin every group and every variable for as long as the
        entry lives.  Only the posterior variables the response graph reads are
        kept, and only the ``posterior`` group: the sweep reads nothing else.

        A response that reads no posterior variable -- a pure function of
        ``pm.Data`` -- keeps the whole tree, as it did before the narrowing,
        because an empty posterior has no ``chain`` or ``draw`` dims to select.
        So does a posterior that lacks a variable the response reads, so the
        conditioned model reports the missing names itself.
        """
        draw_selection = self._draw_indices_for_percentage(posterior_sample_fraction)
        read_names = needed_posterior_names(self.model, [self.model[var_names]])
        posterior = self.idata.posterior.to_dataset()
        if not read_names or any(name not in posterior for name in read_names):
            return self.idata.isel(draw=draw_selection)
        # Selecting names on a DataTree raises, so narrow the dataset.
        narrowed = xr.Dataset({name: posterior[name] for name in read_names})
        return xr.DataTree.from_dict({"/posterior": narrowed.isel(draw=draw_selection)})

    def _sweep_key(
        self,
        var_input: str,
        var_names: str,
        sweep_type: str,
        mask_values: np.ndarray | None,
    ) -> SweepCacheKey:
        """Return the structure a compiled sweep is cached under."""
        return SweepCacheKey(
            var_input=var_input,
            var_names=var_names,
            sweep_type=sweep_type,
            mask_shape=None if mask_values is None else tuple(mask_values.shape),
        )

    def _run_sweep_cached(
        self,
        key: SweepCacheKey,
        idata: xr.DataTree,
        sweep_values: np.ndarray,
        mask_values: np.ndarray | None,
    ) -> tuple[tuple[str, ...], np.ndarray]:
        """Evaluate the sweep through the compiled sweep cache.

        The entry is leased for the duration of the call, so concurrent callers
        on the same key build their own instead of rebinding each other's
        draws.  Every call rebinds the entry to this call's draws, which covers
        ``posterior_sample_fraction`` and a replaced ``mmm.idata``.  An entry
        that cannot be rebound is replaced by one built for this call, and the
        entry it replaces stays out of the cache until the replacement has
        evaluated: only then is the replacement cached and the old entry
        dropped.  A replacement that fails is dropped while the old entry goes
        back, so one posterior the model cannot evaluate cannot empty the
        cache.

        A model :class:`~pymc_marketing.pytensor_utils.ConditionedModel` cannot
        clone falls back to :meth:`_run_sweep_per_call`.  Only the clone step
        falls back, so an error that is not a layout mismatch raised while
        binding or evaluating an entry propagates.  An entry built for this
        call that fails to evaluate is dropped before the error propagates; a
        cached entry that fails stays cached, because the failure is about this
        call's arguments.
        """
        lease, retired = self._lease_bound_to(key, idata)
        if lease is None:
            return self._run_sweep_per_call(key, idata, sweep_values, mask_values)
        try:
            result = (
                lease.value.output_dims,
                lease.value.evaluate(sweep_values, mask_values),
            )
        except BaseException:
            if retired is not None:
                retired.release()
            if not lease.cached:
                lease.discard()
            raise
        else:
            if retired is not None:
                retired.discard()
            return result
        finally:
            lease.release()
            if retired is not None:
                retired.release()

    def _lease(
        self, key: SweepCacheKey, idata: xr.DataTree
    ) -> Lease[SweepEntry] | None:
        """Lease an entry for this call, or ``None`` if the model cannot be cloned.

        ``None`` means the per-call path handles the question.  The cache
        remembers an unclonable model, so the clone is attempted once per
        model rather than once per call.
        """
        try:
            return self._cache.lease(
                key, self.model, lambda: SweepEntry.build(self.model, idata, key)
            )
        except ModelCloneError:
            return None

    def _rebuild(
        self, key: SweepCacheKey, idata: xr.DataTree
    ) -> Lease[SweepEntry] | None:
        """Build this call's entry, or ``None`` if the model cannot be cloned.

        The entry is conditioned on ``idata`` when it is built, so it never
        needs a rebind and whatever the cache holds in the meantime is not
        consulted.
        """
        try:
            return self._cache.rebuild(
                key, self.model, lambda: SweepEntry.build(self.model, idata, key)
            )
        except ModelCloneError:
            return None

    def _lease_bound_to(
        self, key: SweepCacheKey, idata: xr.DataTree
    ) -> tuple[Lease[SweepEntry] | None, Lease[SweepEntry] | None]:
        """Lease an entry whose graphs read the draws in ``idata``.

        Returns the lease to evaluate and, when a cached entry had to be
        replaced, the lease it retires.  The caller drops the retired entry once
        the replacement has evaluated and puts it back if the replacement does
        not, so one bad posterior cannot empty the cache.

        A cached entry that cannot be rebound was built on another posterior
        layout -- renamed or unlabeled coords, another dim set, or labels
        reordered -- which an empty cache would accept.  This call's entry is
        built in its place, without consulting the cache again; a rebuild that
        raises leaves the retired entry untouched, and one that cannot clone the
        model discards it and sends the call to the per-call path.  Both leases
        are ``None`` when the model cannot be cloned.
        """
        lease = self._lease(key, idata)
        if lease is None or not lease.cached:
            return lease, None
        try:
            lease.value.set_posterior(idata)
        except ValueError:
            try:
                replacement = self._rebuild(key, idata)
            except BaseException:
                # The rebind is atomic, so the entry is untouched and still
                # good for the next call.
                lease.release()
                raise
            if replacement is None:
                lease.discard()
                return None, None
            return replacement, lease
        except BaseException:
            # Any other failure leaves the atomic rebind untouched, so the
            # entry is still good for the next call.
            lease.release()
            raise
        return lease, None

    def _run_sweep_per_call(
        self,
        key: SweepCacheKey,
        idata: xr.DataTree,
        sweep_values: np.ndarray,
        mask_values: np.ndarray | None,
    ) -> tuple[tuple[str, ...], np.ndarray]:
        """Extract, compose and compile the sweep on every call.

        This is the path for models :class:`ConditionedModel` cannot clone, and
        it reads the source model's live data, as every call did before the
        cache existed.  The draws are realigned to the model's coords by label,
        as the cached path does, so a permuted posterior gives the same
        numbers; renamed or unlabeled coords are read by position.
        """
        response = self._extract_response_distribution(key.var_names, idata)
        inputs, graph = compose_sweep(
            response=response,
            base_value=self.model[key.var_input],
            sweep_type=key.sweep_type,
            has_mask=mask_values is not None,
            mask_shape=None if mask_values is None else mask_values.shape,
        )
        compiled = function(inputs, graph)
        if mask_values is None:
            return graph.dims, compiled(sweep_values)
        return graph.dims, compiled(sweep_values, mask_values)

    def _sweep_dataarray(
        self,
        dims: tuple[str, ...],
        result: np.ndarray,
        sweep_values: np.ndarray,
    ) -> DataArray:
        """Label the compiled output with the sweep grid and the model coords."""
        coords = {}
        for dim, shape in zip(dims, result.shape, strict=True):
            if dim == SWEEP_DIM:
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
        var_names: str = "channel_contribution",
        sweep_type: Literal[
            "multiplicative", "additive", "absolute"
        ] = "multiplicative",
        posterior_sample_fraction: float = 1.0,
        response_mask: xr.DataArray | np.ndarray | None = None,
        extend_idata: bool = False,
    ) -> xr.DataArray | None:
        """
        Run sweeps by forward-evaluating the response graph (no Jacobian).

        Repeated calls on the same structure reuse a compiled sweep from this
        instance's cache.  On an instance reached through ``mmm.sensitivity``
        that cache belongs to the MMM and is shared by every access, one cache
        per model; the model is not part of the key.  Only the sweep values,
        the mask values and the posterior draws change between calls, and those
        are a function argument or a posterior rebind, so a repeat call does not
        compile unless the posterior layout changed, which rebuilds the entry
        once.

        Parameters
        ----------
        var_input : str
            Name of the pm.Data variable (e.g., "channel_data").
            Expected shape: ``(date, *dims, arbitrary_dim)`` that match var_input dims.
        sweep_values : np.ndarray
            Values to sweep over.
        var_names : str
            The deterministic variable of interest.  It must carry named
            dimensions, for example a ``pmd.Deterministic`` built with
            ``dims=(...)``; a plain ``pm.Deterministic`` has none.
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
            If ``var_names`` is not in the model, or ``idata`` lacks a posterior
            variable the response reads.
        ValueError
            If ``posterior_sample_fraction`` is outside ``(0, 1]``.
            If ``var_names`` has no named dimensions, or no ``date`` dimension.
            If ``response_mask`` does not match the response's dimensions.
            If the posterior cannot be evaluated by the model, for example one
            with a different number of channels than the model has.
        TypeError
            If ``response_mask`` cannot be cast to boolean.

        Notes
        -----
        The posterior in ``idata`` may be replaced between calls, for example by
        assigning ``mmm.idata``.  A posterior stored in another order of the
        model's own labels is realigned by label; renamed or unlabeled coords are
        bound by position.  The number of draws may change, and values are cast
        to the dtype the model expects.  A call answers what an empty cache
        would answer, whether or not it reuses a compiled sweep.
        """
        response = self.model[var_names]
        response_dims = getattr(response, "dims", None)
        if response_dims is None:
            raise ValueError(
                f"{var_names!r} has no dimensions. Sensitivity sweeps require a "
                "response with named dimensions, for example one built with "
                "pmd.Deterministic(..., dims=(...))."
            )
        response_dims = tuple(response_dims)
        if "date" not in response_dims:
            raise ValueError(
                f"{var_names!r} has no 'date' dimension. Sensitivity sweeps "
                "require a response with a 'date' dimension, for example one "
                "built with pmd.Deterministic(..., dims=(...))."
            )

        idata = self._selected_idata(var_names, posterior_sample_fraction)
        mask_values = (
            None
            if response_mask is None
            else self._prepare_response_mask(
                response_mask, response_dims, var_names=var_names
            )
        )
        key = self._sweep_key(var_input, var_names, sweep_type, mask_values)

        dims, result = self._run_sweep_cached(key, idata, sweep_values, mask_values)
        xr_result = self._sweep_dataarray(dims, result, sweep_values)

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
            if not hasattr(self.idata, "sensitivity_analysis"):
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
        xr_result = results.differentiate(coord=SWEEP_DIM)
        if extend_idata:
            if not hasattr(self.idata, "sensitivity_analysis"):
                raise ValueError(
                    "No sensitivity analysis results found in 'self.idata'. "
                    "Run 'mmm.sensitivity.run_sweep()' first."
                )
            self.idata.sensitivity_analysis["marginal_effects"] = xr_result
        return xr_result
