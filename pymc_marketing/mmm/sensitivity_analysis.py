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
    from pymc_marketing.mmm.sensitivity_analysis import SensitivityAnalysis


    def saturation(x, alpha, lam):
        return (alpha * x) / (x + lam)


    coords = {"date": np.arange(52), "channel": list("abcde")}
    with pm.Model(coords=coords) as m:
        X = pm.Data(
            "channel_data",
            np.random.Gamma(13, 12, size=(52, 5)),
            dims=("date", "channel"),
        )
        alpha = pm.Gamma("alpha", 1, 1, dims="channel")
        lam = pm.Gamma("lam", 1, 1, dims="channel")
        contrib = pm.Deterministic(
            "channel_contribution", saturation(X, alpha, lam), dims=("date", "channel")
        )
        mu = pm.Normal("intercept", 0.0, 1.0) + contrib.sum(axis=-1)
        pm.Normal("likelihood", mu=mu, sigma=1.0, dims=("date",))
        idata = pm.sample(1000, tune=500, chains=2, target_accept=0.95, random_seed=42)

    sa = SensitivityAnalysis(m, idata)
    sweeps = np.linspace(0.2, 3.0, 8)
    # Jacobian-based marginal effects with respect to the input; shape: (sample, sweep, channel)
    result = sa.run_sweep(
        sweep_values=sweeps,
        var_input="channel_data",
        var_names="channel_contribution",
        sweep_type="multiplicative",
    )
    # Optional (backwards-compatibility): returns `result` unchanged because `run_sweep`
    # already yields marginal effects. Kept to avoid breaking older code.
    me = SensitivityAnalysis.compute_marginal_effects(result, sweeps)

Notes
-----
- Arbitrary models: As long as the response graph depends (directly or indirectly)
  on the `pm.Data` provided via `var_input`, and you pass the name of the
  deterministic/response via `var_names`, the class builds the Jacobian and
  evaluates it across sweeps automatically.
- Multi-dimensional inputs: If `var_input` has dims like `(date, country, channel)`,
  the output shape is `(sample, sweep, country, channel)`. You can subset with
  `var_names_filter={"country": ["usa"], "channel": ["a", "b"]}`.
- Sweep types: `"multiplicative"`, `"additive"`, and `"absolute"` are supported.
- To persist results, pass `extend_idata=True` to store them under
  `idata.sensitivity_analysis`.
"""

import math
import warnings
from typing import Literal

import arviz as az
import numpy as np
import pytensor.tensor as pt
import xarray as xr
from pymc import Model
from pytensor import function
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph import vectorize_graph

from pymc_marketing.pytensor_utils import extract_response_distribution


class SensitivityAnalysis:
    """SensitivityAnalysis class is used to perform counterfactual analysis on MMM's."""

    def __init__(
        self, pymc_model: Model, idata: az.InferenceData, dims: tuple[str, ...] = ()
    ):
        self.model = pymc_model
        self.idata = idata
        self.dims = dims

    def _compute_dims_order_from_varinput(self, var_input: str) -> list[str]:
        """Compute non-date dims order directly from the model's var_input dims.

        The var_input dims convention is (date, *dims, arbitrary_last_dim_name).
        We drop any occurrence of 'date' and preserve the remaining order as-is.
        """
        var_dims = tuple(self.model.named_vars_to_dims.get(var_input, ()))
        if not var_dims:
            # Fallback: try to infer from the shared variable shape and model coords order
            # This is a best-effort path; in practice PyMC populates named_vars_to_dims
            return list(self.dims)
        non_date_dims = [d for d in var_dims if d != "date"]
        return non_date_dims

    def _coord_to_index(self, dims_order: list[str]) -> dict[str, dict]:
        """Map model coordinate labels to integer indices for all dims in order."""
        mapping: dict[str, dict] = {}
        for dim_name in dims_order:
            if dim_name in self.model.coords:
                coords = list(self.model.coords[dim_name])
                mapping[dim_name] = {label: idx for idx, label in enumerate(coords)}
        return mapping

    def _draw_indices_for_percentage(
        self, posterior_sample_fraction: float
    ) -> slice | np.ndarray:
        if not (0 < posterior_sample_fraction <= 1):
            raise ValueError(
                "'posterior_sample_fraction' must be in the interval (0, 1]."
            )

        draws = int(self.idata.posterior.dims.get("draw", 0))
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
        self, response_variable: str, posterior_sample_fraction: float
    ):
        """Extract response distribution graph conditioned on posterior samples."""
        draw_selection = self._draw_indices_for_percentage(posterior_sample_fraction)
        return extract_response_distribution(
            pymc_model=self.model,
            idata=self.idata.isel(draw=draw_selection),
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

    def _transform_output_to_xarray(
        self,
        result: np.ndarray,
        sweep_values: np.ndarray,
        dims_order: list[str],
    ) -> xr.DataArray:
        """Transform a numpy result into an xarray.DataArray with correct dims/coords.

        Expected input shape: (sample, sweep, *dims_order)
        """
        # Build dims list in the exact order of the numpy result
        dims: list[str] = ["sample", "sweep", *dims_order]

        # Helper for base labels (ignore filtering at data and coord level)
        def _base_labels(dim_name: str, axis_size: int) -> list:
            return list(self.model.coords.get(dim_name, np.arange(axis_size)))

        # Assemble coords aligned with dims
        coords: dict[str, np.ndarray | list] = {}
        coords["sample"] = np.arange(result.shape[0])
        coords["sweep"] = np.asarray(sweep_values)

        # Map remaining dims using model coords and filters
        # Compute axis offsets: after (sample, sweep), axes align with dims_order
        axis_offset = 2
        for i, dim_name in enumerate(dims_order):
            axis_size = result.shape[axis_offset + i]
            coords[dim_name] = _base_labels(dim_name, axis_size)

        return xr.DataArray(result, coords=coords, dims=dims)

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
            self.idata.add_groups({"sensitivity_analysis": dataset})

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
        **kwargs,
    ) -> xr.DataArray | None:
        """
        Run sweeps by forward-evaluating the response graph (no Jacobian).

        Parameters
        ----------
        var_input : str
            Name of the pm.Data variable (e.g., "channel_data").
            Expected shape: (date, *dims, arbitrary_dim) that match var_input dims.
        sweep_values : np.ndarray
            Values to sweep over.
        var_names : str
            The deterministic variable of interest.
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
            (sample, sweep, *dims_order), where `dims_order` are the non-date
            dims of `var_input` in the same order as in the model. The response
            is averaged over the `date` axis as in the draft example. If
            extend_idata is True, stores the result under
            `idata.sensitivity_analysis` and returns None.
        """
        if "posterior_sample_batch" in kwargs:
            legacy_value = kwargs.pop("posterior_sample_batch")
            warnings.warn(
                "'posterior_sample_batch' is deprecated; use 'posterior_sample_fraction' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            legacy_value = float(legacy_value)
            if legacy_value <= 0:
                raise ValueError(
                    "'posterior_sample_batch' must be > 0 for backwards compatibility conversion."
                )
            posterior_sample_fraction = min(1.0, 1.0 / math.sqrt(legacy_value))

        if kwargs:
            raise TypeError(
                f"run_sweep() got unexpected keyword arguments: {', '.join(kwargs)}"
            )

        # Determine dims order from the provided var_input (drop 'date')
        dims_order = self._compute_dims_order_from_varinput(var_input)

        response_dims = tuple(self.model.named_vars_to_dims.get(var_names, ()))

        # 1) Extract response graph (shape typically: (sample, date, *dims_order))
        resp_graph = self._extract_response_distribution(
            response_variable=var_names,
            posterior_sample_fraction=posterior_sample_fraction,
        )

        if response_mask is not None:
            mask_array = self._prepare_response_mask(
                response_mask,
                response_dims,
                var_names=var_names,
            )
            # Add sample dimension and broadcast to match resp_graph shape
            mask_tensor = pt.constant(mask_array, dtype="bool")
            # resp_graph has shape (sample, date, *response_dims)
            # mask_array has shape (*response_dims)
            # We need to add leading dimensions for sample (and date if present)
            pad_dims = resp_graph.ndim - mask_tensor.ndim
            if pad_dims > 0:
                mask_tensor = pt.shape_padleft(mask_tensor, pad_dims)
            mask_tensor = pt.broadcast_to(mask_tensor, resp_graph.shape)
            zeros_resp = pt.zeros_like(resp_graph)
            resp_graph = pt.set_subtensor(
                zeros_resp[mask_tensor],
                resp_graph[mask_tensor],
            )

        # 2) Prepare batched input carrying the sweep axis
        data_shared: SharedVariable = self.model[var_input]
        base_value = data_shared.get_value()
        num_sweeps = int(np.asarray(sweep_values).shape[0])

        # Build a (sweep, 1, 1, ..., 1) array to broadcast against base_value
        sweep_col = np.asarray(sweep_values, dtype=base_value.dtype).reshape(
            (num_sweeps,) + (1,) * base_value.ndim
        )
        if sweep_type == "multiplicative":
            batched_input = sweep_col * base_value[None, ...]
        elif sweep_type == "additive":
            batched_input = sweep_col + base_value[None, ...]
        elif sweep_type == "absolute":
            batched_input = np.broadcast_to(
                sweep_col,
                (num_sweeps, *base_value.shape),
            )
        else:
            raise ValueError(f"Unknown sweep_type {sweep_type!r}")

        # 3) Vectorize the response graph over the new sweep axis by replacing the shared
        #    input with a tensor that carries a leading sweep dimension.
        channel_in = pt.tensor(
            name=f"{var_input}_sweep_in",
            dtype=data_shared.dtype,
            shape=(None, *data_shared.type.shape),
        )
        sweep_graph = vectorize_graph(resp_graph, replace={data_shared: channel_in})
        fn = function([channel_in], sweep_graph)  # (sweep, sample, date, *dims_order)

        evaluated = fn(batched_input)

        # 4) Reduce the date axis (axis=2) as in the draft (mean over time)
        #    Result shape: (sweep, sample, *dims_order)
        evaluated_no_date = evaluated.sum(axis=2)

        # 5) Reorder axes to (sample, sweep, *dims_order)
        #    Move sweep axis (0) to position 1, sample axis becomes 0.
        result = np.moveaxis(evaluated_no_date, 0, 1)

        xr_result = self._transform_output_to_xarray(
            result,
            sweep_values=sweep_values,
            dims_order=dims_order,
        )
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

        Returns
        -------
        xr.DataArray
            The input ``results`` unchanged.
        """
        xr_result = results.differentiate(coord="sweep")
        if extend_idata:
            if not hasattr(self.idata, "sensitivity_analysis"):
                raise ValueError(
                    "No sensitivity analysis results found in 'self.idata'. "
                    "Run 'mmm.sensitivity.run_sweep()' first."
                )
            self.idata.sensitivity_analysis["marginal_effects"] = xr_result
        return xr_result
