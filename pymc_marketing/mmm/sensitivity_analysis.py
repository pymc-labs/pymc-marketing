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

- the name of the input data variable (`varinput`, e.g., `"channel_data"`),
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
    # Jacobian-based effects with respect to the input; shape: (sample, sweep, channel)
    result = sa.run_sweep(
        sweep_values=sweeps,
        varinput="channel_data",
        var_names="channel_contribution",
        sweep_type="multiplicative",
    )
    # Optional: finite-difference marginals along the sweep axis
    me = SensitivityAnalysis.compute_marginal_effects(result, sweeps)

Notes
-----
- Arbitrary models: As long as the response graph depends (directly or indirectly)
  on the `pm.Data` provided via `varinput`, and you pass the name of the
  deterministic/response via `var_names`, the class builds the Jacobian and
  evaluates it across sweeps automatically.
- Multi-dimensional inputs: If `varinput` has dims like `(date, country, channel)`,
  the output shape is `(sample, sweep, country, channel)`. You can subset with
  `var_names_filter={"country": ["usa"], "channel": ["a", "b"]}`.
- Sweep types: `"multiplicative"`, `"additive"`, and `"absolute"` are supported.
- To persist results, pass `extend_idata=True` to store them under
  `idata.sensitivity_analysis`.
"""

from typing import Literal

import arviz as az
import numpy as np
import pytensor.tensor as pt
import xarray as xr
from pymc import Model
from pytensor import function
from pytensor.compile.sharedvalue import SharedVariable

from pymc_marketing.pytensor_utils import extract_response_distribution


class SensitivityAnalysis:
    """SensitivityAnalysis class is used to perform counterfactual analysis on MMM's."""

    def __init__(
        self, pymc_model: Model, idata: az.InferenceData, dims: tuple[str, ...] = ()
    ):
        self.model = pymc_model
        self.idata = idata
        self.dims = dims

    def _compute_dims_order_from_varinput(self, varinput: str) -> list[str]:
        """Compute non-date dims order directly from the model's varinput dims.

        The varinput dims convention is (date, *dims, arbitrary_last_dim_name).
        We drop any occurrence of 'date' and preserve the remaining order as-is.
        """
        var_dims = tuple(self.model.named_vars_to_dims.get(varinput, ()))
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

    def _apply_filter_np(
        self,
        arr: np.ndarray,
        var_names_filter: dict[str, list] | None,
        *,
        dims_order: list[str],
    ) -> np.ndarray:
        """Apply filtering to numpy array along named axes without changing axis order.

        Parameters
        ----------
        arr : np.ndarray
            Array of shape (samples, *dims_order)
        var_names_filter : dict[str, list] | None
            Mapping from dim name to labels/indices to select.
        """
        if not var_names_filter:
            return arr

        coord_to_index = self._coord_to_index(dims_order)
        result = arr
        # axes 1.. correspond to dims_order
        for axis_offset, dim_name in enumerate(dims_order, start=1):
            if dim_name not in var_names_filter:
                continue
            labels = var_names_filter[dim_name]
            # Convert labels to integer indices using model metadata
            indices: list[int] = []
            for label in labels:
                if isinstance(label, int | np.integer):
                    indices.append(int(label))
                else:
                    indices.append(coord_to_index[dim_name][label])
            # Use np.take with an array of indices to preserve the axis (even if length 1)
            result = np.take(
                result, indices=np.array(indices, dtype=int), axis=axis_offset
            )
        return result

    def _extract_response_distribution(
        self, response_variable: str, posterior_sample_batch: int
    ):
        """Extract response distribution graph conditioned on posterior samples."""
        return extract_response_distribution(
            pymc_model=self.model,
            idata=self.idata.isel(draw=slice(None, None, posterior_sample_batch)),
            response_variable=response_variable,
        )

    def _transform_output_to_xarray(
        self,
        result: np.ndarray,
        sweep_values: np.ndarray,
        dims_order: list[str],
        var_names_filter: dict[str, list] | None = None,
    ) -> xr.DataArray:
        """Transform a numpy result into an xarray.DataArray with correct dims/coords.

        Expected input shape: (sample, sweep, *dims_order)
        """
        # Build dims list in the exact order of the numpy result
        dims: list[str] = ["sample", "sweep", *dims_order]

        # Helper to select labels according to optional filters
        def _selected_labels(dim_name: str, axis_size: int) -> list:
            # Base labels from model coordinates if available; otherwise fallback to range
            base_labels = list(self.model.coords.get(dim_name, np.arange(axis_size)))
            if not var_names_filter or dim_name not in var_names_filter:
                return base_labels

            # Map requested entries (labels or indices) to labels
            requested = var_names_filter[dim_name]
            coord_to_index = self._coord_to_index(dims_order).get(dim_name, {})
            indices: list[int] = []
            for req in requested:
                if isinstance(req, int | np.integer):
                    indices.append(int(req))
                else:
                    indices.append(int(coord_to_index[req]))
            return [base_labels[i] for i in indices]

        # Assemble coords aligned with dims
        coords: dict[str, np.ndarray | list] = {}
        coords["sample"] = np.arange(result.shape[0])
        coords["sweep"] = np.asarray(sweep_values)

        # Map remaining dims using model coords and filters
        # Compute axis offsets: after (sample, sweep), axes align with dims_order
        axis_offset = 2
        for i, dim_name in enumerate(dims_order):
            axis_size = result.shape[axis_offset + i]
            coords[dim_name] = _selected_labels(dim_name, axis_size)

        return xr.DataArray(result, coords=coords, dims=dims)

    def _add_to_idata(self, result: xr.DataArray) -> None:
        """Add the result to the idata."""
        self.idata.add_groups({"sensitivity_analysis": result})

    def run_sweep(
        self,
        varinput: str,
        sweep_values: np.ndarray,
        *,
        var_names: str = "channel_contribution",
        sweep_type: Literal[
            "multiplicative", "additive", "absolute"
        ] = "multiplicative",
        posterior_sample_batch: int = 1,
        var_names_filter: dict[str, list] | None = None,
        extend_idata: bool = False,
    ) -> xr.DataArray | None:
        """
        Run sweeps and compute marginal effects.

        Parameters
        ----------
        varinput : str
            Name of the pm.Data variable (e.g., "channel_data").
            Expected shape: (date, *dims, arbitrary_dim) that match varinput dims.
        sweep_values : np.ndarray
            Values to sweep over.
        var_names : str
            The deterministic variable of interest.
        sweep_type : {"multiplicative","additive","absolute"}
            Type of sweep to apply.
        posterior_sample_batch : int
            Subsample posterior draws.

        Returns
        -------
        xarray.DataArray | None
            If extend_idata is False, returns an xarray.DataArray with shape
            (sample, sweep, *dims_order). If extend_idata is True, stores the
            result under `idata.sensitivity_analysis` and returns None.
        """
        # Determine dims order from the provided varinput (drop 'date')
        dims_order = self._compute_dims_order_from_varinput(varinput)

        # 1) Extract response graph
        resp_graph = self._extract_response_distribution(
            response_variable=var_names, posterior_sample_batch=posterior_sample_batch
        )

        # 2) Sum over date dimension → shape (samples, *dims, channel)
        resp_total = resp_graph.sum(axis=1)

        # 3) Input shared variable
        data_shared: SharedVariable = self.model[varinput]
        original_X = data_shared.get_value()

        # 4) Jacobian wrt input (which includes date dim)
        # Collapse ALL non-sample axes to obtain a scalar per sample before differentiation
        # This ensures the Jacobian has shape (samples, date, *dims, channel)
        f_scalar = resp_total.sum(axis=tuple(range(1, resp_total.ndim)))
        J = pt.jacobian(f_scalar, data_shared)

        # 5) Sum out the date axis in the Jacobian → (samples, *dims, channel)
        J_per_channel = J.sum(axis=1)

        jac_fn = function(inputs=[], outputs=J_per_channel)

        # 6) Sweeps
        outs = []
        for s in sweep_values:
            if sweep_type == "multiplicative":
                data_shared.set_value(original_X * s)
            elif sweep_type == "additive":
                data_shared.set_value(original_X + s)
            elif sweep_type == "absolute":
                data_shared.set_value(np.full_like(original_X, s))
            else:
                raise ValueError(f"Unknown sweep_type {sweep_type!r}")
            result = jac_fn()  # (n_samples, *dims, channel)
            # Apply filtering in numpy space using model metadata to preserve axes
            result = self._apply_filter_np(
                result, var_names_filter, dims_order=dims_order
            )
            outs.append(result)

        stacked = np.stack(outs, axis=1)  # (n_samples, n_sweeps, *dims, channel)
        xr_result = self._transform_output_to_xarray(
            stacked,
            sweep_values=sweep_values,
            dims_order=dims_order,
            var_names_filter=var_names_filter,
        )
        if extend_idata:
            self._add_to_idata(xr_result)
            return None
        else:
            return xr_result

    @staticmethod
    def compute_marginal_effects(results, sweep_values) -> xr.DataArray:
        """Compute marginal effects via finite differences from the sweep results."""
        marginal_effects = results.differentiate(coord="sweep")

        return marginal_effects
