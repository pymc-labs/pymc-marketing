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
    # Jacobian-based marginal effects with respect to the input; shape: (sample, sweep, channel)
    result = sa.run_sweep(
        sweep_values=sweeps,
        varinput="channel_data",
        var_names="channel_contribution",
        sweep_type="multiplicative",
    )
    # Optional (backwards-compatibility): returns `result` unchanged because `run_sweep`
    # already yields marginal effects. Kept to avoid breaking older code.
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
        varinput: str,
        sweep_values: np.ndarray,
        *,
        var_names: str = "channel_contribution",
        sweep_type: Literal[
            "multiplicative", "additive", "absolute"
        ] = "multiplicative",
        posterior_sample_batch: int = 1,
        extend_idata: bool = False,
    ) -> xr.DataArray | None:
        """
        Run sweeps by forward-evaluating the response graph (no Jacobian).

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
            (sample, sweep, *dims_order), where `dims_order` are the non-date
            dims of `varinput` in the same order as in the model. The response
            is averaged over the `date` axis as in the draft example. If
            extend_idata is True, stores the result under
            `idata.sensitivity_analysis` and returns None.
        """
        # Determine dims order from the provided varinput (drop 'date')
        dims_order = self._compute_dims_order_from_varinput(varinput)

        # 1) Extract response graph (shape typically: (sample, date, *dims_order))
        resp_graph = self._extract_response_distribution(
            response_variable=var_names, posterior_sample_batch=posterior_sample_batch
        )

        # 2) Prepare batched input carrying the sweep axis
        data_shared: SharedVariable = self.model[varinput]
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
            name=f"{varinput}_sweep_in",
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
        ref: int,
        extend_idata: bool = False,
    ) -> xr.DataArray:
        """Return marginal effects from idata."""
        xr_result = results - ref
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
