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

"""Counterfactual sweeps for Marketing Mix Models (MMM)."""

from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr


class SensitivityAnalysis:
    """SensitivityAnalysis class is used to perform counterfactual analysis on MMM's."""

    def __init__(self, mmm) -> None:
        """
        Initialize the SensitivityAnalysis with a reference to the MMM instance.

        Parameters
        ----------
        mmm : MMM
            The marketing mix model instance used for predictions.
        """
        self.mmm = mmm

    def run_sweep(
        self,
        var_names: list[str],
        sweep_values: np.ndarray,
        sweep_type: Literal[
            "multiplicative", "additive", "absolute"
        ] = "multiplicative",
    ) -> xr.Dataset:
        """Run the model's predict function over the sweep grid and store results.

        Parameters
        ----------
        var_names : list[str]
            List of variable names to intervene on.
        sweep_values : np.ndarray
            Array of sweep values.
        sweep_type : Literal["multiplicative", "additive", "absolute"], optional
            Type of intervention to apply, by default "multiplicative".
            - 'multiplicative': Multiply the original predictor values by each sweep value.
            - 'additive': Add each sweep value to the original predictor values.
            - 'absolute': Set the predictor values directly to each sweep value (ignoring original values).

        Returns
        -------
        xr.Dataset
            Dataset containing the sensitivity analysis results.
        """
        # Validate that idata exists
        if not hasattr(self.mmm, "idata"):
            raise ValueError("idata does not exist. Build the model first and fit.")

        # Store parameters for this run
        self.var_names = var_names
        self.sweep_values = sweep_values
        self.sweep_type = sweep_type

        # TODO: Ideally we can use this --------------------------------------------
        # actual = self.mmm._get_group_predictive_data(
        #     group="posterior_predictive", original_scale=True
        # )["y"]
        actual = self.mmm.idata["posterior_predictive"]["y"]
        # --------------------------------------------------------------------------
        predictions = []
        for sweep_value in self.sweep_values:
            X_new = self.create_intervention(sweep_value)
            counterfac = self.mmm.sample_posterior_predictive(
                X_new, extend_idata=False, combined=False, progressbar=False
            )
            uplift = counterfac - actual
            predictions.append(uplift)

        results = (
            xr.concat(predictions, dim="sweep")
            .assign_coords(sweep=self.sweep_values)
            .transpose(..., "sweep")
        )

        marginal_effects = self.compute_marginal_effects(results, self.sweep_values)

        results = xr.merge(
            [
                results,
                marginal_effects.rename({"y": "marginal_effects"}),
            ]
        ).transpose(..., "sweep")

        # Add metadata to the results
        results.attrs["sweep_type"] = self.sweep_type
        results.attrs["var_names"] = self.var_names

        # Add results to the MMM's idata
        if hasattr(self.mmm.idata, "sensitivity_analysis"):
            delattr(self.mmm.idata, "sensitivity_analysis")
        self.mmm.idata.add_groups({"sensitivity_analysis": results})  # type: ignore

        return results

    def create_intervention(self, sweep_value: float) -> pd.DataFrame:
        """Apply the intervention to the predictors."""
        X_new = self.mmm.X.copy()
        if self.sweep_type == "multiplicative":
            for var_name in self.var_names:
                X_new[var_name] *= sweep_value
        elif self.sweep_type == "additive":
            for var_name in self.var_names:
                X_new[var_name] += sweep_value
        elif self.sweep_type == "absolute":
            for var_name in self.var_names:
                X_new[var_name] = sweep_value
        else:
            raise ValueError(f"Unsupported sweep_type: {self.sweep_type}")
        return X_new

    @staticmethod
    def compute_marginal_effects(results, sweep_values) -> xr.DataArray:
        """Compute marginal effects via finite differences from the sweep results."""
        marginal_effects = results.differentiate(coord="sweep")

        return marginal_effects
