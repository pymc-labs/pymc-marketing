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

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


class CounterfactualSweep:
    """CounterfactualSweep class is used to perform counterfactual analysis on MMM's."""

    def __init__(
        self,
        mmm,
        X: pd.DataFrame,
        predictors: list[str],
        sweep_values: np.ndarray,
        sweep_type: str = "multiplicative",
    ):
        """
        Initialize and run the counterfactual sweep.

        Parameters
        ----------
        - mmm: The marketing mix model instance used for predictions.
        - X: Original design matrix (DataFrame).
        - predictors (list[str]): List of predictors to intervene on.
        - sweep_values (np.ndarray): Array of sweep values.
        - sweep_type (str): 'multiplicative', 'additive', or 'absolute'.
            - 'multiplicative': Multiply the original predictor values by each sweep value.
            - 'additive': Add each sweep value to the original predictor values.
            - 'absolute': Set the predictor values directly to each sweep value (ignoring original values).
        """
        if sweep_type not in ["multiplicative", "additive", "absolute"]:
            raise ValueError(
                "sweep_type must be 'multiplicative', 'additive', or 'absolute'."
            )

        self.mmm = mmm
        self.X = X
        self.predictors = predictors
        self.sweep_values = sweep_values
        self.sweep_type = sweep_type

        # Run sweep and store results
        self.run_sweep()

    def run_sweep(self):
        """Run the model's predict function over the sweep grid and store results."""
        predictions = []
        for sweep_value in self.sweep_values:
            X_new = self.create_intervention(sweep_value)
            counterfac = self.mmm.predict(X_new, extend_idata=False, progressbar=False)
            actual = self.mmm._get_group_predictive_data(
                group="posterior_predictive", original_scale=True
            )["y"]
            uplift = counterfac - actual
            predictions.append(uplift)

        self.results = (
            xr.concat(predictions, dim="sweep")
            .assign_coords(sweep=self.sweep_values)
            .transpose(..., "sweep")
        )

    def create_intervention(self, sweep_value):
        """Apply the intervention to the predictors."""
        X_new = self.X.copy()
        if self.sweep_type == "multiplicative":
            for predictor in self.predictors:
                X_new[predictor] *= sweep_value
        elif self.sweep_type == "additive":
            for predictor in self.predictors:
                X_new[predictor] += sweep_value
        elif self.sweep_type == "absolute":
            for predictor in self.predictors:
                X_new[predictor] = sweep_value
        else:
            raise ValueError(f"Unsupported sweep_type: {self.sweep_type}")
        return X_new

    def plot_uplift(self, hdi_prob: float = 0.94, ax=None) -> plt.Axes:
        """Plot the counterfactual uplift curve."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        x = self.sweep_values
        y = self.results.mean(dim=["chain", "draw"]).sum(dim="date")
        ax.plot(x, y, label="Posterior mean", color="C0")

        az.plot_hdi(
            x,
            self.results.sum(dim="date"),
            hdi_prob=hdi_prob,
            color="C0",
            fill_kwargs={"alpha": 0.5, "label": f"{hdi_prob * 100:.0f}% HDI"},
            plot_kwargs={"color": "C0", "alpha": 0.5},
            smooth=False,
            ax=ax,
        )

        ax.set(title="Counterfactual uplift plot")
        if self.sweep_type == "absolute":
            ax.set_xlabel(f"Absolute value of: {self.predictors}")
        else:
            ax.set_xlabel(
                f"{self.sweep_type.capitalize()} change of: {self.predictors}"
            )
        ax.set_ylabel("Total uplift (sum over dates)")
        plt.legend()
        return ax

    def compute_marginal_effects(self):
        """Compute marginal effects via finite differences from the sweep results."""
        sweep_axis = self.results.get_axis_num("sweep")
        marginal_effects = np.gradient(
            self.results, self.sweep_values, axis=sweep_axis, edge_order=2
        )
        self.marginal_effects = xr.DataArray(
            marginal_effects,
            dims=self.results.dims,
            coords=self.results.coords,
        )
        return self.marginal_effects

    def plot_marginal_effects(self, hdi_prob: float = 0.94, ax=None) -> plt.Axes:
        """Plot the marginal effects curve."""
        if not hasattr(self, "marginal_effects"):
            self.compute_marginal_effects()

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        x = self.sweep_values
        y = self.marginal_effects.mean(dim=["chain", "draw"]).sum(dim="date")
        ax.plot(x, y, label="Posterior mean marginal effect", color="C1")

        az.plot_hdi(
            x,
            self.marginal_effects.sum(dim="date"),
            hdi_prob=hdi_prob,
            color="C1",
            fill_kwargs={"alpha": 0.5, "label": f"{hdi_prob * 100:.0f}% HDI"},
            plot_kwargs={"color": "C1", "alpha": 0.5},
            smooth=False,
            ax=ax,
        )

        ax.set(title="Marginal effects plot")
        if self.sweep_type == "absolute":
            ax.set_xlabel(f"Absolute value of: {self.predictors}")
        else:
            ax.set_xlabel(
                f"{self.sweep_type.capitalize()} change of: {self.predictors}"
            )
        ax.set_ylabel("Marginal effect (dE[Y]/dX)")
        plt.legend()
        return ax
