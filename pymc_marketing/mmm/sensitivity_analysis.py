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

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


class SensitivityAnalysis:
    """SensitivityAnalysis class is used to perform counterfactual analysis on MMM's."""

    def __init__(
        self,
        mmm,
        predictors: list[str],
        sweep_values: np.ndarray,
        sweep_type: Literal[
            "multiplicative", "additive", "absolute"
        ] = "multiplicative",
    ) -> None:
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
        self.mmm = mmm
        self.predictors = predictors
        self.sweep_values = sweep_values
        self.sweep_type = sweep_type

    def run_sweep(self) -> xr.Dataset:
        """Run the model's predict function over the sweep grid and store results."""
        predictions = []
        for sweep_value in self.sweep_values:
            X_new = self.create_intervention(sweep_value)
            counterfac = self.mmm.predict(X_new, extend_idata=False, progressbar=False)
            # TODO: Ideally we can use this --------------------------------------------
            # actual = self.mmm._get_group_predictive_data(
            #     group="posterior_predictive", original_scale=True
            # )["y"]
            actual = self.mmm.idata["posterior_predictive"]["y"]
            # --------------------------------------------------------------------------
            uplift = counterfac - actual
            predictions.append(uplift)

        results = (
            xr.concat(predictions, dim="sweep")
            .assign_coords(sweep=self.sweep_values)
            .transpose(..., "sweep")
        )

        marginal_effects = self.compute_marginal_effects(results, self.sweep_values)

        results = xr.Dataset(
            {
                "y": results,
                "marginal_effects": marginal_effects,
            }
        )
        # Add metadata to the results
        results.attrs["sweep_type"] = self.sweep_type
        results.attrs["predictors"] = self.predictors
        return results

    def create_intervention(self, sweep_value: float) -> pd.DataFrame:
        """Apply the intervention to the predictors."""
        X_new = self.mmm.X.copy()
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

    @staticmethod
    def compute_marginal_effects(results, sweep_values) -> xr.DataArray:
        """Compute marginal effects via finite differences from the sweep results."""
        marginal_effects = results.differentiate(coord="sweep")
        marginal_effects = xr.DataArray(
            marginal_effects,
            dims=results.dims,
            coords=results.coords,
        )
        return marginal_effects

    @staticmethod
    def plot(
        results: xr.Dataset,
        hdi_prob: float = 0.94,
        ax: plt.Axes | None = None,
        marginal: bool = False,
    ) -> plt.Axes:
        """
        Plot the counterfactual uplift or marginal effects curve.

        Parameters
        ----------
        results : xr.Dataset
            The dataset containing the results of the sweep.
        hdi_prob : float, optional
            The probability for computing the highest density interval (HDI). Default is 0.94.
        ax : Optional[plt.Axes], optional
            An optional matplotlib Axes on which to plot. If None, a new Axes is created.
        marginal : bool, optional
            If True, plot marginal effects. If False (default), plot uplift.

        Returns
        -------
        plt.Axes
            The Axes object with the plot.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        x = results.sweep.values
        if marginal:
            y = results.marginal_effects.mean(dim=["chain", "draw"]).sum(dim="date")
            y_hdi = results.marginal_effects.sum(dim="date")
            color = "C1"
            label = "Posterior mean marginal effect"
            title = "Marginal effects plot"
            ylabel = "Marginal effect (dE[Y]/dX)"
        else:
            y = results.y.mean(dim=["chain", "draw"]).sum(dim="date")
            y_hdi = results.y.sum(dim="date")
            color = "C0"
            label = "Posterior mean"
            title = "Sensitivity analysis plot"
            ylabel = "Total uplift (sum over dates)"

        ax.plot(x, y, label=label, color=color)

        az.plot_hdi(
            x,
            y_hdi,
            hdi_prob=hdi_prob,
            color=color,
            fill_kwargs={"alpha": 0.5, "label": f"{hdi_prob * 100:.0f}% HDI"},
            plot_kwargs={"color": color, "alpha": 0.5},
            smooth=False,
            ax=ax,
        )

        ax.set(title=title)
        if results.sweep_type == "absolute":
            ax.set_xlabel(f"Absolute value of: {results.predictors}")
        else:
            ax.set_xlabel(
                f"{results.sweep_type.capitalize()} change of: {results.predictors}"
            )
        ax.set_ylabel(ylabel)
        plt.legend()

        # Set y-axis limits based on the sign of y values
        y_values = y.values if hasattr(y, "values") else np.array(y)
        if np.all(y_values < 0):
            ax.set_ylim(top=0)
        elif np.all(y_values > 0):
            ax.set_ylim(bottom=0)

        # Add reference lines
        if results.sweep_type == "multiplicative":
            ax.axvline(x=1, color="k", linestyle="--", alpha=0.5)
            if not marginal:
                ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
        elif results.sweep_type == "additive":
            ax.axvline(x=0, color="k", linestyle="--", alpha=0.5)

        return ax
