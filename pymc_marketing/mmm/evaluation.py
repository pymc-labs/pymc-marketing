#   Copyright 2024 The PyMC Labs Developers
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
"""Evaluation and diagnostics for MMM models."""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)

from pymc_marketing.mmm.mmm import MMM


class MMMEvaluator:
    """A class to evaluate and diagnose PyMC-Marketing MMMs.

    Parameters
    ----------
    model : MMM
        The PyMC-Marketing MMM object.
    idata : az.InferenceData
        The InferenceData object returned by the sampling method.

    Raises
    ------
    ValueError
        If the model object does not have a non-None 'idata' attribute.
    ValueError
        If the model object does not have a non-None 'model' attribute.

    """

    def __init__(self, model: MMM):
        if not hasattr(model, "idata") or model.idata is None:
            raise ValueError("The model object must have a non-None 'idata' attribute.")
        if not hasattr(model, "model") or model.model is None:
            raise ValueError("The model object must have a non-None 'model' attribute.")

        self.model = model
        self.metric_distributions: dict[str, np.ndarray] = {}
        self.metric_summaries: dict[str, dict[str, float]] = {}

    # Same error metric as Robyn
    @staticmethod
    def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the Normalized Root Mean Square Error (NRMSE).

        Normalization allows for comparison across different data sets and methodologies.
        NRMSE is one of the key metrics used in Robyn MMMs.

        Parameters
        ----------
        y_true : np.ndarray
            True values for target metric
        y_pred : np.ndarray
            Predicted values for target metric

        Returns
        -------
        float
            Normalized root mean square error.
        """
        return root_mean_squared_error(y_true, y_pred) / (y_true.max() - y_true.min())

    @staticmethod
    def nmae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the Normalized Mean Absolute Error (NMAE).

        Normalization allows for comparison across different data sets and methodologies.

        Parameters
        ----------
        y_true : np.ndarray
            True values for target metric
        y_pred : np.ndarray
            Predicted values for target metric

        Returns
        -------
        float
            Normalized mean absolute error.
        """
        return mean_absolute_error(y_true, y_pred) / (y_true.max() - y_true.min())

    def plot_hdi_forest(
        self,
        var_names: list,
        hdi_prob: float = 0.94,
        figsize: tuple = (12, 8),
        **plot_kwargs,
    ) -> plt.Figure:
        """Plot a forest plot to compare high-density intervals (HDIs).

        Plot a forest plot to compare high-density intervals (HDIs) from a given set of
        posterior distributions, as well as their r-hat statistics.

        Parameters
        ----------
        var_names : list
            List of variable names to include in the forest plot.
        hdi_prob : float, optional
            The probability mass of the highest density interval. Defaults to 0.94.
        figsize : tuple, optional
            Figure size in inches. Defaults to (12, 8).
        **plot_kwargs
            Additional keyword arguments to pass to the az.plot_forest method.

        Returns
        -------
        plt.Figure
            The matplotlib figure object.

        Raises
        ------
        ValueError
            If the required attributes (idata) are not found.
        """
        if not hasattr(self.model, "idata") or self.model.idata is None:
            raise ValueError(
                "The model object must have a non-None 'idata' attribute. "
                "Ensure you've called model.fit() before running diagnostics."
            )

        fig, ax = plt.subplots(figsize=figsize, ncols=2)
        az.plot_forest(
            data=self.model.idata,
            var_names=var_names,
            combined=True,
            ax=ax,
            hdi_prob=hdi_prob,
            r_hat=True,
            **plot_kwargs,
        )

        fig.suptitle(f"Posterior Distributions: {hdi_prob:.1%} HDI")

        return fig

    def plot_prior_vs_posterior(
        self,
        var_name: str,
        alphabetical_sort: bool = True,
        figsize: tuple[int, int] | None = None,
    ) -> plt.Figure:
        """
        Plot the prior vs posterior distribution for a specified variable in a 3 columngrid layout.

        This function generates KDE plots for each MMM channel, showing the prior predictive
        and posterior distributions with their respective means highlighted.
        It sorts the plots either alphabetically or based on the difference between the
        posterior and prior means, with the largest difference (posterior - prior) at the top.

        Parameters
        ----------
        var_name: str
            The variable to analyze (e.g., 'adstock_alpha').
        alphabetical_sort: bool, optional
            Whether to sort the channels alphabetically (True) or by the difference
            between the posterior and prior means (False). Default is True.
        figsize : tuple of int, optional
            Figure size in inches. If None, it will be calculated based on the number of channels.

        Returns
        -------
        fig : plt.Figure
            The matplotlib figure object

        Raises
        ------
        ValueError
            If the required attributes (prior, posterior) were not found.
        ValueError
            If var_name is not a string.
        """
        if (
            self.model.idata is None
            or not hasattr(self.model.idata, "prior")
            or not hasattr(self.model.idata, "posterior")
        ):
            raise ValueError(
                "Required attributes (prior, posterior) not found.\
                            Ensure you've called model.fit() and mmm.sample_prior_predictive()"
            )

        if not isinstance(var_name, str):
            raise ValueError(
                "var_name must be a string. Please provide a single variable name."
            )

        # Determine the number of channels and set up the grid
        num_channels = len(self.model.channel_columns)
        num_cols = 3
        num_rows = (num_channels + num_cols - 1) // num_cols  # Calculate rows needed

        if figsize is None:
            figsize = (25, 5 * num_rows)

        # Calculate prior and posterior means for sorting
        channel_means = []
        for channel in self.model.channel_columns:
            prior_mean = (
                self.model.idata.prior[var_name].sel(channel=channel).mean().values
            )
            posterior_mean = (
                self.model.idata.posterior[var_name].sel(channel=channel).mean().values
            )
            difference = posterior_mean - prior_mean
            channel_means.append((channel, prior_mean, posterior_mean, difference))

        # Choose how to sort the channels
        if alphabetical_sort:
            sorted_channels = sorted(channel_means, key=lambda x: x[0])
        else:
            # Otherwise, sort on difference between posterior and prior means
            sorted_channels = sorted(channel_means, key=lambda x: x[3], reverse=True)

        fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
        axs = axs.flatten()  # Flatten the array for easy iteration

        # Plot for each channel
        for i, (channel, prior_mean, posterior_mean, difference) in enumerate(
            sorted_channels
        ):
            # Extract prior samples for the current channel
            prior_samples = (
                self.model.idata.prior[var_name].sel(channel=channel).values.flatten()
            )

            # Plot the prior predictive distribution
            sns.kdeplot(
                prior_samples,
                ax=axs[i],
                label="Prior Predictive",
                color="blue",
                fill=True,
            )

            # Add a vertical line for the mean of the prior distribution
            axs[i].axvline(
                prior_mean,
                color="blue",
                linestyle="--",
                linewidth=2,
                label=f"Prior Mean: {prior_mean:.2f}",
            )

            # Extract posterior samples for the current channel
            posterior_samples = (
                self.model.idata.posterior[var_name]
                .sel(channel=channel)
                .values.flatten()
            )

            # Plot the prior predictive distribution
            sns.kdeplot(
                posterior_samples,
                ax=axs[i],
                label="Posterior Predictive",
                color="red",
                fill=True,
                alpha=0.15,
            )

            # Add a vertical line for the mean of the posterior distribution
            axs[i].axvline(
                posterior_mean,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Posterior Mean: {posterior_mean:.2f} (Diff: {difference:.2f})",
            )

            # Set titles and labels
            axs[i].set_title(channel)  # Subplot title is just the channel name
            axs[i].set_xlabel(var_name.capitalize())
            axs[i].set_ylabel("Density")
            axs[i].legend(loc="upper right")

        # Set the overall figure title
        fig.suptitle(f"Prior Predictive Distributions for {var_name}", fontsize=16)

        # Hide any unused subplots
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to fit the title

        return fig

    def calculate_metric_distributions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics_to_calculate: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Calculate distributions of evaluation metrics for posterior samples.

        Parameters
        ----------
        y_true : np.ndarray
            True values for the dataset. Shape: (date,)
        y_pred : np.ndarray
            Posterior predictive samples. Shape: (date, sample)
        metrics_to_calculate : list of str or None, optional
            List of metrics to calculate. Options include:
                * `r_squared`: Bayesian R-squared.
                * `rmse`: Root Mean Squared Error.
                * `nrmse`: Normalized Root Mean Squared Error.
                * `mae`: Mean Absolute Error.
                * `nmae`: Normalized Mean Absolute Error.
                * `mape`: Mean Absolute Percentage Error.
            Defaults to all metrics if None.

        Returns
        -------
        dict of str to np.ndarray
            A dictionary containing calculated metric distributions.

        Sets
        ----
        self.metric_distributions : dict of str to np.ndarray
            Stores the calculated metric distributions as an instance attribute.
        """
        metric_functions = {
            "r_squared": lambda y_true, y_pred: az.r2_score(y_true, y_pred)["r2"],
            "rmse": root_mean_squared_error,
            "nrmse": self.nrmse,
            "mae": mean_absolute_error,
            "nmae": self.nmae,
            "mape": mean_absolute_percentage_error,
        }

        if metrics_to_calculate is None:
            metrics_to_calculate = list(metric_functions.keys())
        else:
            invalid_metrics = set(metrics_to_calculate) - set(metric_functions.keys())
            if invalid_metrics:
                raise ValueError(
                    f"Invalid metrics: {invalid_metrics}. "
                    f"Valid options are: {list(metric_functions.keys())}"
                )

        results = {}
        for metric in metrics_to_calculate:
            metric_values = np.array(
                [
                    metric_functions[metric](
                        y_true, y_pred[:, i]
                    )  # Calculate along date dimension
                    for i in range(y_pred.shape[1])
                ]
            )
            results[metric] = metric_values

        self.metric_distributions = results
        return results

    def summarize_metric_distributions(
        self,
        metric_distributions: dict[str, np.ndarray] | None = None,
        hdi_prob: float = 0.94,
    ) -> dict[str, dict[str, float]]:
        """Summarize metric distributions with point estimates and HDIs.

        Parameters
        ----------
        metric_distributions : dict of str to np.ndarray
            Dictionary of metric distributions as returned by calculate_metric_distributions.
        hdi_prob : float, optional
            The probability mass of the highest density interval. Defaults to 0.94.

        Returns
        -------
        dict of str to dict
            A dictionary containing summary statistics for each metric.
            List of summary statistics calculated for each metric:
                * `mean`: Mean of the metric distribution.
                * `median`: Median of the metric distribution.
                * `std`: Standard deviation of the metric distribution.
                * `min`: Minimum value of the metric distribution.
                * `max`: Maximum value of the metric distribution.
                * `hdi_lower`: Lower bound of the Highest Density Interval.
                * `hdi_upper`: Upper bound of the Highest Density Interval.
        Sets
        ----
        self.metric_summaries : dict of str to dict
            Stores the calculated metric summaries as an instance attribute.
        """
        if metric_distributions is None:
            if self.metric_distributions is None:
                raise ValueError(
                    "Metric distributions have not been calculated.\
                                  Call `calculate_metric_distributions` first."
                )
            metric_distributions = self.metric_distributions

        metric_summaries = {}
        for metric, distribution in metric_distributions.items():
            hdi = az.hdi(distribution, hdi_prob=hdi_prob)
            metric_summaries[metric] = {
                "mean": np.mean(distribution),
                "median": np.median(distribution),
                "std": np.std(distribution),
                "min": np.min(distribution),
                "max": np.max(distribution),
                f"{hdi_prob:.0%}_hdi_lower": hdi[0],
                f"{hdi_prob:.0%}_hdi_upper": hdi[1],
            }

        self.metric_summaries = metric_summaries
        return metric_summaries

    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics_to_calculate: list[str] | None = None,
        hdi_prob: float = 0.94,
    ) -> dict[str, dict[str, float]]:
        """Evaluate the model by calculating metric distributions and summarizing them.

        This method combines the functionality of `calculate_metric_distributions` and
        `summarize_metric_distributions`.

        Parameters
        ----------
        y_true : np.ndarray
            The true values of the target variable.
        y_pred : np.ndarray
            The predicted values of the target variable.
        metrics_to_calculate : list of str or None, optional
            List of metrics to calculate. Options include:
                * `r_squared`: Bayesian R-squared.
                * `rmse`: Root Mean Squared Error.
                * `nrmse`: Normalized Root Mean Squared Error.
                * `mae`: Mean Absolute Error.
                * `nmae`: Normalized Mean Absolute Error.
                * `mape`: Mean Absolute Percentage Error.
            Defaults to all metrics if None.
        hdi_prob : float, optional
            The probability mass of the highest density interval. Defaults to 0.94.

        Returns
        -------
        dict of str to dict
            A dictionary containing summary statistics for each metric.
            List of summary statistics calculated for each metric:
                * `mean`: Mean of the metric distribution.
                * `median`: Median of the metric distribution.
                * `std`: Standard deviation of the metric distribution.
                * `min`: Minimum value of the metric distribution.
                * `max`: Maximum value of the metric distribution.
                * `hdi_lower`: Lower bound of the Highest Density Interval.
                * `hdi_upper`: Upper bound of the Highest Density Interval.

        Sets
        ----
        self.metric_distributions : dict
            A dictionary containing the distributions of calculated metrics.
        self.metric_summaries : dict
            A dictionary containing summary statistics for each calculated metric.

        Examples
        --------
        Evaluation (error and model metrics) for a PyMC-Marketing MMM.

        .. code-block:: python
            import pandas as pd
            from pymc_marketing.mmm import (
                GeometricAdstock,
                LogisticSaturation,
                MMM,
            )
            from pymc_marketing.mmm.evaluation import MMMEvaluator

            # Usual PyMC-Marketing demo model code
            data_url = "https://raw.githubusercontent.com/pymc-labs/pymc-marketing/main/data/mmm_example.csv"
            data = pd.read_csv(data_url, parse_dates=["date_week"])

            X = data.drop("y",axis=1)
            y = data["y"]
            mmm = MMM(
                adstock=GeometricAdstock(l_max=8),
                saturation=LogisticSaturation(),
                date_column="date_week",
                channel_columns=["x1", "x2"],
                control_columns=[
                    "event_1",
                    "event_2",
                    "t",
                ],
                yearly_seasonality=2,
            )
            mmm.fit(X, y)

            # Create an evaluator
            evaluator = MMMEvaluator(mmm)

            # Generate posterior predictive samples
            posterior_preds = mmm.sample_posterior_predictive(X)

            # Evaluate the model
            results = evaluator.evaluate_model(
                y_true=mmm.y,
                y_pred=posterior_preds.y,
                metrics_to_calculate=['r_squared', 'rmse', 'mae'],
                hdi_prob=0.89
            )

            # Print the results neatly
            for metric, stats in results.items():
                print(f"{metric}:")
                for stat, value in stats.items():
                    print(f"  {stat}: {value:.4f}")
                print()
        """
        self.calculate_metric_distributions(y_true, y_pred, metrics_to_calculate)
        self.summarize_metric_distributions(hdi_prob=hdi_prob)
        return self.metric_summaries
