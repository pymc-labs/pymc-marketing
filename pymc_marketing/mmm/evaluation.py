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
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)

from pymc_marketing.metrics import nmae, nrmse


def calculate_metric_distributions(
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
    """
    metric_functions = {
        "r_squared": lambda y_true, y_pred: az.r2_score(y_true, y_pred)["r2"],
        "rmse": root_mean_squared_error,
        "nrmse": nrmse,
        "mae": mean_absolute_error,
        "nmae": nmae,
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

    return results


def summarize_metric_distributions(
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
        raise ValueError(
            "Metric distributions have not been calculated.\
            Call `calculate_metric_distributions` first."
        )

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

    return metric_summaries


def compute_summary_metrics(
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
        from pymc_marketing.mmm.evaluation import evaluate_model

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

        # Generate posterior predictive samples
        posterior_preds = mmm.sample_posterior_predictive(X)

        # Evaluate the model
        results = evaluate_model(
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
    metric_distributions = calculate_metric_distributions(
        y_true, y_pred, metrics_to_calculate
    )
    metric_summaries = summarize_metric_distributions(metric_distributions, hdi_prob)
    return metric_summaries
