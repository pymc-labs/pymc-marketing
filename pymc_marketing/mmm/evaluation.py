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
import pymc as pm
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)

from pymc_marketing.mmm.delayed_saturated_mmm import MMM


# Same error metric as Robyn
def nrmse(y_true, y_pred):
    """Calculate the Normalized Root Mean Square Error (NRMSE).

    Normalization allows for comparison across different data sets and methodologies.
    NRMSE is one of the key metrics used in Robyn MMMs.

    Parameters
    ----------
    y_true : np.array
        Test samples.
    y_pred : np.array
        Predicted samples.

    Returns
    -------
    float
        Normalized root mean square error.
    """
    return root_mean_squared_error(y_true, y_pred) / (y_true.max() - y_true.min())


def nmae(y_true, y_pred):
    """Calculate the Normalized Mean Absolute Error (NMAE).

    Normalization allows for comparison across different data sets and methodologies.

    Parameters
    ----------
    y_true : np.array
        Test samples.
    y_pred : np.array
        Predicted samples.

    Returns
    -------
    float
        Normalized mean absolute error.
    """
    return mean_absolute_error(y_true, y_pred) / (y_true.max() - y_true.min())


def calc_model_diagnostics(mmm: MMM) -> tuple[dict[str, float], az.ELPDData]:
    """Calculate model diagnostics including divergences and Bayesian LOOCV metrics.

    Parameters
    ----------
    mmm : MMM
        Model object with inference data.

    Returns
    -------
    tuple of (dict of str to float, az.ELPDData)
        A tuple containing a dictionary of model diagnostics and the model_loo object.
    """
    # Ensure that mmm has both idata and model attributes and that they are not None
    if not hasattr(mmm, "idata") or mmm.idata is None:
        raise ValueError(
            "The model object 'mmm' must have a non-None 'idata' attribute. "
            "Ensure you've called mmm.fit() before running diagnostics."
        )
    if not hasattr(mmm, "model") or mmm.model is None:
        raise ValueError(
            "The model object 'mmm' must have a non-None 'model' attribute. "
            "Ensure you've called mmm.fit() before running diagnostics."
        )

    # Log divergences
    divergences = mmm.idata["sample_stats"]["diverging"].sum().item()
    step_size = mmm.idata["sample_stats"]["step_size"].mean().item()
    model_diagnostics = {
        "model_diagnostic_divergences": divergences,
        "model_diagnostic_divergences_pct": divergences
        / mmm.idata["sample_stats"]["diverging"].size,
        "model_diagnostic_step_size": step_size,
    }
    if divergences != 0:
        print(f"Model has {divergences} divergences")
    else:
        print("No divergences!")

    # Calculate elemwise log_likelihood of model given posteriors
    pm.compute_log_likelihood(
        mmm.idata, model=mmm.model, progressbar=False
    )  # Used for LOOCV
    model_loo = az.loo(mmm.idata)

    # Log LOOCV metrics
    loocv_metrics = {
        "loocv_elpd_loo": model_loo.elpd_loo,  # expected log pointwise predictive density
        "loocv_se": model_loo.se,  # standard error of elpd
        "loocv_p_loo": model_loo.p_loo,  # effective number of parameters
    }

    # Display LOOCV metrics (including the Pareto k values which aren't logged in MLFlow)
    print(f"LOOCV metrics: {model_loo}")
    # Combine diagnostics and LOOCV metrics
    model_diagnostics.update(loocv_metrics)

    return model_diagnostics, model_loo


def plot_hdi_forest(mmm: MMM, var_names: list) -> plt.Figure:
    """Plot a forest plot to compare high-density intervals (HDIs).

    Plot a forest plot to compare high-density intervals (HDIs) from a given set of
    posterior distributions, as well as their r-hat statistics.

    Parameters
    ----------
    mmm : MMM
        Model object with inference data.
    var_names : list
        List of variable names to include in the forest plot.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    # Ensure that mmm has the required 'idata' attribute and is not None
    if not hasattr(mmm, "idata") or mmm.idata is None:
        raise ValueError(
            "The model object 'mmm' must have a non-None 'idata' attribute. "
            "Ensure you've called mmm.fit() before running diagnostics."
        )
    # Create the forest plot
    fig, ax = plt.subplots(figsize=(12, 8), ncols=2)
    az.plot_forest(
        data=mmm.idata,
        var_names=var_names,
        combined=True,
        ax=ax,
        hdi_prob=0.94,
        # Also plot the split R-hat statistic
        r_hat=True,
    )

    # Set the title for the figure
    fig.suptitle("Posterior Distributions: 94.0% HDI")

    # Return the figure
    return fig


def calc_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics_to_calculate: list[str] | None = None,
    prefix: str | None = None,
) -> dict[str, float]:
    """Calculate evaluation metrics for a given true and predicted dataset.

    Parameters
    ----------
    y_true : np.ndarray
        True values for the dataset.
    y_pred : np.ndarray
        Predictions for the dataset.
    metrics_to_calculate : list of str or None, optional
        List of metrics to calculate. Options include:
            * `r_squared`: Bayesian R-squared.
            * `rmse`: Root Mean Squared Error.
            * `nrmse`: Normalized Root Mean Squared Error.
            * `mae`: Mean Absolute Error.
            * `nmae`: Normalized Mean Absolute Error.
            * `mape`: Mean Absolute Percentage Error.
        Defaults to all metrics if None.
    prefix : str or None, optional
        Prefix to label the metrics (e.g., 'train', 'test'). Defaults to None.

    Returns
    -------
    dict of str to float
        A dictionary containing calculated metrics.
    """
    if prefix is None:
        prefix = ""
    else:
        prefix += "_"

    if metrics_to_calculate is None:
        metrics_to_calculate = ["r_squared", "rmse", "nrmse", "mae", "nmae", "mape"]

    metric_functions = {
        "r_squared": lambda y_true, y_pred: az.r2_score(y_true, y_pred)["r2"],
        "rmse": root_mean_squared_error,
        "nrmse": nrmse,
        "mae": mean_absolute_error,
        "nmae": nmae,
        "mape": mean_absolute_percentage_error,
    }

    metrics = {}

    for metric in metrics_to_calculate:
        if metric in metric_functions:
            metrics[f"{prefix}{metric}"] = metric_functions[metric](y_true, y_pred)
            # Print the metrics in more human-readable format
            print(
                f"{prefix.replace('_', ' ').title()} {metric.upper()} = {metrics[f'{prefix}{metric}'] * 100:.2f}%"
                if metric in ["r_squared", "nrmse", "nmae", "mape"]
                else f"{metrics[f'{prefix}{metric}']:.4f}"
            )

    return metrics
