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

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from loguru import logger
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)

from pymc_marketing.mmm.delayed_saturated_mmm import MMM


# Same error metric as Robyn
def nrmse(y_true, y_pred):
    """
    Normalized Root Mean Square Error. Normalization allows for comparison across
    different data sets and methodologies. e.g. NRMSE is one of the key metrics used
    in Robyn MMMs.
    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples
    Returns:
        [float]: normalized root mean square error
    """
    return root_mean_squared_error(y_true, y_pred) / (y_true.max() - y_true.min())


def nmae(y_true, y_pred):
    """
    Normalized Mean Absolute Error. Normalization allows for comparison across
    different data sets and methodologies.
    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples
    Returns:
        [float]: normalized mean absolute error
    """
    return mean_absolute_error(y_true, y_pred) / (y_true.max() - y_true.min())


def calc_model_diagnostics(mmm: MMM) -> tuple[dict[str, float], az.ELPDData]:
    """
    Calculate model diagnostics including divergences and Bayesian LOOCV metrics.

    Parameters:
    - mmm: Model object with inference data

    Returns:
    - Tuple[Dict[str, float], az.ELPDData]: A tuple containing a dictionary of
      model diagnostics and the model_loo object

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
        logger.error(f"Model has {divergences} divergences")
    else:
        logger.success("No divergences!")

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
    logger.info(f"LOOCV metrics: {model_loo}")
    # Combine diagnostics and LOOCV metrics
    model_diagnostics.update(loocv_metrics)

    return model_diagnostics, model_loo


def plot_hdi_forest(mmm: MMM, var_names: list) -> plt.Figure:
    """
    Plot a forest plot to compare high-density intervals (HDIs) from a given set of
    posterior distributions, as well as their r-hat statistic.

    Parameters:
    - mmm: Model object with inference data
    - var_names: list: List of variable names to include in the forest plot

    Returns:
    - plt.Figure: The matplotlib figure object
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
        hdi_prob=0.95,
        # Also plot the split R-hat statistic
        r_hat=True,
    )

    # Set the title for the figure
    fig.suptitle("Posterior Distributions: 95.0% HDI")

    # Return the figure
    return fig


def calc_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, prefix: str | None = None
) -> dict[str, float]:
    """
    Calculate metrics including R-squared, RMSE, NRMSE, MAE, NMAE, and MAPE for a given true and predicted dataset.

    Parameters:
    - y_true: np.ndarray: True values for the dataset.
    - y_pred: np.ndarray: Predictions for the dataset.
    - prefix: Optional[str]: Prefix to label the metrics (e.g., 'train', 'test'). Defaults to None.

    Returns:
    - Dict[str, float]: A dictionary containing calculated metrics.
    """
    if prefix is None:
        prefix = ""
    else:
        prefix += "_"

    metrics = {
        f"{prefix}r_squared": np.nan,
        f"{prefix}rmse": np.nan,
        f"{prefix}nrmse": np.nan,
        f"{prefix}mae": np.nan,
        f"{prefix}nmae": np.nan,
        f"{prefix}mape": np.nan,
    }

    try:
        # Calculate Bayesian R-squared
        metrics[f"{prefix}r_squared"] = az.r2_score(y_true, y_pred)["r2"]
        logger.info(
            f"{prefix.replace('_', ' ').title()} R-Squared = {metrics[f'{prefix}r_squared'] * 100:.2f}%"
        )

        # Calculate RMSE
        metrics[f"{prefix}rmse"] = root_mean_squared_error(y_true, y_pred)
        logger.info(
            f"{prefix.replace('_', ' ').title()} RMSE = {metrics[f'{prefix}rmse']:.4f}"
        )

        # Calculate NRMSE
        metrics[f"{prefix}nrmse"] = nrmse(y_true, y_pred)
        logger.info(
            f"{prefix.replace('_', ' ').title()} NRMSE = {metrics[f'{prefix}nrmse'] * 100:.2f}%"
        )

        # Calculate MAE
        metrics[f"{prefix}mae"] = mean_absolute_error(y_true, y_pred)
        logger.info(
            f"{prefix.replace('_', ' ').title()} MAE = {metrics[f'{prefix}mae']:.4f}"
        )

        # Calculate NMAE
        metrics[f"{prefix}nmae"] = nmae(y_true, y_pred)
        logger.info(
            f"{prefix.replace('_', ' ').title()} NMAE = {metrics[f'{prefix}nmae'] * 100:.2f}%"
        )

        # Calculate MAPE
        metrics[f"{prefix}mape"] = mean_absolute_percentage_error(y_true, y_pred)
        logger.info(
            f"{prefix.replace('_', ' ').title()} MAPE = {metrics[f'{prefix}mape'] * 100:.2f}%"
        )

    except ValueError:
        logger.error("Some NaNs were present")

    return metrics
