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

# import arviz as az
# import matplotlib.pyplot as plt
# import numpy as np
# import numpy.typing as npt
# from sklearn.metrics import (
#     mean_absolute_error,
#     mean_absolute_percentage_error,
#     root_mean_squared_error,
# )

# from pymc_marketing.mmm.mmm import MMM


# class MMMEvaluator:
#     """A class to evaluate and diagnose PyMC-Marketing MMMs.

#     Parameters
#     ----------
#     model : MMM
#         The PyMC-Marketing MMM object.
#     idata : az.InferenceData
#         The InferenceData object returned by the sampling method.

#     """

#     def __init__(self, model: MMM):
#         if not hasattr(model, "idata") or model.idata is None:
#             raise ValueError("The model object must have a non-None 'idata' attribute.")
#         if not hasattr(model, "model") or model.model is None:
#             raise ValueError("The model object must have a non-None 'model' attribute.")

#         self.model = model
#         self.model_diagnostics = None
#         self.model_loo = None

#     # Same error metric as Robyn
#     def nrmse(self, y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
#         """Calculate the Normalized Root Mean Square Error (NRMSE).

#         Normalization allows for comparison across different data sets and methodologies.
#         NRMSE is one of the key metrics used in Robyn MMMs.

#         Parameters
#         ----------
#         y_true : npt.ArrayLike
#             True values for target metric
#         y_pred : npt.ArrayLike
#             Predicted values for target metric

#         Returns
#         -------
#         float
#             Normalized root mean square error.
#         """
#         return root_mean_squared_error(y_true, y_pred) / (y_true.max() - y_true.min())

#     def nmae(self, y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
#         """Calculate the Normalized Mean Absolute Error (NMAE).

#         Normalization allows for comparison across different data sets and methodologies.

#         Parameters
#         ----------
#         y_true : npt.ArrayLike
#             True values for target metric
#         y_pred : npt.ArrayLike
#             Predicted values for target metric

#         Returns
#         -------
#         float
#             Normalized mean absolute error.
#         """
#         return mean_absolute_error(y_true, y_pred) / (y_true.max() - y_true.min())

#     def plot_hdi_forest(self, var_names: list, **plot_kwargs) -> plt.Figure:
#         """Plot a forest plot to compare high-density intervals (HDIs).

#         Plot a forest plot to compare high-density intervals (HDIs) from a given set of
#         posterior distributions, as well as their r-hat statistics.

#         Parameters
#         ----------
#         var_names : list
#             List of variable names to include in the forest plot.
#         **plot_kwargs
#             Additional keyword arguments to pass to the az.plot_forest method.

#         Returns
#         -------
#         plt.Figure
#             The matplotlib figure object.
#         """
#         # Ensure that the model has the required 'idata' attribute and is not None
#         if not hasattr(self.model, "idata") or self.model.idata is None:
#             raise ValueError(
#                 "The model object must have a non-None 'idata' attribute. "
#                 "Ensure you've called model.fit() before running diagnostics."
#             )
#         # Create the forest plot
#         fig, ax = plt.subplots(figsize=(12, 8), ncols=2)
#         az.plot_forest(
#             data=self.model.idata,
#             var_names=var_names,
#             combined=True,
#             ax=ax,
#             hdi_prob=0.94,
#             # Also plot the split R-hat statistic
#             r_hat=True,
#             **plot_kwargs,
#         )

#         # Set the title for the figure
#         fig.suptitle("Posterior Distributions: 94.0% HDI")

#         # Return the figure
#         return fig

#     def calc_metrics(
#         self,
#         y_true: np.ndarray,
#         y_pred: np.ndarray,
#         metrics_to_calculate: list[str] | None = None,
#         prefix: str | None = None,
#     ) -> dict[str, float]:
#         """Calculate evaluation metrics for a given true and predicted dataset.

#         Parameters
#         ----------
#         y_true : np.ndarray
#             True values for the dataset.
#         y_pred : np.ndarray
#             Predictions for the dataset.
#         metrics_to_calculate : list of str or None, optional
#             List of metrics to calculate. Options include:
#                 * `r_squared`: Bayesian R-squared.
#                 * `rmse`: Root Mean Squared Error.
#                 * `nrmse`: Normalized Root Mean Squared Error.
#                 * `mae`: Mean Absolute Error.
#                 * `nmae`: Normalized Mean Absolute Error.
#                 * `mape`: Mean Absolute Percentage Error.
#             Defaults to all metrics if None.
#         prefix : str or None, optional
#             Prefix to label the metrics (e.g., 'train', 'test'). Defaults to None.

#         Returns
#         -------
#         dict of str to float
#             A dictionary containing calculated metrics.
#         """
#         if prefix is None:
#             prefix = ""
#         else:
#             prefix += "_"

#         if metrics_to_calculate is None:
#             metrics_to_calculate = ["r_squared", "rmse", "nrmse", "mae", "nmae", "mape"]

#         metric_functions = {
#             "r_squared": lambda y_true, y_pred: az.r2_score(y_true, y_pred)["r2"],
#             "rmse": self.root_mean_squared_error,
#             "nrmse": self.nrmse,
#             "mae": mean_absolute_error,
#             "nmae": self.nmae,
#             "mape": mean_absolute_percentage_error,
#         }

#         return {
#             f"{prefix}{metric}": metric_functions[metric](y_true, y_pred)
#             for metric in metrics_to_calculate
#         }
