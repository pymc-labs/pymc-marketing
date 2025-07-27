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
"""MLflow logging utilities for PyMC models.

This module provides utilities to log various aspects of PyMC models to MLflow
which is then extended to PyMC-Marketing models.

Autologging is supported for PyMC models and PyMC-Marketing models. This including
logging of sampler diagnostics, model information, data used in the model, and
InferenceData objects.

The autologging can be enabled by calling the `autolog` function. The following functions
are patched:

- `pymc.sample`:
    - :func:`log_versions`: Log the versions of PyMC-Marketing, PyMC, and ArviZ to MLflow.
    - :func:`log_model_derived_info`: Log types of parameters, coords, model graph, etc.
    - :func:`log_sample_diagnostics`: Log information derived from the InferenceData object.
    - :func:`log_arviz_summary`: Log table of summary statistics about estimated parameters
    - :func:`log_metadata`: Log the metadata of the data used in the model.
    - :func:`log_error`: Log the traceback and exception if an error occurs during sampling.
- `pymc.find_MAP`:
    - :func:`log_model_derived_info`: Log types of parameters, coords, model graph, etc.
- `MMM.fit`:
    - All parameters, metrics, and artifacts from `pymc.sample`
    - :func:`log_mmm_configuration`: Log the configuration of the MMM model.
- `CLVModel.fit`:
    - Information dependent on fit method used (MCMC or MAP)
    - Model type and fit method

Examples
--------
Autologging for a PyMC model:

.. code-block:: python

    import mlflow

    import pymc as pm

    import pymc_marketing.mlflow

    pymc_marketing.mlflow.autolog()

    # Usual PyMC model code
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=1)
        obs = pm.Normal("obs", mu=mu, sigma=1, observed=[1, 2, 3])

    # Incorporate into MLflow workflow
    mlflow.set_experiment("PyMC Experiment")

    with mlflow.start_run():
        idata = pm.sample(model=model)

Autologging for a PyMC-Marketing MMM:

.. code-block:: python

    import pandas as pd

    import mlflow

    from pymc_marketing.mmm import (
        GeometricAdstock,
        LogisticSaturation,
        MMM,
    )
    from pymc_marketing.paths import data_dir
    import pymc_marketing.mlflow

    pymc_marketing.mlflow.autolog(log_mmm=True)

    # Usual PyMC-Marketing model code

    file_path = data_dir / "mmm_example.csv"
    data = pd.read_csv(file_path, parse_dates=["date_week"])

    X = data.drop("y", axis=1)
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

    # Incorporate into MLflow workflow

    mlflow.set_experiment("MMM Experiment")

    with mlflow.start_run():
        idata = mmm.fit(X, y)

        # Additional specific logging
        fig = mmm.plot_components_contributions()
        mlflow.log_figure(fig, "components.png")

Autologging for a PyMC-Marketing CLV model:

.. code-block:: python

    import pandas as pd

    import mlflow

    from pymc_marketing.clv import BetaGeoModel
    from pymc_marketing.paths import data_dir

    import pymc_marketing.mlflow

    pymc_marketing.mlflow.autolog(log_clv=True)

    mlflow.set_experiment("CLV Experiment")

    file_path = data_dir / "clv_quickstart.csv"
    data = pd.read_csv(file_path)
    data["customer_id"] = data.index

    model = BetaGeoModel(data=data)

    with mlflow.start_run():
        model.fit()

"""

import logging
import os
import tempfile
import traceback
import warnings
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, Literal

import arviz as az
import numpy.typing as npt
import pandas as pd
import pymc as pm
import xarray as xr
from pymc.model.core import Model
from pytensor.tensor import TensorVariable

try:
    import mlflow
except ImportError:  # pragma: no cover
    msg = "This module requires mlflow. Install using `pip install mlflow`"
    raise ImportError(msg)

from mlflow.utils.autologging_utils import autologging_integration

from pymc_marketing.clv.models.basic import CLVModel
from pymc_marketing.mmm import MMM
from pymc_marketing.mmm.evaluation import compute_summary_metrics
from pymc_marketing.version import __version__

FLAVOR_NAME = "pymc"


PYMC_MARKETING_ISSUE = "https://github.com/pymc-labs/pymc-marketing/issues/new"
warning_msg = (
    "This functionality is experimental and subject to change. "
    "If you encounter any issues or have suggestions, please raise them at: "
    f"{PYMC_MARKETING_ISSUE}"
)
warnings.warn(warning_msg, FutureWarning, stacklevel=1)


def _exclude_tuning(func):
    def callback(trace, draw):
        if draw.tuning:
            return

        return func(trace, draw)

    return callback


def _take_every(n: int):
    def decorator(func):
        def callback(trace, draw):
            if draw.draw_idx % n != 0:
                return

            return func(trace, draw)

        return callback

    return decorator


def create_log_callback(
    stats: list[str] | None = None,
    parameters: list[str] | None = None,
    exclude_tuning: bool = True,
    take_every: int = 100,
):
    """Create callback function to log sample stats and parameter values to MLflow during sampling.

    This callback only works for the "pymc" sampler.

    Parameters
    ----------
    stats : list of str, optional
        List of sample statistics to log from the Draw
    parameters : list of str, optional
        List of parameters to log from the Draw
    exclude_tuning : bool, optional
        Whether to exclude tuning steps from logging. Defaults to True.
    take_every : int, optional
        Specifies the interval at which to log values. Defaults to 100.

    Returns
    -------
    callback : Callable
        The callback function to log sample stats and parameter values to MLflow during sampling

    Examples
    --------
    Create example model:

    .. code-block:: python

        import pymc as pm

        with pm.Model() as model:
            mu = pm.Normal("mu")
            sigma = pm.HalfNormal("sigma")
            obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=[1, 2, 3])

    Log off divergences and logp every 100th draw:

    .. code-block:: python

        import mlflow

        from pymc_marketing.mlflow import create_log_callback

        callback = create_log_callback(
            stats=["diverging", "model_logp"],
            take_every=100,
        )

        mlflow.set_experiment("Live Tracking Stats")

        with mlflow.start_run():
            idata = pm.sample(model=model, callback=callback)

    Log the parameters `mu` and `sigma_log__` every 100th draw:

    .. code-block:: python

        import mlflow

        from pymc_marketing.mlflow import create_log_callback

        callback = create_log_callback(
            parameters=["mu", "sigma_log__"],
            take_every=100,
        )

        mlflow.set_experiment("Live Tracking Parameters")

        with mlflow.start_run():
            idata = pm.sample(model=model, callback=callback)

    """
    if not stats and not parameters:
        raise ValueError("At least one of `stats` or `parameters` must be provided.")

    def callback(_, draw):
        prefix = f"chain_{draw.chain}"
        for stat in stats or []:
            mlflow.log_metric(
                key=f"{prefix}/{stat}",
                value=draw.stats[0][stat],
                step=draw.draw_idx,
            )

        for parameter in parameters or []:
            mlflow.log_metric(
                key=f"{prefix}/{parameter}",
                value=draw.point[parameter],
                step=draw.draw_idx,
            )

    if exclude_tuning:
        callback = _exclude_tuning(callback)

    if take_every:
        callback = _take_every(n=take_every)(callback)

    return callback


def _log_and_remove_artifact(path: str | Path) -> None:
    """Log an artifact to MLflow and then remove the local file.

    Parameters
    ----------
    path : str | Path
        Path to the artifact file to log and remove.
    """
    mlflow.log_artifact(str(path))
    os.remove(path)


def _force_load_idata_groups(idata: az.InferenceData) -> None:
    """Force load all groups into memory since ArviZ does lazy loading.

    Parameters
    ----------
    idata : az.InferenceData
        The InferenceData object to force load.
    """
    for group in idata.groups():
        # Convert each group to an in-memory dataset
        if hasattr(idata, group):
            group_data = getattr(idata, group)
            if hasattr(group_data, "load"):
                group_data.load()


def log_arviz_summary(
    idata: az.InferenceData,
    path: str | Path,
    var_names: list[str] | None = None,
    **summary_kwargs,
) -> None:
    """Log the ArviZ summary as an artifact on MLflow.

    Automatically removes the file after logging.

    Parameters
    ----------
    idata : az.InferenceData
        The InferenceData object returned by the sampling method.
    path : str | Path
        The path to save the summary as HTML.
    var_names : list[str], optional
        The names of the variables to include in the summary. Default is
        all the variables in the InferenceData object.
    summary_kwargs : dict
        Additional keyword arguments to pass to `az.summary`.

    """
    df_summary = az.summary(idata, var_names=var_names, **summary_kwargs)
    df_summary.to_html(path)
    mlflow.log_artifact(str(path))
    os.remove(path)


def log_metadata(model: Model, idata: az.InferenceData) -> None:
    """Log the metadata of the data used in the model to MLflow.

    Saved in the form of numpy arrays based on all the constant and observed data
    in the model.

    Parameters
    ----------
    model : Model
        The PyMC model object.
    idata : az.InferenceData
        The InferenceData object returned by the sampling method.

    """
    data_vars: list[TensorVariable] = model.data_vars

    if "constant_data" in idata:
        features = {
            var.name: idata.constant_data[var.name].to_numpy()
            for var in data_vars
            if var.name in idata.constant_data
        }
    else:
        features = {}

    targets = {
        var.name: idata.observed_data[var.name].to_numpy()
        for var in model.observed_RVs
        if var.name in idata.observed_data
    }

    if not features and not targets:
        return

    data = mlflow.data.from_numpy(features=features, targets=targets)
    mlflow.log_input(data, context="sample")


def log_model_graph(model: Model, path: str | Path) -> None:
    """Log the model graph PDF as artifact on MLflow.

    Automatically removes the file after logging.

    Parameters
    ----------
    model : Model
        The PyMC model object.
    path : str | Path
        The path to save the model graph

    """
    try:
        graph = pm.model_to_graphviz(model)
    except ImportError as e:
        msg = (
            "Unable to render the model graph. Please install the graphviz package. "
            f"{e}"
        )
        logging.info(msg)

        return None

    try:
        saved_path = graph.render(path)
    except Exception as e:
        msg = f"Unable to render the model graph. {e}"
        logging.info(msg)
        return None
    else:
        _log_and_remove_artifact(saved_path)
        os.remove(path)


def _get_random_variable_name(rv) -> str:
    # Taken from new version of pymc/model_graph.py
    symbol = rv.owner.op.__class__.__name__

    if symbol.endswith("RV"):
        symbol = symbol[:-2]

    return symbol


def log_types_of_parameters(model: Model) -> None:
    """Log the types of parameters in a PyMC model to MLflow.

    Parameters
    ----------
    model : Model
        The PyMC model object.

    """
    mlflow.log_param("n_free_RVs", len(model.free_RVs))
    mlflow.log_param("n_observed_RVs", len(model.observed_RVs))
    mlflow.log_param("n_deterministics", len(model.deterministics))
    mlflow.log_param("n_potentials", len(model.potentials))


def log_likelihood_type(model: Model) -> None:
    """Save the likelihood type of the model to MLflow.

    Parameters
    ----------
    model : Model
        The PyMC model object.

    """
    observed_RVs_types = [_get_random_variable_name(rv) for rv in model.observed_RVs]
    if len(observed_RVs_types) == 1:
        mlflow.log_param("likelihood", observed_RVs_types[0])
    elif len(observed_RVs_types) > 1:
        mlflow.log_param("observed_RVs_types", observed_RVs_types)


def log_model_derived_info(model: Model) -> None:
    """Log various model derived information to MLflow.

    Includes:

    - The types of parameters in the model.
    - The likelihood type of the model.
    - The model representation (str).
    - The model coordinates (coords.json).

    Parameters
    ----------
    model : Model
        The PyMC model object.

    """
    log_types_of_parameters(model)

    mlflow.log_text(model.str_repr(), "model_repr.txt")

    if model.coords:
        mlflow.log_dict(model.coords, "coords.json")

    log_model_graph(model, "model_graph")
    log_likelihood_type(model)


def log_sample_diagnostics(
    idata: az.InferenceData,
    tune: int | None = None,
) -> None:
    """Log sample diagnostics to MLflow.

    Includes:

    - The total number of divergences
    - The total sampling time in seconds (if available)
    - The time per draw in seconds (if available)
    - The number of tuning steps (if available)
    - The number of draws
    - The number of chains
    - The inference library used
    - The version of the inference library
    - The version of ArviZ

    Parameters
    ----------
    idata : az.InferenceData
        The InferenceData object returned by the sampling method.
    tune : int, optional
        The number of tuning steps used in sampling. Derived from the
        inference data if not provided.

    """
    if "posterior" not in idata:
        raise KeyError("InferenceData object does not contain the group posterior.")

    if "sample_stats" not in idata:
        raise KeyError("InferenceData object does not contain the group sample_stats.")

    posterior = idata["posterior"]
    sample_stats = idata["sample_stats"]

    diverging = sample_stats["diverging"]

    chains = posterior.sizes["chain"]
    draws = posterior.sizes["draw"]
    posterior_samples = chains * draws

    tuning_step = sample_stats.attrs.get("tuning_steps", tune)
    if tuning_step is not None:
        tuning_samples = tuning_step * chains
        mlflow.log_param("tuning_steps", tuning_step)
        mlflow.log_param("tuning_samples", tuning_samples)

    total_divergences = diverging.sum().item()
    mlflow.log_metric("total_divergences", total_divergences)
    if sampling_time := sample_stats.attrs.get("sampling_time"):
        mlflow.log_metric("sampling_time", sampling_time)
        mlflow.log_metric(
            "time_per_draw",
            sampling_time / posterior_samples,
        )

    mlflow.log_param("draws", draws)
    mlflow.log_param("chains", chains)
    mlflow.log_param("posterior_samples", posterior_samples)

    if inference_library := posterior.attrs.get("inference_library"):
        mlflow.log_param("inference_library", inference_library)
        mlflow.log_param(
            "inference_library_version",
            posterior.attrs["inference_library_version"],
        )


def log_inference_data(
    idata: az.InferenceData,
    save_file: str | Path = "idata.nc",
) -> None:
    """Log the InferenceData to MLflow.

    Parameters
    ----------
    idata : az.InferenceData
        The InferenceData object returned by the sampling method.
    save_file : str | Path
        The path to save the InferenceData object as a netCDF file.

    """
    idata.to_netcdf(str(save_file))
    _log_and_remove_artifact(save_file)


def log_mmm_evaluation_metrics(
    y_true: npt.NDArray | pd.Series,
    y_pred: npt.NDArray | xr.DataArray,
    metrics_to_calculate: list[str] | None = None,
    hdi_prob: float = 0.94,
    prefix: str = "",
) -> None:
    """Log evaluation metrics produced by `pymc_marketing.mmm.evaluation.compute_summary_metrics()` to MLflow.

    Parameters
    ----------
    y_true : npt.NDArray | pd.Series
        The true values of the target variable.
    y_pred : npt.NDArray | xr.DataArray
        The predicted values of the target variable.
    metrics_to_calculate : list of str or None, optional
        List of metrics to calculate. If None, all available metrics will be calculated.
        Options include:
            * `r_squared`: Bayesian R-squared.
            * `rmse`: Root Mean Squared Error.
            * `nrmse`: Normalized Root Mean Squared Error.
            * `mae`: Mean Absolute Error.
            * `nmae`: Normalized Mean Absolute Error.
            * `mape`: Mean Absolute Percentage Error.
    hdi_prob : float, optional
        The probability mass of the highest density interval. Defaults to 0.94.
    prefix : str, optional
        Prefix to add to the metric names. Defaults to "".

    Examples
    --------
    Log in-sample evaluation metrics for a PyMC-Marketing MMM model:

    .. code-block:: python

        import mlflow

        from pymc_marketing.mmm import MMM

        mmm = MMM(...)
        mmm.fit(X, y)

        predictions = mmm.sample_posterior_predictive(X)

        with mlflow.start_run():
            log_mmm_evaluation_metrics(y, predictions["y"])

    """
    metric_summaries = compute_summary_metrics(
        y_true=y_true,
        y_pred=y_pred,
        metrics_to_calculate=metrics_to_calculate,
        hdi_prob=hdi_prob,
    )

    if prefix and not prefix.endswith("_"):
        prefix = f"{prefix}_"

    for metric, stats in metric_summaries.items():
        for stat, value in stats.items():
            # mlflow doesn't support % in metric names
            mlflow.log_metric(f"{prefix}{metric}_{stat.replace('%', '')}", value)


class MMMWrapper(mlflow.pyfunc.PythonModel):
    """A class to prepare a PyMC-Marketing Mix Model (MMM) for logging and registering in MLflow.

    This class extends MLflow's PythonModel to handle prediction tasks using a PyMC-based MMM.
    It supports several prediction methods, including point-prediction, posterior and prior predictive sampling.

    Parameters
    ----------
    model : pymc_marketing.mmm.MMM
        The marketing mix model to be registered and used for predictions.
    predict_method : str, optional, default="predict"
        The default prediction method to use, such as "predict",
        "sample_posterior_predictive", or "sample_prior_predictive".
    extend_idata : bool, default=False
        Boolean determining whether the predictions should be added to inference data object. Defaults to False.
    combined : bool, default=True
        Combine chain and draw dims into sample. Won't work if a dim named sample already exists. Defaults to True.
    include_last_observations : bool, default=False
        Boolean determining whether to include the last observations of the training data in order to carry over
        costs with the adstock transformation. Assumes that X are the next predictions following the
        training data. Defaults to False.
    original_scale : bool, default=True
        Boolean determining whether to return the predictions in the original scale of the target variable.
    var_names : list of str, optional, default=None
        The variable names to include in the predictions.
    sample_kwargs : dict, optional
        Additional keyword arguments to pass to the selected sampling methods.

    """

    def __init__(
        self,
        model: MMM,
        predict_method: Literal[
            "predict", "sample_posterior_predictive", "sample_prior_predictive"
        ] = "predict",
        extend_idata: bool = False,
        combined: bool = True,
        include_last_observations: bool = False,
        original_scale: bool = True,
        var_names: list[str] | None = None,
        **sample_kwargs: dict,
    ):
        self.model = model
        self.predict_method = predict_method
        self.extend_idata = extend_idata
        self.combined = combined
        self.include_last_observations = include_last_observations
        self.original_scale = original_scale
        self.var_names = (
            var_names if var_names is not None else [model.output_var]
        )  # Initialize if not provided
        self.sample_kwargs = sample_kwargs

    def predict(
        self, context: Any, model_input, params: dict[str, Any] | None = None
    ) -> Any:
        """Perform predictions or sampling using the specified prediction method.

        Parameters
        ----------
        context : Any
            The context in which the model is running. Isn't specified by users but is passed by MLflow.
        model_input : array, shape (n_pred, n_features)
            The input data used for prediction.
        params : dict, optional
            A dictionary of parameters to specify the prediction method.

        Returns
        -------
        ndarray or InferenceData
            The predictions or samples generated by the model.

        Raises
        ------
        ValueError
            If an unsupported prediction method is specified.

        """
        # Use the class-level predict_method if params is not provided or doesn't contain 'predict_method'
        params = params or {"predict_method": "predict"}
        predict_method = params.get("predict_method", self.predict_method)

        if predict_method == "predict":
            return self.model.predict(
                model_input,
                extend_idata=self.extend_idata,
                include_last_observations=self.include_last_observations,
                original_scale=self.original_scale,
                var_names=self.var_names,
                **self.sample_kwargs,  # type: ignore[arg-type]
            )
        elif predict_method == "sample_posterior_predictive":
            return self.model.sample_posterior_predictive(
                model_input,
                extend_idata=self.extend_idata,
                combined=self.combined,
                include_last_observations=self.include_last_observations,
                original_scale=self.original_scale,
                var_names=self.var_names,
                **self.sample_kwargs,  # type: ignore[arg-type]
            )
        elif predict_method == "sample_prior_predictive":
            return self.model.sample_prior_predictive(
                model_input,
                extend_idata=self.extend_idata,
                combined=self.combined,
                var_names=self.var_names,
                **self.sample_kwargs,  # type: ignore[arg-type]
            )
        else:
            raise ValueError(
                f"The prediction method '{predict_method}' is not supported."
            )


def log_mmm(
    mmm: MMM,
    artifact_path: str = "model",
    registered_model_name: str | None = None,
    extend_idata: bool = False,
    combined: bool = True,
    include_last_observations: bool = False,
    original_scale: bool = True,
) -> None:
    """Log a PyMC-Marketing MMM as a native MLflow model for the current run.

    Parameters
    ----------
    mmm : MMM
        The MMM to be logged.
    artifact_path : str, optional
        The path to the artifact to be logged. Defaults to "mmm_model".
    conda_env : dict, optional
        A dictionary representation of a Conda environment. Defaults to the default conda environment.
    registered_model_name : str, optional
        The name of the registered model to be logged. Defaults to None.
        If specified, the model will be registered under this name, otherwise it will not be registered.
    extend_idata : bool, optional
        Whether to extend the inference data with predictions. Used for all prediction methods.
        Defaults to False.
    combined : bool, optional
        Whether to combine chain and draw dims into sample. Won't work if a dim named sample
        already exists. Used for posterior/prior predictive sampling. Defaults to True.
    include_last_observations : bool, optional
        Whether to include the last observations of training data for adstock transformation.
        Assumes X are next predictions following training data. Used for all prediction
        methods. Defaults to False.
    original_scale : bool, optional
        Whether to return predictions in original scale of target variable. Used for all
        prediction methods. Defaults to True.

    Notes
    -----
    This function logs the model as a native MLflow model, this is different to the full model object,
    which includes the InferenceData. Doing this allows for the model to be stored in the MLFlow registry,
    helping with model versioning and deployment.

    Examples
    --------
    MLFlow Registering for a PyMC-Marketing MMM:

    .. code-block:: python

        import pandas as pd

        import mlflow

        from pymc_marketing.mmm import (
            GeometricAdstock,
            LogisticSaturation,
            MMM,
        )
        from pymc_marketing.paths import data_dir
        import pymc_marketing.mlflow
        from pymc_marketing.mlflow import log_mmm

        pymc_marketing.mlflow.autolog(log_mmm=True)

        # Usual PyMC-Marketing model code

        file_path = data_dir / "mmm_example.csv"
        data = pd.read_csv(file_path, parse_dates=["date_week"])

        X = data.drop("y", axis=1)
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

        mlflow.set_experiment("MMM Experiment")

        with mlflow.start_run():
            idata = mmm.fit(X, y)

            # Additional specific logging
            fig = mmm.plot_components_contributions()
            mlflow.log_figure(fig, "components.png")

            model_info = log_mmm(
                mmm=mmm,
                registered_model_name="my_amazing_mmm",
                include_last_observations=True,
                original_scale=False,
            )
    """
    # Incorporate MMM into MLflow workflow
    mlflow_mmm = MMMWrapper(
        model=mmm,
        extend_idata=extend_idata,
        combined=combined,
        include_last_observations=include_last_observations,
        original_scale=original_scale,
    )

    mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=mlflow_mmm,
    )
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/{artifact_path}"

    if registered_model_name:
        mlflow.register_model(model_uri, registered_model_name)


def load_mmm(
    run_id: str,
    full_model: bool = False,
    keep_idata: bool = False,
    artifact_path: str = "model",
    dst_path: str | None = None,
) -> mlflow.pyfunc.PyFuncModel | MMM:
    """
    Load a PyMC-Marketing MMM model from MLflow.

    Can either load the full model including the InferenceData, or just the lighter PyFuncModel version.

    Parameters
    ----------
    run_id : str
        The MLflow run ID from which to load the model.
    full_model : bool, default=True
        If True, load the full MMM model including the InferenceData.
    keep_idata : bool, default=False
        If True, keep the downloaded InferenceData saved locally.
    artifact_path : str, default="model"
        The artifact path within the run where the model is stored.
    dst_path : str | None, default=None
        The local destination path where the InferenceData will be downloaded.
        If None, defaults to "idata_{run_id}" to avoid conflicts when loading multiple models.

    Returns
    -------
    model : mlflow.pyfunc.PyFuncModel | MMM
        The loaded MLflow PyFuncModel or MMM model.


    Examples
    --------
    .. code-block:: python

        # Load model using run_id
        model = load_mmm(run_id="your_run_id", full_model=True, keep_idata=True)
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"

    if not full_model:
        model = mlflow.pyfunc.load_model(model_uri)
        return model

    # Create unique destination path if not provided
    if dst_path is None:
        dst_path = f"idata_{run_id}"

    idata_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="idata.nc", dst_path=dst_path
    )

    model = MMM.load(idata_path)

    if not keep_idata:
        _force_load_idata_groups(model.idata)

        try:
            os.remove(idata_path)
            os.rmdir(dst_path)
        except OSError:
            warnings.warn(
                f"Could not remove temporary files at {dst_path}. You may want to remove them manually.",
                UserWarning,
                stacklevel=2,
            )

    return model


def log_versions() -> None:
    """Log the versions of PyMC-Marketing, PyMC, and ArviZ to MLflow."""
    mlflow.log_param("pymc_marketing_version", __version__)
    mlflow.log_param("pymc_version", pm.__version__)
    mlflow.log_param("arviz_version", az.__version__)


def log_mmm_configuration(mmm: MMM) -> None:
    """Log the configuration of the MMM model to MLflow."""
    attrs = mmm.create_idata_attrs()
    mlflow.log_params(attrs)

    mlflow.log_param("adstock_name", mmm.adstock.lookup_name)
    mlflow.log_param("saturation_name", mmm.saturation.lookup_name)


def log_error(func: Callable, file_name: str):
    """Log arbitrary caught error and traceback to MLflow.

    .. note::

        The error will still be raised with the program. It is just logged
        to MLflow

    Parameters
    ----------
    func : Callable
        Arbitrary function
    file_name : str
        The name of the MLflow artifact

    Examples
    --------

    .. code-block:: python

        import mlflow

        from pymc_marketing.mlflow import log_error


        def raising_function():
            raise NotImplementedError("Sorry. Not implemented")


        func = log_error(raising_function, file_name="raising-function")

        with mlflow.start_run():
            func()

    """

    @wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            with tempfile.TemporaryDirectory() as tmp_dir:
                path = Path(tmp_dir) / file_name
                with path.open("w") as f:
                    traceback.print_exc(file=f)

                mlflow.log_artifact(str(path))
            raise e

    return wrapped


@autologging_integration(FLAVOR_NAME)
def autolog(
    log_sampler_info: bool = True,
    log_metadata_info: bool = True,
    log_model_info: bool = True,
    sample_error_file: str | None = "sample-error.txt",
    summary_var_names: list[str] | None = None,
    arviz_summary_kwargs: dict | None = None,
    log_mmm: bool = True,
    log_clv: bool = True,
    disable: bool = False,
    silent: bool = False,
) -> None:
    """Autologging support for PyMC models and PyMC-Marketing models.

    Includes logging of sampler diagnostics, model information, data used in the
    model, and InferenceData objects upon sampling the models.

    For more information about MLflow, see
    https://mlflow.org/docs/latest/python_api/mlflow.html

    Parameters
    ----------
    log_sampler_info : bool, optional
        Whether to log sampler diagnostics. Default is True.
    log_metadata_info : bool, optional
        Whether to log the metadata of inputs used in the model. Default is True.
    log_model_info : bool, optional
        Whether to log model information. Default is True.
    sample_error_file : str, optional
        The name of the file to log the error if an error occurs during sampling. If
        None, the error will not be logged. Default is "sample-error.txt".
    summary_var_names : list[str], optional
        The names of the variables to include in the ArviZ summary. Default is
        all the variables in the InferenceData object.
    arviz_summary_kwargs : dict, optional
        Additional keyword arguments to pass to `az.summary`.
    log_mmm : bool, optional
        Whether to log PyMC-Marketing MMM models. Default is True.
    log_clv : bool, optional
        Whether to log PyMC-Marketing CLV models. Default is True.
    disable : bool, optional
        Whether to disable autologging. Default is False.
    silent : bool, optional
        Whether to suppress all warnings. Default is False.

    Examples
    --------
    Autologging for a PyMC model:

    .. code-block:: python

        import mlflow

        import pymc as pm

        import pymc_marketing.mlflow

        pymc_marketing.mlflow.autolog()

        # Usual PyMC model code
        with pm.Model() as model:
            mu = pm.Normal("mu", mu=0, sigma=1)
            obs = pm.Normal("obs", mu=mu, sigma=1, observed=[1, 2, 3])

        # Incorporate into MLflow workflow
        mlflow.set_experiment("PyMC Experiment")

        with mlflow.start_run():
            idata = pm.sample(model=model)

    Autologging for a PyMC-Marketing MMM:

    .. code-block:: python

        import pandas as pd

        import mlflow

        from pymc_marketing.mmm import (
            GeometricAdstock,
            LogisticSaturation,
            MMM,
        )
        from pymc_marketing.paths import data_dir
        import pymc_marketing.mlflow

        pymc_marketing.mlflow.autolog(log_mmm=True)

        # Usual PyMC-Marketing model code

        file_path = data_dir / "mmm_example.csv"
        data = pd.read_csv(file_path, parse_dates=["date_week"])

        X = data.drop("y", axis=1)
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

        # Incorporate into MLflow workflow

        mlflow.set_experiment("MMM Experiment")

        with mlflow.start_run():
            idata = mmm.fit(X, y)
            posterior_preds = mmm.sample_posterior_predictive(X)

            # Additional specific logging
            fig = mmm.plot_components_contributions()
            mlflow.log_figure(fig, "components.png")

    Autologging for a PyMC-Marketing CLV model:

    .. code-block:: python

        import pandas as pd

        import mlflow

        from pymc_marketing.clv import BetaGeoModel
        from pymc_marketing.paths import data_dir

        import pymc_marketing.mlflow

        pymc_marketing.mlflow.autolog(log_clv=True)

        mlflow.set_experiment("CLV Experiment")

        file_path = data_dir / "clv_quickstart.csv"
        data = pd.read_csv(file_path)
        data["customer_id"] = data.index

        model = BetaGeoModel(data=data)

        with mlflow.start_run():
            model.fit()

        with mlflow.start_run():
            model.fit(fit_method="map")

    """
    arviz_summary_kwargs = arviz_summary_kwargs or {}

    def patch_sample(sample: Callable) -> Callable:
        @wraps(sample)
        def new_sample(*args, **kwargs):
            log_versions()

            model = pm.modelcontext(kwargs.get("model"))

            mlflow.log_param("nuts_sampler", kwargs.get("nuts_sampler", "pymc"))

            if log_model_info:
                log_model_derived_info(model)

            idata = sample(*args, **kwargs)

            # Align with the default values in pymc.sample
            tune = kwargs.get("tune", 1000)

            if log_sampler_info:
                log_sample_diagnostics(idata, tune=tune)
                log_arviz_summary(
                    idata,
                    "summary.html",
                    var_names=summary_var_names,
                    **arviz_summary_kwargs,
                )

            if log_metadata_info:
                log_metadata(model=model, idata=idata)

            return idata

        if sample_error_file:
            new_sample = log_error(new_sample, sample_error_file)

        return new_sample

    pm.sample = patch_sample(pm.sample)

    def patch_find_MAP(find_MAP):
        @wraps(find_MAP)
        def new_find_MAP(*args, **kwargs):
            model = pm.modelcontext(kwargs.get("model"))

            if log_model_info:
                log_model_derived_info(model)

            return find_MAP(*args, **kwargs)

        return new_find_MAP

    pm.find_MAP = patch_find_MAP(pm.find_MAP)

    def patch_mmm_fit(fit: Callable) -> Callable:
        @wraps(fit)
        def new_fit(self, *args, **kwargs):
            log_mmm_configuration(self)

            idata = fit(self, *args, **kwargs)

            log_inference_data(idata, save_file="idata.nc")

            return idata

        return new_fit

    if log_mmm:
        MMM.fit = patch_mmm_fit(MMM.fit)

    def patch_clv_fit(fit):
        @wraps(fit)
        def new_fit(self, fit_method: str = "mcmc", **kwargs):
            mlflow.log_param("model_type", self._model_type)
            mlflow.log_param("fit_method", fit_method)
            idata = fit(self, fit_method, **kwargs)
            mlflow.log_params(
                idata.attrs,
            )
            log_inference_data(idata, save_file="idata.nc")

            return idata

        return new_fit

    if log_clv:
        CLVModel.fit = patch_clv_fit(CLVModel.fit)
