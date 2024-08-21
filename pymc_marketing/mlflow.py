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
"""MLflow logging utilities for PyMC models.

This module provides utilities to log various aspects of PyMC models to MLflow
which is then extended to PyMC-Marketing models.

Autologging is supported for PyMC models and PyMC-Marketing models. This including
logging of sampler diagnostics, model information, data used in the model, and
InferenceData objects.

The autologging can be enabled by calling the `autolog` function. This function
patches the `pymc.sample` and `MMM.fit` calls to log the required information.

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

Autologging for a PyMC-Marketing model:

.. code-block:: python

    import pandas as pd

    import mlflow

    from pymc_marketing.mmm import (
        GeometricAdstock,
        LogisticSaturation,
        MMM,
    )
    import pymc_marketing.mlflow

    pymc_marketing.mlflow.autolog(log_mmm=True)

    # Usual PyMC-Marketing model code

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

    # Incorporate into MLflow workflow

    mlflow.set_experiment("MMM Experiment")

    with mlflow.start_run():
        idata = mmm.fit(X, y)

        # Additional specific logging
        fig = mmm.plot_components_contributions()
        mlflow.log_figure(fig, "components.png")

"""

import json
import logging
import os
import warnings
from functools import wraps
from pathlib import Path

import arviz as az
import pymc as pm
from pymc.model.core import Model
from pytensor.tensor import TensorVariable

try:
    import mlflow
except ImportError:  # pragma: no cover
    msg = "This module requires mlflow. Install using `pip install mlflow`"
    raise ImportError(msg)

from mlflow.utils.autologging_utils import autologging_integration

from pymc_marketing.mmm import MMM
from pymc_marketing.version import __version__

FLAVOR_NAME = "pymc"


PYMC_MARKETING_ISSUE = "https://github.com/pymc-labs/pymc-marketing/issues/new"
warning_msg = (
    "This functionality is experimental and subject to change. "
    "If you encounter any issues or have suggestions, please raise them at: "
    f"{PYMC_MARKETING_ISSUE}"
)
warnings.warn(warning_msg, FutureWarning, stacklevel=1)


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


def _backwards_compatiable_data_vars(model: Model) -> list[TensorVariable]:
    # TODO: Remove with PyMC update
    non_data = (
        model.observed_RVs + model.free_RVs + model.deterministics + model.potentials
    )
    vars = {
        key: value for key, value in model.named_vars.items() if value not in non_data
    }

    return list(vars.values())


def log_data(model: Model, idata: az.InferenceData) -> None:
    """Log the data used in the model to MLflow.

    Saved in the form of numpy arrays based on all the constant and observed data
    in the model.

    Parameters
    ----------
    model : Model
        The PyMC model object.
    idata : az.InferenceData
        The InferenceData object returned by the sampling method.

    """

    data_vars: list[TensorVariable] = (
        _backwards_compatiable_data_vars(model)
        if not hasattr(model, "data_vars")
        else model.data_vars
    )

    features = {
        var.name: idata.constant_data[var.name].to_numpy()
        for var in data_vars
        if var.name in idata.constant_data
    }
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
        mlflow.log_artifact(saved_path)
        os.remove(saved_path)
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
    mlflow.log_param(
        "n_deterministics",
        len(model.deterministics),
    )
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
        mlflow.log_dict(
            model.coords,
            "coords.json",
        )

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
    mlflow.log_param("arviz_version", posterior.attrs["arviz_version"])


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
    mlflow.log_artifact(local_path=str(save_file))
    os.remove(save_file)


@autologging_integration(FLAVOR_NAME)
def autolog(
    log_sampler_info: bool = True,
    log_datasets: bool = True,
    log_model_info: bool = True,
    summary_var_names: list[str] | None = None,
    arviz_summary_kwargs: dict | None = None,
    log_mmm: bool = True,
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
    log_datasets : bool, optional
        Whether to log the data used in the model. Default is True.
    log_model_info : bool, optional
        Whether to log model information. Default is True.
    summary_var_names : list[str], optional
        The names of the variables to include in the ArviZ summary. Default is
        all the variables in the InferenceData object.
    arviz_summary_kwargs : dict, optional
        Additional keyword arguments to pass to `az.summary`.
    log_mmm : bool, optional
        Whether to log PyMC-Marketing MMM models. Default is True.
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

    Autologging for a PyMC-Marketing model:

    .. code-block:: python

        import pandas as pd

        import mlflow

        from pymc_marketing.mmm import (
            GeometricAdstock,
            LogisticSaturation,
            MMM,
        )
        import pymc_marketing.mlflow

        pymc_marketing.mlflow.autolog(log_mmm=True)

        # Usual PyMC-Marketing model code

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

        # Incorporate into MLflow workflow

        mlflow.set_experiment("MMM Experiment")

        with mlflow.start_run():
            idata = mmm.fit(X, y)

            # Additional specific logging
            fig = mmm.plot_components_contributions()
            mlflow.log_figure(fig, "components.png")

    """

    arviz_summary_kwargs = arviz_summary_kwargs or {}

    def patch_sample(sample):
        @wraps(sample)
        def new_sample(*args, **kwargs):
            idata = sample(*args, **kwargs)
            mlflow.log_param("pymc_marketing_version", __version__)
            mlflow.log_param("pymc_version", pm.__version__)
            mlflow.log_param("nuts_sampler", kwargs.get("nuts_sampler", "pymc"))

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

            model = pm.modelcontext(kwargs.get("model"))
            if log_model_info:
                log_model_derived_info(model)

            if log_datasets:
                log_data(model=model, idata=idata)

            return idata

        return new_sample

    pm.sample = patch_sample(pm.sample)

    def patch_mmm_fit(fit):
        @wraps(fit)
        def new_fit(*args, **kwargs):
            idata = fit(*args, **kwargs)

            mlflow.log_params(
                idata.attrs,
            )
            mlflow.log_param(
                "adstock_name",
                json.loads(idata.attrs["adstock"])["lookup_name"],
            )
            mlflow.log_param(
                "saturation_name",
                json.loads(idata.attrs["saturation"])["lookup_name"],
            )
            log_inference_data(idata, save_file="idata.nc")

            return idata

        return new_fit

    if log_mmm:
        MMM.fit = patch_mmm_fit(MMM.fit)
