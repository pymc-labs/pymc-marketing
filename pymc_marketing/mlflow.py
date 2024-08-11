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
import json
import os
from functools import wraps
from pathlib import Path

import arviz as az
import pymc as pm
from pymc.model.core import Model

try:
    import mlflow
except ImportError:
    msg = "This module requires mlflow. Install using `pip install mlflow`"
    raise ImportError(msg)

from mlflow.utils.autologging_utils import autologging_integration

from pymc_marketing.mmm import MMM

FLAVOR_NAME = "pymc"


def save_arviz_summary(idata: az.InferenceData, path: str | Path, var_names) -> None:
    df_summary = az.summary(idata, var_names=var_names)
    df_summary.to_html(path)
    mlflow.log_artifact(str(path))
    os.remove(path)


def save_data(model: Model, idata: az.InferenceData) -> None:
    features = {
        var.name: idata.constant_data[var.name].to_numpy()
        for var in model.data_vars
        if var.name in idata.constant_data
    }
    targets = {
        var.name: idata.observed_data[var.name].to_numpy()
        for var in model.observed_RVs
        if var.name in idata.observed_data
    }

    data = mlflow.data.from_numpy(features=features, targets=targets)
    mlflow.log_input(data, context="sample")


def save_model_graph(model: Model, path: str | Path) -> None:
    try:
        graph = pm.model_to_graphviz(model)
    except ImportError:
        return None

    try:
        saved_path = graph.render(path)
    except Exception:
        return None
    else:
        mlflow.log_artifact(saved_path)
        os.remove(saved_path)
        os.remove(path)


def get_random_variable_name(rv) -> str:
    # Taken from new version of pymc/model_graph.py
    symbol = rv.owner.op.__class__.__name__

    if symbol.endswith("RV"):
        symbol = symbol[:-2]

    return symbol


def save_types_of_parameters(model: Model) -> None:
    mlflow.log_param("n_free_RVs", len(model.free_RVs))
    mlflow.log_param("n_observed_RVs", len(model.observed_RVs))
    mlflow.log_param(
        "n_deterministics",
        len(model.deterministics),
    )
    mlflow.log_param("n_potentials", len(model.potentials))


def save_likelihood_type(model: Model) -> None:
    observed_RVs_types = [get_random_variable_name(rv) for rv in model.observed_RVs]
    if len(observed_RVs_types) == 1:
        mlflow.log_param("likelihood", observed_RVs_types[0])
    elif len(observed_RVs_types) > 1:
        mlflow.log_param("observed_RVs_types", observed_RVs_types)


def log_model_info(model: Model) -> None:
    save_types_of_parameters(model)

    mlflow.log_text(model.str_repr(), "model_repr.txt")
    mlflow.log_dict(
        model.coords,
        "coords.json",
    )

    save_model_graph(model, "model_graph")

    save_likelihood_type(model)


def diagnostics_sample(idata: az.InferenceData, var_names) -> None:
    posterior = idata.posterior
    sample_stats = idata.sample_stats
    diverging = sample_stats["diverging"]

    total_divergences = diverging.sum().item()
    mlflow.log_metric("total_divergences", total_divergences)
    if sampling_time := sample_stats.attrs.get("sampling_time"):
        mlflow.log_metric("sampling_time", sampling_time)
        mlflow.log_metric(
            "time_per_draw",
            sampling_time / (posterior.sizes["draw"] * posterior.sizes["chain"]),
        )

    if tuning_step := sample_stats.attrs.get("tuning_steps"):
        mlflow.log_param("tuning_steps", tuning_step)
    mlflow.log_param("draws", posterior.sizes["draw"])
    mlflow.log_param("chains", posterior.sizes["chain"])

    if inference_library := posterior.attrs.get("inference_library"):
        mlflow.log_param("inference_library", inference_library)
        mlflow.log_param(
            "inference_library_version",
            posterior.attrs["inference_library_version"],
        )
    mlflow.log_param("arviz_version", posterior.attrs["arviz_version"])

    save_arviz_summary(idata, "summary.html", var_names=var_names)


@autologging_integration(FLAVOR_NAME)
def autolog(
    log_datasets: bool = True,
    sampling_diagnostics: bool = True,
    model_info: bool = True,
    end_run_after_sample: bool = False,
    summary_var_names: list[str] | None = None,
    log_mmm: bool = True,
    disable: bool = False,
    silent: bool = False,
) -> None:
    def patch_sample(sample):
        @wraps(sample)
        def new_sample(*args, **kwargs):
            idata = sample(*args, **kwargs)
            if sampling_diagnostics:
                diagnostics_sample(idata, var_names=summary_var_names)

            model = pm.modelcontext(kwargs.get("model"))
            if model_info:
                log_model_info(model)

            if log_datasets:
                save_data(model=model, idata=idata)

            mlflow.log_param("pymc_version", pm.__version__)
            mlflow.log_param("nuts_sampler", kwargs.get("nuts_sampler", "pymc"))

            if end_run_after_sample:
                mlflow.end_run()

            return idata

        return new_sample

    pm.sample = patch_sample(pm.sample)

    def patch_mmm_fit(fit):
        @wraps(fit)
        def new_fit(*args, **kwargs):
            idata = fit(*args, **kwargs)
            if not log_mmm:
                return idata

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
            save_file = "idata.nc"
            idata.to_netcdf(save_file)
            mlflow.log_artifact(local_path=save_file)
            os.remove(save_file)

            return idata

        return new_fit

    MMM.fit = patch_mmm_fit(MMM.fit)
