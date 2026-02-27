#   Copyright 2022 - 2026 The PyMC Labs Developers
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
"""CLV Model base class."""

import warnings
from collections.abc import Sequence
from typing import Literal, cast

import arviz as az
import pandas as pd
import pymc as pm
from pydantic import ConfigDict, InstanceOf, validate_call
from pymc.backends import NDArray
from pymc.backends.base import MultiTrace
from pymc.model.core import Model

from pymc_marketing.model_builder import DifferentModelError, ModelBuilder
from pymc_marketing.model_config import ModelConfig, parse_model_config


class CLVModel(ModelBuilder):
    """CLV Model base class."""

    _model_type = "CLVModel"

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        data: pd.DataFrame | None = None,
        *,
        model_config: InstanceOf[ModelConfig] | None = None,
        sampler_config: dict | None = None,
        non_distributions: list[str] | None = None,
    ):
        if data is not None:
            warnings.warn(
                f"'{self._model_type}(data)' is deprecated and will be removed in version 1.0. "
                f"Use '{self._model_type}.build_model(data)' or '{self._model_type}.fit(data)' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.data = data

        model_config = model_config or {}

        deprecated_keys = [key for key in model_config if key.endswith("_prior")]
        for key in deprecated_keys:
            new_key = key.replace("_prior", "")
            warnings.warn(
                f"The key '{key}' in model_config is deprecated and will be removed in future versions."
                f"Use '{new_key}' instead.",
                DeprecationWarning,
                stacklevel=2,
            )

            model_config[new_key] = model_config.pop(key)

        super().__init__(model_config, sampler_config)

        # Parse model config after merging with defaults
        self.model_config = parse_model_config(
            self.model_config,
            non_distributions=non_distributions,
        )

    @staticmethod
    def _validate_cols(
        data: pd.DataFrame,
        required_cols: Sequence[str],
        must_be_unique: Sequence[str] = (),
        must_be_homogenous: Sequence[str] = (),
    ):
        missing = set(required_cols).difference(data.columns)
        if missing:
            raise ValueError(
                "The following required columns are missing from the "
                f"input data: {sorted(list(missing))}"
            )

        n = data.shape[0]

        for col in required_cols:
            if col in must_be_unique:
                if data[col].nunique() != n:
                    raise ValueError(f"Column {col} has duplicate entries")
            if col in must_be_homogenous:
                if data[col].nunique() != 1:
                    raise ValueError(f"Column {col} has non-homogeneous entries")

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate and prepare data for model building.

        This method should be overridden by child classes to specify
        required columns and model-specific validation rules.

        Parameters
        ----------
        data : pd.DataFrame
            The input data to validate

        Raises
        ------
        ValueError
            If data doesn't meet model requirements
        """
        pass

    def __repr__(self) -> str:
        """Representation of the model."""
        if not hasattr(self, "model"):
            return self._model_type
        else:
            return f"{self._model_type}\n{self.model.str_repr()}"

    def _add_fit_data_group(self, data: pd.DataFrame) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="The group fit_data is not defined in the InferenceData scheme",
            )
            assert self.idata is not None  # noqa: S101
            self.idata.add_groups(fit_data=data.to_xarray())

    def fit(  # type: ignore
        self,
        data: pd.DataFrame | None = None,
        method: str = "mcmc",
        fit_method: str | None = None,
        **kwargs,
    ) -> az.InferenceData:
        """Infer model posterior.

        Parameters
        ----------
        data : pd.DataFrame, optional
            The input data for model fitting. If not provided, uses data
            from model initialization (deprecated) or previously built model.
        method: str
            Method used to fit the model. Options are:
            - "mcmc": Samples from the posterior via `pymc.sample` (default)
            - "map": Finds maximum a posteriori via `pymc.find_MAP`
            - "demz": Samples from the posterior via `pymc.sample` using DEMetropolisZ
            - "advi": Samples from the posterior via `pymc.fit(method="advi")` and `pymc.sample`
            - "fullrank_advi": Samples from the posterior via `pymc.fit(method="fullrank_advi")` and `pymc.sample`
        kwargs:
            Other keyword arguments passed to the underlying PyMC routines

        """
        # Handle deprecated fit_method parameter
        if fit_method:
            warnings.warn(
                "'fit_method' is deprecated and will be removed in version 1.0. "
                "Use 'method' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            method = fit_method

        # TODO: Delete this logic when old API is removed in 1.0.
        # Handle data parameter
        if data is not None:
            self.build_model(data)  # type: ignore
        elif hasattr(self, "data") and self.data is not None:
            # Using old API data - build model if not already built
            if not hasattr(self, "model"):
                self.build_model()  # type: ignore
        else:
            # No data available anywhere
            if not hasattr(self, "model"):
                raise ValueError(
                    "Data must be provided either to fit(data=...) or "
                    "model must be built with build_model(data=...) first."
                )

        approx = None
        match method:
            case "mcmc":
                idata = self._fit_mcmc(**kwargs)
            case "map":
                idata = self._fit_MAP(**kwargs)
            case "demz":
                idata = self._fit_DEMZ(**kwargs)
            case "advi":
                approx, idata = self._fit_approx(method="advi", **kwargs)
            case "fullrank_advi":
                approx, idata = self._fit_approx(method="fullrank_advi", **kwargs)
            case _:
                raise ValueError(
                    f"Fit method options are ['mcmc', 'map', 'demz', 'advi', 'fullrank_advi'], got: {method}"
                )

        self.idata = idata
        if approx:
            self.approx = approx
        self.set_idata_attrs(self.idata)
        if self.data is not None:
            self._add_fit_data_group(self.data)

        return self.idata

    def _fit_mcmc(self, **kwargs) -> az.InferenceData:
        """Fit a model with NUTS."""
        sampler_config = {}
        if self.sampler_config is not None:
            sampler_config = self.sampler_config.copy()
        sampler_config.update(**kwargs)
        return pm.sample(**sampler_config, model=self.model)

    def _fit_MAP(self, **kwargs) -> az.InferenceData:
        """Find model maximum a posteriori using scipy optimizer."""
        model = self.model
        map_res = pm.find_MAP(model=model, **kwargs)
        # Filter non-value variables
        value_vars_names = set(v.name for v in cast(Model, model).value_vars)
        map_res = {k: v for k, v in map_res.items() if k in value_vars_names}
        # Convert map result to InferenceData
        map_strace = NDArray(model=model)
        map_strace.setup(draws=1, chain=0)
        map_strace.record(map_res)
        map_strace.close()
        trace = MultiTrace([map_strace])
        return pm.to_inference_data(trace, model=model)

    def _fit_DEMZ(self, **kwargs) -> az.InferenceData:
        """Fit a model with DEMetropolisZ gradient-free sampler."""
        sampler_config = {}
        if self.sampler_config is not None:
            sampler_config = self.sampler_config.copy()
        sampler_config.update(**kwargs)
        with self.model:
            return pm.sample(step=pm.DEMetropolisZ(), **sampler_config)

    def _fit_approx(
        self, method: Literal["advi", "fullrank_advi"] = "advi", **kwargs
    ) -> az.InferenceData:
        """Fit a model with ADVI."""
        sampler_config = {}
        if self.sampler_config is not None:
            sampler_config = self.sampler_config.copy()

        sampler_config = {**sampler_config, **kwargs}
        if sampler_config.get("method") is not None:
            raise ValueError(
                "The 'method' parameter is set in sampler_config. Cannot be called with 'advi'."
            )

        if sampler_config.get("chains", 1) > 1:
            warnings.warn(
                "The 'chains' parameter must be 1 with 'advi'. Sampling only 1 chain despite the provided parameter.",
                UserWarning,
                stacklevel=2,
            )

        with self.model:
            approx = pm.fit(
                method=method,
                callbacks=[pm.callbacks.CheckParametersConvergence(diff="absolute")],
                **{
                    k: v
                    for k, v in sampler_config.items()
                    if k
                    in [
                        "n",
                        "random_seed",
                        "inf_kwargs",
                        "start",
                        "start_sigma",
                        "score",
                        "callbacks",
                        "progressbar",
                        "progressbar_theme",
                        "obj_n_mc",
                        "tf_n_mc",
                        "obj_optimizer",
                        "test_optimizer",
                        "more_obj_params",
                        "more_tf_params",
                        "more_updates",
                        "total_grad_norm_constraint",
                        "fn_kwargs",
                        "more_replacements",
                    ]
                },
            )
            return approx, approx.sample(
                **{
                    k: v
                    for k, v in sampler_config.items()
                    if k in ["draws", "random_seed", "return_inferencedata"]
                }
            )

    @classmethod
    def idata_to_init_kwargs(cls, idata: az.InferenceData) -> dict:
        """Create the initialization kwargs from an InferenceData object."""
        kwargs = cls.attrs_to_init_kwargs(idata.attrs)
        kwargs["data"] = idata.fit_data.to_dataframe()

        return kwargs

    @classmethod
    def build_from_idata(cls, idata: az.InferenceData) -> None:
        """Build the model from the InferenceData object."""
        kwargs = cls.idata_to_init_kwargs(idata)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
            )
            model = cls(**kwargs)

        model.idata = idata
        model._rename_posterior_variables()

        # Extract data from fit_data group if it exists
        if hasattr(idata, "fit_data"):
            model.data = idata.fit_data.to_dataframe()

        model.build_model()  # type: ignore
        if model.id != idata.attrs["id"]:
            msg = (
                "The model id in the InferenceData does not match the model id. "
                "There was no error loading the inference data, but the model may "
                "be different. "
                "Investigate if the model structure or configuration has changed."
            )
            raise DifferentModelError(msg)
        return model

    # TODO: Remove in 2026Q1?
    def _rename_posterior_variables(self):
        """Rename variables in the posterior group to remove the _prior suffix.

        This is used to support the old model configuration format, which used
        to include a _prior suffix for each parameter.
        """
        prior_vars = [
            var for var in self.idata.posterior.data_vars if var.endswith("_prior")
        ]
        rename_dict = {var: var.replace("_prior", "") for var in prior_vars}
        self.idata.posterior = self.idata.posterior.rename(rename_dict)
        return self.idata.posterior

    def thin_fit_result(self, keep_every: int):
        """Return a copy of the model with a thinned fit result.

        This is useful when computing summary statistics that may require too much memory per posterior draw.

        Examples
        --------

        .. code-block:: python

            fitted_gg = ...
            fitted bg = ...

            fitted_gg_thinned = fitted_gg.thin_fit_result(keep_every=10)
            fitted_bg_thinned = fitted_bg.thin_fit_result(keep_every=10)

            clv_thinned = fitted_gg_thinned.expected_customer_lifetime_value(
                transaction_model=fitted_bg_thinned,
                customer_id=t.index,
                frequency=t["frequency"],
                recency=t["recency"],
                T=t["T"],
                mean_transaction_value=t["monetary_value"],
            )

        """
        self.fit_result  # noqa: B018 (Raise Error if fit didn't happen yet)
        assert self.idata is not None  # noqa: S101
        new_idata = self.idata.isel(draw=slice(None, None, keep_every)).copy()
        return self.build_from_idata(new_idata)

    @property
    def default_sampler_config(self) -> dict:
        """Default sampler configuration."""
        return {}

    @property
    def _serializable_model_config(self) -> dict:
        return self.model_config

    def fit_summary(self, **kwargs):
        """Compute the summary of the fit result."""
        res = self.fit_result
        # Map fitting only gives one value, so we return it. We use arviz
        # just to get it nicely into a DataFrame
        if res.chain.size == 1 and res.draw.size == 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = az.summary(self.fit_result, **kwargs, kind="stats")
            return res["mean"].rename("value")
        else:
            return az.summary(self.fit_result, **kwargs)
