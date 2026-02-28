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
"""Multidimensional Marketing Mix Model (MMM).

Examples
--------
Basic MMM fit:

.. code-block:: python

    from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
    from pymc_marketing.mmm.multidimensional import MMM
    import pandas as pd

    X = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=8, freq="W-MON"),
            "C1": [100, 120, 90, 110, 105, 115, 98, 102],
            "C2": [80, 70, 95, 85, 90, 88, 92, 94],
        }
    )
    y = pd.Series([230, 260, 220, 240, 245, 255, 235, 238], name="y")

    mmm = MMM(
        date_column="date",
        channel_columns=["C1", "C2"],
        adstock=GeometricAdstock(l_max=10),
        saturation=LogisticSaturation(),
    )
    mmm.fit(X, y)

    # Optional: posterior predictive and plots
    mmm.sample_posterior_predictive(X)
    _ = mmm.plot.contributions_over_time(var=["channel_contribution"])

Multi-dimensional (panel) with dims:

.. code-block:: python

    X = pd.DataFrame(
        {
            "date": ["2025-01-06", "2025-01-13"] * 2,
            "country": ["A", "A", "B", "B"],
            "C1": [100, 120, 90, 110],
            "C2": [80, 70, 95, 85],
        }
    )
    y = pd.Series([230, 260, 220, 240], name="y")

    mmm = MMM(
        date_column="date",
        channel_columns=["C1", "C2"],
        adstock=GeometricAdstock(l_max=10),
        saturation=LogisticSaturation(),
        dims=("country",),
    )
    mmm.fit(X, y)

Time-varying parameters and seasonality:

.. code-block:: python

    from pymc_marketing.mmm import SoftPlusHSGP

    mmm = MMM(
        date_column="date",
        channel_columns=["C1", "C2"],
        adstock=GeometricAdstock(l_max=10),
        saturation=LogisticSaturation(),
        time_varying_intercept=True,
        time_varying_media=True,  # or SoftPlusHSGP(...)
        yearly_seasonality=4,
    )
    mmm.fit(X, y)

Controls (additional regressors):

.. code-block:: python

    X["price_index"] = [1.0, 1.02, 0.99, 1.01]
    mmm = MMM(
        date_column="date",
        channel_columns=["C1", "C2"],
        control_columns=["price_index"],
        adstock=GeometricAdstock(l_max=10),
        saturation=LogisticSaturation(),
    )
    mmm.fit(X, y)

Events:

.. code-block:: python

    from pymc_extras.prior import Prior
    from pymc_marketing.mmm.events import EventEffect, GaussianBasis
    import pandas as pd

    df_events = pd.DataFrame(
        {
            "name": ["Promo", "Holiday"],
            "start_date": pd.to_datetime(["2025-02-01", "2025-03-20"]),
            "end_date": pd.to_datetime(["2025-02-03", "2025-03-25"]),
        }
    )
    effect = EventEffect(
        basis=GaussianBasis(
            priors={"sigma": Prior("Gamma", mu=7, sigma=1, dims="event")}
        ),
        effect_size=Prior("Normal", mu=0, sigma=1, dims="event"),
        dims=("event",),
    )

    mmm = MMM(
        date_column="date",
        channel_columns=["C1", "C2"],
        adstock=GeometricAdstock(l_max=10),
        saturation=LogisticSaturation(),
    )
    mmm.add_events(df_events=df_events, prefix="event", effect=effect)
    mmm.fit(X, y)

Save, load, and plot:

.. code-block:: python

    mmm.save("mmm.nc")
    loaded = MMM.load("mmm.nc")
    _ = loaded.plot.posterior_predictive()

Notes
-----
- X must include `date`, the `channel_columns`, and any extra `dims` columns.
- y is a Series with name equal to `target_column`.
- Call `add_events` before fitting/building.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Sequence
from copy import deepcopy
from functools import singledispatch
from typing import Annotated, Any, cast

import arviz as az
import numpy as np
import numpy.typing as npt
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pydantic import Field, InstanceOf, StrictBool, validate_call
from pymc.model.fgraph import clone_model as cm
from pymc.util import RandomState
from pymc_extras.deserialize import deserialize
from pymc_extras.prior import Prior, create_dim_handler
from scipy.optimize import OptimizeResult

from pymc_marketing.data.idata.mmm_wrapper import MMMIDataWrapper
from pymc_marketing.data.idata.utils import subsample_draws
from pymc_marketing.hsgp_kwargs import HSGPKwargs
from pymc_marketing.mmm import SoftPlusHSGP
from pymc_marketing.mmm.additive_effect import (
    EventAdditiveEffect,
    MuEffect,
    safe_to_datetime,
)
from pymc_marketing.mmm.budget_optimizer import OptimizerCompatibleModelWrapper
from pymc_marketing.mmm.causal import CausalGraphModel
from pymc_marketing.mmm.components.adstock import (
    AdstockTransformation,
    adstock_from_dict,
)
from pymc_marketing.mmm.components.saturation import (
    SaturationTransformation,
    saturation_from_dict,
)
from pymc_marketing.mmm.events import EventEffect
from pymc_marketing.mmm.experiment import (
    ExperimentResult,
    ExperimentType,
    run_experiment,
)
from pymc_marketing.mmm.fourier import YearlyFourier
from pymc_marketing.mmm.hsgp import HSGPBase, hsgp_from_dict
from pymc_marketing.mmm.incrementality import Incrementality
from pymc_marketing.mmm.lift_test import (
    add_cost_per_target_potentials,
    add_lift_measurements_to_likelihood_from_saturation,
    scale_lift_measurements,
)
from pymc_marketing.mmm.plot import MMMPlotSuite
from pymc_marketing.mmm.scaling import Scaling, VariableScaling
from pymc_marketing.mmm.sensitivity_analysis import SensitivityAnalysis
from pymc_marketing.mmm.tvp import create_hsgp_from_config, infer_time_index
from pymc_marketing.mmm.utility import UtilityFunctionType, average_response
from pymc_marketing.mmm.utils import (
    add_noise_to_channel_allocation,
    create_zero_dataset,
)
from pymc_marketing.model_builder import (
    RegressionModelBuilder,
    _handle_deprecate_pred_argument,
)
from pymc_marketing.model_config import parse_model_config
from pymc_marketing.model_graph import deterministics_to_flat


@singledispatch
def _serialize_mu_effect(effect: MuEffect) -> dict[str, Any]:
    """Serialize a MuEffect to JSON-compatible dict.

    Default implementation uses Pydantic's model_dump for unknown types.
    Register new types with: @_serialize_mu_effect.register(YourEffect)
    """
    # Check if the effect has model_dump (Pydantic BaseModel)
    if not hasattr(effect, "model_dump"):
        raise TypeError(
            f"Cannot serialize MuEffect of type '{effect.__class__.__name__}': "
            f"MuEffect subclasses must inherit from pydantic.BaseModel. "
            f"Update your custom effect class to inherit from MuEffect (which is now a BaseModel). "
            f"See pymc_marketing.mmm.additive_effect.MuEffect for the new base class definition."
        )

    return {
        "class": effect.__class__.__name__,
        **effect.model_dump(mode="json"),
    }


def _deserialize_mu_effect(data: dict[str, Any]) -> MuEffect:
    """Deserialize a MuEffect from dict using class name."""
    class_name = data.get("class")
    if class_name not in _MUEFFECT_DESERIALIZERS:
        raise ValueError(
            f"Unknown MuEffect class: {class_name}. "
            f"Registered types: {list(_MUEFFECT_DESERIALIZERS.keys())}"
        )
    return _MUEFFECT_DESERIALIZERS[class_name](data)


# Deserialization registry: maps class name -> deserializer function
_MUEFFECT_DESERIALIZERS: dict[str, Callable[[dict[str, Any]], MuEffect]] = {}


# Register serializers/deserializers at module load
# Imports are inside function to avoid circular dependencies
def _register_mu_effect_handlers():
    """Register all known MuEffect serialization handlers."""
    from pymc_marketing.mmm import fourier as fourier_module
    from pymc_marketing.mmm.additive_effect import (
        EventAdditiveEffect,
        FourierEffect,
        LinearTrendEffect,
    )
    from pymc_marketing.mmm.linear_trend import LinearTrend

    # FourierEffect
    @_serialize_mu_effect.register(FourierEffect)
    def _(effect: FourierEffect) -> dict[str, Any]:
        return {
            "class": "FourierEffect",
            "fourier": effect.fourier.to_dict(),
            "date_dim_name": effect.date_dim_name,
        }

    def _deser_fourier(data: dict[str, Any]) -> FourierEffect:
        # Get Fourier class from module by name and use its from_dict method
        fourier_data = data["fourier"]
        fourier_class_name = fourier_data.get("class")
        fourier_class = getattr(fourier_module, fourier_class_name, None)
        if fourier_class is None:
            raise ValueError(
                f"Unknown Fourier class: {fourier_class_name}. "
                f"Not found in pymc_marketing.mmm.fourier module."
            )

        fourier = fourier_class.from_dict(fourier_data)
        return FourierEffect(
            fourier=fourier,
            date_dim_name=data.get("date_dim_name", "date"),
        )

    # LinearTrendEffect
    @_serialize_mu_effect.register(LinearTrendEffect)
    def _(effect: LinearTrendEffect) -> dict[str, Any]:
        # Serialize trend data, handling priors separately
        trend_data = effect.trend.model_dump(mode="json", exclude={"priors"})
        # Manually serialize priors using Prior.to_dict()
        if effect.trend.priors is not None:
            trend_data["priors"] = {
                key: prior.to_dict() for key, prior in effect.trend.priors.items()
            }
        return {
            "class": "LinearTrendEffect",
            "trend": trend_data,
            "prefix": effect.prefix,
            "date_dim_name": effect.date_dim_name,
        }

    def _deser_linear_trend(data: dict[str, Any]) -> LinearTrendEffect:
        # Deserialize priors separately using generic deserialize()
        # to support both Prior and SpecialPrior (e.g., LogNormalPrior)
        trend_data = data["trend"].copy()
        if "priors" in trend_data and trend_data["priors"] is not None:
            trend_data["priors"] = {
                key: deserialize(prior_dict)
                for key, prior_dict in trend_data["priors"].items()
            }
        return LinearTrendEffect(
            trend=LinearTrend.model_validate(trend_data),
            prefix=data["prefix"],
            date_dim_name=data.get("date_dim_name", "date"),
        )

    # EventAdditiveEffect
    @_serialize_mu_effect.register(EventAdditiveEffect)
    def _(effect: EventAdditiveEffect) -> dict[str, Any]:
        result = {
            "class": "EventAdditiveEffect",
            **effect.model_dump(mode="json", exclude={"df_events", "effect"}),
        }
        result["event_names"] = effect.df_events["name"].tolist()
        return result

    def _deser_event(data: dict[str, Any]) -> EventAdditiveEffect:
        raise ValueError(
            "EventAdditiveEffect deserialization not supported: "
            "requires original df_events DataFrame. "
            f"Event names in saved model: {data.get('event_names', [])}"
        )

    # Populate deserialization registry
    _MUEFFECT_DESERIALIZERS.update(
        {
            "FourierEffect": _deser_fourier,
            "LinearTrendEffect": _deser_linear_trend,
            "EventAdditiveEffect": _deser_event,
        }
    )


# Register handlers at module load
_register_mu_effect_handlers()


class MMM(RegressionModelBuilder):
    """Marketing Mix Model class for estimating the impact of marketing channels on a target variable.

    This class implements the core functionality of a Marketing Mix Model (MMM), allowing for the
    specification of various marketing channels, adstock transformations, saturation effects,
    and time-varying parameters. It provides methods for fitting the model to data, making
    predictions, and visualizing the results.

    Attributes
    ----------
    date_column : str
        The name of the column representing the date in the dataset.
    channel_columns : list[str]
        A list of column names representing the marketing channels.
    target_column : str, optional
        The name of the column representing the target variable in the dataset. Defaults to `y`.
    adstock : AdstockTransformation
        The adstock transformation to apply to the channel data.
    saturation : SaturationTransformation
        The saturation transformation to apply to the channel data.
    time_varying_intercept : bool
        Whether to use a time-varying intercept in the model.
    time_varying_media : bool
        Whether to use time-varying effects for media channels.
    dims : tuple | None
        Additional batch-dimensions for the model.
        One categorical-like column with the name of each batch dimension should be present in the dataset.
        This is used to identify which batch-dimension(s) are associated with each row of data.
        Data must be rectangular these batch dimensions (i.e., same dates and length for each combination)
    scaling : Scaling | dict | None
        Scaling methods to be used for the target variable and the marketing channels.
        Defaults to max scaling for both.
    model_config : dict | None
        Configuration settings for the model.
    sampler_config : dict | None
        Configuration settings for the sampler.
    control_columns : list[str] | None
        A list of control variables to include in the model.
    yearly_seasonality : int | None
        The number of yearly seasonalities to include in the model.
    adstock_first : bool
        Whether to apply adstock transformations before saturation.
    """

    _model_type: str = "MMMM (Multi-Dimensional Marketing Mix Model)"
    version: str = "0.0.2"
    output_var = "y"

    @validate_call
    def __init__(
        self,
        *,
        date_column: str = Field(..., description="Column name of the date variable."),
        channel_columns: list[str] = Field(
            min_length=1, description="Column names of the media channel variables."
        ),
        target_column: str = Field("y", description="The name of the target column."),
        adstock: InstanceOf[AdstockTransformation] = Field(
            ..., description="Type of adstock transformation to apply."
        ),
        saturation: InstanceOf[SaturationTransformation] = Field(
            ...,
            description="The saturation transformation to apply to the channel data.",
        ),
        time_varying_intercept: Annotated[
            StrictBool | InstanceOf[HSGPBase],
            Field(
                description=(
                    "Whether to use a time-varying intercept, or pass an HSGP instance "
                    "(e.g., SoftPlusHSGP) specifying dims and priors."
                ),
            ),
        ] = False,
        time_varying_media: Annotated[
            StrictBool | InstanceOf[HSGPBase],
            Field(
                description=(
                    "Whether to use time-varying media effects, or pass an HSGP instance "
                    "(e.g., SoftPlusHSGP) specifying dims and priors."
                ),
            ),
        ] = False,
        dims: tuple[str, ...] | None = Field(
            None, description="Additional dimensions for the model."
        ),
        scaling: InstanceOf[Scaling] | dict | None = Field(
            None, description="Scaling configuration for the model."
        ),
        model_config: dict | None = Field(
            None, description="Configuration settings for the model."
        ),
        sampler_config: dict | None = Field(
            None, description="Configuration settings for the sampler."
        ),
        control_columns: Annotated[
            list[str] | None,
            Field(
                min_length=1,
                description="A list of control variables to include in the model.",
            ),
        ] = None,
        yearly_seasonality: Annotated[
            int | None,
            Field(
                gt=0,
                description="The number of yearly seasonalities to include in the model.",
            ),
        ] = None,
        adstock_first: Annotated[
            bool,
            Field(strict=True, description="Apply adstock before saturation?"),
        ] = True,
        dag: str | None = Field(
            None,
            description="Optional DAG provided as a string Dot format for causal identification.",
        ),
        treatment_nodes: list[str] | tuple[str] | None = Field(
            None,
            description="Column names of the variables of interest to identify causal effects on outcome.",
        ),
        outcome_node: str | None = Field(
            None, description="Name of the outcome variable."
        ),
    ) -> None:
        """Define the constructor method."""
        # Your existing initialization logic
        self.control_columns = control_columns
        self.time_varying_intercept = time_varying_intercept
        self.time_varying_media = time_varying_media
        self.date_column = date_column
        self.adstock_first = adstock_first

        dims = dims if dims is not None else ()
        core_dims = {"date", "channel", "control", "fourier_mode"}
        if invalid_dims := core_dims & set(dims):
            raise ValueError(
                f"Dims {sorted(invalid_dims)} are reserved for internal use"
            )

        self.dims = dims

        if isinstance(scaling, dict):
            scaling = deepcopy(scaling)

            if "channel" not in scaling:
                scaling["channel"] = VariableScaling(method="max", dims=self.dims)
            if "target" not in scaling:
                scaling["target"] = VariableScaling(method="max", dims=self.dims)

            scaling = Scaling(**scaling)

        self.scaling: Scaling = scaling or Scaling(
            target=VariableScaling(method="max", dims=self.dims),
            channel=VariableScaling(method="max", dims=self.dims),
        )

        if set(self.scaling.target.dims).difference([*self.dims, "date"]):
            raise ValueError(
                f"Target scaling dims {self.scaling.target.dims} must contain {self.dims} and 'date'"
            )

        if set(self.scaling.channel.dims).difference([*self.dims, "channel", "date"]):
            raise ValueError(
                f"Channel scaling dims {self.scaling.channel.dims} must contain {self.dims}, 'channel', and 'date'"
            )

        model_config = model_config if model_config is not None else {}
        sampler_config = sampler_config
        model_config = parse_model_config(
            model_config,  # type: ignore
            hsgp_kwargs_fields=["intercept_tvp_config", "media_tvp_config"],
        )

        self.adstock, self.saturation = adstock, saturation
        del adstock, saturation
        if model_config is not None:
            # self.default_model_config accesses self.adstock and self.saturation
            self.adstock = self.adstock.with_updated_priors(
                {**self.default_model_config, **model_config}
            )
            self.saturation = self.saturation.with_updated_priors(
                {**self.default_model_config, **model_config}
            )
        self.adstock = self.adstock.with_default_prior_dims((*self.dims, "channel"))
        self.saturation = self.saturation.with_default_prior_dims(
            (*self.dims, "channel")
        )

        self._check_compatible_media_dims()

        self.date_column = date_column
        self.target_column = target_column
        self.channel_columns = channel_columns
        self.yearly_seasonality = yearly_seasonality

        # Causal graph configuration
        self.dag = dag
        self.treatment_nodes = treatment_nodes
        self.outcome_node = outcome_node

        # Initialize causal graph if provided
        if self.dag is not None and self.outcome_node is not None:
            if self.treatment_nodes is None:
                self.treatment_nodes = self.channel_columns
                warnings.warn(
                    "No treatment nodes provided, using channel columns as treatment nodes.",
                    stacklevel=2,
                )
            self.causal_graphical_model = CausalGraphModel.build_graphical_model(
                graph=self.dag,
                treatment=self.treatment_nodes,
                outcome=self.outcome_node,
            )

            self.control_columns = self.causal_graphical_model.compute_adjustment_sets(
                control_columns=self.control_columns,
                channel_columns=self.channel_columns,
            )

            # Only apply yearly seasonality adjustment if an adjustment set was computed
            if hasattr(self.causal_graphical_model, "adjustment_set") and (
                self.causal_graphical_model.adjustment_set is not None
            ):
                if (
                    "yearly_seasonality"
                    not in self.causal_graphical_model.adjustment_set
                ):
                    warnings.warn(
                        "Yearly seasonality excluded as it's not required for adjustment.",
                        stacklevel=2,
                    )
                    self.yearly_seasonality = None

        super().__init__(model_config=model_config, sampler_config=sampler_config)

        if self.yearly_seasonality is not None:
            self.yearly_fourier = YearlyFourier(
                n_order=self.yearly_seasonality,
                prefix="fourier_mode",
                prior=self.model_config["gamma_fourier"],
                variable_name="gamma_fourier",
            )

        self.mu_effects: list[MuEffect] = []

    def __eq__(self, other: object) -> bool:
        """Compare two MMM instances for equivalence.

        Compares all configuration attributes including:
        - Core configuration (date, channels, target, dims, scaling)
        - Transformations (adstock, saturation, adstock_first)
        - Time-varying effects (time_varying_intercept, time_varying_media)
        - Additive effects (mu_effects)
        - Causal graph (dag, treatment_nodes, outcome_node)
        - Control columns and seasonality settings
        - Model and sampler configuration
        - Model ID (which validates full config consistency)

        Parameters
        ----------
        other : object
            The other object to compare with.

        Returns
        -------
        bool
            True if all configuration attributes are equal, False otherwise.

        """
        if not isinstance(other, MMM):
            return False

        # Core configuration
        if (
            self.date_column != other.date_column
            or self.channel_columns != other.channel_columns
            or self.target_column != other.target_column
            or self.dims != other.dims
            or self.control_columns != other.control_columns
            or self.adstock_first != other.adstock_first
        ):
            return False

        # Transformations - compare by type and serialized form
        if self.adstock.__class__ is not other.adstock.__class__:
            return False
        if hasattr(self.adstock, "to_dict"):
            if self.adstock.to_dict() != other.adstock.to_dict():
                return False

        if self.saturation.__class__ is not other.saturation.__class__:
            return False
        if hasattr(self.saturation, "to_dict"):
            if self.saturation.to_dict() != other.saturation.to_dict():
                return False

        # Time-varying effects
        if (
            self.time_varying_intercept.__class__
            is not other.time_varying_intercept.__class__
        ):
            return False
        if isinstance(self.time_varying_intercept, HSGPBase):
            if (
                self.time_varying_intercept.to_dict()
                != other.time_varying_intercept.to_dict()
            ):
                return False
        else:
            if self.time_varying_intercept != other.time_varying_intercept:
                return False

        if self.time_varying_media.__class__ is not other.time_varying_media.__class__:
            return False
        if isinstance(self.time_varying_media, HSGPBase):
            if self.time_varying_media.to_dict() != other.time_varying_media.to_dict():
                return False
        else:
            if self.time_varying_media != other.time_varying_media:
                return False

        # Additive effects (mu_effects)
        if len(self.mu_effects) != len(other.mu_effects):
            return False
        # Length check above ensures zip lengths match, suppressing B905 warning
        for self_effect, other_effect in zip(self.mu_effects, other.mu_effects):  # noqa: B905
            if self_effect.__class__ is not other_effect.__class__:
                return False
            if hasattr(self_effect, "model_dump") and hasattr(
                other_effect, "model_dump"
            ):
                if self_effect.model_dump() != other_effect.model_dump():
                    return False

        # Causal graph
        if (
            self.dag != other.dag
            or self.treatment_nodes != other.treatment_nodes
            or self.outcome_node != other.outcome_node
        ):
            return False

        # Seasonality
        if self.yearly_seasonality != other.yearly_seasonality:
            return False

        # Scaling configuration
        if self.scaling.__class__ is not other.scaling.__class__:
            return False
        if hasattr(self.scaling, "model_dump"):
            if self.scaling.model_dump() != other.scaling.model_dump():
                return False

        # Model and sampler config (validated by ID comparison)
        if self.sampler_config != other.sampler_config:
            return False

        # Final validation: model IDs must match
        # This is a content-based hash that validates the entire config
        if self.id != other.id:
            return False

        return True

    def _check_compatible_media_dims(self) -> None:
        allowed_dims = set(self.dims).union({"channel"})

        if not set(self.adstock.combined_dims).issubset(allowed_dims):
            raise ValueError(
                f"Adstock effect dims {self.adstock.combined_dims} must contain {allowed_dims}"
            )

        if not set(self.saturation.combined_dims).issubset(allowed_dims):
            raise ValueError(
                f"Saturation effect dims {self.saturation.combined_dims} must contain {allowed_dims}"
            )

    @property
    def default_sampler_config(self) -> dict:
        """Default sampler configuration."""
        return {}

    def _data_setter(self, X, y=None): ...

    def add_events(
        self,
        df_events: pd.DataFrame,
        prefix: str,
        effect: EventEffect,
    ) -> None:
        """Add event effects to the model.

        This must be called before building the model.

        Parameters
        ----------
        df_events : pd.DataFrame
            The DataFrame containing the event data.
                * `name`: name of the event. Used as the model coordinates.
                * `start_date`: start date of the event
                * `end_date`: end date of the event
        prefix : str
            The prefix to use for the event effect and associated variables.
        effect : EventEffect
            The event effect to apply.

        Raises
        ------
        ValueError
            If the event effect dimensions do not contain the prefix and model dimensions.

        """
        if not set(effect.dims).issubset((prefix, *self.dims)):
            raise ValueError(
                f"Event effect dims {effect.dims} must contain {prefix} and {self.dims}"
            )

        event_effect = EventAdditiveEffect(
            df_events=df_events,
            prefix=prefix,
            effect=effect,
        )
        self.mu_effects.append(event_effect)

    @property
    def _serializable_model_config(self) -> dict[str, Any]:
        def serialize_value(value):
            """Recursively serialize values to JSON-compatible types."""
            if isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, dict):
                return {k: serialize_value(v) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                return [serialize_value(v) for v in value]
            elif hasattr(value, "to_dict"):
                # Handle any object with a to_dict method (Prior, SpecialPrior, etc.)
                return value.to_dict()
            elif hasattr(value, "model_dump"):
                # Handle Pydantic models
                return value.model_dump()
            else:
                return value

        serializable_config = {}
        for key, value in self.model_config.items():
            serializable_config[key] = serialize_value(value)

        return serializable_config

    def create_idata_attrs(self) -> dict[str, str]:
        """Return the idata attributes for the model."""
        attrs = super().create_idata_attrs()
        attrs["dims"] = json.dumps(self.dims)
        attrs["date_column"] = self.date_column
        attrs["adstock"] = json.dumps(self.adstock.to_dict())
        attrs["saturation"] = json.dumps(self.saturation.to_dict())
        attrs["adstock_first"] = json.dumps(self.adstock_first)
        attrs["control_columns"] = json.dumps(self.control_columns)
        attrs["channel_columns"] = json.dumps(self.channel_columns)
        attrs["yearly_seasonality"] = json.dumps(self.yearly_seasonality)
        attrs["time_varying_intercept"] = json.dumps(
            self.time_varying_intercept
            if not isinstance(self.time_varying_intercept, HSGPBase)
            else {
                **self.time_varying_intercept.to_dict(),
                **{"hsgp_class": self.time_varying_intercept.__class__.__name__},
            }
        )
        attrs["time_varying_media"] = json.dumps(
            self.time_varying_media
            if not isinstance(self.time_varying_media, HSGPBase)
            else {
                **self.time_varying_media.to_dict(),
                **{"hsgp_class": self.time_varying_media.__class__.__name__},
            }
        )
        attrs["target_column"] = self.target_column
        attrs["scaling"] = json.dumps(self.scaling.model_dump(mode="json"))
        attrs["dag"] = json.dumps(getattr(self, "dag", None))
        attrs["treatment_nodes"] = json.dumps(getattr(self, "treatment_nodes", None))
        attrs["outcome_node"] = json.dumps(getattr(self, "outcome_node", None))

        # Serialize mu_effects
        mu_effects_list = [_serialize_mu_effect(effect) for effect in self.mu_effects]
        attrs["mu_effects"] = json.dumps(mu_effects_list)

        return attrs

    @classmethod
    def _model_config_formatting(cls, model_config: dict) -> dict:
        """Format the model configuration.

        Because of json serialization, model_config values that were originally tuples
        or numpy are being encoded as lists. This function converts them back to tuples
        and numpy arrays to ensure correct id encoding.

        Parameters
        ----------
        model_config : dict
            The model configuration to format.

        Returns
        -------
        dict
            The formatted model configuration.

        """

        def format_nested_dict(d: dict) -> dict:
            for key, value in d.items():
                if isinstance(value, dict):
                    d[key] = format_nested_dict(value)
                elif isinstance(value, list):
                    # Check if the key is "dims" to convert it to tuple
                    if key == "dims":
                        d[key] = tuple(value)
                    # Convert all other lists to numpy arrays
                    else:
                        d[key] = np.array(value)
            return d

        return format_nested_dict(model_config.copy())

    @classmethod
    def attrs_to_init_kwargs(cls, attrs: dict[str, str]) -> dict[str, Any]:
        """Convert the idata attributes to the model initialization kwargs."""
        return {
            "model_config": cls._model_config_formatting(
                json.loads(attrs["model_config"])
            ),
            "date_column": attrs["date_column"],
            "control_columns": json.loads(attrs["control_columns"]),
            "channel_columns": json.loads(attrs["channel_columns"]),
            "adstock": adstock_from_dict(json.loads(attrs["adstock"])),
            "saturation": saturation_from_dict(json.loads(attrs["saturation"])),
            "adstock_first": json.loads(attrs.get("adstock_first", "true")),
            "yearly_seasonality": json.loads(attrs["yearly_seasonality"]),
            "time_varying_intercept": hsgp_from_dict(
                json.loads(attrs.get("time_varying_intercept", "false"))
            ),
            "target_column": attrs["target_column"],
            "time_varying_media": hsgp_from_dict(
                json.loads(attrs.get("time_varying_media", "false"))
            ),
            "sampler_config": json.loads(attrs["sampler_config"]),
            "dims": tuple(json.loads(attrs.get("dims", "[]"))),
            "scaling": json.loads(attrs.get("scaling", "null")),
            "dag": json.loads(attrs.get("dag", "null")),
            "treatment_nodes": json.loads(attrs.get("treatment_nodes", "null")),
            "outcome_node": json.loads(attrs.get("outcome_node", "null")),
        }

    @property
    def plot(self) -> MMMPlotSuite:
        """Use the MMMPlotSuite to plot the results."""
        self._validate_model_was_built()
        self._validate_idata_exists()
        return MMMPlotSuite(idata=self.idata)

    @property
    def plot_interactive(self):  # type: ignore[no-any-return]
        """Access interactive Plotly plotting functionality.

        Returns a factory for creating interactive plots using Plotly.
        Automatically integrates with Component 2 (MMMSummaryFactory)
        to fetch data and apply faceting for custom dimensions.

        Returns
        -------
        MMMPlotlyFactory
            Factory for creating interactive plots

        Examples
        --------
        .. code-block:: python

            # Interactive posterior predictive plot
            fig = mmm.plot_interactive.posterior_predictive()
            fig.show()

            # Contributions with faceting
            fig = mmm.plot_interactive.contributions(facet_col="country")
            fig.show()

            # ROAS bar chart
            fig = mmm.plot_interactive.roas()
            fig.show()

            # Saturation curves
            fig = mmm.plot_interactive.saturation_curves()
            fig.show()

            # Adstock curves
            fig = mmm.plot_interactive.adstock_curves()
            fig.show()

        See Also
        --------
        MMMPlotSuite : Static matplotlib plotting functionality
        MMMPlotlyFactory : Interactive plotting class documentation
        """
        try:
            from pymc_marketing.mmm.plot_interactive import MMMPlotlyFactory
        except ImportError:
            raise ImportError(
                "Plotly is required for interactive plotting. "
                "Install it with: pip install pymc-marketing[plotly]"
            )

        self._validate_model_was_built()
        self._validate_idata_exists()

        return MMMPlotlyFactory(summary=self.summary)

    @property
    def data(self) -> Any:  # type: ignore[no-any-return]
        """Get data wrapper for InferenceData access and manipulation.

        Returns a fresh wrapper on each access. The wrapper is lightweight
        and wraps the current state of self.idata.

        Validation is explicit - call `.validate()` or `.validate_or_raise()`
        to check idata structure after modifications.

        Returns
        -------
        MMMIDataWrapper
            Wrapper providing validated access and transformations

        Examples
        --------
        .. code-block:: python

            # Access observed data
            observed = mmm.data.get_target()

            # Get contributions in original scale
            contributions = mmm.data.get_contributions(original_scale=True)

            # Validate after modifications
            mmm.add_original_scale_contribution_variable(["channel_contribution"])
            mmm.data.validate_or_raise()

            # Filter and aggregate
            monthly = mmm.data.filter_dates("2024-01-01", "2024-12-31").aggregate_time(
                "monthly"
            )
        """
        self._validate_idata_exists()

        return MMMIDataWrapper.from_mmm(self)

    @property
    def summary(self) -> Any:  # type: ignore[no-any-return]
        """Access summary DataFrame generation functionality.

        Returns a factory for creating summary DataFrames from the model's
        InferenceData with configurable defaults for HDI levels and output format.

        Returns a fresh factory on each access. The factory includes both
        data and model, enabling all summary methods including transformation curves.

        Returns
        -------
        MMMSummaryFactory
            Factory providing methods for different summary types

        Examples
        --------
        .. code-block:: python

            # Get contribution summary (default: pandas, 94% HDI)
            df = mmm.summary.contributions()

            # Get ROAS summary
            df = mmm.summary.roas()

            # Get saturation curves (requires model - provided automatically)
            df = mmm.summary.saturation_curves(n_points=50)

            # Get adstock curves
            df = mmm.summary.adstock_curves(max_lag=15)

            # Get posterior predictive with custom settings
            df = mmm.summary.posterior_predictive(
                hdi_probs=[0.80, 0.94], frequency="monthly", output_format="polars"
            )

            # Create factory with different defaults (direct instantiation)
            from pymc_marketing.mmm.summary import MMMSummaryFactory

            polars_factory = MMMSummaryFactory(
                mmm.data, model=mmm, hdi_probs=[0.50, 0.94], output_format="polars"
            )
            df = polars_factory.contributions()  # Uses configured defaults

            # Get change over time
            df = mmm.summary.change_over_time()

        See Also
        --------
        MMMSummaryFactory : Factory class documentation
        pymc_marketing.mmm.summary : Module with all factory functions
        """
        from pymc_marketing.mmm.summary import MMMSummaryFactory

        self._validate_idata_exists()
        return MMMSummaryFactory(self.data, model=self)  # Pass both data and model

    @property
    def default_model_config(self) -> dict:
        """Define the default model configuration."""
        base_config = {
            "intercept": Prior("Normal", mu=0, sigma=2, dims=self.dims),
            "likelihood": Prior(
                "Normal",
                sigma=Prior("HalfNormal", sigma=2, dims=self.dims),
                dims=("date", *self.dims),
            ),
            "gamma_control": Prior(
                "Normal", mu=0, sigma=2, dims=(*self.dims, "control")
            ),
            "gamma_fourier": Prior(
                "Laplace", mu=0, b=1, dims=(*self.dims, "fourier_mode")
            ),
        }

        if self.time_varying_intercept:
            base_config["intercept_tvp_config"] = HSGPKwargs(
                m=200,
                L=None,
                eta_lam=1,
                ls_mu=5,
                ls_sigma=10,
                cov_func=None,
            )
        if self.time_varying_media:
            base_config["media_tvp_config"] = HSGPKwargs(
                m=200,
                L=None,
                eta_lam=1,
                ls_mu=5,
                ls_sigma=10,
                cov_func=None,
            )

        return {
            **base_config,
            **self.adstock.model_config,
            **self.saturation.model_config,
        }

    def post_sample_model_transformation(self) -> None:
        """Post-sample model transformation in order to store the HSGP state from fit."""
        names = []
        if self.time_varying_intercept:
            names.extend(
                SoftPlusHSGP.deterministics_to_replace("intercept_latent_process")
            )
        if self.time_varying_media:
            names.extend(
                SoftPlusHSGP.deterministics_to_replace(
                    "media_temporal_latent_multiplier"
                )
            )
        if not names:
            return

        self.model = deterministics_to_flat(self.model, names=names)

    def _validate_idata_exists(self) -> None:
        """Validate that the idata exists."""
        if not hasattr(self, "idata") or self.idata is None:
            raise ValueError("idata does not exist. Build the model first and fit.")

    def _validate_dims_in_multiindex(
        self, index: pd.MultiIndex, dims: tuple[str, ...], date_column: str
    ) -> list[str]:
        """Validate that dimensions exist in the MultiIndex.

        Parameters
        ----------
        index : pd.MultiIndex
            The MultiIndex to check
        dims : tuple[str, ...]
            The dimensions to validate
        date_column : str
            The name of the date column

        Returns
        -------
        list[str]
            List of valid dimensions found in the index

        Raises
        ------
        ValueError
            If date_column is not in the index
        """
        if date_column not in index.names:
            raise ValueError(f"date_column '{date_column}' not found in index")

        valid_dims = [dim for dim in dims if dim in index.names]
        return valid_dims

    def _validate_dims_in_dataframe(
        self, df: pd.DataFrame, dims: tuple[str, ...], date_column: str
    ) -> list[str]:
        """Validate that dimensions exist in the DataFrame columns.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to check
        dims : tuple[str, ...]
            The dimensions to validate
        date_column : str
            The name of the date column

        Returns
        -------
        list[str]
            List of valid dimensions found in the DataFrame

        Raises
        ------
        ValueError
            If date_column is not in the DataFrame
        """
        if date_column not in df.columns:
            raise ValueError(f"date_column '{date_column}' not found in DataFrame")

        valid_dims = [dim for dim in dims if dim in df.columns]
        return valid_dims

    def _validate_metrics(
        self, data: pd.DataFrame | pd.Series, metric_list: list[str]
    ) -> list[str]:
        """Validate that metrics exist in the data.

        Parameters
        ----------
        data : pd.DataFrame | pd.Series
            The data to check
        metric_list : list[str]
            The metrics to validate

        Returns
        -------
        list[str]
            List of valid metrics found in the data
        """
        if isinstance(data, pd.DataFrame):
            return [metric for metric in metric_list if metric in data.columns]
        else:  # pd.Series
            return [metric for metric in metric_list if metric in data.index.names]

    def _process_multiindex_series(
        self,
        series: pd.Series,
        date_column: str,
        valid_dims: list[str],
        metric_coordinate_name: str,
    ) -> xr.Dataset:
        """Process a MultiIndex Series into an xarray Dataset.

        Parameters
        ----------
        series : pd.Series
            The MultiIndex Series to process
        date_column : str
            The name of the date column
        valid_dims : list[str]
            List of valid dimensions
        metric_coordinate_name : str
            Name for the metric coordinate

        Returns
        -------
        xr.Dataset
            The processed xarray Dataset
        """
        # Reset index to get a DataFrame with all index levels as columns
        df = series.reset_index()

        # The series values become the metric values
        df_long = pd.DataFrame(
            {
                **{col: df[col] for col in [date_column, *valid_dims]},
                metric_coordinate_name: series.name,
                f"_{metric_coordinate_name}": series.values,
            }
        )

        # Drop duplicates to avoid non-unique MultiIndex
        df_long = df_long.drop_duplicates(
            subset=[date_column, *valid_dims, metric_coordinate_name]
        )

        # Convert to xarray, renaming date_column to "date" for internal consistency
        if valid_dims:
            df_long = df_long.rename(columns={date_column: "date"})
            return df_long.set_index(
                ["date", *valid_dims, metric_coordinate_name]
            ).to_xarray()
        df_long = df_long.rename(columns={date_column: "date"})
        return df_long.set_index(["date", metric_coordinate_name]).to_xarray()

    def _process_dataframe(
        self,
        df: pd.DataFrame,
        date_column: str,
        valid_dims: list[str],
        valid_metrics: list[str],
        metric_coordinate_name: str,
    ) -> xr.Dataset:
        """Process a DataFrame into an xarray Dataset.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to process
        date_column : str
            The name of the date column
        valid_dims : list[str]
            List of valid dimensions
        valid_metrics : list[str]
            List of valid metrics
        metric_coordinate_name : str
            Name for the metric coordinate

        Returns
        -------
        xr.Dataset
            The processed xarray Dataset
        """
        # Reshape DataFrame to long format
        df_long = df.melt(
            id_vars=[date_column, *valid_dims],
            value_vars=valid_metrics,
            var_name=metric_coordinate_name,
            value_name=f"_{metric_coordinate_name}",
        )

        # Drop duplicates to avoid non-unique MultiIndex
        df_long = df_long.drop_duplicates(
            subset=[date_column, *valid_dims, metric_coordinate_name]
        )

        # Convert to xarray, renaming date_column to "date" for internal consistency
        df_long = df_long.rename(columns={date_column: "date"})
        if valid_dims:
            return df_long.set_index(
                ["date", *valid_dims, metric_coordinate_name]
            ).to_xarray()
        return df_long.set_index(["date", metric_coordinate_name]).to_xarray()

    def _create_xarray_from_pandas(
        self,
        data: pd.DataFrame | pd.Series,
        date_column: str,
        dims: tuple[str, ...],
        metric_list: list[str],
        metric_coordinate_name: str,
    ) -> xr.Dataset:
        """Create an xarray Dataset from a DataFrame or Series.

        This method handles both DataFrame and MultiIndex Series inputs, reshaping them
        into a long format and converting into an xarray Dataset. It validates dimensions
        and metrics, ensuring they exist in the input data.

        Parameters
        ----------
        data : pd.DataFrame | pd.Series
            The input data to transform
        date_column : str
            The name of the date column
        dims : tuple[str, ...]
            The dimensions to include
        metric_list : list[str]
            List of metrics to include
        metric_coordinate_name : str
            Name for the metric coordinate in the output

        Returns
        -------
        xr.Dataset
            The transformed data in xarray format

        Raises
        ------
        ValueError
            If date_column is not found in the data
        """
        # Validate dimensions based on input type
        if isinstance(data, pd.Series):
            valid_dims = self._validate_dims_in_multiindex(
                index=data.index,  # type: ignore
                dims=dims,  # type: ignore
                date_column=date_column,  # type: ignore
            )
            return self._process_multiindex_series(
                series=data,
                date_column=date_column,
                valid_dims=valid_dims,
                metric_coordinate_name=metric_coordinate_name,
            )
        else:  # pd.DataFrame
            valid_dims = self._validate_dims_in_dataframe(
                df=data,
                dims=dims,
                date_column=date_column,  # type: ignore
            )
            valid_metrics = self._validate_metrics(data, metric_list)
            return self._process_dataframe(
                df=data,
                date_column=date_column,
                valid_dims=valid_dims,
                valid_metrics=valid_metrics,
                metric_coordinate_name=metric_coordinate_name,
            )

    def _generate_and_preprocess_model_data(
        self,
        X: pd.DataFrame,  # type: ignore
        y: pd.Series,  # type: ignore
    ):
        self.X = X  # type: ignore
        self.y = y  # type: ignore

        dataarrays = []

        X_dataarray = self._create_xarray_from_pandas(
            data=X,
            date_column=self.date_column,
            dims=self.dims,
            metric_list=self.channel_columns,
            metric_coordinate_name="channel",
        )
        dataarrays.append(X_dataarray)

        # Create a temporary DataFrame to properly handle the y data transformation
        temp_y_df = pd.concat([self.X[[self.date_column, *self.dims]], self.y], axis=1)
        y_dataarray = self._create_xarray_from_pandas(
            data=temp_y_df.set_index([self.date_column, *self.dims])[
                self.target_column
            ],
            date_column=self.date_column,
            dims=self.dims,
            metric_list=[self.target_column],
            metric_coordinate_name="target",
        ).sum("target")
        dataarrays.append(y_dataarray)

        if self.control_columns is not None:
            control_dataarray = self._create_xarray_from_pandas(
                data=X,
                date_column=self.date_column,
                dims=self.dims,
                metric_list=self.control_columns,
                metric_coordinate_name="control",
            )
            dataarrays.append(control_dataarray)

        self.xarray_dataset = xr.merge(dataarrays).fillna(0)

        self.xarray_dataset["_channel"] = self.xarray_dataset["_channel"].astype(float)

        self.model_coords = {
            dim: self.xarray_dataset.coords[dim].values
            for dim in self.xarray_dataset.coords.dims
        }

        if bool(self.time_varying_intercept) or bool(self.time_varying_media):
            self._time_index = np.arange(0, X[self.date_column].unique().shape[0])
            self._time_index_mid = X[self.date_column].unique().shape[0] // 2
            self._time_resolution = (
                X[self.date_column].iloc[1] - X[self.date_column].iloc[0]
            ).days

    def forward_pass(
        self,
        x: pt.TensorVariable | npt.NDArray[np.float64],
        dims: tuple[str, ...],
    ) -> pt.TensorVariable:
        """Transform channel input into target contributions of each channel.

        This method handles the ordering of the adstock and saturation
        transformations.

        This method must be called from without a pm.Model context but not
        necessarily in the instance's model. A dim named "channel" is required
        associated with the number of columns of `x`.

        Parameters
        ----------
        x : pt.TensorVariable | npt.NDArray[np.float64]
            The channel input which could be spends or impressions

        Returns
        -------
        The contributions associated with the channel input

        Examples
        --------
        .. code-block:: python

            mmm = MMM(
                date_column="date_week",
                channel_columns=["channel_1", "channel_2"],
                target_column="target",
            )

        """
        first, second = (
            (self.adstock, self.saturation)
            if self.adstock_first
            else (self.saturation, self.adstock)
        )

        return second.apply(x=first.apply(x=x, dims=dims), dims=dims)

    def _compute_scales(self) -> None:
        """Compute and save scaling factors for channels and target."""
        self.scalers = xr.Dataset()

        channel_method = getattr(
            self.xarray_dataset["_channel"],
            self.scaling.channel.method,
        )
        self.scalers["_channel"] = channel_method(
            dim=("date", *self.scaling.channel.dims)
        )

        target_method = getattr(
            self.xarray_dataset["_target"],
            self.scaling.target.method,
        )
        self.scalers["_target"] = target_method(dim=("date", *self.scaling.target.dims))

    def get_scales_as_xarray(self) -> dict[str, xr.DataArray]:
        """Return the saved scaling factors as xarray DataArrays.

        Returns
        -------
        dict[str, xr.DataArray]
            A dictionary containing the scaling factors for channels and target.

        Examples
        --------
        .. code-block:: python

            mmm = MMM(
                date_column="date_week",
                channel_columns=["channel_1", "channel_2"],
                target_column="target",
            )
            mmm.build_model(X, y)
            mmm.get_scales_as_xarray()

        """
        if not hasattr(self, "scalers"):
            raise ValueError(
                "Scales have not been computed yet. Build the model first."
            )

        return {
            "channel_scale": self.scalers._channel,
            "target_scale": self.scalers._target,
        }

    def _validate_model_was_built(self) -> None:
        """Validate that the model was built."""
        if not hasattr(self, "model"):
            raise ValueError(
                "Model was not built. Build the model first using MMM.build_model()"
            )

    def _validate_contribution_variable(self, var: str) -> None:
        """Validate that the variable ends with "_contribution" and is in the model."""
        if not (var.endswith("_contribution") or var == self.output_var):
            raise ValueError(
                f"Variable {var} must end with '_contribution' or be {self.output_var}"
            )

        if var not in self.model.named_vars:
            raise ValueError(f"Variable {var} is not in the model")

    def add_original_scale_contribution_variable(self, var: list[str]) -> None:
        """Add a pm.Deterministic variable to the model that multiplies by the scaler.

        Restricted to the model parameters. Only make it possible for "_contribution" variables.

        Parameters
        ----------
        var : list[str]
            The variables to add the original scale contribution variable.

        Examples
        --------
        .. code-block:: python

            model.add_original_scale_contribution_variable(
                var=["channel_contribution", "total_media_contribution", "y"]
            )

        """
        self._validate_model_was_built()
        target_dims = self.scalers._target.dims
        with self.model:
            for v in var:
                self._validate_contribution_variable(v)
                var_dims = self.model.named_vars_to_dims.get(v, ())
                mmm_dims_order = ("date", *self.dims)

                if v == "channel_contribution":
                    mmm_dims_order += ("channel",)
                elif v == "control_contribution":
                    mmm_dims_order += ("control",)
                elif v == "fourier_contribution":
                    mmm_dims_order += ("fourier_mode",)
                elif v == "yearly_seasonality_contribution":
                    pass  # Only has date dim
                elif v == "intercept_contribution":
                    pass  # Only has date dim

                deterministic_dims = tuple(
                    [
                        dim
                        for dim in mmm_dims_order
                        if dim in set(target_dims).union(var_dims)
                    ]
                )
                dim_handler = create_dim_handler(deterministic_dims)

                pm.Deterministic(
                    name=v + "_original_scale",
                    var=dim_handler(self.model[v], var_dims)
                    * dim_handler(
                        self.model["target_scale"],
                        target_dims,
                    ),
                    dims=deterministic_dims,
                )

    def build_model(  # type: ignore[override]
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        **kwargs,
    ) -> None:
        """Build a probabilistic model using PyMC for marketing mix modeling.

        The model incorporates channels, control variables, and Fourier components, applying
        adstock and saturation transformations to the channel data. The final model is
        constructed with multiple factors contributing to the response variable.

        Parameters
        ----------
        X : pd.DataFrame
            The input data for the model, which should include columns for channels,
            control variables (if applicable), and Fourier components (if applicable).

        y : Union[pd.Series, np.ndarray]
            The target/response variable for the modeling.

        **kwargs : dict
            Additional keyword arguments that might be required by underlying methods or utilities.

        Attributes Set
        ---------------
        model : pm.Model
            The PyMC model object containing all the defined stochastic and deterministic variables.

        Examples
        --------
        Initialize model with custom configuration

        .. code-block:: python

            from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
            from pymc_marketing.mmm.multidimensional import MMM
            from pymc_extras.prior import Prior

            custom_config = {
                "intercept": Prior("Normal", mu=0, sigma=2),
                "saturation_beta": Prior("Gamma", mu=1, sigma=3),
                "saturation_lambda": Prior("Beta", alpha=3, beta=1),
                "adstock_alpha": Prior("Beta", alpha=1, beta=3),
                "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=2)),
                "gamma_control": Prior("Normal", mu=0, sigma=2, dims="control"),
                "gamma_fourier": Prior("Laplace", mu=0, b=1, dims="fourier_mode"),
            }

            model = MMM(
                date_column="date_week",
                channel_columns=["x1", "x2"],
                adstock=GeometricAdstock(l_max=8),
                saturation=LogisticSaturation(),
                control_columns=[
                    "event_1",
                    "event_2",
                    "t",
                ],
                yearly_seasonality=2,
                model_config=custom_config,
            )

        """
        self._generate_and_preprocess_model_data(
            X=X,  # type: ignore
            y=y,  # type: ignore
        )
        # Compute and save scales
        self._compute_scales()

        with pm.Model(
            coords=self.model_coords,
        ) as self.model:
            _channel_scale = pm.Data(
                "channel_scale",
                self.scalers._channel.values,
                dims=self.scalers._channel.dims,
            )
            _target_scale = pm.Data(
                "target_scale",
                self.scalers._target,
                dims=self.scalers._target.dims,
            )

            _channel_data = pm.Data(
                name="channel_data",
                value=self.xarray_dataset._channel.transpose(
                    "date", *self.dims, "channel"
                ).values,
                dims=("date", *self.dims, "channel"),
            )

            _target = pm.Data(
                name="target_data",
                value=(
                    self.xarray_dataset._target.transpose("date", *self.dims).values
                ),
                dims=("date", *self.dims),
            )

            # Scale `channel_data` and `target`
            channel_dim_handler = create_dim_handler(("date", *self.dims, "channel"))
            channel_data_ = _channel_data / channel_dim_handler(
                _channel_scale,
                self.scalers._channel.dims,
            )
            channel_data_ = pt.switch(pt.isnan(channel_data_), 0.0, channel_data_)
            channel_data_.name = "channel_data_scaled"
            channel_data_.dims = ("date", *self.dims, "channel")

            target_dim_handler = create_dim_handler(("date", *self.dims))

            target_data_scaled = _target / target_dim_handler(
                _target_scale, self.scalers._target.dims
            )
            target_data_scaled.name = "target_scaled"
            target_data_scaled.dims = ("date", *self.dims)
            ## TODO: Find a better way to save it or access it in the pytensor graph.
            self.target_data_scaled = target_data_scaled

            for mu_effect in self.mu_effects:
                mu_effect.create_data(self)

            if bool(self.time_varying_intercept) or bool(self.time_varying_media):
                time_index = pm.Data(
                    name="time_index",
                    value=self._time_index,
                    dims="date",
                )

            # Add intercept logic
            if (
                isinstance(self.time_varying_intercept, bool)
                and self.time_varying_intercept
            ):
                intercept_baseline = self.model_config["intercept"].create_variable(
                    "intercept_baseline"
                )

                intercept_latent_process = create_hsgp_from_config(
                    X=time_index,
                    dims=("date", *self.dims),
                    config=self.model_config["intercept_tvp_config"],
                ).create_variable("intercept_latent_process")

                intercept = pm.Deterministic(
                    name="intercept_contribution",
                    var=intercept_baseline[None, ...] * intercept_latent_process,
                    dims=("date", *self.dims),
                )

            elif isinstance(self.time_varying_intercept, HSGPBase):
                intercept_baseline = self.model_config["intercept"].create_variable(
                    "intercept_baseline"
                )

                # Register internal time index and build latent process
                self.time_varying_intercept.register_data(time_index)
                intercept_latent_process = self.time_varying_intercept.create_variable(
                    "intercept_latent_process"
                )

                intercept = pm.Deterministic(
                    name="intercept_contribution",
                    var=intercept_baseline[None, ...] * intercept_latent_process,
                    dims=("date", *self.dims),
                )
            else:
                intercept = self.model_config["intercept"].create_variable(
                    name="intercept_contribution"
                )

            # Add media logic
            if isinstance(self.time_varying_media, bool) and self.time_varying_media:
                baseline_channel_contribution = pm.Deterministic(
                    name="baseline_channel_contribution",
                    var=self.forward_pass(
                        x=channel_data_, dims=(*self.dims, "channel")
                    ),
                    dims=("date", *self.dims, "channel"),
                )

                media_latent_process = create_hsgp_from_config(
                    X=time_index,
                    dims=("date", *self.dims),
                    config=self.model_config["media_tvp_config"],
                ).create_variable("media_temporal_latent_multiplier")

                channel_contribution = pm.Deterministic(
                    name="channel_contribution",
                    var=baseline_channel_contribution * media_latent_process[..., None],
                    dims=("date", *self.dims, "channel"),
                )
            elif isinstance(self.time_varying_media, HSGPBase):
                baseline_channel_contribution = self.forward_pass(
                    x=channel_data_, dims=(*self.dims, "channel")
                )
                baseline_channel_contribution.name = "baseline_channel_contribution"
                baseline_channel_contribution.dims = (
                    "date",
                    *self.dims,
                    "channel",
                )

                # Register internal time index and build latent process
                self.time_varying_media.register_data(time_index)
                media_latent_process = self.time_varying_media.create_variable(
                    "media_temporal_latent_multiplier"
                )

                # Determine broadcasting over channel axis
                media_dims = pm.modelcontext(None).named_vars_to_dims[
                    media_latent_process.name
                ]
                if "channel" in media_dims:
                    media_broadcast = media_latent_process
                else:
                    media_broadcast = media_latent_process[..., None]

                channel_contribution = pm.Deterministic(
                    name="channel_contribution",
                    var=baseline_channel_contribution * media_broadcast,
                    dims=("date", *self.dims, "channel"),
                )
            else:
                channel_contribution = pm.Deterministic(
                    name="channel_contribution",
                    var=self.forward_pass(
                        x=channel_data_, dims=(*self.dims, "channel")
                    ),
                    dims=("date", *self.dims, "channel"),
                )

            dim_handler = create_dim_handler(("date", *self.dims))
            pm.Deterministic(
                name="total_media_contribution_original_scale",
                var=(
                    channel_contribution.sum(axis=-1)
                    * dim_handler(_target_scale, self.scalers._target.dims)
                ).sum(),
                dims=(),
            )

            # Add other contributions and likelihood
            mu_var = intercept + channel_contribution.sum(axis=-1)

            if self.control_columns is not None and len(self.control_columns) > 0:
                gamma_control = self.model_config["gamma_control"].create_variable(
                    name="gamma_control"
                )

                control_data_ = pm.Data(
                    name="control_data",
                    value=self.xarray_dataset._control.transpose(
                        "date", *self.dims, "control"
                    ).values,
                    dims=("date", *self.dims, "control"),
                )

                control_contribution = pm.Deterministic(
                    name="control_contribution",
                    var=control_data_ * gamma_control,
                    dims=("date", *self.dims, "control"),
                )

                mu_var += control_contribution.sum(axis=-1)

            if self.yearly_seasonality is not None:
                dayofyear = pm.Data(
                    name="dayofyear",
                    value=pd.to_datetime(
                        self.model_coords["date"]
                    ).dayofyear.to_numpy(),
                    dims="date",
                )

                def create_deterministic(x: pt.TensorVariable) -> None:
                    pm.Deterministic(
                        "fourier_contribution",
                        x,
                        dims=("date", *self.yearly_fourier.prior.dims),
                    )

                yearly_seasonality_contribution = pm.Deterministic(
                    name="yearly_seasonality_contribution",
                    var=self.yearly_fourier.apply(
                        dayofyear, result_callback=create_deterministic
                    ),
                    dims=("date", *self.dims),
                )
                mu_var += yearly_seasonality_contribution

            for mu_effect in self.mu_effects:
                mu_var += mu_effect.create_effect(self)

            mu_var.name = "mu"
            mu_var.dims = ("date", *self.dims)

            self.model_config["likelihood"].dims = ("date", *self.dims)
            self.model_config["likelihood"].create_likelihood_variable(
                name=self.output_var,
                mu=mu_var,
                observed=target_data_scaled,
            )

    def _validate_date_overlap_with_include_last_observations(
        self, X: pd.DataFrame, include_last_observations: bool
    ) -> None:
        """Validate that include_last_observations is not used with overlapping dates.

        Parameters
        ----------
        X : pd.DataFrame
            The input data for prediction.
        include_last_observations : bool
            Whether to include the last observations of the training data.

        Raises
        ------
        ValueError
            If include_last_observations=True and input dates overlap with training dates.
        """
        if not include_last_observations:
            return

        # Get training dates and input dates
        training_dates = safe_to_datetime(self.model_coords["date"], "date")
        input_dates = safe_to_datetime(X[self.date_column].unique(), self.date_column)

        # Check for overlap
        overlapping_dates = set(training_dates).intersection(set(input_dates))

        if overlapping_dates:
            overlapping_dates_str = ", ".join(
                sorted([str(d.date()) for d in overlapping_dates])
            )
            raise ValueError(
                f"Cannot use include_last_observations=True when input dates overlap with training dates. "
                f"Overlapping dates found: {overlapping_dates_str}. "
                f"Either set include_last_observations=False or use input dates that don't overlap with training data."
            )

    def _posterior_predictive_data_transformation(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        include_last_observations: bool = False,
    ) -> xr.Dataset:
        """Transform the data for posterior predictive sampling.

        Parameters
        ----------
        X : pd.DataFrame
            The input data for prediction.
        y : pd.Series, optional
            The target data for prediction.
        include_last_observations : bool, optional
            Whether to include the last observations of the training data for continuity.

        Returns
        -------
        xr.Dataset
            The transformed data in xarray format.
        """
        # Validate that include_last_observations is not used with overlapping dates
        self._validate_date_overlap_with_include_last_observations(
            X, include_last_observations
        )

        dataarrays = []
        if include_last_observations:
            last_obs = self.xarray_dataset.isel(date=slice(-self.adstock.l_max, None))
            dataarrays.append(last_obs)

        # Transform X and y_pred to xarray
        X_xarray = self._create_xarray_from_pandas(
            data=X,
            date_column=self.date_column,
            dims=self.dims,
            metric_list=self.channel_columns,
            metric_coordinate_name="channel",
        ).transpose("date", *self.dims, "channel")
        dataarrays.append(X_xarray)

        if self.control_columns is not None:
            control_dataarray = self._create_xarray_from_pandas(
                data=X,
                date_column=self.date_column,
                dims=self.dims,
                metric_list=self.control_columns,
                metric_coordinate_name="control",
            ).transpose("date", *self.dims, "control")
            dataarrays.append(control_dataarray)

        if y is not None:
            y_xarray = (
                self._create_xarray_from_pandas(
                    data=y,
                    date_column=self.date_column,
                    dims=self.dims,
                    metric_list=[self.target_column],
                    metric_coordinate_name="target",
                )
                .sum("target")
                .transpose("date", *self.dims)
            )
        else:
            # Return empty xarray with same dimensions as the target but full of zeros
            # Use the same dtype as the existing target data to avoid dtype mismatches
            target_dtype = self.xarray_dataset._target.dtype
            y_xarray = xr.DataArray(
                np.zeros(
                    (
                        X[self.date_column].nunique(),
                        *[len(self.xarray_dataset.coords[dim]) for dim in self.dims],
                    ),
                    dtype=target_dtype,
                ),
                dims=("date", *self.dims),
                coords={
                    "date": X[self.date_column].unique(),
                    **{dim: self.xarray_dataset.coords[dim] for dim in self.dims},
                },
                name="_target",
            ).to_dataset()

        dataarrays.append(y_xarray)
        return xr.merge(dataarrays, join="outer", compat="no_conflicts").fillna(0)

    def _set_xarray_data(
        self,
        dataset_xarray: xr.Dataset,
        clone_model: bool = True,
    ) -> pm.Model:
        """Set xarray data into the model.

        Parameters
        ----------
        dataset_xarray : xr.Dataset
            Input data for channels and other variables.
        clone_model : bool, optional
            Whether to clone the model. Defaults to True.

        Returns
        -------
        None
        """
        model = cm(self.model) if clone_model else self.model

        # Get channel data and handle dtype conversion
        channel_values = dataset_xarray._channel.transpose(
            "date", *self.dims, "channel"
        )
        if "channel_data" in model.named_vars:
            original_dtype = model.named_vars["channel_data"].type.dtype
            channel_values = channel_values.astype(original_dtype)

        data = {"channel_data": channel_values}
        coords = self.model.coords.copy()
        coords["date"] = dataset_xarray["date"].to_numpy()

        if "_control" in dataset_xarray:
            control_values = dataset_xarray["_control"].transpose(
                "date", *self.dims, "control"
            )
            if "control_data" in model.named_vars:
                original_dtype = model.named_vars["control_data"].type.dtype
                control_values = control_values.astype(original_dtype)
            data["control_data"] = control_values
            coords["control"] = dataset_xarray["control"].to_numpy()
        if self.yearly_seasonality is not None:
            data["dayofyear"] = dataset_xarray["date"].dt.dayofyear.to_numpy()

        if self.time_varying_intercept or self.time_varying_media:
            data["time_index"] = infer_time_index(
                pd.Series(dataset_xarray["date"]),
                pd.Series(self.model_coords["date"]),
                self._time_resolution,
            )

        if "_target" in dataset_xarray:
            target_values = dataset_xarray._target.transpose("date", *self.dims)
            # Get the original dtype from the model's shared variable
            if "target_data" in model.named_vars:
                original_dtype = model.named_vars["target_data"].type.dtype
                # Convert to the original dtype to avoid precision loss errors
                data["target_data"] = target_values.astype(original_dtype)
            else:
                data["target_data"] = target_values

        self.new_updated_data = data
        self.new_updated_coords = coords
        self.new_updated_model = model

        with model:
            pm.set_data(data, coords=coords)

        return model

    def sample_posterior_predictive(
        self,
        X: pd.DataFrame | None = None,  # type: ignore
        extend_idata: bool = True,  # type: ignore
        combined: bool = True,  # type: ignore
        include_last_observations: bool = False,  # type: ignore
        clone_model: bool = True,  # type: ignore
        **sample_posterior_predictive_kwargs,  # type: ignore
    ) -> xr.DataArray:
        """Sample from the model's posterior predictive distribution.

        Parameters
        ----------
        X : pd.DataFrame
            Input data for prediction, with the same structure as the training data.
        y : pd.Series, optional
            Optional target data for validation or alignment. Default is None.
        extend_idata : bool, optional
            Whether to add predictions to the inference data object. Defaults to True.
        combined : bool, optional
            Combine chain and draw dimensions into a single sample dimension. Defaults to True.
        include_last_observations : bool, optional
            Whether to include the last observations of the training data for continuity
            (useful for adstock transformations). Defaults to False.
        clone_model : bool, optional
            Whether to clone the model. Defaults to True.
        **sample_posterior_predictive_kwargs
            Additional arguments for `pm.sample_posterior_predictive`.

        Returns
        -------
        xr.DataArray
            Posterior predictive samples.
        """
        X = _handle_deprecate_pred_argument(X, "X", sample_posterior_predictive_kwargs)
        # Update model data with xarray
        if X is None:
            raise ValueError("X values must be provided")
        dataset_xarray = self._posterior_predictive_data_transformation(
            X=X,
            include_last_observations=include_last_observations,
        )
        model = self._set_xarray_data(
            dataset_xarray=dataset_xarray,
            clone_model=clone_model,
        )

        for mu_effect in self.mu_effects:
            mu_effect.set_data(self, model, dataset_xarray)

        with model:
            # Sample from posterior predictive
            post_pred = pm.sample_posterior_predictive(
                self.idata, **sample_posterior_predictive_kwargs
            )

            if extend_idata and self.idata is not None:
                self.idata.add_groups(
                    posterior_predictive=post_pred.posterior_predictive,
                    posterior_predictive_constant_data=post_pred.constant_data,
                )  # type: ignore

        group = "posterior_predictive"
        posterior_predictive_samples = az.extract(post_pred, group, combined=combined)

        if include_last_observations:
            # Remove extra observations used for adstock continuity
            posterior_predictive_samples = posterior_predictive_samples.isel(
                date=slice(self.adstock.l_max, None)
            )

        return posterior_predictive_samples

    @validate_call(config={"arbitrary_types_allowed": True})
    def sample_saturation_curve(
        self,
        max_value: float = Field(
            1.0, gt=0, description="Maximum value for curve (in scaled space)."
        ),
        num_points: int = Field(100, gt=0, description="Number of points."),
        num_samples: int | None = Field(
            500, gt=0, description="Number of posterior samples to use."
        ),
        random_state: RandomState | None = None,
        original_scale: bool = Field(
            True, description="Whether to return curve in original scale."
        ),
        idata: InstanceOf[az.InferenceData] | None = Field(
            None, description="Optional InferenceData to sample from."
        ),
    ) -> xr.DataArray:
        """Sample saturation curves from posterior parameters.

        This method samples the saturation transformation curves using posterior
        parameters from the fitted model. It allows visualization of the
        diminishing returns relationship between media spend and contribution.

        Parameters
        ----------
        max_value : float, optional
            Maximum value for the curve x-axis, in scaled space (consistent with
            model internals). By default 1.0. This represents the maximum spend
            level in scaled units. To convert from original scale, divide by
            channel_scale:
            ``max_scaled = original_max / mmm.data.get_channel_scale().mean()``
        num_points : int, optional
            Number of points between 0 and max_value to evaluate the curve at.
            By default 100. Higher values give smoother curves but take longer.
        num_samples : int or None, optional
            Number of posterior samples to use for generating curves. By default 500.
            Samples are drawn randomly from the full posterior (across all chains
            and draws). Using fewer samples speeds up computation and reduces memory
            usage while still capturing posterior uncertainty. If None, all posterior
            samples are used without subsampling.
        random_state : int, np.random.Generator, or None, optional
            Random state for reproducible subsampling. Can be an integer seed,
            a numpy Generator instance, or None for non-reproducible sampling.
            Only used when num_samples is not None and less than total available
            samples.
        original_scale : bool, optional
            Whether to return curve y-values in original scale. If True (default),
            y-axis values (contribution) are multiplied by target_scale to convert
            from scaled to original units. If False, values remain in scaled space
            as used internally by the model. Note that x-axis values always remain
            in scaled space consistent with the max_value parameter.
        idata : az.InferenceData or None, optional
            Optional InferenceData to sample from. If None (default), uses
            self.idata. This allows sampling curves from different posterior
            distributions, such as from a different model or a subset of samples.

        Returns
        -------
        xr.DataArray
            Sampled saturation curves with dimensions:
            - Simple model: (x, channel, sample)
            - Panel model: (x, *custom_dims, channel, sample)

            The "sample" dimension indexes the posterior samples used.
            The "x" coordinate represents spend levels in scaled space (consistent
            with max_value). Y-values are in original scale when original_scale=True,
            otherwise in scaled space.

        Raises
        ------
        ValueError
            If called before model is fitted (idata doesn't exist) and no idata provided
        ValueError
            If original_scale=True but scale factors not found in constant_data

        Examples
        --------
        Sample curves with default parameters (original scale):

        >>> curves = mmm.sample_saturation_curve()
        >>> curves.dims
        ('sample', 'x', 'channel')

        Sample curves using all posterior samples:

        >>> curves_all = mmm.sample_saturation_curve(num_samples=None)

        Sample curves in scaled space:

        >>> curves_scaled = mmm.sample_saturation_curve(original_scale=False)

        Sample curves with custom max value and reproducible sampling:

        >>> channel_scale = mmm.data.get_channel_scale()
        >>> max_original = 10000  # $10,000
        >>> max_scaled = max_original / float(channel_scale.mean())
        >>> curves = mmm.sample_saturation_curve(
        ...     max_value=max_scaled, num_points=200, num_samples=1000, random_state=42
        ... )

        Sample curves from a different InferenceData:

        >>> external_idata = az.from_netcdf("other_model.nc")
        >>> curves = mmm.sample_saturation_curve(idata=external_idata)


        Notes
        -----
        - The max_value parameter is always in **scaled space**, consistent with how
          the model operates internally. This matches the pattern of other MMM methods.
        - For panel models, curves are generated for each combination of custom
          dimensions (e.g., each country) and channel.
        - The returned array includes a "sample" dimension for uncertainty
          quantification. Use `.mean(dim='sample')` for point estimates and
          `.quantile()` for credible intervals.
        - Posterior samples are drawn randomly without replacement when num_samples
          is less than the total available samples, otherwise all samples are used.
        """
        # Use provided idata or fall back to self.idata
        if idata is None:
            self._validate_idata_exists()
            idata = cast(az.InferenceData, self.idata)

        # Validate that posterior exists (model was fitted, not just prior sampled)
        if not hasattr(idata, "posterior") or idata.posterior is None:
            raise ValueError(
                "posterior not found in idata. "
                "The model must be fitted (call .fit()) before sampling saturation curves."
            )

        # Subsample posterior if needed
        parameters = subsample_draws(
            idata.posterior, num_samples=num_samples, random_state=random_state
        )

        # Sample curve using transformation's method
        curve = self.saturation.sample_curve(
            parameters=parameters,
            max_value=max_value,
            num_points=num_points,
        )

        # Flatten chain and draw dimensions to sample dimension
        curve = curve.stack(sample=("chain", "draw"))

        # Convert to original scale if requested
        if original_scale:
            # Scale y values (contribution) to original target units
            # Note: x coordinates remain in scaled space (same as max_value input)
            # since converting to original scale would require per-channel scaling
            # which complicates plotting and interpretation
            target_scale = MMMIDataWrapper(idata).get_target_scale()
            # Multiply by target_scale since saturation affects target variable
            curve = curve * target_scale

        return curve

    @validate_call(config={"arbitrary_types_allowed": True})
    def sample_adstock_curve(
        self,
        amount: float = Field(
            1.0, gt=0, description="Amount to apply the adstock transformation to."
        ),
        num_samples: int | None = Field(
            500, gt=0, description="Number of posterior samples to use."
        ),
        random_state: RandomState | None = None,
        idata: InstanceOf[az.InferenceData] | None = Field(
            None, description="Optional InferenceData to sample from."
        ),
    ) -> xr.DataArray:
        """Sample adstock curves from posterior parameters.

        This method samples the adstock transformation curves using posterior
        parameters from the fitted model. It allows visualization of the
        carryover effect of media exposure over time.

        Parameters
        ----------
        amount : float, optional
            Amount to apply the adstock transformation to. By default 1.0.
            This represents an impulse of spend at time 0, and the curve
            shows how this effect decays over subsequent time periods.
        num_samples : int or None, optional
            Number of posterior samples to use for generating curves. By default 500.
            Samples are drawn randomly from the full posterior (across all chains
            and draws). Using fewer samples speeds up computation and reduces memory
            usage while still capturing posterior uncertainty. If None, all posterior
            samples are used without subsampling.
        random_state : int, np.random.Generator, or None, optional
            Random state for reproducible subsampling. Can be an integer seed,
            a numpy Generator instance, or None for non-reproducible sampling.
            Only used when num_samples is not None and less than total available
            samples.
        idata : az.InferenceData or None, optional
            Optional InferenceData to sample from. If None (default), uses
            self.idata. This allows sampling curves from different posterior
            distributions, such as from a different model or a subset of samples.

        Returns
        -------
        xr.DataArray
            Sampled adstock curves with dimensions:
            - Simple model: (time since exposure, channel, sample)
            - Panel model: (time since exposure, *custom_dims, channel, sample)

            The "sample" dimension indexes the posterior samples used.
            The "time since exposure" coordinate represents time periods from 0
            to l_max (the maximum lag for the adstock transformation).

        Raises
        ------
        ValueError
            If called before model is fitted (idata doesn't exist) and no idata provided
        ValueError
            If idata exists but no posterior (model not fitted)

        Examples
        --------
        Sample curves with default parameters:

        >>> curves = mmm.sample_adstock_curve()
        >>> curves.dims
        ('sample', 'time since exposure', 'channel')

        Sample curves using all posterior samples:

        >>> curves_all = mmm.sample_adstock_curve(num_samples=None)

        Sample curves with custom amount and reproducible sampling:

        >>> curves = mmm.sample_adstock_curve(
        ...     amount=100.0, num_samples=1000, random_state=42
        ... )

        Sample curves from a different InferenceData:

        >>> external_idata = az.from_netcdf("other_model.nc")
        >>> curves = mmm.sample_adstock_curve(idata=external_idata)

        Notes
        -----
        - The adstock curve shows the carryover effect of a single impulse of
          media exposure over time, unlike saturation curves which show
          diminishing returns.
        - For panel models, curves are generated for each combination of custom
          dimensions (e.g., each country) and channel.
        - The returned array includes a "sample" dimension for uncertainty
          quantification. Use `.mean(dim='sample')` for point estimates and
          `.quantile()` for credible intervals.
        - Posterior samples are drawn randomly without replacement when num_samples
          is less than the total available samples.
        """
        # Use provided idata or fall back to self.idata
        if idata is None:
            self._validate_idata_exists()
            idata = cast(az.InferenceData, self.idata)

        # Validate that posterior exists
        if not hasattr(idata, "posterior") or idata.posterior is None:
            raise ValueError(
                "posterior not found in idata. "
                "The model must be fitted (call .fit()) before sampling adstock curves."
            )

        # Subsample posterior if needed
        parameters = subsample_draws(
            idata.posterior, num_samples=num_samples, random_state=random_state
        )

        # Sample curve using transformation's method
        curve = self.adstock.sample_curve(
            parameters=parameters,
            amount=amount,
        )

        # Flatten chain and draw dimensions to sample dimension
        curve = curve.stack(sample=("chain", "draw"))

        return curve

    @property
    def sensitivity(self) -> SensitivityAnalysis:
        """Access sensitivity analysis functionality.

        Returns a SensitivityAnalysis instance that can be used to run
        counterfactual sweeps on the model.

        Returns
        -------
        SensitivityAnalysis
            An instance configured with this MMM model.

        Examples
        --------
        .. code-block:: python

            mmm.sensitivity.run_sweep(
                var_names=["channel_1", "channel_2"],
                sweep_values=np.linspace(0.5, 2.0, 10),
                sweep_type="multiplicative",
            )

        """
        # Provide the underlying PyMC model, the model's inference data, and dims
        return SensitivityAnalysis(
            pymc_model=self.model, idata=self.idata, dims=self.dims
        )

    @property
    def incrementality(self) -> Incrementality:
        """Access incrementality and counterfactual analysis functionality.

        Returns an Incrementality instance for computing incremental contributions,
        ROAS, and CAC using counterfactual analysis with proper adstock carryover
        handling.

        Returns
        -------
        Incrementality
            An instance configured with this MMM model for computing
            incremental contributions, ROAS, and CAC.

        Examples
        --------
        Compute incremental contributions:

        >>> incremental = mmm.incrementality.compute_incremental_contribution(
        ...     start_date="2024-01-01",
        ...     end_date="2024-03-31",
        ...     frequency="weekly",
        ... )
        """
        self._validate_idata_exists()
        return Incrementality(model=self, idata=self.idata)

    def _make_channel_transform(
        self, df_lift_test: pd.DataFrame
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Create a function for transforming the channel data into the same scale as in the model.

        Parameters
        ----------
        df_lift_test : pd.DataFrame
            Lift test measurements.

        Returns
        -------
        Callable[[np.ndarray], np.ndarray]
            The function for scaling the channel data.
        """
        # The transformer will be passed a np.ndarray of data corresponding to this index.
        index_cols = [*list(self.dims), "channel"]
        # We reconstruct the input dataframe following the transformations performed within
        # `lift_test.scale_channel_lift_measurements()``.
        input_df = (
            df_lift_test.loc[:, [*index_cols, "x", "delta_x"]]
            .set_index(index_cols, append=True)
            .stack()
            .unstack(level=-2)
            .reindex(self.channel_columns, axis=1)  # type: ignore
            .fillna(0)
        )

        def channel_transform(input: np.ndarray) -> np.ndarray:
            """Transform lift test channel data to the same scale as in the model."""
            # reconstruct the df corresponding to the input np.ndarray.
            reconstructed = (
                pd.DataFrame(data=input, index=input_df.index, columns=input_df.columns)
                .stack()
                .unstack(level=-2)
            )
            return (
                (
                    # Scale the data according to the scaler coords.
                    reconstructed.to_xarray() / self.scalers._channel
                )
                .to_dataframe()
                .fillna(0)
                .stack()
                .unstack(level=-2)
                .loc[input_df.index, :]
                .values
            )

        # Finally return the scaled data as a np.ndarray corresponding to the input index order.
        return channel_transform

    def _make_target_transform(
        self, df_lift_test: pd.DataFrame
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Create a function for transforming the target measurements into the same scale as in the model.

        Parameters
        ----------
        df_lift_test : pd.DataFrame
            Lift test measurements.

        Returns
        -------
        Callable[[np.ndarray], np.ndarray]
            The function for scaling the target data.
        """
        # These are the same order as in the original lift test measurements.
        index_cols = [*list(self.dims), "channel"]
        input_idx = df_lift_test.set_index(index_cols, append=True).index

        def target_transform(input: np.ndarray) -> np.ndarray:
            """Transform lift test measurements and sigma to the same scale as in the model."""
            # Reconstruct the input df column with the correct index.
            reconstructed = pd.DataFrame(
                data=input, index=input_idx, columns=["target"]
            )
            return (
                (
                    # Scale the measurements.
                    reconstructed.to_xarray() / self.scalers._target
                )
                .to_dataframe()
                .loc[input_idx, :]
                .values
            )

        # Finally, return the scaled measurements as a np.ndarray corresponding to
        # the input index order.
        return target_transform

    def add_lift_test_measurements(
        self,
        df_lift_test: pd.DataFrame,
        dist: type[pm.Distribution] = pm.Gamma,
        name: str = "lift_measurements",
    ) -> None:
        """Add lift tests to the model.

        The model for the difference of a channel's saturation curve is created
        from `x` and `x + delta_x` for each channel. This random variable is
        then conditioned using the empirical lift, `delta_y`, and `sigma` of the lift test
        with the specified distribution `dist`.

        The pseudo-code for the lift test is as follows:

        .. code-block:: python

            model_estimated_lift = saturation_curve(x + delta_x) - saturation_curve(x)
            empirical_lift = delta_y
            dist(abs(model_estimated_lift), sigma=sigma, observed=abs(empirical_lift))


        The model has to be built before adding the lift tests.

        Parameters
        ----------
        df_lift_test : pd.DataFrame
            DataFrame with lift test results with at least the following columns:
                * `DIM_NAME`: dimension name. One column per dimension in `mmm.dims`.
                * `channel`: channel name. Must be present in `channel_columns`.
                * `x`: x axis value of the lift test.
                * `delta_x`: change in x axis value of the lift test.
                * `delta_y`: change in y axis value of the lift test.
                * `sigma`: standard deviation of the lift test.
        dist : pm.Distribution, optional
            The distribution to use for the likelihood, by default pm.Gamma
        name : str, optional
            The name of the likelihood of the lift test contribution(s),
            by default "lift_measurements". Name change required if calling
            this method multiple times.

        Raises
        ------
        RuntimeError
            If the model has not been built yet.
        KeyError
            If the 'channel' column or any of the model dimensions is not present
            in df_lift_test.

        Examples
        --------
        Build the model first then add lift test measurements.

        .. code-block:: python

            import pandas as pd
            import numpy as np

            from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation

            from pymc_marketing.mmm.multidimensional import MMM

            model = MMM(
                date_column="date",
                channel_columns=["x1", "x2"],
                target_column="target",
                adstock=GeometricAdstock(l_max=8),
                saturation=LogisticSaturation(),
                yearly_seasonality=2,
                dims=("geo",),
            )

            X = pd.DataFrame(
                {
                    "date": np.tile(
                        pd.date_range(start="2025-01-01", end="2025-05-01", freq="W"), 2
                    ),
                    "x1": np.random.rand(34),
                    "x2": np.random.rand(34),
                    "target": np.random.rand(34),
                    "geo": 17 * ["FIN"] + 17 * ["SWE"],
                }
            )
            y = X["target"]

            model.build_model(X.drop(columns=["target"]), y)

            df_lift_test = pd.DataFrame(
                {
                    "channel": ["x1", "x1"],
                    "geo": ["FIN", "SWE"],
                    "x": [1, 1],
                    "delta_x": [0.1, 0.2],
                    "delta_y": [0.1, 0.1],
                    "sigma": [0.1, 0.1],
                }
            )

            model.add_lift_test_measurements(df_lift_test)

        """
        if not hasattr(self, "model"):
            raise RuntimeError(
                "The model has not been built yet. Please, build the model first."
            )

        if "channel" not in df_lift_test.columns:
            raise KeyError(
                "The 'channel' column is required to map the lift measurements to the model."
            )

        for dim in self.dims:
            if dim not in df_lift_test.columns:
                raise KeyError(
                    f"The {dim} column is required to map the lift measurements to the model."
                )

        # Function to scale "delta_y", and "sigma" to same scale as target in model.
        target_transform = self._make_target_transform(df_lift_test)

        # Function to scale "x" and "delta_x" to the same scale as their respective channels.
        channel_transform = self._make_channel_transform(df_lift_test)

        df_lift_test_scaled = scale_lift_measurements(
            df_lift_test=df_lift_test,
            channel_col="channel",
            channel_columns=self.channel_columns,  # type: ignore
            channel_transform=channel_transform,
            target_transform=target_transform,
            dim_cols=list(self.dims),
        )
        # This is coupled with the name of the
        # latent process Deterministic
        time_varying_var_name = (
            "media_temporal_latent_multiplier" if self.time_varying_media else None
        )
        add_lift_measurements_to_likelihood_from_saturation(
            df_lift_test=df_lift_test_scaled,
            saturation=self.saturation,
            time_varying_var_name=time_varying_var_name,
            model=self.model,
            dist=dist,
            name=name,
        )

    def add_cost_per_target_calibration(
        self,
        data: pd.DataFrame,
        calibration_data: pd.DataFrame,
        name_prefix: str = "cpt_calibration",
    ) -> None:
        """Calibrate cost-per-target using constraints via ``pm.Potential``.

        This adds a deterministic ``cpt_variable_name`` computed as
        ``channel_data_spend / channel_contribution_original_scale`` and creates
        per-row penalty terms based on ``calibration_data`` using a quadratic penalty:

        ``penalty = - |cpt_mean - target|^2 / (2 * sigma^2)``.

        Parameters
        ----------
        data : pd.DataFrame
            Feature-like DataFrame with columns matching training ``X`` but with
            channel values representing spend (original units). Must include the
            same ``date`` and any model ``dims`` columns.
        calibration_data : pd.DataFrame
            DataFrame with rows specifying calibration targets. Must include:
              - ``channel``: channel name in ``self.channel_columns``
              - ``cost_per_target``: desired CPT value
              - ``sigma``: accepted deviation; larger => weaker penalty
            and one column per dimension in ``self.dims``.
        cpt_variable_name : str
            Name for the cost-per-target Deterministic in the model.
        name_prefix : str
            Prefix to use for generated potential names.

        Examples
        --------
        Build a model and calibrate CPT for selected (dims, channel):

        .. code-block:: python

            # spend data in original scale with the same structure as X
            spend_df = X.copy()
            # e.g., if X contains impressions, replace with monetary spend
            # spend_df[channels] = ...

            calibration_df = pd.DataFrame(
                {
                    "channel": ["C1", "C2"],
                    "geo": ["US", "US"],  # dims columns as needed
                    "cost_per_target": [30.0, 45.0],
                    "sigma": [2.0, 3.0],
                }
            )

            mmm.add_cost_per_target_calibration(
                data=spend_df,
                calibration_data=calibration_df,
                name_prefix="cpt_calibration",
            )
        """
        if not hasattr(self, "model"):
            raise RuntimeError("Model must be built before adding calibration.")

        # Validate required columns in calibration_data
        if "channel" not in calibration_data.columns:
            raise KeyError("'channel' column missing in calibration_data")
        for dim in self.dims:
            if dim not in calibration_data.columns:
                raise KeyError(
                    f"The {dim} column is required in calibration_data to map to model dims."
                )

        # Prepare spend data as xarray (original units)
        spend_ds = (
            self._create_xarray_from_pandas(
                data=data,
                date_column=self.date_column,
                dims=self.dims,
                metric_list=self.channel_columns,
                metric_coordinate_name="channel",
            )
            .transpose("date", *self.dims, "channel")
            .fillna(0)
        )

        spend_array = spend_ds._channel
        # Compute expected shape from the model
        channel_data_dims = self.model.named_vars_to_dims["channel_data"]
        expected_shape = tuple(len(self.model.coords[dim]) for dim in channel_data_dims)

        # Align spend array to the models dim order
        spend_aligned = spend_array.transpose(*channel_data_dims)

        # Now the check will fail when a coord (e.g., a country) is missing
        if spend_aligned.shape != expected_shape:
            raise ValueError(
                "Spend data shape does not match channel data dims in the model: "
                f"expected {expected_shape}, got {spend_aligned.shape}"
            )

        for dim in channel_data_dims:
            spend_labels = np.asarray(spend_aligned.coords[dim].values)
            model_labels = np.asarray(self.model.coords[dim])
            if not np.array_equal(spend_labels, model_labels):
                raise ValueError(
                    f"Spend data coordinates for dim {dim!r} do not match model coords: "
                    f"expected {model_labels.tolist()}, got {spend_labels.tolist()}"
                )

        spend_tensor = pt.as_tensor_variable(spend_aligned.values)

        with self.model:
            # Ensure original-scale contribution exists
            if "channel_contribution_original_scale" not in self.model.named_vars:
                raise ValueError(
                    "`channel_contribution_original_scale` is not in the model."
                    "Please, add the original scale contribution variable using the method "
                    "`add_original_scale_contribution_variable` before adding the cost-per-target calibration."
                )

            denom = pt.clip(
                self.model["channel_contribution_original_scale"], 1e-12, np.inf
            )
            cpt_tensor = spend_tensor / denom

        add_cost_per_target_potentials(
            calibration_df=calibration_data,
            model=self.model,
            cpt_value=cpt_tensor,
            name_prefix=name_prefix,
        )

    def experiment(
        self,
        experiment_type: str | ExperimentType,
        data: pd.DataFrame,
        **kwargs: Any,
    ) -> ExperimentResult:
        """Run a CausalPy causal inference experiment.

        Provides a convenient interface to run quasi-experimental analyses
        (Interrupted Time Series, Synthetic Control, Difference-in-Differences,
        Regression Discontinuity) using CausalPy, with results that can be
        directly converted to lift test calibration data.

        Parameters
        ----------
        experiment_type : str or ExperimentType
            The type of experiment to run. Accepts string aliases:

            - ``"its"``: Interrupted Time Series
            - ``"sc"``: Synthetic Control
            - ``"did"``: Difference-in-Differences
            - ``"rd"``: Regression Discontinuity

        data : pd.DataFrame
            The experiment data to pass to CausalPy.
        **kwargs
            Additional keyword arguments passed to the CausalPy experiment
            constructor. Common arguments include:

            - ``treatment_time``: When the treatment/intervention started.
            - ``formula``: Patsy formula for the model specification.
            - ``model``: A CausalPy model (e.g.
              ``causalpy.pymc_models.LinearRegression()``).

        Returns
        -------
        ExperimentResult
            A wrapped result with methods for summarizing, plotting,
            and converting to lift test format via :meth:`ExperimentResult.to_lift_test`.

        Raises
        ------
        ImportError
            If ``causalpy`` is not installed. Install it via
            ``pip install pymc-marketing[experiment]``.
        ValueError
            If ``experiment_type`` is not a valid experiment type.

        See Also
        --------
        add_lift_test : Add an experiment result as lift test calibration.
        add_lift_test_measurements : Add raw lift test measurements.
        ExperimentResult : The result wrapper class.

        Examples
        --------
        Run a Synthetic Control experiment and use it to calibrate the MMM:

        .. code-block:: python

            import causalpy as cp

            result = mmm.experiment(
                experiment_type="sc",
                data=df_experiment,
                treatment_time=70,
                formula="actual ~ 0 + a + b + c",
                model=cp.pymc_models.WeightedSumFitter(
                    sample_kwargs={"random_seed": 42}
                ),
            )

            # Visualize the experiment results
            result.summary()
            fig, ax = result.plot()

            # Convert to lift test and add to model
            mmm.add_lift_test(
                experiment=result,
                channel="tv",
                x=1000.0,
                delta_x=200.0,
            )

        """
        return run_experiment(
            experiment_type=experiment_type,
            data=data,
            **kwargs,
        )

    def add_lift_test(
        self,
        experiment: ExperimentResult,
        channel: str,
        x: float,
        delta_x: float,
        dist: type[pm.Distribution] = pm.Gamma,
        name: str = "lift_measurements",
        **dim_kwargs: str,
    ) -> None:
        """Add a CausalPy experiment result as a lift test calibration.

        Convenience method that converts an :class:`ExperimentResult` into the
        lift test DataFrame format and passes it to
        :meth:`add_lift_test_measurements`.

        Parameters
        ----------
        experiment : ExperimentResult
            The experiment result from :meth:`experiment` or
            :func:`~pymc_marketing.mmm.experiment.run_experiment`.
        channel : str
            The marketing channel name. Must be present in
            ``channel_columns``.
        x : float
            The baseline spend level for the channel during the experiment.
        delta_x : float
            The change in channel spend during the experiment.
        dist : type[pm.Distribution], optional
            The distribution to use for the likelihood, by default
            ``pm.Gamma``.
        name : str, optional
            The name of the likelihood contribution, by default
            ``"lift_measurements"``.
        **dim_kwargs : str
            Dimension values for the lift test, e.g. ``geo="US"``.
            Keys must match the model's ``dims``.

        Raises
        ------
        RuntimeError
            If the model has not been built yet.
        KeyError
            If the channel or dimension values don't match the model.

        See Also
        --------
        experiment : Run a CausalPy experiment.
        add_lift_test_measurements : Add raw lift test measurements.

        Examples
        --------
        .. code-block:: python

            import causalpy as cp

            result = mmm.experiment(
                experiment_type="its",
                data=df_experiment,
                treatment_time=pd.Timestamp("2024-01-01"),
                formula="y ~ 1 + t",
                model=cp.pymc_models.LinearRegression(),
            )

            mmm.add_lift_test(
                experiment=result,
                channel="tv",
                x=1000.0,
                delta_x=200.0,
                geo="US",
            )

        """
        df_lift_test = experiment.to_lift_test(
            channel=channel,
            x=x,
            delta_x=delta_x,
            **dim_kwargs,
        )
        self.add_lift_test_measurements(
            df_lift_test=df_lift_test,
            dist=dist,
            name=name,
        )

    def create_fit_data(
        self,
        X: pd.DataFrame | xr.Dataset | xr.DataArray,
        y: np.ndarray | pd.Series | xr.DataArray,
    ) -> xr.Dataset:
        """Create a fit dataset aligned on date and present dimensions.

        Builds and returns an xarray ``Dataset`` that contains:

        - data variables from ``X`` (all non-coordinate columns),
        - the target variable from ``y`` under ``self.output_var``, and
        - coordinates on ``(self.date_column, *dims present in X)``.

        Parameters
        ----------
        X : pd.DataFrame | xr.Dataset | xr.DataArray
            Feature data. If an xarray object is provided, it is converted to a
            DataFrame via ``to_dataframe().reset_index()`` before processing.
        y : np.ndarray | pd.Series | xr.DataArray
            Target values. Must align with ``X`` either by position (same length)
            or via a MultiIndex that includes ``(self.date_column, *dims present in X)``.

        Returns
        -------
        xr.Dataset
            Dataset indexed by ``(self.date_column, *dims present in X)`` with the
            feature variables and a target variable named ``self.output_var``.

        Raises
        ------
        ValueError
            - If ``self.date_column`` is missing in ``X``.
            - If ``y`` is a ``np.ndarray`` and its length does not match ``X``.
            - If ``y`` cannot be aligned to ``X`` by index or position.
        RuntimeError
            If the target column is missing after alignment.

        Notes
        -----
        - The original date column name is preserved (``self.date_column``).
        - Coordinates are assigned only for dimensions present in ``X``.
        - Data is sorted by ``(self.date_column, *dims present in X)`` prior to
          conversion to xarray.

        Examples
        --------
        .. code-block:: python

            ds = mmm.create_fit_data(X, y)

        """
        # --- Coerce X to DataFrame ---
        if isinstance(X, xr.Dataset):
            X_df = X.to_dataframe().reset_index()
        elif isinstance(X, xr.DataArray):
            X_df = X.to_dataframe(name=X.name or "value").reset_index()
        else:
            X_df = X.copy()

        if self.date_column not in X_df.columns:
            raise ValueError(f"'{self.date_column}' not in X columns")

        # --- Coerce y to Series ---
        if isinstance(y, xr.DataArray):
            y_s = y.to_series()
        elif isinstance(y, np.ndarray):
            if len(y) != len(X_df):
                raise ValueError("y length must match X when passed as ndarray")
            y_s = pd.Series(y, index=X_df.index)
        else:
            y_s = y.copy()
        y_s.name = self.target_column

        dims_in_X = [d for d in self.dims if d in X_df.columns]
        coord_cols = [self.date_column, *dims_in_X]

        # Alignment strategies
        if isinstance(y_s.index, pd.MultiIndex) and set(coord_cols).issubset(
            y_s.index.names
        ):
            # Align via MultiIndex
            X_mi = X_df.set_index(coord_cols)
            aligned = y_s.reindex(X_mi.index)
            if aligned.isna().any():  # fallback merge if mismatch
                X_df = X_df.merge(
                    y_s.reset_index(),
                    on=coord_cols,
                    how="left",
                )
            else:
                X_df[self.target_column] = aligned.values
        elif len(y_s) == len(X_df):
            # Positional
            X_df[self.target_column] = y_s.to_numpy()
        else:
            # Try merge if y has columns as index levels
            if isinstance(y_s.index, pd.MultiIndex) and set(coord_cols).issubset(
                y_s.index.names
            ):
                X_df = X_df.merge(y_s.reset_index(), on=coord_cols, how="left")
            else:
                raise ValueError(
                    "Cannot align y with X; incompatible indices / lengths"
                )

        if self.target_column not in X_df.columns:
            raise RuntimeError(
                f"Target column {self.target_column} missing after alignment"
            )

        ds = X_df.sort_values(coord_cols).set_index(coord_cols).to_xarray()
        return ds

    def build_from_idata(self, idata: az.InferenceData) -> None:
        """Rebuild the model from an ``InferenceData`` object.

        Uses the stored fit dataset in ``idata`` to reconstruct the model graph by
        calling :meth:`build_model`. This is commonly used as part of a ``load``
        workflow to restore a model prior to sampling predictive quantities.

        Parameters
        ----------
        idata : az.InferenceData
                Inference data containing the fit dataset under the ``fit_data`` group.

        Returns
        -------
        None

        Notes
        -----
        - Expects ``idata.fit_data`` to exist and contain both features and the
            target column named ``self.output_var``.
        - This rebuilds the model structure; it does not attach posterior samples.
            Assign ``self.idata = idata`` separately if you need to reuse samples.

        Examples
        --------
        .. code-block:: python

            mmm.build_from_idata(idata)

        """
        # Restore mu_effects from idata attrs if present
        if "mu_effects" in idata.attrs:
            mu_effects_data = json.loads(idata.attrs["mu_effects"])
            self.mu_effects = []
            for effect_data in mu_effects_data:
                try:
                    effect = _deserialize_mu_effect(effect_data)
                    self.mu_effects.append(effect)
                except Exception as e:
                    # Log warning but continue - don't fail the load for unsupported effects
                    # Catches ValueError, KeyError, AttributeError, pydantic.ValidationError, etc.
                    warnings.warn(f"Could not deserialize mu_effect: {e}", stacklevel=2)

        dataset = idata.fit_data.to_dataframe()

        if isinstance(dataset.index, pd.MultiIndex) or isinstance(
            dataset.index, pd.DatetimeIndex
        ):
            dataset = dataset.reset_index()
        # type: ignore
        X = dataset.drop(columns=[self.target_column])
        y = dataset[self.target_column]

        self.build_model(X, y)  # type: ignore


def create_sample_kwargs(
    sampler_config: dict[str, Any] | None,
    progressbar: bool | None,
    random_seed: RandomState | None,
    **kwargs,
) -> dict[str, Any]:
    """Create the dictionary of keyword arguments for `pm.sample`.

    Parameters
    ----------
    sampler_config : dict | None
        The configuration dictionary for the sampler. If None, defaults to an empty dict.
    progressbar : bool, optional
        Whether to show the progress bar during sampling. Defaults to True.
    random_seed : RandomState, optional
        The random seed for the sampler.
    **kwargs : Any
        Additional keyword arguments to pass to the sampler.

    Returns
    -------
    dict
        The dictionary of keyword arguments for `pm.sample`.
    """
    # Ensure sampler_config is a dictionary
    sampler_config = sampler_config.copy() if sampler_config is not None else {}

    # Handle progress bar configuration
    sampler_config["progressbar"] = (
        progressbar
        if progressbar is not None
        else sampler_config.get("progressbar", True)
    )

    # Add random seed if provided
    if random_seed is not None:
        sampler_config["random_seed"] = random_seed

    # Update with additional keyword arguments
    sampler_config.update(kwargs)
    return sampler_config


class MultiDimensionalBudgetOptimizerWrapper(OptimizerCompatibleModelWrapper):
    """Wrapper for the BudgetOptimizer to handle multi-dimensional model."""

    def __init__(
        self,
        model: MMM,
        start_date: str,
        end_date: str,
        compile_kwargs: dict | None = None,
    ):
        self.model_class = model
        self.start_date = start_date
        self.end_date = end_date
        # Compute the number of periods to allocate budget for
        self.zero_data = create_zero_dataset(
            model=self.model_class,
            start_date=start_date,
            end_date=end_date,
            include_carryover=False,
        )
        self.num_periods = len(self.zero_data[self.model_class.date_column].unique())
        self.compile_kwargs = compile_kwargs
        # Adding missing dependencies for compatibility with BudgetOptimizer
        self._channel_scales = 1.0

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped MMM model."""
        try:
            # First, try to get the attribute from the wrapper itself
            return object.__getattribute__(self, name)
        except AttributeError:
            # If not found, delegate to the wrapped model
            try:
                return getattr(self.model_class, name)
            except AttributeError as e:
                # Raise an AttributeError if the attribute is not found in either
                raise AttributeError(
                    f"'{type(self).__name__}' object and its wrapped 'MMM' object have no attribute '{name}'"
                ) from e

    def _set_predictors_for_optimization(self, num_periods: int) -> pm.Model:
        """Return the respective PyMC model with any predictors set for optimization."""
        # Use the model's method for transformation
        dataset_xarray = self._posterior_predictive_data_transformation(
            X=self.zero_data,
            include_last_observations=False,
        )

        # Use the model's method to set data
        pymc_model = self._set_xarray_data(
            dataset_xarray=dataset_xarray,
            clone_model=True,  # Ensure we work on a clone
        )

        # Use the model's mu_effects and set data using the model instance
        for mu_effect in self.mu_effects:
            mu_effect.set_data(self, pymc_model, dataset_xarray)

        return pymc_model

    def optimize_budget(
        self,
        budget: float | int,
        budget_bounds: xr.DataArray | None = None,
        response_variable: str = "total_media_contribution_original_scale",
        utility_function: UtilityFunctionType = average_response,
        constraints: Sequence[dict[str, Any]] = (),
        default_constraints: bool = True,
        budgets_to_optimize: xr.DataArray | None = None,
        budget_distribution_over_period: xr.DataArray | None = None,
        callback: bool = False,
        **minimize_kwargs,
    ) -> (
        tuple[xr.DataArray, OptimizeResult]
        | tuple[xr.DataArray, OptimizeResult, list[dict[str, Any]]]
    ):
        """Optimize the budget allocation for the model.

        Parameters
        ----------
        budget : float | int
            Total budget to allocate.
        budget_bounds : xr.DataArray | None
            Budget bounds per channel.
        response_variable : str
            Response variable to optimize.
        utility_function : UtilityFunctionType
            Utility function to maximize.
        constraints : Sequence[dict[str, Any]]
            Custom constraints for the optimizer.
        default_constraints : bool
            Whether to add default constraints.
        budgets_to_optimize : xr.DataArray | None
            Mask defining which budgets to optimize.
        budget_distribution_over_period : xr.DataArray | None
            Distribution factors for budget allocation over time. Should have dims ("date", *budget_dims)
            where date dimension has length num_periods. Values along date dimension should sum to 1 for
            each combination of other dimensions. If None, budget is distributed evenly across periods.
        callback : bool
            Whether to return callback information tracking optimization progress.
        **minimize_kwargs
            Additional arguments for the optimizer.

        Returns
        -------
        tuple
            Optimal budgets and optimization result. If callback=True, also returns
            a list of dictionaries with optimization information at each iteration.
        """
        from pymc_marketing.mmm.budget_optimizer import BudgetOptimizer

        allocator = BudgetOptimizer(
            num_periods=self.num_periods,
            utility_function=utility_function,
            response_variable=response_variable,
            custom_constraints=constraints,
            default_constraints=default_constraints,
            budgets_to_optimize=budgets_to_optimize,
            budget_distribution_over_period=budget_distribution_over_period,
            model=self,  # Pass the wrapper instance itself to the BudgetOptimizer
            compile_kwargs=self.compile_kwargs,
        )

        return allocator.allocate_budget(
            total_budget=budget,
            budget_bounds=budget_bounds,
            callback=callback,
            **minimize_kwargs,
        )

    def _apply_budget_distribution_pattern(
        self,
        data_with_noise: pd.DataFrame,
        budget_distribution: xr.DataArray,
    ) -> pd.DataFrame:
        """Apply budget distribution pattern to noisy data.

        This method multiplies the channel values in data_with_noise by the
        corresponding values in budget_distribution, aligning by date, dimensions,
        and channels. Works like a left join where all dimensions must match.

        Parameters
        ----------
        data_with_noise : pd.DataFrame
            DataFrame with noise added to channel allocations.
        budget_distribution : xr.DataArray
            Distribution factors with dims ("date", *budget_dims) and "channel".

        Returns
        -------
        pd.DataFrame
            The data_with_noise DataFrame with channel values multiplied by
            the distribution pattern where dimensions match.
        """
        # Set index to match the expected dimensions
        index_cols = [self.date_column, *list(self.dims)]

        # Store original index to restore later
        original_index = data_with_noise.index

        # Set MultiIndex for proper alignment
        data_with_noise_indexed = data_with_noise.set_index(index_cols)

        # Convert DataFrame channel columns to xarray
        data_xr = data_with_noise_indexed[self.channel_columns].to_xarray()

        # Stack channel columns into a 'channel' dimension to match budget_distribution format
        data_xr_stacked = data_xr.to_array(dim="channel")

        # Rename date column to 'date' for consistency with budget_distribution
        if self.date_column != "date":
            data_xr_stacked = data_xr_stacked.rename({self.date_column: "date"})

        # Handle date coordinate alignment
        # If budget_distribution has integer date indices, map them to actual dates
        if np.issubdtype(budget_distribution.coords["date"].dtype, np.integer):
            # Get unique dates from data_xr_stacked
            unique_dates = safe_to_datetime(
                data_xr_stacked.coords["date"].values, "date"
            )
            unique_dates_sorted = sorted(unique_dates.unique())

            # Map integer indices to actual dates
            date_mapping = {i: date for i, date in enumerate(unique_dates_sorted)}

            # Create new coordinates with actual dates
            new_coords = dict(budget_distribution.coords)
            new_coords["date"] = [
                date_mapping[i] for i in budget_distribution.coords["date"].values
            ]

            # Recreate budget_distribution with new date coordinates
            _budget_distribution = xr.DataArray(
                budget_distribution.values,
                dims=budget_distribution.dims,
                coords=new_coords,
            )
        else:
            # If dates are already in the correct format, use as is
            _budget_distribution = budget_distribution

        # Multiply by budget distribution (xarray will automatically align dimensions)
        # Only matching channels and dates will be multiplied
        data_xr_multiplied = data_xr_stacked * (_budget_distribution * self.num_periods)

        # Convert back to DataFrame format
        # First unstack the channel dimension
        data_xr_unstacked = data_xr_multiplied.to_dataset(dim="channel")

        # Rename 'date' back to original date column name if needed
        if self.date_column != "date":
            data_xr_unstacked = data_xr_unstacked.rename({"date": self.date_column})

        # Convert to DataFrame
        multiplied_df = data_xr_unstacked.to_dataframe()

        # Update the channel columns in the indexed DataFrame
        for channel in self.channel_columns:
            if channel in multiplied_df.columns:
                data_with_noise_indexed.loc[:, channel] = multiplied_df[channel]

        # Reset to original index structure
        data_with_noise = data_with_noise_indexed.reset_index()
        data_with_noise.index = original_index

        return data_with_noise

    def _apply_carryover_effect(
        self,
        data_with_noise: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply carryover effect by zeroing out last observations.

        Parameters
        ----------
        data_with_noise : pd.DataFrame
            DataFrame with channel allocations

        Returns
        -------
        pd.DataFrame
            DataFrame with carryover effect applied
        """
        from pymc_marketing.mmm.utils import _convert_frequency_to_timedelta

        # Get date series and infer frequency
        date_series = safe_to_datetime(
            data_with_noise[self.date_column], self.date_column
        )
        inferred_freq = pd.infer_freq(date_series.unique())

        if inferred_freq is None:  # fall-back if inference fails
            warnings.warn(
                f"Could not infer frequency from '{self.date_column}'. Using weekly ('W').",
                UserWarning,
                stacklevel=2,
            )
            inferred_freq = "W"

        # Calculate the cutoff date
        cutoff_date = data_with_noise[
            self.date_column
        ].max() - _convert_frequency_to_timedelta(self.adstock.l_max, inferred_freq)

        # Zero out channel values after the cutoff date
        data_with_noise.loc[
            data_with_noise[self.date_column] > cutoff_date,
            self.channel_columns,
        ] = 0

        return data_with_noise

    def sample_response_distribution(
        self,
        allocation_strategy: xr.DataArray,
        noise_level: float = 0.001,
        additional_var_names: list[str] | None = None,
        include_last_observations: bool = False,
        include_carryover: bool = True,
        budget_distribution_over_period: xr.DataArray | None = None,
    ) -> az.InferenceData:
        """Generate synthetic dataset and sample posterior predictive based on allocation.

        Parameters
        ----------
        allocation_strategy : DataArray
            The allocation strategy for the channels.
        noise_level : float
            The relative level of noise to add to the data allocation.
        additional_var_names : list[str] | None
            Additional variable names to include in the posterior predictive sampling.
        include_last_observations : bool
            Whether to include the last observations for continuity.
        include_carryover : bool
            Whether to include carryover effects.
        budget_distribution_over_period : xr.DataArray | None
            Distribution factors for budget allocation over time. Should have dims ("date", *budget_dims)
            where date dimension has length num_periods. Values along date dimension should sum to 1 for
            each combination of other dimensions. If provided, multiplies the noise values by this distribution.

        Returns
        -------
        az.InferenceData
            The posterior predictive samples based on the synthetic dataset.
        """
        data = create_zero_dataset(
            model=self,
            start_date=self.start_date,
            end_date=self.end_date,
            channel_xr=allocation_strategy.to_dataset(dim="channel"),
            include_carryover=include_carryover,
        )

        data_with_noise = add_noise_to_channel_allocation(
            df=data,
            channels=self.channel_columns,
            rel_std=noise_level,
            seed=42,
        )

        # Apply budget distribution pattern if provided
        if budget_distribution_over_period is not None:
            data_with_noise = self._apply_budget_distribution_pattern(
                data_with_noise=data_with_noise,
                budget_distribution=budget_distribution_over_period,
            )

        if include_carryover:
            data_with_noise = self._apply_carryover_effect(data_with_noise)

        constant_data = allocation_strategy.to_dataset(name="allocation")
        _dataset = data_with_noise.set_index([self.date_column, *list(self.dims)])[
            self.channel_columns
        ].to_xarray()

        var_names = [
            self.output_var,
            "channel_contribution",
            "total_media_contribution_original_scale",
        ]
        if additional_var_names is not None:
            var_names.extend(additional_var_names)

        return (
            self.sample_posterior_predictive(
                X=data_with_noise,
                extend_idata=False,
                include_last_observations=include_last_observations,
                var_names=var_names,
                progressbar=False,
            )
            .merge(constant_data)
            .merge(_dataset)
        )
