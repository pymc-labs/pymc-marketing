from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pymc as pm
from pytensor.tensor import TensorVariable

from pymc_marketing.mmm.models.base import BaseMMM

from pymc_marketing.mmm.utils import (
    _get_distribution,
    _create_likelihood_distribution
)

from pymc_marketing.mmm.models.components.lagging import _get_lagging_function
from pymc_marketing.mmm.models.components.saturation import _get_saturation_function
from pymc_marketing.mmm.models.components.calibration import (
    add_hill_empirical_lift_measurements_to_likelihood,
    add_menten_empirical_lift_measurements_to_likelihood
)

from pymc_marketing.mmm.models.components.time import (
    HSGPTVP,
    compute_beta_guess_for_inverse_gamma,
    compute_mass_within_bounds_for_inverse_gamma,
    find_c_and_m
)

class HierarchicalDelayedSaturatedMMM(BaseMMM):
    _model_type = "HierarchicalDelayedSaturatedMMM"

    def __init__(
        self,
        data: pd.DataFrame = None,
        date_column: str = None,
        channel_columns: List[str] = None,
        control_columns: List[str] = None,
        hierarchy_column: str = None,
        target_column: str = None,
        lagging_function: str = "geometric",
        saturation_function: str = "michaelis_menten",
        model_config: Optional[Dict] = None,
        sampler_config: Optional[Dict] = None,
        model_coordinates: Dict = None,
        adstock_max_lagging: int = 10,
        hsgp_config: Dict = None,
        **kwargs,
    ) -> None:
        self.adstock_max_lagging = adstock_max_lagging
        self.date_column = date_column
        self.channel_columns = channel_columns
        self.control_columns = control_columns
        self.hierarchy_column = hierarchy_column
        self.target_column = target_column
        self.lagging_function = lagging_function
        self.saturation_function = saturation_function

        self.model_config = model_config
        self.sampler_config = sampler_config
        self.hsgp_config = hsgp_config
        self.coords = model_coordinates

        super().__init__(data=data, **kwargs)

        if isinstance(self.model_config, type(None)):
            self.model_config = self._default_model_config()

        if isinstance(self.coords, type(None)):
            self.coords = self._default_model_coordinates()
        
        if isinstance(self.hsgp_config, type(None)):
            self.hsgp_config = self._default_hsgp_config()

    def add_lift_measurements(self, df_lift_tests: pd.DataFrame) -> None:
        if self.saturation_function == "hill":
            kwargs = {
                "sigma_name": "saturation_sigma_channel",
                "lambda_name": "saturation_lambda_channel",
                "beta_name": "saturation_beta_channel",
            }
            func = add_hill_empirical_lift_measurements_to_likelihood
        elif self.saturation_function == "michaelis_menten":
            kwargs = {
                "alpha_name": "saturation_alpha_channel",
                "lambda_name": "saturation_lambda_channel",
            }
            func = add_menten_empirical_lift_measurements_to_likelihood
        else:
            msg = (
                f"Saturation function {self.saturation_function}"
                " is not supported for lift test."
            )
            raise ValueError(msg)

        with self.model:
            func(df_lift_tests, model=self.model, **kwargs)

    def _default_model_config(self) -> Dict:
        model_config = {}

        if self.lagging_function == "geometric":
            model_config["adstock_offset"] = {
                "dist": "Normal",
                "kwargs": {"mu": 0.5, "sigma": 0.5},
            }
            model_config["adstock_mu"] = {"dist": "HalfNormal", "kwargs": {"sigma": 2}}
            model_config["adstock_sigma"] = {
                "dist": "HalfNormal",
                "kwargs": {"sigma": 2},
            }

        if self.saturation_function == "hill":
            model_config["saturation_sigma_offset"] = {
                "dist": "Normal",
                "kwargs": {"mu": 1, "sigma": 2},
            }
            model_config["saturation_sigma_mu"] = {
                "dist": "HalfNormal",
                "kwargs": {"sigma": 2},
            }
            model_config["saturation_sigma_sigma"] = {
                "dist": "HalfNormal",
                "kwargs": {"sigma": 2},
            }
            model_config["saturation_lambda_offset"] = {
                "dist": "Normal",
                "kwargs": {"mu": 1, "sigma": 2},
            }
            model_config["saturation_lambda_mu"] = {
                "dist": "HalfNormal",
                "kwargs": {"sigma": 2},
            }
            model_config["saturation_lambda_sigma"] = {
                "dist": "HalfNormal",
                "kwargs": {"sigma": 2},
            }
            model_config["saturation_beta_offset"] = {
                "dist": "Normal",
                "kwargs": {"mu": 1, "sigma": 2},
            }
            model_config["saturation_beta_mu"] = {
                "dist": "HalfNormal",
                "kwargs": {"sigma": 2},
            }
            model_config["saturation_beta_sigma"] = {
                "dist": "HalfNormal",
                "kwargs": {"sigma": 2},
            }
        
        elif self.saturation_function == "michaelis_menten":
            model_config["saturation_alpha_offset"] = {
                "dist": "Normal",
                "kwargs": {"mu": 1, "sigma": 2},
            }
            model_config["saturation_alpha_mu"] = {
                "dist": "HalfNormal",
                "kwargs": {"sigma": 2},
            }
            model_config["saturation_alpha_sigma"] = {
                "dist": "HalfNormal",
                "kwargs": {"sigma": 2},
            }
            model_config["saturation_lambda_offset"] = {
                "dist": "Normal",
                "kwargs": {"mu": 1, "sigma": 2},
            }
            model_config["saturation_lambda_mu"] = {
                "dist": "HalfNormal",
                "kwargs": {"sigma": 2},
            }
            model_config["saturation_lambda_sigma"] = {
                "dist": "HalfNormal",
                "kwargs": {"sigma": 2},
            }

        model_config["intercept"] = {"dist": "Gamma", "kwargs": {"mu": 1, "sigma": 2}}
        model_config["intercept_mu"] = {"dist": "HalfNormal", "kwargs": {"sigma": 1}}
        model_config["intercept_sigma"] = {"dist": "HalfNormal", "kwargs": {"sigma": 2}}
        model_config["beta_control"] = {
            "dist": "Normal",
            "kwargs": {"mu": 0, "sigma": 1},
        }
        model_config["likelihood"] = {
            "dist": "Normal",
            "kwargs": {
                "sigma": {"dist": "HalfNormal", "kwargs": {"sigma": 2}},
            },
        }

        return model_config
    
    def _default_hsgp_config(self) -> Dict:
        return {
            "lower_landscape": 8,
            "upper_landscape": 15,
            "alpha_landscape": 70,
            "mass_probability": 0.05
        }

    def _default_model_coordinates(self):
        if isinstance(self.date_column, type(None)):
            raise ValueError("Date column must be provided.")
        elif isinstance(self.channel_columns, type(None)):
            raise ValueError("Channel columns must be provided.")
        elif isinstance(self.hierarchy_column, type(None)):
            raise ValueError("Hierarchy column must be provided.")

        return {
            "date": self.data[self.date_column].unique(),
            "channel": self.channel_columns,
            "hierarchy": self.data[self.hierarchy_column].unique(),
        }

    def _transform_data(self):
        self.data.rename(
            columns={
                self.date_column: "date",
                self.hierarchy_column: "hierarchy",
                self.target_column: "y",
            },
            inplace=True,
        )

        self.x_channel_data = self.data.melt(
            id_vars=["date", "hierarchy"],
            value_vars=self.channel_columns,
            var_name="channel",
            value_name="value",
        )
        self.x_channel_data = (
            self.x_channel_data.set_index(["date", "channel", "hierarchy"])
            .to_xarray()
            .to_dataarray()
            .squeeze("variable", drop=True)
            .sel(channel=self.channel_columns)
        )

        if isinstance(self.control_columns, List):
            self.x_control_data = self.data.melt(
                id_vars=["date", "hierarchy"],
                value_vars=self.control_columns,
                var_name="control",
                value_name="value",
            )
            self.x_control_data = (
                self.x_control_data.set_index(["date", "control", "hierarchy"])
                .to_xarray()
                .to_dataarray()
                .squeeze("variable", drop=True)
                .sel(control=self.control_columns)
            )

        self.y = (
            self.data[["date", "hierarchy", "y"]]
            .melt(
                id_vars=["date", "hierarchy"],
                value_vars=["y"],
                var_name="y",
                value_name="value",
            )[["date", "hierarchy", "value"]]
            .rename(columns={"value": "y"})
            .fillna(0)
        )
        self.y = (
            self.y.set_index(["date", "hierarchy"])
            .to_xarray()
            .to_dataarray()
            .squeeze("variable", drop=True)
            .sel(hierarchy=self.data[self.hierarchy_column].unique())
        )

        return self

    def build_model(self):
        self._transform_data()

        self.intercept_dist = _get_distribution(dist=self.model_config["intercept"])
        self.intercept_mu_dist = _get_distribution(dist=self.model_config["intercept_mu"])
        self.intercept_sigma_dist = _get_distribution(dist=self.model_config["intercept_sigma"])
        
        if isinstance(self.control_columns, List):
            self.beta_control_dist = _get_distribution(
                dist=self.model_config["beta_control"]
            )

        beta_landscape = compute_beta_guess_for_inverse_gamma(
            self.hsgp_config["alpha_landscape"], 
            self.hsgp_config["lower_landscape"], 
            self.hsgp_config["upper_landscape"]
        )
        landscape_mass = compute_mass_within_bounds_for_inverse_gamma(
            self.hsgp_config["alpha_landscape"], 
            beta_landscape, 
            self.hsgp_config["lower_landscape"], 
            self.hsgp_config["upper_landscape"]
        )

        with pm.Model() as hierarchical_delayed_saturated_mmm:
            self.lag_function = _get_lagging_function(
                name=self.lagging_function,
                max_lagging=self.adstock_max_lagging,
                model_config=self.model_config,
                model=hierarchical_delayed_saturated_mmm,
            )

            self.sat_function = _get_saturation_function(
                name=self.saturation_function,
                model_config=self.model_config,
                model=hierarchical_delayed_saturated_mmm,
            )

            for coord, value in self.coords.items():
                hierarchical_delayed_saturated_mmm.add_coord(coord, value, mutable=True)

            channel_data = pm.MutableData(
                name="channel_data",
                value=self.x_channel_data.values,
                dims=("date", "channel", "hierarchy"),
            )

            target_y = pm.MutableData(
                name="target_y", value=self.y.values, dims=("date", "hierarchy")
            )

            # INTERCEPT
            prior_intercept = pm.find_constrained_prior(
                self.intercept_dist,
                lower=0.1,
                upper=0.9,
                init_guess={"mu": 0.5, "sigma": 0.5},
            )

            intercept_offset = self.intercept_dist(
                'intercept_offset',
                **prior_intercept,
                dims=("hierarchy")
            )

            intercept_mu = self.intercept_mu_dist(
                name="intercept_mu", 
                **self.model_config["intercept_mu"]["kwargs"]
            )
            intercept_sigma = self.intercept_sigma_dist(
                name="intercept_sigma", 
                **self.model_config["intercept_sigma"]["kwargs"]
            )

            intercept = pm.Deterministic(
                "intercept",
                var=intercept_mu + intercept_offset * intercept_sigma,
                dims=("hierarchy")
            )

            # LAGGING FUNCTION
            adstock_contribution = self.lag_function.apply(data=channel_data)

            # SATURATION FUNCTION
            contribution = self.sat_function.apply(data=adstock_contribution)

            # HSGP | TIME VARYING COEFFICIENT
            ls = pm.InverseGamma(
                "ls",
                alpha=self.hsgp_config["alpha_landscape"],
                beta=beta_landscape,
            )
            eta = pm.Exponential(
                f"eta", lam=-np.log(self.hsgp_config["mass_probability"])
            )  #probability of mass greater than 1
            cov = eta**2 * pm.gp.cov.Matern52(1, ls=ls)
            m, c = find_c_and_m(
                self.hsgp_config["lower_landscape"], 
                self.hsgp_config["upper_landscape"], 
                landscape_mass, 
                N=len(self.coords["date"])
            )

            time_varying_factor = HSGPTVP(
                name="time_varying_factor", 
                m=m, 
                c=c, 
                cov=cov, 
                dims=("date", "hierarchy"), 
                size=(
                    len(self.coords["date"]), 
                    len(self.coords["hierarchy"])
                )
            ).build()
            
            time_varying_factor_positive = pm.Deterministic(
                "time_varying_factor_positive", 
                var=pm.math.log(1 + pm.math.exp(time_varying_factor)), 
                dims=("date", "hierarchy")
            )

            channel_varying_contribution = pm.Deterministic(
                "channel_varying_contribution",
                var=contribution * time_varying_factor_positive[:, None, :],  # Broadcasting to apply across channels
                dims=("date", "channel", "hierarchy")
            )

            varying_contribution = pm.Deterministic(
                    "varying_contribution", 
                    var=channel_varying_contribution.sum(axis=1),
                    dims=("date", "hierarchy")
            )

            dep_var = (
                intercept + 
                varying_contribution
            )

            # control contribution
            if isinstance(self.control_columns, List):
                control_data = pm.MutableData(
                    name="control_data",
                    value=self.x_control_data.values,
                    dims=("date", "control", "hierarchy"),
                )

                beta_control_params = pm.find_constrained_prior(
                    self.beta_control_dist,
                    lower=0.1,
                    upper=0.8,
                    init_guess={"mu": 0.3, "sigma": 0.5},
                )
                beta_control = self.beta_control_dist(
                    "beta_control", **beta_control_params, dims=("control", "hierarchy")
                )

                control_contribution = pm.Deterministic(
                    "control_contribution",
                    var=(control_data * beta_control),
                    dims=("date", "control", "hierarchy"),
                )

                dep_var += control_contribution.sum(axis=1)

            _create_likelihood_distribution(
                dist=self.model_config["likelihood"],
                mu=dep_var,
                observed=target_y,
                dims=("date", "hierarchy"),
            )
        self.model = hierarchical_delayed_saturated_mmm
        return self
