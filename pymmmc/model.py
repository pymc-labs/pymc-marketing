from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy.typing as npt
import pandas as pd
import pymc as pm
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from pymmmc.transformers import geometric_adstock_vectorized, logistic_saturation

ModelInputDataFormat = Union[
    pd.DataFrame, pd.Series, Tuple[npt.ArrayLike, npt.ArrayLike]
]


@dataclass(frozen=True)
class DataContainer(ABC):
    raw_data: Union[pd.Series, pd.DataFrame]

    def __post_init__(self) -> None:
        self.__validate_raw_data()

    def __validate_raw_data(self) -> None:
        if not isinstance(self.raw_data, (pd.Series, pd.DataFrame)):
            raise TypeError(
                f"raw_data must be a pandas.Series or pandas.DataFrame, "
                f"but got {type(self.raw_data)}"
            )
        if self.raw_data.empty:
            raise ValueError("raw_data must not be empty")
        if self.raw_data.isna().any().any():
            raise ValueError("raw_data must not contain NaN")

    @abstractmethod
    def get_preprocessed_data(self) -> ModelInputDataFormat:
        raise NotImplementedError("data processing method must be implemented.")


@dataclass(frozen=True)
class ContinuousDataContainer(DataContainer):
    transformer: Optional[BaseEstimator] = None

    def get_preprocessed_data(self) -> pd.DataFrame:
        if self.transformer is None:
            return self.raw_data.copy()
        self.transformer.fit(self.raw_data)
        return pd.DataFrame(
            data=self.transformer.transform(self.raw_data),
            index=self.raw_data.index,
            columns=self.raw_data.columns,
        )


@dataclass(frozen=True)
class MediaDataContainer(ContinuousDataContainer):
    transformer: Optional[MinMaxScaler] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if (self.raw_data < 0.0).any().any():
            raise ValueError("media must not contain negative values")


@dataclass(frozen=True)
class CategoryDataContainer(DataContainer):
    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.raw_data, pd.Series):
            raise TypeError(
                f"raw_data must be a pandas.Series for categorical variables, "
                f"but got {type(self.raw_data)}"
            )
        if self.raw_data.nunique() < 2:
            raise ValueError("category must have at least two unique values")

    def get_preprocessed_data(self) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        return self.raw_data_df.factorize(sort=True)


class BaseMMModel(ABC):
    def __init__(
        self,
        train_df: pd.DataFrame,
        target: str,
        date_col: str,
        media_columns: List[str],
    ) -> None:
        self.train_df = train_df.copy()
        self.target = target
        self.date_col = date_col
        self.media_columns = media_columns
        self._validate_input_data()

    def _validate_input_data(self) -> None:
        raise NotImplementedError("validation method must be implemented.")

    @abstractmethod
    def _build_model(self) -> pm.Model:
        raise NotImplementedError(
            "model building method depending on `model_name` must be implemented."
        )


class AdstockGeometricLogistiSaturation(BaseMMModel):
    def __init__(
        self,
        train_df: pd.DataFrame,
        target: str,
        date_col: str,
        media_columns: List[str],
    ) -> None:
        super().__init__(train_df, target, date_col, media_columns)

    def _prepare_data(
        self, train_df: pd.DataFrame
    ) -> Tuple[npt.ArrayLike, pd.Series, pd.DataFrame]:
        dates = train_df[self.date_col].to_numpy()
        target_dc = ContinuousDataContainer(
            raw_data=train_df[self.target], transformer=StandardScaler()
        )
        target_data = target_dc.get_preprocessed_data()
        media_dc = MediaDataContainer(raw_data=train_df[self.media_columns])
        media_data = media_dc.get_preprocessed_data()
        return dates, target_data, media_data

    def _build_model(
        self, dates: npt.ArrayLike, target_data: pd.Series, media_data: pd.DataFrame
    ) -> pm.Model:

        coords: Dict[str, Any] = {
            "date": dates,
            "channel": media_data.columns,
        }

        with pm.Model(coords=coords) as model:

            # --- Priors ---
            intercept = pm.Normal(name="intercept", mu=0, sigma=2)
            # coefficients marketing channels
            beta = pm.HalfNormal(name="beta", sigma=2, dims="channel")
            # adstock parameter
            alpha = pm.Beta(name="alpha", alpha=1, beta=3, dims="channel")
            # saturation parameter
            lam = pm.Gamma(name="lam", alpha=3, beta=1, dims="channel")
            # likelihood standard deviation
            sigma = pm.HalfNormal(name="sigma", sigma=2)
            # degrees of freedom
            nu = pm.Gamma(name="nu", alpha=25, beta=2)

            # --- Model Parametrization ---
            # marketing channel effects
            channels_saturated = pm.Deterministic(
                name="channels_saturated",
                var=logistic_saturation(x=media_data.data, lam=lam),
                dims=("date_week", "channel"),
            )
            channels_saturated_adstock = pm.Deterministic(
                name="channels_saturated_adstock",
                var=geometric_adstock_vectorized(
                    x=channels_saturated,
                    alpha=alpha,
                    l_max=12,
                    normalize=True,
                ),
                dims=("date_week", "channel"),
            )
            channels_effects = pm.Deterministic(
                name="channels_effects",
                var=pm.math.dot(channels_saturated_adstock, beta),
                dims=("date_week"),
            )

            mu = pm.Deterministic(
                name="mu",
                var=intercept + channels_effects,
                dims="date_week",
            )

            # --- Likelihood ---
            likelihood = pm.StudentT(  # noqa F841
                "likelihood",
                mu=mu,
                nu=nu,
                sigma=sigma,
                observed=target_data,
                dims="date",
            )

        return model
