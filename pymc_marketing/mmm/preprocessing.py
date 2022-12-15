from typing import Callable

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler

__all__ = [
    "preprocessing_method",
    "MixMaxScaleTarget",
    "MaxAbsScaleChannels",
    "StandardizeControls",
]


def preprocessing_method(method: Callable) -> Callable:
    if not hasattr(method, "_tags"):
        method._tags = {}
    method._tags["preprocessing"] = True
    return method


class MixMaxScaleTarget:
    @preprocessing_method
    def min_max_scale_target_data(self, data_df: pd.DataFrame) -> pd.DataFrame:
        target_vector = data_df[self.target_column].to_numpy().reshape(-1, 1)
        transformers = [("scaler", MinMaxScaler())]
        pipeline = Pipeline(steps=transformers)
        self.target_transformer: Pipeline = pipeline.fit(X=target_vector)
        data_df[self.target_column] = self.target_transformer.transform(
            X=target_vector
        ).flatten()
        return data_df


class MaxAbsScaleChannels:
    @preprocessing_method
    def max_abs_scale_channel_data(self, data_df: pd.DataFrame) -> pd.DataFrame:
        channel_data: pd.DataFrame = data_df[self.channel_columns]
        transformers = [("scaler", MaxAbsScaler())]
        pipeline: Pipeline = Pipeline(steps=transformers)
        self.channel_transformer: Pipeline = pipeline.fit(X=channel_data.to_numpy())
        data_df[self.channel_columns] = self.channel_transformer.transform(
            channel_data.to_numpy()
        )
        return data_df


class StandardizeControls:
    @preprocessing_method
    def standardize_control_data(self, data_df: pd.DataFrame) -> pd.DataFrame:
        control_data: pd.DataFrame = data_df[self.control_columns]
        transformers = [("scaler", StandardScaler())]
        pipeline: Pipeline = Pipeline(steps=transformers)
        self.control_transformer: Pipeline = pipeline.fit(X=control_data.to_numpy())
        data_df[self.control_columns] = self.control_transformer.transform(
            control_data.to_numpy()
        )
        return data_df
