from unittest.mock import patch

import numpy as np
import pandas as pd
import pymc as pm
import pytest

from pymc_marketing.mmm.base import MMM
from pymc_marketing.mmm.preprocessing import (
    preprocessing_method_X,
    preprocessing_method_y,
)
from pymc_marketing.mmm.validating import validation_method_X, validation_method_y

seed: int = sum(map(ord, "pymc_marketing"))
rng: np.random.Generator = np.random.default_rng(seed=seed)
date_data: pd.DatetimeIndex = pd.date_range(
    start="2019-06-01", end="2021-12-31", freq="W-MON"
)

n: int = date_data.size

toy_X = pd.DataFrame(
    data={
        "date": date_data,
        "channel_1": rng.integers(low=0, high=400, size=n),
        "channel_2": rng.integers(low=0, high=50, size=n),
        "control_1": rng.gamma(shape=1000, scale=500, size=n),
        "control_2": rng.gamma(shape=100, scale=5, size=n),
        "other_column_1": rng.integers(low=0, high=100, size=n),
        "other_column_2": rng.normal(loc=0, scale=1, size=n),
    }
)
toy_y = pd.Series(data=rng.integers(low=0, high=100, size=n))


class TestMMM:
    @patch("pymc_marketing.mmm.base.MMM.validate_target")
    @patch("pymc_marketing.mmm.base.MMM.validate_date_col")
    @patch("pymc_marketing.mmm.base.MMM.validate_channel_columns")
    @pytest.mark.parametrize(
        argnames="channel_prior",
        argvalues=[None, pm.HalfNormal.dist(sigma=5)],
        ids=["no_channel_prior", "channel_prior"],
    )
    @pytest.mark.parametrize(
        argnames="channel_columns",
        argvalues=[
            (["channel_1"]),
            (["channel_1", "channel_2"]),
        ],
        ids=[
            "single_channel",
            "multiple_channel",
        ],
    )
    def test_init(
        self,
        validate_channel_columns,
        validate_date_col,
        validate_target,
        channel_columns,
        channel_prior,
    ) -> None:
        validate_channel_columns.configure_mock(_tags={"validation_X": True})
        validate_date_col.configure_mock(_tags={"validation_X": True})
        validate_target.configure_mock(_tags={"validation_y": True})
        toy_validation_X_count = 0
        toy_validation_y_count = 0
        toy_preprocess_X_count = 0
        toy_preprocess_y_count = 0

        class ToyMMM(MMM):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.X = None
                self.y = None
                self.preprocessed_data = {"X": None, "y": None}

            def build_model(*args, **kwargs):
                pass

            def generate_and_preprocess_model_data(self, X, y):
                self.validate("X", X)
                self.validate("y", y)
                self.preprocessed_data["X"] = self.preprocess("X", X)
                self.preprocessed_data["y"] = self.preprocess("y", y)
                self.X = X
                self.y = y

            @property
            def default_model_config(self):
                pass

            @property
            def default_sampler_config(self):
                pass

            def _data_setter(self):
                pass

            def _serializable_model_config(self):
                pass

            @validation_method_X
            def toy_validation_X(self, data):
                nonlocal toy_validation_X_count
                toy_validation_X_count += 1
                pd.testing.assert_frame_equal(data, toy_X)
                return None

            @validation_method_y
            def toy_validation_y(self, data):
                nonlocal toy_validation_y_count
                toy_validation_y_count += 1
                pd.testing.assert_series_equal(data, toy_y)
                return None

            @preprocessing_method_X
            def toy_preprocessing_X(self, data):
                nonlocal toy_preprocess_X_count
                toy_preprocess_X_count += 1
                pd.testing.assert_frame_equal(data, toy_X)
                return data

            @preprocessing_method_y
            def toy_preprocessing_y(self, data):
                nonlocal toy_preprocess_y_count
                toy_preprocess_y_count += 1
                pd.testing.assert_series_equal(data, toy_y)
                return data

        instance = ToyMMM(
            date_column="date",
            channel_columns=channel_columns,
            channel_prior=channel_prior,
        )
        instance.generate_and_preprocess_model_data(toy_X, toy_y)
        pd.testing.assert_frame_equal(instance.X, toy_X)
        pd.testing.assert_frame_equal(instance.preprocessed_data["X"], toy_X)
        pd.testing.assert_series_equal(instance.y, toy_y)
        pd.testing.assert_series_equal(instance.preprocessed_data["y"], toy_y)
        validate_target.assert_called_once_with(instance, toy_y)
        validate_date_col.assert_called_once_with(instance, toy_X)
        validate_channel_columns.assert_called_once_with(instance, toy_X)

        assert toy_validation_X_count == 1
        assert toy_validation_y_count == 1
        assert toy_preprocess_X_count == 1
        assert toy_preprocess_y_count == 1
