from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pymc_marketing.mmm.base import MMM
from pymc_marketing.mmm.preprocessing import preprocessing_method
from pymc_marketing.mmm.validating import validation_method

seed: int = sum(map(ord, "pymc_marketing"))
rng: np.random.Generator = np.random.default_rng(seed=seed)
date_data: pd.DatetimeIndex = pd.date_range(
    start="2019-06-01", end="2021-12-31", freq="W-MON"
)

n: int = date_data.size

toy_df = pd.DataFrame(
    data={
        "date": date_data,
        "y": rng.integers(low=0, high=100, size=n),
        "channel_1": rng.integers(low=0, high=400, size=n),
        "channel_2": rng.integers(low=0, high=50, size=n),
        "control_1": rng.gamma(shape=1000, scale=500, size=n),
        "control_2": rng.gamma(shape=100, scale=5, size=n),
        "other_column_1": rng.integers(low=0, high=100, size=n),
        "other_column_2": rng.normal(loc=0, scale=1, size=n),
    }
)


class TestMMM:
    @patch("pymc_marketing.mmm.base.MMM.validate_target")
    @patch("pymc_marketing.mmm.base.MMM.validate_date_col")
    @patch("pymc_marketing.mmm.base.MMM.validate_channel_columns")
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
    ) -> None:
        validate_channel_columns.configure_mock(_tags={"validation": True})
        validate_date_col.configure_mock(_tags={"validation": True})
        validate_target.configure_mock(_tags={"validation": True})
        toy_validation_count = 0
        toy_preprocess_count = 0
        build_model_count = 0

        class ToyMMM(MMM):
            def build_model(*args, **kwargs):
                nonlocal build_model_count
                build_model_count += 1
                pd.testing.assert_frame_equal(kwargs["data_df"], toy_df)
                return None

            @validation_method
            def toy_validation(self, data):
                nonlocal toy_validation_count
                toy_validation_count += 1
                pd.testing.assert_frame_equal(data, toy_df)
                return None

            @preprocessing_method
            def toy_preprocessing(self, data):
                nonlocal toy_preprocess_count
                toy_preprocess_count += 1
                pd.testing.assert_frame_equal(data, toy_df)
                return data

        instance = ToyMMM(
            data_df=toy_df,
            target_column="y",
            date_column="date",
            channel_columns=channel_columns,
        )
        pd.testing.assert_frame_equal(instance.data_df, toy_df)
        pd.testing.assert_frame_equal(instance.preprocessed_data, toy_df)
        validate_target.assert_called_once_with(instance, toy_df)
        validate_date_col.assert_called_once_with(instance, toy_df)
        validate_channel_columns.assert_called_once_with(instance, toy_df)

        assert toy_validation_count == 1
        assert toy_preprocess_count == 1
        assert build_model_count == 1
