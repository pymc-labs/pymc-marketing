from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pymmmc.mmm.base import RescaledMMM, preprocessing_method, validation_method

seed: int = sum(map(ord, "pymmmc"))
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


@pytest.fixture(scope="module")
def toy_mmm_class():
    class ToyMMM(RescaledMMM):
        def build_model(*args, **kwargs):
            return None

    return ToyMMM


@pytest.fixture(
    scope="module",
    params=[["channel_1"], ["channel_1", "channel_2"]],
    ids=[
        "single_channel",
        "multiple_channel",
    ],
)
def channel_columns(request):
    return request.param


@pytest.fixture(scope="module")
def toy_mmm(channel_columns, toy_mmm_class):
    return toy_mmm_class(
        data_df=toy_df,
        target_column="y",
        date_column="date",
        channel_columns=channel_columns,
    )


class TestRescaledMMM:
    @patch("pymmmc.mmm.RescaledMMM.validate_target")
    @patch("pymmmc.mmm.RescaledMMM.validate_date_col")
    @patch("pymmmc.mmm.RescaledMMM.validate_channel_columns")
    @patch("pymmmc.mmm.RescaledMMM.min_max_scale_target_data", return_value=toy_df)
    @patch("pymmmc.mmm.RescaledMMM.max_abs_scale_channel_data", return_value=toy_df)
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
        max_abs_scale_channel_data,
        min_max_scale_target_data,
        validate_channel_columns,
        validate_date_col,
        validate_target,
        channel_columns,
    ) -> None:
        max_abs_scale_channel_data.configure_mock(_tags={"preprocessing": True})
        min_max_scale_target_data.configure_mock(_tags={"preprocessing": True})
        validate_channel_columns.configure_mock(_tags={"validation": True})
        validate_date_col.configure_mock(_tags={"validation": True})
        validate_target.configure_mock(_tags={"validation": True})
        toy_validation_count = 0
        toy_preprocess_count = 0
        build_model_count = 0

        class ToyMMM(RescaledMMM):
            def build_model(*args, **kwargs):
                nonlocal build_model_count
                build_model_count += 1
                return None

            @validation_method
            def toy_validation(self, data):
                nonlocal toy_validation_count
                toy_validation_count += 1
                return None

            @preprocessing_method
            def toy_preprocessing(self, data):
                nonlocal toy_preprocess_count
                toy_preprocess_count += 1
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
        min_max_scale_target_data.assert_called_once()
        max_abs_scale_channel_data.assert_called_once()

        call_arg_0, call_arg_1 = min_max_scale_target_data.call_args_list[0][0]
        assert call_arg_0 is instance
        pd.testing.assert_frame_equal(call_arg_1, toy_df)
        call_arg_0, call_arg_1 = max_abs_scale_channel_data.call_args_list[0][0]
        assert call_arg_0 is instance
        pd.testing.assert_frame_equal(call_arg_1, toy_df)

        assert toy_validation_count == 1
        assert toy_preprocess_count == 1
        assert build_model_count == 1

    def test_validate_target(self, toy_mmm):
        with pytest.raises(ValueError, match="target y not in data_df"):
            toy_mmm.validate_target(toy_df.drop(columns=["y"]))

    def test_validate_date_col(self, toy_mmm):
        with pytest.raises(ValueError, match="date_col date not in data_df"):
            toy_mmm.validate_date_col(toy_df.drop(columns=["date"]))
        with pytest.raises(ValueError, match="date_col date has repeated values"):
            toy_mmm.validate_date_col(
                pd.concat([toy_df, toy_df], ignore_index=True, axis=0)
            )

    def test_channel_columns(self, toy_mmm_class):
        global toy_df
        with pytest.raises(ValueError, match="channel_columns must be a list or tuple"):
            toy_mmm_class(
                data_df=toy_df,
                target_column="y",
                date_column="date",
                channel_columns={},
            )
        with pytest.raises(ValueError, match="channel_columns must not be empty"):
            toy_mmm_class(
                data_df=toy_df,
                target_column="y",
                date_column="date",
                channel_columns=[],
            )
        with pytest.raises(
            ValueError,
            match="channel_columns \['out_of_columns'\] not in data_df",  # noqa: W605
        ):
            toy_mmm_class(
                data_df=toy_df,
                target_column="y",
                date_column="date",
                channel_columns=["out_of_columns"],
            )
        with pytest.raises(
            ValueError,
            match="channel_columns \['channel_1', 'channel_1'\] contains duplicates",  # noqa: E501, W605
        ):
            toy_mmm_class(
                data_df=toy_df,
                target_column="y",
                date_column="date",
                channel_columns=["channel_1", "channel_1"],
            )
        with pytest.raises(
            ValueError,
            match="channel_columns \['channel_1'\] contains negative values",  # noqa: E501, W605
        ):
            new_toy_df = toy_df.copy()
            new_toy_df["channel_1"] -= 1e4
            toy_mmm_class(
                data_df=new_toy_df,
                target_column="y",
                date_column="date",
                channel_columns=["channel_1"],
            )

    def test_preprocessing(self, toy_mmm):
        assert toy_mmm.preprocessed_data["y"].min() == 0
        assert toy_mmm.preprocessed_data["y"].max() == 1
        assert np.all(toy_mmm.preprocessed_data[toy_mmm.channel_columns].max() == 1)
