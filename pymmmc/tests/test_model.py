import re
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pymc as pm
import pytest
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from pymmmc.model import (
    AdstockGeometricLogistiSaturation,
    BaseMMModel,
    CategoryDataContainer,
    ContinuousDataContainer,
    DataContainer,
    MediaDataContainer,
)


class MyDummyDataContainer(DataContainer):
    def get_preprocessed_data(self) -> pd.DataFrame:
        return self.raw_data.copy()


class MyDummyMMM(BaseMMModel):
    def _build_model(self) -> pm.Model:
        return pm.Model()


@pytest.fixture
def synthetic_data() -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(
        "pymmmc/tests/fixtures/synthetic_data.csv",
        parse_dates=["date"],
        index_col="index",
    )
    df["cat1"] = pd.Categorical(df["cat1"])
    df["cat2"] = pd.Categorical(df["cat2"])
    return df


@pytest.fixture
def synthetic_data_with_missing_values(synthetic_data: pd.DataFrame) -> pd.DataFrame:
    df = synthetic_data.copy()
    df.loc[0, "z"] = np.nan
    return df


class TestBaseDataContainer:
    def test_missing_values_input_data(
        self, synthetic_data_with_missing_values: pd.DataFrame
    ) -> None:
        with pytest.raises(
            expected_exception=ValueError, match="raw_data must not contain NaN"
        ):
            MyDummyDataContainer(raw_data=synthetic_data_with_missing_values)

    def test_empty_input_data(self) -> None:
        with pytest.raises(
            expected_exception=ValueError, match="raw_data must not be empty"
        ):
            MyDummyDataContainer(raw_data=pd.DataFrame())

    def test_bad_data_input_type(self) -> None:
        with pytest.raises(
            expected_exception=TypeError,
            match="raw_data must be a pandas.Series or pandas.DataFrame",
        ):
            MyDummyDataContainer(raw_data=1)

    def test_no_reassign_data(self, synthetic_data: pd.DataFrame) -> None:
        data_container = MyDummyDataContainer(raw_data=synthetic_data)
        with pytest.raises(
            expected_exception=FrozenInstanceError,
            match="cannot assign to field 'raw_data'",
        ):
            data_container.raw_data = pd.DataFrame()


class TestContinuousDataContainer:
    def test_no_transformation(self, synthetic_data: pd.DataFrame) -> None:
        raw_data = synthetic_data.filter(["z", "trend"])
        dc = ContinuousDataContainer(raw_data=raw_data)
        pd.testing.assert_frame_equal(left=raw_data, right=dc.get_preprocessed_data())

    @pytest.mark.parametrize(
        argnames="scaler",
        argvalues=[MinMaxScaler(), StandardScaler()],
        ids=["MinMaxScaler", "StandardScaler"],
    )
    def test__scaler(self, synthetic_data: pd.DataFrame, scaler: BaseEstimator) -> None:
        raw_data = synthetic_data.filter(["z", "trend"])
        dc = ContinuousDataContainer(raw_data=raw_data, transformer=scaler)
        processed_data: pd.DataFrame = dc.get_preprocessed_data()
        assert processed_data.shape == raw_data.shape
        assert processed_data.columns.equals(raw_data.columns)
        assert processed_data.index.equals(raw_data.index)
        assert not processed_data.isna().any().any()
        if scaler.__class__() == StandardScaler():
            np.testing.assert_allclose(processed_data.mean(axis=0), 0.0, atol=1e-5)
            np.testing.assert_allclose(processed_data.std(axis=0), 1.0, atol=1e-2)
        if scaler.__class__() == MinMaxScaler():
            np.testing.assert_allclose(processed_data.max(axis=0), 1.0, atol=1e-3)
            np.testing.assert_allclose(processed_data.min(axis=0), 0.0, atol=1e-3)


class TestMediaDataContainer:
    def test_no_transformation(self, synthetic_data: pd.DataFrame) -> None:
        raw_data = synthetic_data.filter(["z", "trend"])
        dc = ContinuousDataContainer(raw_data=raw_data)
        pd.testing.assert_frame_equal(left=raw_data, right=dc.get_preprocessed_data())

    def test_minmax_scaler(self, synthetic_data: pd.DataFrame) -> None:
        raw_data = synthetic_data.filter(["z", "trend"])
        dc = MediaDataContainer(raw_data=raw_data, transformer=MinMaxScaler())
        processed_data: pd.DataFrame = dc.get_preprocessed_data()
        np.testing.assert_allclose(processed_data.max(axis=0), 1.0, atol=1e-3)
        np.testing.assert_allclose(processed_data.min(axis=0), 0.0, atol=1e-3)

    def test_bad_negative_data(self) -> None:
        raw_data = pd.DataFrame({"z": [-1, -2, -3]})
        with pytest.raises(
            expected_exception=ValueError,
            match="media must not contain negative values",
        ):
            MediaDataContainer(raw_data=raw_data)


class TestCategoryDataContainer:
    @pytest.mark.parametrize(
        argnames="col_name",
        argvalues=["cat1", "cat2"],
        ids=["cat1", "cat2"],
    )
    def test_good_data(self, synthetic_data: pd.DataFrame, col_name: str) -> None:
        dc = CategoryDataContainer(raw_data=synthetic_data[col_name])
        assert dc.get_preprocessed_data()

    @pytest.mark.parametrize(
        argnames="col_name",
        argvalues=["cat1", "cat2"],
        ids=["cat1", "cat2"],
    )
    def test_bad_data_type_dataframe(
        self, synthetic_data: pd.DataFrame, col_name: str
    ) -> None:
        with pytest.raises(
            expected_exception=TypeError,
        ):
            CategoryDataContainer(raw_data=synthetic_data[[col_name]])

    @pytest.mark.parametrize(
        argnames="col_name",
        argvalues=["date", "z", "trend"],
        ids=["date_col", "z", "trend"],
    )
    def test_bad_data_type(self, synthetic_data: pd.DataFrame, col_name: str) -> None:
        with pytest.raises(
            expected_exception=TypeError,
        ):
            CategoryDataContainer(raw_data=synthetic_data[col_name])


class TestBaseMMModel:
    def test_good_input_data(self, synthetic_data: pd.DataFrame) -> None:
        model = MyDummyMMM(
            train_df=synthetic_data, target="y", date_col="date", media_columns=["z"]
        )
        assert model

    def test_bad_input_empty_data(self) -> None:
        with pytest.raises(
            expected_exception=ValueError,
            match="train_df must not be empty",
        ):
            MyDummyMMM(
                train_df=pd.DataFrame(),
                target="y",
                date_col="date",
                media_columns=["z"],
            )

    def test_bad_input_no_target(self, synthetic_data: pd.DataFrame) -> None:
        with pytest.raises(
            expected_exception=ValueError,
            match="target bad_target not in train_df",
        ):
            MyDummyMMM(
                train_df=synthetic_data,
                target="bad_target",
                date_col="date",
                media_columns=["z"],
            )

        with pytest.raises(
            expected_exception=ValueError,
            match="target must not be None",
        ):
            MyDummyMMM(
                train_df=synthetic_data,
                target=None,
                date_col="date",
                media_columns=["z"],
            )

    def test_bad_input_no_date_col(self, synthetic_data: pd.DataFrame) -> None:
        with pytest.raises(
            expected_exception=ValueError,
            match="date_col bad_date_col not in train_df",
        ):
            MyDummyMMM(
                train_df=synthetic_data,
                target="y",
                date_col="bad_date_col",
                media_columns=["z"],
            )

        with pytest.raises(
            expected_exception=ValueError,
            match="date_col must not be None",
        ):
            MyDummyMMM(
                train_df=synthetic_data,
                target="y",
                date_col=None,
                media_columns=["z"],
            )

    def test_bad_input_no_media_col(self, synthetic_data: pd.DataFrame) -> None:
        with pytest.raises(
            expected_exception=ValueError,
            match=re.escape("media_columns ['bad_media_col'] not in train_df"),
        ):
            MyDummyMMM(
                train_df=synthetic_data,
                target="y",
                date_col="date",
                media_columns=["bad_media_col"],
            )

        with pytest.raises(
            expected_exception=ValueError,
            match="media_columns must not be None",
        ):
            MyDummyMMM(
                train_df=synthetic_data,
                target="y",
                date_col="date",
                media_columns=None,
            )

        with pytest.raises(
            expected_exception=ValueError,
            match="media_columns must not be empty",
        ):
            MyDummyMMM(
                train_df=synthetic_data,
                target="y",
                date_col="date",
                media_columns=[],
            )


class TestAdstockGeometricLogistiSaturation:
    def test_good_input_data(self, synthetic_data: pd.DataFrame) -> None:
        model = AdstockGeometricLogistiSaturation(
            train_df=synthetic_data, target="y", date_col="date", media_columns=["z"]
        )
        assert model
