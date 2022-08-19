from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from pymmmc.model import ContinuousDataContainer, DataContainer


class MyDummyDataContainer(DataContainer):
    def get_preprocessed_data(self) -> pd.DataFrame:
        return self.raw_data.copy()


@pytest.fixture
def synthetic_data() -> pd.DataFrame:
    return pd.read_csv("pymmmc/tests/fixtures/synthetic_data.csv", parse_dates=["date"])


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

    @pytest.mark.parametrize("scaler", [MinMaxScaler(), StandardScaler()])
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


class TestModels:
    def test_dummy(self) -> None:
        pass
