#   Copyright 2022 - 2025 The PyMC Labs Developers
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
import json
import re
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pymc as pm
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FunctionTransformer, Pipeline

from pymc_marketing.mmm.base import BaseValidateMMM as MMM
from pymc_marketing.mmm.preprocessing import (
    preprocessing_method_X,
    preprocessing_method_y,
)
from pymc_marketing.mmm.validating import (
    validation_method_X,
    validation_method_y,
)

seed: int = sum(map(ord, "pymc_marketing"))
rng: np.random.Generator = np.random.default_rng(seed=seed)


@pytest.fixture(scope="module")
def toy_X() -> pd.DataFrame:
    date_data: pd.DatetimeIndex = pd.date_range(
        start="2019-06-01", end="2021-12-31", freq="W-MON"
    )
    n: int = date_data.size

    return pd.DataFrame(
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


@pytest.fixture(scope="module")
def toy_y(toy_X) -> pd.Series:
    return pd.Series(rng.integers(low=0, high=100, size=toy_X.shape[0]), name="y")


@pytest.fixture(scope="module")
def toy_mmm(request, toy_X, toy_y):
    channel_columns = request.param["channel_columns"]

    class ToyMMM(MMM):
        def __init__(
            self,
            date_column: str,
            channel_columns,
            model_config=None,
            sampler_config=None,
        ) -> None:
            super().__init__(
                date_column=date_column,
                channel_columns=channel_columns,
                model_config=model_config,
                sampler_config=sampler_config,
            )

            self.X = None
            self.y = None
            self.preprocessed_data = {"X": None, "y": None}

        def create_idata_attrs(self) -> dict[str, str]:
            attrs = super().create_idata_attrs()
            attrs["date_column"] = self.data_column
            attrs["channel_columns"] = self.channel_columns

            return attrs

        def build_model(*args, **kwargs):
            pass

        def _generate_and_preprocess_model_data(self, X, y):
            self.validate("X", X)
            self.validate("y", y)
            self.preprocessed_data["X"] = self.preprocess("X", X)
            self.preprocessed_data["y"] = self.preprocess("y", y)
            self.X = X
            self.y = y

        @property
        def default_model_config(self):
            return {}

        @property
        def default_sampler_config(self):
            return {}

        @property
        def output_var(self):
            pass

        def _data_setter(self, X, y=None):
            pass

        def _serializable_model_config(self):
            pass

        @validation_method_X
        def toy_validation_X(self, data):
            pd.testing.assert_frame_equal(data, toy_X)
            return None

        @validation_method_y
        def toy_validation_y(self, data):
            pd.testing.assert_series_equal(data, toy_y)
            return None

        @preprocessing_method_X
        def toy_preprocessing_X(self, data):
            pd.testing.assert_frame_equal(data, toy_X)
            return data

        @preprocessing_method_y
        def toy_preprocessing_y(self, data):
            pd.testing.assert_series_equal(data, toy_y)
            return data

    return ToyMMM(
        date_column="date",
        channel_columns=channel_columns,
    )


class TestMMM:
    @patch("pymc_marketing.mmm.validating.ValidateTargetColumn.validate_target")
    @patch("pymc_marketing.mmm.validating.ValidateDateColumn.validate_date_col")
    @patch(
        "pymc_marketing.mmm.validating.ValidateChannelColumns.validate_channel_columns"
    )
    @pytest.mark.parametrize(
        "toy_mmm",
        [
            {"channel_columns": ["channel_1"]},
            {"channel_columns": ["channel_1", "channel_2"]},
        ],
        indirect=True,
    )
    def test_init(
        self,
        validate_channel_columns,
        validate_date_col,
        validate_target,
        toy_mmm,
        toy_X,
        toy_y,
    ) -> None:
        validate_channel_columns.configure_mock(_tags={"validation_X": True})
        validate_date_col.configure_mock(_tags={"validation_X": True})
        validate_target.configure_mock(_tags={"validation_y": True})
        toy_mmm._generate_and_preprocess_model_data(toy_X, toy_y)
        pd.testing.assert_frame_equal(toy_mmm.X, toy_X)
        pd.testing.assert_frame_equal(toy_mmm.preprocessed_data["X"], toy_X)
        pd.testing.assert_series_equal(toy_mmm.y, toy_y)
        pd.testing.assert_series_equal(toy_mmm.preprocessed_data["y"], toy_y)
        validate_target.assert_called_once_with(toy_mmm, toy_y)
        validate_date_col.assert_called_once_with(toy_mmm, toy_X)
        validate_channel_columns.assert_called_once_with(toy_mmm, toy_X)


@pytest.fixture(scope="module")
def test_mmm():
    class ToyMMM(MMM):
        mock_method1 = Mock()
        mock_method2 = Mock()
        validation_methods = [(mock_method1,), (mock_method2,)]

        def __init__(
            self,
            date_column: str,
            channel_columns,
            model_config=None,
            sampler_config=None,
        ) -> None:
            super().__init__(
                date_column=date_column,
                channel_columns=channel_columns,
                model_config=model_config,
                sampler_config=sampler_config,
            )

            self.X = None
            self.y = None
            self.preprocessed_data = {"X": None, "y": None}

        def create_idata_attrs(self) -> dict[str, str]:
            attrs = super().create_idata_attrs()
            attrs["date_column"] = self.date_column
            attrs["channel_columns"] = json.dumps(self.channel_columns)

            return attrs

        def build_model(self, toy_X, *args, **kwargs):
            with pm.Model() as self.model:
                intercept = pm.Normal("intercept", mu=0, sigma=1)
                sigma = pm.HalfNormal("sigma", sigma=1)
                slope = pm.Normal("slope", mu=0, sigma=1)
                mu = intercept + slope
                pm.Normal("y", mu=mu, sigma=sigma)

        def _generate_and_preprocess_model_data(self, toy_X, toy_y):
            self.validate("X", toy_X)
            self.validate("y", toy_y)
            self.preprocessed_data["X"] = self.preprocess("X", toy_X)
            self.preprocessed_data["y"] = self.preprocess("y", toy_y)
            self.X = toy_X
            self.y = toy_y

        @property
        def default_model_config(self):
            return {"model": "model"}

        @property
        def default_sampler_config(self):
            return {"draws": 1000, "tune": 1000}

        @property
        def output_var(self):
            return "y"

        def _data_setter(self, X, y=None):
            pass

        @property
        def _serializable_model_config(self):
            return {"model": "model"}

    return ToyMMM(date_column="date", channel_columns=["channel_1"])


class MyScaler(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1):
        self.factor = factor

    def fit(self, X, y=None):
        return self  # Nothing happens in fit, so just return self

    def transform(self, X):
        return X * self.factor


def test_validate_and_preprocess(toy_X, toy_y, test_mmm):
    test_mmm.validate("X", toy_X)
    test_mmm.mock_method1.assert_called_once_with(test_mmm, toy_X)

    test_mmm.validate("y", toy_y)
    test_mmm.mock_method2.assert_called_once_with(test_mmm, toy_y)

    with pytest.raises(ValueError, match="Target must be either 'X' or 'y'"):
        test_mmm.validate("invalid", toy_X)
    with pytest.raises(ValueError, match="Target must be either 'X' or 'y'"):
        test_mmm.preprocess("invalid", toy_X)


def test_get_target_transformer_when_set(test_mmm):
    # Arrange
    mmm = test_mmm
    expected_transformer = Pipeline(steps=[("your_step", MyScaler(10))])
    mmm.target_transformer = expected_transformer

    # Act
    actual_transformer = mmm.get_target_transformer()

    # Assert
    assert actual_transformer == expected_transformer


def test_get_target_transformer_when_not_set(test_mmm):
    # Arrange
    mmm = test_mmm
    if hasattr(mmm, "target_transformer"):
        del mmm.target_transformer
    # Act
    actual_transformer = mmm.get_target_transformer()

    # Assert
    assert isinstance(actual_transformer, Pipeline)
    assert isinstance(actual_transformer.named_steps["scaler"], FunctionTransformer)


def test_calling_prior_predictive_before_fit_raises_error(test_mmm, toy_X, toy_y):
    # Arrange
    test_mmm.idata = None
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "The model hasn't been sampled yet, call .sample_prior_predictive() first"
        ),
    ):
        test_mmm.prior_predictive


def test_calling_fit_result_before_fit_raises_error(
    test_mmm,
    toy_X,
    toy_y,
    mock_pymc_sample,
):
    # Arrange
    test_mmm.idata = None
    with pytest.raises(
        RuntimeError,
        match=re.escape("The model hasn't been fit yet, call .fit() first"),
    ):
        test_mmm.fit_result
    test_mmm.fit(toy_X, toy_y)
    test_mmm.fit_result
    assert test_mmm.idata is not None
    assert "posterior" in test_mmm.idata


def test_calling_prior_before_sample_prior_predictive_raises_error(
    test_mmm, toy_X, toy_y
):
    # Arrange
    test_mmm.idata = None
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "The model hasn't been sampled yet, call .sample_prior_predictive() first",
        ),
    ):
        test_mmm.prior


def test_plot_prior_predictive_no_fitted(test_mmm) -> None:
    with pytest.raises(
        RuntimeError,
        match="Make sure the model has been fitted and the prior_predictive has been sampled!",
    ):
        test_mmm.plot_prior_predictive()


def test_plot_posterior_predictive_no_fitted(test_mmm) -> None:
    with pytest.raises(
        RuntimeError,
        match="Make sure the model has been fitted and the posterior_predictive has been sampled!",
    ):
        test_mmm.plot_posterior_predictive()


def test_get_errors_raises_not_fitted(test_mmm) -> None:
    with pytest.raises(
        RuntimeError,
        match="Make sure the model has been fitted and the posterior_predictive has been sampled!",
    ):
        test_mmm.get_errors()
