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
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pymc_marketing.mmm.experiment import (
    _CAUSALPY_IMPORT_ERROR_MSG,
    ExperimentResult,
    ExperimentType,
    _import_causalpy,
    run_experiment,
)

# ---------------------------------------------------------------------------
# ExperimentType enum tests
# ---------------------------------------------------------------------------


class TestExperimentType:
    def test_enum_values(self) -> None:
        assert ExperimentType.ITS.value == "its"
        assert ExperimentType.SC.value == "sc"
        assert ExperimentType.DID.value == "did"
        assert ExperimentType.RD.value == "rd"

    def test_from_string(self) -> None:
        assert ExperimentType("its") is ExperimentType.ITS
        assert ExperimentType("sc") is ExperimentType.SC
        assert ExperimentType("did") is ExperimentType.DID
        assert ExperimentType("rd") is ExperimentType.RD

    def test_invalid_string_raises(self) -> None:
        with pytest.raises(ValueError, match="'invalid'"):
            ExperimentType("invalid")

    @patch("pymc_marketing.mmm.experiment._import_causalpy")
    def test_get_experiment_class_its(self, mock_import: MagicMock) -> None:
        mock_cp = MagicMock()
        mock_import.return_value = mock_cp
        cls = ExperimentType.ITS.get_experiment_class()
        assert cls is mock_cp.InterruptedTimeSeries

    @patch("pymc_marketing.mmm.experiment._import_causalpy")
    def test_get_experiment_class_sc(self, mock_import: MagicMock) -> None:
        mock_cp = MagicMock()
        mock_import.return_value = mock_cp
        cls = ExperimentType.SC.get_experiment_class()
        assert cls is mock_cp.SyntheticControl

    @patch("pymc_marketing.mmm.experiment._import_causalpy")
    def test_get_experiment_class_did(self, mock_import: MagicMock) -> None:
        mock_cp = MagicMock()
        mock_import.return_value = mock_cp
        cls = ExperimentType.DID.get_experiment_class()
        assert cls is mock_cp.DifferenceInDifferences

    @patch("pymc_marketing.mmm.experiment._import_causalpy")
    def test_get_experiment_class_rd(self, mock_import: MagicMock) -> None:
        mock_cp = MagicMock()
        mock_import.return_value = mock_cp
        cls = ExperimentType.RD.get_experiment_class()
        assert cls is mock_cp.RegressionDiscontinuity


# ---------------------------------------------------------------------------
# Lazy import tests
# ---------------------------------------------------------------------------


class TestImportCausalpy:
    @patch.dict("sys.modules", {"causalpy": MagicMock()})
    def test_import_succeeds_when_installed(self) -> None:
        result = _import_causalpy()
        assert result is not None

    @patch.dict("sys.modules", {"causalpy": None})
    def test_import_fails_with_helpful_message(self) -> None:
        with pytest.raises(ImportError, match="CausalPy is required"):
            _import_causalpy()


# ---------------------------------------------------------------------------
# ExperimentResult tests
# ---------------------------------------------------------------------------


def _make_mock_its_result() -> MagicMock:
    """Create a mock CausalPy ITS result with realistic attributes."""
    mock = MagicMock()

    # post_impact is an xarray DataArray with an obs_ind dimension
    impact_data = np.array([10.0, 12.0, 8.0, 11.0])
    mock.post_impact = xr.DataArray(impact_data, dims=["obs_ind"])

    return mock


def _make_mock_did_result() -> MagicMock:
    """Create a mock CausalPy DiD result with realistic attributes."""
    mock = MagicMock()

    # causal_impact is an xarray DataArray of posterior samples
    samples = np.random.default_rng(42).normal(5.0, 1.0, size=100)
    mock.causal_impact = xr.DataArray(samples, dims=["sample"])

    return mock


def _make_mock_rd_result() -> MagicMock:
    """Create a mock CausalPy RD result with realistic attributes."""
    mock = MagicMock()

    effect_table = pd.DataFrame(
        {
            "mean": [3.5],
            "median": [3.4],
            "hdi_lower": [1.74],
            "hdi_upper": [5.26],
            "p_gt_0": [0.99],
        }
    )
    mock_effect = MagicMock()
    mock_effect.table = effect_table
    mock.effect_summary.return_value = mock_effect

    return mock


class TestExperimentResult:
    def test_summary_delegates(self) -> None:
        mock_result = MagicMock()
        er = ExperimentResult(result=mock_result, experiment_type=ExperimentType.ITS)
        er.summary()
        mock_result.summary.assert_called_once()

    def test_effect_summary_delegates(self) -> None:
        mock_result = MagicMock()
        er = ExperimentResult(result=mock_result, experiment_type=ExperimentType.ITS)
        er.effect_summary(direction="increase", alpha=0.05)
        mock_result.effect_summary.assert_called_once_with(
            direction="increase", alpha=0.05
        )

    def test_plot_delegates(self) -> None:
        mock_result = MagicMock()
        mock_result.plot.return_value = ("fig", "ax")
        er = ExperimentResult(result=mock_result, experiment_type=ExperimentType.ITS)
        fig, ax = er.plot()
        mock_result.plot.assert_called_once()
        assert fig == "fig"
        assert ax == "ax"

    def test_idata_property(self) -> None:
        mock_result = MagicMock()
        mock_result.idata = "mock_idata"
        er = ExperimentResult(result=mock_result, experiment_type=ExperimentType.ITS)
        assert er.idata == "mock_idata"

    # ---- _get_causal_impact tests ----

    def test_get_causal_impact_its(self) -> None:
        mock_result = _make_mock_its_result()
        er = ExperimentResult(result=mock_result, experiment_type=ExperimentType.ITS)
        mean, std = er._get_causal_impact()

        expected_total = np.array([10.0, 12.0, 8.0, 11.0]).sum()
        assert mean == pytest.approx(expected_total)
        assert std == pytest.approx(0.0, abs=1e-10)

    def test_get_causal_impact_sc(self) -> None:
        mock_result = _make_mock_its_result()
        er = ExperimentResult(result=mock_result, experiment_type=ExperimentType.SC)
        mean, _std = er._get_causal_impact()

        expected_total = np.array([10.0, 12.0, 8.0, 11.0]).sum()
        assert mean == pytest.approx(expected_total)

    def test_get_causal_impact_did(self) -> None:
        mock_result = _make_mock_did_result()
        er = ExperimentResult(result=mock_result, experiment_type=ExperimentType.DID)
        mean, std = er._get_causal_impact()

        rng = np.random.default_rng(42)
        expected_samples = rng.normal(5.0, 1.0, size=100)
        assert mean == pytest.approx(float(expected_samples.mean()), rel=1e-5)
        assert std == pytest.approx(float(expected_samples.std()), rel=1e-5)

    def test_get_causal_impact_rd(self) -> None:
        mock_result = _make_mock_rd_result()
        er = ExperimentResult(result=mock_result, experiment_type=ExperimentType.RD)
        mean, std = er._get_causal_impact()

        assert mean == pytest.approx(3.5)
        # std ~ (5.26 - 1.74) / (2 * 1.88) = 3.52 / 3.76 ~ 0.936
        assert std == pytest.approx((5.26 - 1.74) / (2 * 1.88), rel=1e-3)

    # ---- to_lift_test tests ----

    def test_to_lift_test_basic(self) -> None:
        mock_result = _make_mock_its_result()
        er = ExperimentResult(result=mock_result, experiment_type=ExperimentType.ITS)

        df = er.to_lift_test(channel="tv", x=1000.0, delta_x=200.0)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert set(df.columns) == {"channel", "x", "delta_x", "delta_y", "sigma"}
        assert df["channel"].iloc[0] == "tv"
        assert df["x"].iloc[0] == 1000.0
        assert df["delta_x"].iloc[0] == 200.0
        assert df["delta_y"].iloc[0] == pytest.approx(41.0)

    def test_to_lift_test_with_dims(self) -> None:
        mock_result = _make_mock_its_result()
        er = ExperimentResult(result=mock_result, experiment_type=ExperimentType.ITS)

        df = er.to_lift_test(
            channel="tv", x=1000.0, delta_x=200.0, geo="US", country="USA"
        )

        assert "geo" in df.columns
        assert "country" in df.columns
        assert df["geo"].iloc[0] == "US"
        assert df["country"].iloc[0] == "USA"


# ---------------------------------------------------------------------------
# run_experiment tests
# ---------------------------------------------------------------------------


class TestRunExperiment:
    @patch("pymc_marketing.mmm.experiment._import_causalpy")
    def test_run_experiment_its(self, mock_import: MagicMock) -> None:
        mock_cp = MagicMock()
        mock_import.return_value = mock_cp

        mock_its_instance = _make_mock_its_result()
        mock_cp.InterruptedTimeSeries.return_value = mock_its_instance

        df = pd.DataFrame({"y": [1, 2, 3], "t": [1, 2, 3]})
        result = run_experiment(
            experiment_type="its",
            data=df,
            treatment_time=2,
            formula="y ~ 1 + t",
        )

        assert isinstance(result, ExperimentResult)
        assert result.experiment_type is ExperimentType.ITS
        mock_cp.InterruptedTimeSeries.assert_called_once_with(
            data=df,
            treatment_time=2,
            formula="y ~ 1 + t",
        )

    @patch("pymc_marketing.mmm.experiment._import_causalpy")
    def test_run_experiment_sc(self, mock_import: MagicMock) -> None:
        mock_cp = MagicMock()
        mock_import.return_value = mock_cp

        mock_sc_instance = _make_mock_its_result()
        mock_cp.SyntheticControl.return_value = mock_sc_instance

        df = pd.DataFrame({"actual": [1, 2, 3], "a": [1, 2, 3]})
        result = run_experiment(
            experiment_type="sc",
            data=df,
            treatment_time=2,
            formula="actual ~ 0 + a",
        )

        assert isinstance(result, ExperimentResult)
        assert result.experiment_type is ExperimentType.SC

    @patch("pymc_marketing.mmm.experiment._import_causalpy")
    def test_run_experiment_with_enum(self, mock_import: MagicMock) -> None:
        mock_cp = MagicMock()
        mock_import.return_value = mock_cp

        mock_did_instance = _make_mock_did_result()
        mock_cp.DifferenceInDifferences.return_value = mock_did_instance

        df = pd.DataFrame({"y": [1, 2, 3]})
        result = run_experiment(
            experiment_type=ExperimentType.DID,
            data=df,
            formula="y ~ 1 + group*post_treatment",
        )

        assert isinstance(result, ExperimentResult)
        assert result.experiment_type is ExperimentType.DID

    @patch("pymc_marketing.mmm.experiment._import_causalpy")
    def test_run_experiment_invalid_type(self, mock_import: MagicMock) -> None:
        mock_cp = MagicMock()
        mock_import.return_value = mock_cp

        df = pd.DataFrame({"y": [1, 2, 3]})
        with pytest.raises(ValueError, match="Invalid experiment type"):
            run_experiment(experiment_type="invalid", data=df)

    @patch("pymc_marketing.mmm.experiment._import_causalpy")
    def test_run_experiment_case_insensitive(self, mock_import: MagicMock) -> None:
        mock_cp = MagicMock()
        mock_import.return_value = mock_cp

        mock_its_instance = _make_mock_its_result()
        mock_cp.InterruptedTimeSeries.return_value = mock_its_instance

        df = pd.DataFrame({"y": [1, 2, 3]})
        result = run_experiment(experiment_type="ITS", data=df, treatment_time=2)

        assert result.experiment_type is ExperimentType.ITS


# ---------------------------------------------------------------------------
# Import error message test
# ---------------------------------------------------------------------------


def test_import_error_message_content() -> None:
    assert "pip install causalpy" in _CAUSALPY_IMPORT_ERROR_MSG
    assert "pymc-marketing[experiment]" in _CAUSALPY_IMPORT_ERROR_MSG
