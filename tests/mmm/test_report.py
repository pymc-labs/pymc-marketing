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
"""Tests for MMM report module."""

from __future__ import annotations

from dataclasses import dataclass

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import xarray as xr

from pymc_marketing.mmm.report import MMMReport, ReportConfig
from pymc_marketing.mmm.report._notebook import build_notebook
from pymc_marketing.mmm.report._sections import PARITY_MATRIX

_CHANNELS = ["TV", "Search"]
_DATE_RANGE = pd.date_range("2024-01-01", periods=4, freq="D")


@dataclass
class _FakeSensitivity:
    model: _FakeMMM

    def run_sweep(self, *args, **kwargs):
        data = xr.DataArray(
            np.ones((2, 5, 4, 2)),
            dims=("chain", "draw", "date", "channel"),
            coords={"date": _DATE_RANGE, "channel": _CHANNELS},
        )
        self.model.idata.add_groups(
            {"sensitivity_analysis": xr.Dataset({"channel_contribution": data})}
        )


class _FakeIncrementality:
    def contribution_over_spend(self, **kwargs):
        return xr.DataArray(
            np.random.uniform(0.5, 2.0, size=(2, 5, 2)),
            dims=("chain", "draw", "channel"),
            coords={"channel": _CHANNELS},
        )


class _FakeData:
    def aggregate_time(self, frequency):
        return self

    def get_elementwise_roas(self, original_scale=True):
        return xr.DataArray(
            np.random.uniform(0.5, 2.0, size=(2, 5, 2)),
            dims=("chain", "draw", "channel"),
            coords={"channel": _CHANNELS},
        )


class _FakeSummary:
    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=4, freq="D"),
                "channel": ["TV", "Search", "TV", "Search"],
                "mean": [1.0, 2.0, 1.5, 1.8],
                "median": [1.0, 2.0, 1.4, 1.7],
                "abs_error_94_lower": [0.8, 1.7, 1.2, 1.4],
                "abs_error_94_upper": [1.2, 2.3, 1.8, 2.1],
            }
        )

    def posterior_predictive(self, **kwargs):
        return self._frame().drop(columns=["channel"])

    def total_contribution(self, **kwargs):
        out = self._frame().drop(columns=["channel"])
        out["component"] = ["media", "baseline", "media", "baseline"]
        return out

    def contributions(self, **kwargs):
        return self._frame()

    def roas(self, **kwargs):
        return self._frame()

    def saturation_curves(self, **kwargs):
        out = self._frame()
        out["x"] = [0.1, 0.2, 0.3, 0.4]
        return out


class _FakePlot:
    @staticmethod
    def _fig():
        fig, ax = plt.subplots()
        ax.plot([1, 2], [1, 3])
        return fig, np.array([[ax]])

    def posterior_predictive(self, **kwargs):
        return self._fig()

    def waterfall_components_decomposition(self, **kwargs):
        return self._fig()

    def contributions_over_time(self, **kwargs):
        return self._fig()

    def channel_contribution_share_hdi(self, **kwargs):
        return self._fig()

    def saturation_scatterplot(self, **kwargs):
        return self._fig()

    def sensitivity_analysis(self, **kwargs):
        return self._fig()


class _FakePlotInteractive:
    @staticmethod
    def _fig():
        return go.Figure(data=[go.Bar(x=["TV", "Search"], y=[1, 2])])

    def posterior_predictive(self, **kwargs):
        return self._fig()

    def contributions(self, **kwargs):
        return self._fig()

    def roas(self, **kwargs):
        return self._fig()

    def saturation_curves(self, **kwargs):
        return self._fig()


class _FakeMMM:
    def __init__(self):
        posterior = xr.Dataset(
            {
                "intercept_contribution": xr.DataArray(
                    np.random.normal(size=(2, 5)),
                    dims=("chain", "draw"),
                ),
                "y_sigma": xr.DataArray(
                    np.abs(np.random.normal(size=(2, 5))),
                    dims=("chain", "draw"),
                ),
                "y_original_scale": xr.DataArray(
                    np.random.normal(size=(2, 5, 4)),
                    dims=("chain", "draw", "date"),
                    coords={"date": _DATE_RANGE},
                ),
                "channel_contribution_original_scale": xr.DataArray(
                    np.random.normal(size=(2, 5, 4, 2)),
                    dims=("chain", "draw", "date", "channel"),
                    coords={"date": _DATE_RANGE, "channel": _CHANNELS},
                ),
                "control_contribution_original_scale": xr.DataArray(
                    np.random.normal(size=(2, 5, 4)),
                    dims=("chain", "draw", "date"),
                    coords={"date": _DATE_RANGE},
                ),
                "intercept_contribution_original_scale": xr.DataArray(
                    np.random.normal(size=(2, 5, 4)),
                    dims=("chain", "draw", "date"),
                    coords={"date": _DATE_RANGE},
                ),
            }
        )
        sample_stats = xr.Dataset(
            {
                "diverging": xr.DataArray(
                    np.zeros((2, 5), dtype=int),
                    dims=("chain", "draw"),
                )
            }
        )
        self.idata = az.InferenceData(posterior=posterior, sample_stats=sample_stats)
        self.fit_result = posterior
        self.date_column = "date"
        self.target_column = "y"
        self.channel_columns = _CHANNELS
        self.control_columns = ["promo"]
        self.dims = ("geo",)
        self.adstock = object()
        self.saturation = object()
        self.X = pd.DataFrame({"date": _DATE_RANGE})
        self.summary = _FakeSummary()
        self.plot = _FakePlot()
        self.plot_interactive = _FakePlotInteractive()
        self.sensitivity = _FakeSensitivity(self)
        self.incrementality = _FakeIncrementality()
        self.data = _FakeData()


@pytest.fixture
def fake_mmm():
    return _FakeMMM()


def test_report_config_normalizes_hdi_probs():
    config = ReportConfig(hdi_probs=[0.94, 0.5, 0.94])
    assert config.hdi_probs == (0.5, 0.94)


def test_report_config_rejects_invalid_hdi_prob():
    with pytest.raises(ValueError, match="open interval"):
        ReportConfig(hdi_probs=(1.0,))


def test_parity_matrix_contains_motivational_notebooks():
    expected = {
        "docs/source/notebooks/mmm/mmm_case_study.ipynb",
        "docs/source/notebooks/mmm/mmm_example.ipynb",
        "docs/source/notebooks/mmm/plot_interactive.ipynb",
    }
    flat = {item for values in PARITY_MATRIX.values() for item in values}
    assert expected.issubset(flat)


def test_to_dataframe_contains_point_forecast_columns(fake_mmm):
    report = MMMReport(fake_mmm)
    tables = report.to_dataframe()
    assert "roas_elementwise" in tables
    assert "point_forecast" in tables["roas_elementwise"].columns
    assert "posterior_predictive" in tables


def test_build_notebook_has_header_and_report_code_cells(fake_mmm):
    report = MMMReport(fake_mmm)
    nb = build_notebook(report.report_data, include_interactive=True)
    assert len(nb.cells) > 2
    assert "MMM Report" in nb.cells[0].source
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert code_cells
    assert "report-code" in code_cells[0].metadata["tags"]


def test_to_html_writes_output(fake_mmm, tmp_path):
    pytest.importorskip("nbconvert")
    report = MMMReport(fake_mmm)
    path = tmp_path / "report.html"
    html = report.to_html(file_name=str(path))
    assert path.exists()
    assert "MMM Report" in html


def test_diagnostics_has_no_trace_plot(fake_mmm):
    report = MMMReport(fake_mmm)
    diagnostics = report.report_data.sections["diagnostics"]
    assert not diagnostics.static_figures


def test_roas_section_has_forest_plots(fake_mmm):
    report = MMMReport(fake_mmm)
    roas = report.report_data.sections["roas"]
    assert "roas_forest_elementwise" in roas.static_figures
    assert "roas_forest_incremental" in roas.static_figures


def test_component_section_has_waterfall_and_original_scale(fake_mmm):
    report = MMMReport(fake_mmm)
    components = report.report_data.sections["component_contributions"]
    assert "waterfall_components_decomposition" in components.static_figures
    assert "contributions_original_scale" in components.static_figures


def test_static_figures_have_titles(fake_mmm):
    report = MMMReport(fake_mmm)
    import matplotlib.figure

    for section in report.report_data.sections.values():
        for name, fig in section.static_figures.items():
            if isinstance(fig, matplotlib.figure.Figure):
                suptitle = fig._suptitle
                assert suptitle is not None and suptitle.get_text(), (
                    f"Figure '{name}' in section '{section.title}' has no suptitle"
                )


def test_to_excel_writes_output(fake_mmm, tmp_path):
    pytest.importorskip("openpyxl")
    report = MMMReport(fake_mmm)
    path = tmp_path / "report.xlsx"
    report.to_excel(file_name=str(path))
    assert path.exists()
