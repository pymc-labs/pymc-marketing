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
from pymc_marketing.mmm.report._exporters import _make_sheet_name
from pymc_marketing.mmm.report._notebook import build_notebook

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


# --- ReportConfig validation tests ---


def test_report_config_normalizes_hdi_probs():
    config = ReportConfig(hdi_probs=[0.94, 0.5, 0.94])
    assert config.hdi_probs == (0.5, 0.94)


def test_report_config_rejects_invalid_hdi_prob():
    with pytest.raises(ValueError, match="open interval"):
        ReportConfig(hdi_probs=(1.0,))


# --- Report data / section tests ---


def test_to_dataframe_contains_point_forecast_columns(fake_mmm):
    report = MMMReport(fake_mmm)
    tables = report.to_dataframe()
    assert "posterior_predictive" in tables
    assert "point_forecast" in tables["posterior_predictive"].columns
    assert "roas_elementwise" in tables
    assert "mean" in tables["roas_elementwise"].columns


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


# --- Excel export tests ---


def test_to_excel_writes_output(fake_mmm, tmp_path):
    pytest.importorskip("openpyxl")
    report = MMMReport(fake_mmm)
    path = tmp_path / "report.xlsx"
    report.to_excel(file_name=str(path))
    assert path.exists()


def test_to_excel_one_sheet_per_dataframe(fake_mmm, tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    report = MMMReport(fake_mmm)
    path = tmp_path / "report.xlsx"
    report.to_excel(file_name=str(path))

    wb = openpyxl.load_workbook(str(path))
    total_dfs = sum(
        len(section.dataframes) for section in report.report_data.sections.values()
    )
    assert len(wb.sheetnames) == 1 + total_dfs
    assert wb.sheetnames[0] == "Cover"


# --- _make_sheet_name unit tests ---


def test_make_sheet_name_passthrough():
    used: set[str] = set()
    assert _make_sheet_name("model_overview", used) == "model_overview"
    assert "model_overview" in used


def test_make_sheet_name_truncates_long_names():
    used: set[str] = set()
    long_name = "a" * 50
    result = _make_sheet_name(long_name, used)
    assert len(result) <= 31
    assert result in used


def test_make_sheet_name_collision_suffix():
    used: set[str] = {"my_table"}
    result = _make_sheet_name("my_table", used)
    assert result == "my_table_2"
    assert "my_table_2" in used


def test_make_sheet_name_sanitizes_illegal_chars():
    used: set[str] = set()
    result = _make_sheet_name("data[0]:summary*", used)
    assert "[" not in result
    assert "]" not in result
    assert ":" not in result
    assert "*" not in result
    assert result == "data_0__summary_"


# --- display_dataframes / slim HTML tests ---


def test_notebook_excludes_time_series_tables(fake_mmm):
    """Time-series sections should render no HTML tables in the notebook."""
    report = MMMReport(fake_mmm)
    nb = build_notebook(report.report_data, include_interactive=False)

    sections_with_tables: set[str] = set()
    current_section: str | None = None
    for cell in nb.cells:
        if cell.cell_type == "markdown" and cell.source.startswith("## "):
            current_section = cell.source.split("\n")[0].lstrip("# ").strip()
        if cell.cell_type == "code" and current_section is not None:
            for out in cell.outputs:
                if "text/html" in out.get("data", {}):
                    sections_with_tables.add(current_section)

    time_series_sections = {
        "Posterior Predictive Fit",
        "Component Contributions",
        "Saturation Curves",
    }
    assert not sections_with_tables & time_series_sections, (
        f"Time-series sections should have no HTML tables: "
        f"{sections_with_tables & time_series_sections}"
    )
    assert "Model Overview" in sections_with_tables
    assert "Diagnostics" in sections_with_tables


def test_roas_dataframe_is_aggregated(fake_mmm):
    """ROAS DataFrames should have one row per channel with az.summary cols."""
    report = MMMReport(fake_mmm)
    roas_section = report.report_data.sections["roas"]

    for key in ("roas_elementwise", "roas_incremental"):
        df = roas_section.dataframes[key]
        assert len(df) == len(_CHANNELS)
        assert "mean" in df.columns
        assert "sd" in df.columns
        assert "date" not in df.columns

    assert roas_section.display_dataframes.keys() == roas_section.dataframes.keys()
