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
"""Public MMM report interface.

Exposes the high-level :class:`MMMReport` class that wraps a fitted MMM
and provides convenience methods for exporting the report to HTML, PDF,
Excel, or raw DataFrames.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, Literal

import pandas as pd

from pymc_marketing.mmm.report._contracts import ReportConfig, ReportData
from pymc_marketing.mmm.report._exporters import export_excel, export_html, export_pdf
from pymc_marketing.mmm.report._sections import build_report_data


class MMMReport:
    """Generate standardized MMM reports across multiple output formats.

    The report data (tables and figures) is computed lazily on first access
    to :attr:`report_data` and cached for subsequent exports.

    Parameters
    ----------
    mmm : MMM
        A fitted media-mix model instance.
    hdi_probs : tuple of float
        HDI probability levels for uncertainty intervals.
    point_estimate : {"mean", "median"}
        Which point estimate to highlight.
    frequency : str
        Time-aggregation frequency forwarded to summary methods.
    roas_methods : tuple of {"elementwise", "incremental"}
        ROAS computation methods to include.
    dims : dict or None
        Optional dimension filters applied to summary DataFrames.
    sensitivity_sweep_values : tuple of float or None
        Multipliers for the sensitivity-analysis sweep.
    include_interactive : bool
        Whether to generate Plotly interactive figures.
    num_samples : int or None
        Number of posterior samples for stochastic summaries.
    random_state : int, RandomState, or None
        Random state for reproducibility.

    Examples
    --------
    .. code-block:: python

        report = MMMReport(mmm, hdi_probs=(0.94,))
        report.to_html("report.html")
        report.to_excel("report.xlsx")
    """

    def __init__(
        self,
        mmm: Any,
        *,
        hdi_probs: tuple[float, ...] = (0.94,),
        point_estimate: Literal["mean", "median"] = "mean",
        frequency: str = "all_time",
        roas_methods: tuple[Literal["elementwise", "incremental"], ...] = (
            "elementwise",
            "incremental",
        ),
        dims: dict[str, str | int | list[str | int]] | None = None,
        sensitivity_sweep_values: tuple[float, ...] | None = None,
        include_interactive: bool = True,
        num_samples: int | None = None,
        random_state: Any | None = None,
    ) -> None:
        self.mmm = mmm
        self.config = ReportConfig(
            hdi_probs=hdi_probs,
            point_estimate=point_estimate,
            frequency=frequency,
            roas_methods=roas_methods,
            dims=dims,
            sensitivity_sweep_values=sensitivity_sweep_values,
            include_interactive=include_interactive,
            num_samples=num_samples,
            random_state=random_state,
        )

    @cached_property
    def report_data(self) -> ReportData:
        """Lazily computed report payload with metadata, tables, and figures.

        Returns
        -------
        ReportData
            The full report payload, computed once and cached.
        """
        return build_report_data(self.mmm, self.config)

    def to_dataframe(self) -> dict[str, pd.DataFrame]:
        """Return all report tables as a flat dictionary.

        Returns
        -------
        dict of str to pd.DataFrame
            Keys are the table names (e.g. ``"roas_elementwise"``); values
            are independent copies of the underlying DataFrames.
        """
        result: dict[str, pd.DataFrame] = {}
        for section in self.report_data.sections.values():
            for table_key, df in section.dataframes.items():
                result[table_key] = df.copy()
        return result

    def to_html(
        self,
        file_name: str | None = None,
        *,
        save_intermediate_notebook: str | None = None,
    ) -> str:
        """Export the report as an HTML string.

        Parameters
        ----------
        file_name : str or None
            If given, the HTML is also written to this path.
        save_intermediate_notebook : str or None
            If given, the intermediate ``.ipynb`` is saved to this path.

        Returns
        -------
        str
            The rendered HTML.
        """
        return export_html(
            self.report_data,
            file_name=file_name,
            save_intermediate_notebook=save_intermediate_notebook,
        )

    def to_pdf(
        self,
        file_name: str,
        *,
        engine: Literal["auto", "latex", "webpdf"] = "auto",
    ) -> None:
        """Export the report as a PDF file.

        Parameters
        ----------
        file_name : str
            Destination path for the PDF.
        engine : {"auto", "latex", "webpdf"}
            PDF rendering backend.
        """
        export_pdf(self.report_data, file_name=file_name, engine=engine)

    def to_excel(self, file_name: str) -> None:
        """Export report tables to an Excel workbook.

        Parameters
        ----------
        file_name : str
            Destination path for the ``.xlsx`` file.
        """
        export_excel(self.report_data, file_name=file_name)
