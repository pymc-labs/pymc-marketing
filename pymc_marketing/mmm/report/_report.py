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
"""Public MMM report interface."""

from __future__ import annotations

from functools import cached_property
from typing import Any, Literal

import pandas as pd

from pymc_marketing.mmm.report._contracts import ReportConfig, ReportData
from pymc_marketing.mmm.report._exporters import export_excel, export_html, export_pdf
from pymc_marketing.mmm.report._sections import build_report_data


class MMMReport:
    """Generate standardized MMM reports across multiple output formats."""

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
        """Computed report payload with metadata, tables, and figures."""
        return build_report_data(self.mmm, self.config)

    def to_dataframe(self) -> dict[str, pd.DataFrame]:
        """Return all report tables in a deterministic dictionary."""
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
        """Export report to HTML and optionally write it to disk."""
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
        """Export report to PDF with configurable engine selection."""
        export_pdf(self.report_data, file_name=file_name, engine=engine)

    def to_excel(self, file_name: str) -> None:
        """Export report tables to Excel."""
        export_excel(self.report_data, file_name=file_name)
