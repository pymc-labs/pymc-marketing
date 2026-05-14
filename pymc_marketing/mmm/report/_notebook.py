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
"""Notebook construction for MMM reports.

Builds an in-memory Jupyter notebook (:mod:`nbformat` v4) whose cells contain
pre-rendered outputs (DataFrames as HTML, figures as base-64 PNG, interactive
charts as Plotly JSON).  The notebook is the intermediate representation used
by the HTML and PDF exporters.
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import nbformat

from pymc_marketing.mmm.report._contracts import ReportData


def _fig_to_png_base64(fig: Any) -> str:
    """Render a matplotlib figure to a base-64 encoded PNG string.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to render.

    Returns
    -------
    str
        Base-64 encoded PNG bytes (ASCII).
    """
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _load_logo_base64() -> str | None:
    """Load the PyMC-Marketing logo as a base-64 string.

    Returns
    -------
    str or None
        Base-64 encoded JPEG bytes, or ``None`` if the logo file is not
        found on disk.
    """
    repo_root = Path(__file__).resolve().parents[3]
    logo_path = repo_root / "docs" / "source" / "_static" / "marketing-logo-light.jpg"
    if not logo_path.exists():
        return None
    return base64.b64encode(logo_path.read_bytes()).decode("ascii")


def _build_header_markdown(report_data: ReportData) -> str:
    """Build the Markdown header cell for the report notebook.

    Parameters
    ----------
    report_data : ReportData
        Report payload whose metadata populates the header.

    Returns
    -------
    str
        Markdown source including the logo, title, and metadata bullet list.
    """
    metadata = report_data.metadata
    logo_b64 = _load_logo_base64()
    logo_html = ""
    if logo_b64 is not None:
        logo_html = (
            "<div style='float:right; margin-left:16px;'>"
            f"<img alt='PyMC-Marketing Logo' src='data:image/jpeg;base64,{logo_b64}' width='180' />"
            "</div>"
        )

    return (
        f"{logo_html}\n"
        "# MMM Report\n\n"
        f"- **Created (UTC):** {metadata.created_at.isoformat()}\n"
        f"- **PyMC-Marketing Version:** {metadata.package_version}\n"
        f"- **Model:** {metadata.model_name}\n"
        f"- **Date Range:** {metadata.start_date} to {metadata.end_date}\n"
        f"- **Posterior Samples:** chains={metadata.chains}, draws={metadata.draws}\n"
        f"- **Channels:** {', '.join(metadata.channels)}\n"
        f"- **Controls:** {', '.join(metadata.controls) if metadata.controls else 'None'}\n"
    )


def build_notebook(
    report_data: ReportData, *, include_interactive: bool = True
) -> nbformat.NotebookNode:
    """Build a Jupyter notebook with pre-computed cell outputs.

    Parameters
    ----------
    report_data : ReportData
        Fully populated report payload.
    include_interactive : bool
        Whether to embed interactive Plotly figures as cell outputs.

    Returns
    -------
    nbformat.NotebookNode
        An nbformat v4 notebook ready for export or serialisation.
    """
    nb = nbformat.v4.new_notebook()
    nb.cells = []

    nb.cells.append(nbformat.v4.new_markdown_cell(_build_header_markdown(report_data)))

    for section_key, section in report_data.sections.items():
        nb.cells.append(
            nbformat.v4.new_markdown_cell(
                f"## {section.title}\n\n{section.description}\n\n`section_key`: `{section_key}`"
            )
        )
        cell = nbformat.v4.new_code_cell(section.source_code)
        cell.metadata["tags"] = ["report-code"]
        cell.metadata["jupyter"] = {"source_hidden": True}
        cell.outputs = []

        for name, df in section.display_dataframes.items():
            cell.outputs.append(
                nbformat.v4.new_output(
                    output_type="display_data",
                    data={
                        "text/plain": name,
                        "text/html": df.to_html(index=False, classes="dataframe"),
                    },
                    metadata={},
                )
            )

        for _, fig in section.static_figures.items():
            cell.outputs.append(
                nbformat.v4.new_output(
                    output_type="display_data",
                    data={"image/png": _fig_to_png_base64(fig)},
                    metadata={},
                )
            )

        if include_interactive:
            for _, fig in section.interactive_figures.items():
                cell.outputs.append(
                    nbformat.v4.new_output(
                        output_type="display_data",
                        data={"application/vnd.plotly.v1+json": fig.to_dict()},
                        metadata={},
                    )
                )

        nb.cells.append(cell)

    return nb
