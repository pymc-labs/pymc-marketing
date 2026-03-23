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
"""Export helpers for MMM reports."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import nbformat
import pandas as pd

from pymc_marketing.mmm.report._contracts import ReportData
from pymc_marketing.mmm.report._notebook import build_notebook


def _check_notebook_deps() -> None:
    try:
        import nbconvert  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "HTML/PDF report export requires nbconvert. "
            "Install with: pip install pymc-marketing[report]"
        ) from err


def _check_excel_deps() -> None:
    try:
        import openpyxl  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "Excel report export requires openpyxl. "
            "Install with: pip install pymc-marketing[report]"
        ) from err


def export_html(
    report_data: ReportData,
    *,
    file_name: str | None = None,
    save_intermediate_notebook: str | None = None,
) -> str:
    """Render report HTML from notebook representation."""
    _check_notebook_deps()
    from nbconvert import HTMLExporter

    nb = build_notebook(report_data, include_interactive=True)
    if save_intermediate_notebook is not None:
        Path(save_intermediate_notebook).write_text(
            nbformat.writes(nb), encoding="utf-8"
        )

    template_dir = Path(__file__).resolve().parent / "templates"
    exporter = HTMLExporter(template_name="report_html")
    exporter.extra_template_basedirs = [str(template_dir)]
    html, _ = exporter.from_notebook_node(nb)
    if file_name is not None:
        Path(file_name).write_text(html, encoding="utf-8")
    return html


def export_pdf(
    report_data: ReportData,
    *,
    file_name: str,
    engine: Literal["auto", "latex", "webpdf"] = "auto",
) -> None:
    """Export report as PDF, preferring webpdf then latex in auto mode."""
    _check_notebook_deps()
    from nbconvert import PDFExporter, WebPDFExporter
    from traitlets.config import Config

    nb = build_notebook(report_data, include_interactive=False)
    config = Config()
    config.TagRemovePreprocessor.enabled = True
    config.TagRemovePreprocessor.remove_input_tags = {"report-code"}
    config.TagRemovePreprocessor.remove_cell_tags = set()

    errors: list[str] = []

    def _try_export(exporter_cls) -> bytes | None:
        try:
            exporter = exporter_cls(config=config)
            body, _ = exporter.from_notebook_node(nb)
            return body
        except Exception as err:  # pragma: no cover - dependent on local system tooling
            errors.append(f"{exporter_cls.__name__}: {err}")
            return None

    body: bytes | None = None
    if engine == "webpdf":
        body = _try_export(WebPDFExporter)
    elif engine == "latex":
        body = _try_export(PDFExporter)
    else:
        body = _try_export(WebPDFExporter) or _try_export(PDFExporter)

    if body is None:
        error_text = "\n".join(errors)
        raise RuntimeError(
            "Unable to export PDF with selected engine(s). "
            "For webpdf install playwright chromium; for latex install a TeX distribution.\n"
            f"Underlying errors:\n{error_text}"
        )

    Path(file_name).write_bytes(body)


def export_excel(report_data: ReportData, *, file_name: str) -> None:
    """Export report tables to Excel with a metadata cover sheet."""
    _check_excel_deps()
    from openpyxl.drawing.image import Image

    metadata = report_data.metadata
    with pd.ExcelWriter(file_name, engine="openpyxl") as writer:
        cover = pd.DataFrame(
            {
                "field": [
                    "created_at_utc",
                    "package_version",
                    "model_name",
                    "date_range",
                    "chains",
                    "draws",
                    "channels",
                    "controls",
                ],
                "value": [
                    metadata.created_at.isoformat(),
                    metadata.package_version,
                    metadata.model_name,
                    f"{metadata.start_date} to {metadata.end_date}",
                    metadata.chains,
                    metadata.draws,
                    ", ".join(metadata.channels),
                    ", ".join(metadata.controls) if metadata.controls else "None",
                ],
            }
        )
        cover.to_excel(writer, sheet_name="Cover", index=False)

        for section_key, section in report_data.sections.items():
            for table_key, df in section.dataframes.items():
                sheet_name = f"{section_key[:12]}_{table_key[:18]}"[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        ws = writer.book["Cover"]
        repo_root = Path(__file__).resolve().parents[3]
        logo_path = (
            repo_root / "docs" / "source" / "_static" / "marketing-logo-light.jpg"
        )
        if logo_path.exists():  # pragma: no branch
            image = Image(str(logo_path))
            image.width = 220
            image.height = 90
            ws.add_image(image, "D1")
