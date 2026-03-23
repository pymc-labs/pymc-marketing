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
"""Section builders for MMM report generation."""

from __future__ import annotations

import warnings
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pymc_marketing.mmm.report._contracts import (
    ReportConfig,
    ReportData,
    ReportMetadata,
    ReportSection,
)
from pymc_marketing.version import __version__

NOTEBOOK_CASE_STUDY = "docs/source/notebooks/mmm/mmm_case_study.ipynb"
NOTEBOOK_EXAMPLE = "docs/source/notebooks/mmm/mmm_example.ipynb"
NOTEBOOK_INTERACTIVE = "docs/source/notebooks/mmm/plot_interactive.ipynb"

PARITY_MATRIX: dict[str, tuple[str, ...]] = {
    "model_overview": (NOTEBOOK_CASE_STUDY, NOTEBOOK_EXAMPLE),
    "diagnostics": (NOTEBOOK_CASE_STUDY, NOTEBOOK_EXAMPLE),
    "posterior_predictive": (
        NOTEBOOK_CASE_STUDY,
        NOTEBOOK_EXAMPLE,
        NOTEBOOK_INTERACTIVE,
    ),
    "component_contributions": (
        NOTEBOOK_CASE_STUDY,
        NOTEBOOK_EXAMPLE,
        NOTEBOOK_INTERACTIVE,
    ),
    "roas": (NOTEBOOK_CASE_STUDY, NOTEBOOK_EXAMPLE, NOTEBOOK_INTERACTIVE),
    "saturation_curves": (
        NOTEBOOK_CASE_STUDY,
        NOTEBOOK_EXAMPLE,
        NOTEBOOK_INTERACTIVE,
    ),
    "sensitivity": (NOTEBOOK_CASE_STUDY, NOTEBOOK_EXAMPLE),
}


def _finalize_figure(fig: Any, title: str) -> Any:
    """Set suptitle and tight_layout on a matplotlib figure."""
    fig.suptitle(title, fontsize=14, fontweight="bold")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "The figure layout has changed to tight")
        fig.tight_layout()
    return fig


def _ensure_pandas(df: Any) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df.copy()
    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    return pd.DataFrame(df)


def _add_point_forecast(df: pd.DataFrame, point_estimate: str) -> pd.DataFrame:
    out = df.copy()
    if point_estimate in out.columns:
        out["point_forecast"] = out[point_estimate]
    elif "mean" in out.columns:
        out["point_forecast"] = out["mean"]
    elif "median" in out.columns:
        out["point_forecast"] = out["median"]
    return out


def _apply_dims_filter(
    df: pd.DataFrame,
    dims: Mapping[str, str | int | Iterable[str | int]] | None,
) -> pd.DataFrame:
    if dims is None or df.empty:
        return df
    out = df.copy()
    for col, value in dims.items():
        if col not in out.columns:
            continue
        if isinstance(value, (list, tuple)):
            out = out[out[col].isin(value)]
        else:
            out = out[out[col] == value]
    return out


def _build_metadata(mmm: Any) -> ReportMetadata:
    idata = mmm.idata
    posterior = idata.posterior
    chains = int(posterior.sizes.get("chain", 0))
    draws = int(posterior.sizes.get("draw", 0))

    x_df: pd.DataFrame | None = getattr(mmm, "X", None)
    date_column = getattr(mmm, "date_column", "date")
    start_date: str | None = None
    end_date: str | None = None
    if (
        isinstance(x_df, pd.DataFrame)
        and date_column in x_df.columns
        and not x_df.empty
    ):
        start_date = str(pd.to_datetime(x_df[date_column]).min())
        end_date = str(pd.to_datetime(x_df[date_column]).max())

    return ReportMetadata.now(
        package_version=__version__,
        model_name=type(mmm).__name__,
        date_column=date_column,
        start_date=start_date,
        end_date=end_date,
        chains=chains,
        draws=draws,
        channels=getattr(mmm, "channel_columns", []),
        controls=getattr(mmm, "control_columns", []) or [],
    )


def _section_model_overview(mmm: Any) -> ReportSection:
    rows = [
        ("model_name", type(mmm).__name__),
        ("date_column", getattr(mmm, "date_column", "")),
        ("target_column", getattr(mmm, "target_column", "")),
        ("channels", ", ".join(getattr(mmm, "channel_columns", []))),
        ("controls", ", ".join(getattr(mmm, "control_columns", []) or [])),
        ("dims", ", ".join(getattr(mmm, "dims", ()) or ())),
        ("adstock", type(getattr(mmm, "adstock", None)).__name__),
        ("saturation", type(getattr(mmm, "saturation", None)).__name__),
    ]
    overview_df = pd.DataFrame(rows, columns=["field", "value"])
    return ReportSection(
        title="Model Overview",
        description="Model metadata and configuration.",
        source_code="# Derived from MMM model attributes",
        dataframes={"model_overview": overview_df},
    )


def _section_diagnostics(mmm: Any, config: ReportConfig) -> ReportSection:
    divergences = int(mmm.idata["sample_stats"]["diverging"].sum().item())
    diagnostics_df = pd.DataFrame({"metric": ["divergences"], "value": [divergences]})
    var_names = [
        name
        for name in [
            "intercept_contribution",
            "y_sigma",
            "saturation_beta",
            "saturation_lam",
            "adstock_alpha",
            "gamma_control",
            "gamma_fourier",
        ]
        if name in mmm.fit_result.data_vars
    ]
    summary_df = az.summary(data=mmm.fit_result, var_names=var_names).reset_index()

    return ReportSection(
        title="Diagnostics",
        description="Sampling diagnostics and divergence summary.",
        source_code=(
            "divergences = mmm.idata['sample_stats']['diverging'].sum().item()\n"
            "az.summary(data=mmm.fit_result, var_names=[...])"
        ),
        dataframes={
            "diagnostics": diagnostics_df,
            "arviz_summary": _add_point_forecast(summary_df, config.point_estimate),
        },
    )


def _section_posterior_predictive(mmm: Any, config: ReportConfig) -> ReportSection:
    pp_df = _ensure_pandas(
        mmm.summary.posterior_predictive(
            hdi_probs=config.hdi_probs,
            frequency="original",
            output_format="pandas",
        )
    )
    pp_df = _add_point_forecast(pp_df, config.point_estimate)
    pp_df = _apply_dims_filter(pp_df, config.dims)

    hdi_prob = max(config.hdi_probs)
    fig_static, _ = mmm.plot.posterior_predictive(
        var=["y_original_scale"] if "y_original_scale" in mmm.idata.posterior else None,
        hdi_prob=hdi_prob,
    )
    _finalize_figure(fig_static, "Posterior Predictive Fit")

    interactive_figures: dict[str, Any] = {}
    if config.include_interactive:
        interactive_figures["posterior_predictive"] = (
            mmm.plot_interactive.posterior_predictive(
                hdi_prob=hdi_prob,
                frequency="original",
            )
        )

    return ReportSection(
        title="Posterior Predictive Fit",
        description="Observed target against posterior predictive uncertainty.",
        source_code=(
            "df = mmm.summary.posterior_predictive(hdi_probs=..., frequency='original')\n"
            "mmm.plot.posterior_predictive(hdi_prob=max(hdi_probs))"
        ),
        dataframes={"posterior_predictive": pp_df},
        static_figures={"posterior_predictive": fig_static},
        interactive_figures=interactive_figures,
    )


def _section_component_contributions(mmm: Any, config: ReportConfig) -> ReportSection:
    total_df = _ensure_pandas(
        mmm.summary.total_contribution(
            hdi_probs=config.hdi_probs,
            frequency=config.frequency,
            output_format="pandas",
        )
    )
    channel_df = _ensure_pandas(
        mmm.summary.contributions(
            component="channel",
            hdi_probs=config.hdi_probs,
            frequency=config.frequency,
            output_format="pandas",
        )
    )
    total_df = _add_point_forecast(
        _apply_dims_filter(total_df, config.dims), config.point_estimate
    )
    channel_df = _add_point_forecast(
        _apply_dims_filter(channel_df, config.dims), config.point_estimate
    )

    hdi_prob = max(config.hdi_probs)
    fig_waterfall, _ = mmm.plot.waterfall_components_decomposition()
    _finalize_figure(fig_waterfall, "Waterfall Components Decomposition")

    fig_over_time, _ = mmm.plot.contributions_over_time(
        var=["channel_contribution"],
        hdi_prob=hdi_prob,
        dims=config.dims,
    )
    _finalize_figure(fig_over_time, "Channel Contributions Over Time")

    fig_share, _ = mmm.plot.channel_contribution_share_hdi(
        hdi_prob=hdi_prob, dims=config.dims
    )
    _finalize_figure(fig_share, "Channel Contribution Share")

    original_scale_vars = [
        v
        for v in [
            "channel_contribution_original_scale",
            "control_contribution_original_scale",
            "intercept_contribution_original_scale",
            "yearly_seasonality_contribution_original_scale",
        ]
        if v in mmm.idata.posterior
    ]
    fig_original, axes_original = mmm.plot.contributions_over_time(
        var=original_scale_vars,
        combine_dims=True,
        hdi_prob=hdi_prob,
        figsize=(12, 7),
    )
    legend = np.asarray(axes_original).ravel()[0].get_legend()
    if legend is not None:
        legend.set_bbox_to_anchor((0.8, -0.12))
    _finalize_figure(fig_original, "Component Contributions (Original Scale)")

    interactive_figures: dict[str, Any] = {}
    if config.include_interactive:
        interactive_figures["channel_contributions"] = (
            mmm.plot_interactive.contributions(
                hdi_prob=hdi_prob,
                component="channel",
                frequency=config.frequency,
            )
        )

    return ReportSection(
        title="Component Contributions",
        description="Baseline, controls, and channel contribution decomposition.",
        source_code=(
            "total_df = mmm.summary.total_contribution(...)\n"
            "channel_df = mmm.summary.contributions(component='channel', ...)\n"
            "mmm.plot.waterfall_components_decomposition()\n"
            "mmm.plot.contributions_over_time(var=[...], combine_dims=True)"
        ),
        dataframes={
            "total_contributions": total_df,
            "channel_contributions": channel_df,
        },
        static_figures={
            "waterfall_components_decomposition": fig_waterfall,
            "contributions_over_time": fig_over_time,
            "channel_contribution_share_hdi": fig_share,
            "contributions_original_scale": fig_original,
        },
        interactive_figures=interactive_figures,
    )


def _roas_forest_figure(
    mmm: Any,
    method: str,
    config: ReportConfig,
) -> Any:
    """Build an ``az.plot_forest`` figure for ROAS.

    Always uses ``frequency="all_time"`` so the forest plot shows one
    entry per channel (matching the case-study style).
    """
    if method == "incremental":
        roas_xr = mmm.incrementality.contribution_over_spend(
            frequency="all_time",
            num_samples=config.num_samples,
            random_state=config.random_state,
        ).rename("roas")
    else:
        data = mmm.data.aggregate_time("all_time")
        roas_xr = data.get_elementwise_roas(original_scale=True).rename("roas")

    fig, ax = plt.subplots(figsize=(10, 6))
    az.plot_forest(roas_xr, combined=True, ax=ax)
    title = (
        "Return on Ad Spend (Incremental)"
        if method == "incremental"
        else "Return on Ad Spend (Elementwise)"
    )
    _finalize_figure(fig, title)
    return fig


def _section_roas(mmm: Any, config: ReportConfig) -> ReportSection:
    hdi_prob = max(config.hdi_probs)
    dfs: dict[str, pd.DataFrame] = {}
    static_figures: dict[str, Any] = {}
    interactive_figures: dict[str, Any] = {}

    for method in config.roas_methods:
        roas_df = _ensure_pandas(
            mmm.summary.roas(
                hdi_probs=config.hdi_probs,
                frequency=config.frequency,
                method=method,
                num_samples=config.num_samples,
                random_state=config.random_state,
                output_format="pandas",
            )
        )
        roas_df = _add_point_forecast(
            _apply_dims_filter(roas_df, config.dims), config.point_estimate
        )
        dfs[f"roas_{method}"] = roas_df
        static_figures[f"roas_forest_{method}"] = _roas_forest_figure(
            mmm, method, config
        )
        if config.include_interactive:
            interactive_figures[f"roas_{method}"] = mmm.plot_interactive.roas(
                hdi_prob=hdi_prob,
                frequency=config.frequency,
                method=method,
                num_samples=config.num_samples,
                random_state=config.random_state,
            )

    return ReportSection(
        title="ROAS",
        description="Elementwise and incremental return-on-ad-spend summaries.",
        source_code=(
            "roas = mmm.incrementality.contribution_over_spend(...).rename('roas')\n"
            "az.plot_forest(roas, combined=True)"
        ),
        dataframes=dfs,
        static_figures=static_figures,
        interactive_figures=interactive_figures,
    )


def _section_saturation_curves(mmm: Any, config: ReportConfig) -> ReportSection:
    saturation_df = _ensure_pandas(
        mmm.summary.saturation_curves(
            hdi_probs=config.hdi_probs,
            output_format="pandas",
            num_samples=config.num_samples or 500,
            random_state=config.random_state,
            original_scale=True,
        )
    )
    saturation_df = _add_point_forecast(
        _apply_dims_filter(saturation_df, config.dims), config.point_estimate
    )
    fig_static, _ = mmm.plot.saturation_scatterplot(
        original_scale=True, dims=config.dims
    )
    _finalize_figure(fig_static, "Saturation Curves")
    interactive_figures: dict[str, Any] = {}
    if config.include_interactive:
        interactive_figures["saturation_curves"] = (
            mmm.plot_interactive.saturation_curves(
                hdi_prob=max(config.hdi_probs),
                num_samples=config.num_samples or 500,
                random_state=config.random_state,
                original_scale=True,
            )
        )
    return ReportSection(
        title="Saturation Curves",
        description="Spend-response behavior and uncertainty by channel.",
        source_code=(
            "saturation_df = mmm.summary.saturation_curves(...)\n"
            "mmm.plot.saturation_scatterplot(original_scale=True)"
        ),
        dataframes={"saturation_curves": saturation_df},
        static_figures={"saturation_scatterplot": fig_static},
        interactive_figures=interactive_figures,
    )


def _section_sensitivity(mmm: Any, config: ReportConfig) -> ReportSection | None:
    if config.sensitivity_sweep_values is None:
        return None

    sweep_values = np.asarray(config.sensitivity_sweep_values, dtype=float)
    mmm.sensitivity.run_sweep(
        sweep_values=sweep_values,
        var_input="channel_data",
        var_names="channel_contribution",
        extend_idata=True,
    )
    fig_relative, _ = mmm.plot.sensitivity_analysis(
        hdi_prob=max(config.hdi_probs),
        x_sweep_axis="relative",
    )
    _finalize_figure(fig_relative, "Sensitivity Analysis (Relative)")
    fig_absolute, _ = mmm.plot.sensitivity_analysis(
        hdi_prob=max(config.hdi_probs),
        x_sweep_axis="absolute",
        hue_dim="channel",
    )
    _finalize_figure(fig_absolute, "Sensitivity Analysis (Absolute)")
    sweep_df = (
        mmm.idata["sensitivity_analysis"]["x"].to_dataframe(name="value").reset_index()
    )
    sweep_df = _apply_dims_filter(sweep_df, config.dims)
    return ReportSection(
        title="Sensitivity Analysis",
        description="Counterfactual sweep across spend multipliers.",
        source_code=(
            "mmm.sensitivity.run_sweep(...)\n"
            "mmm.plot.sensitivity_analysis(x_sweep_axis='relative')\n"
            "mmm.plot.sensitivity_analysis(x_sweep_axis='absolute')"
        ),
        dataframes={"sensitivity_analysis": sweep_df},
        static_figures={
            "sensitivity_relative": fig_relative,
            "sensitivity_absolute": fig_absolute,
        },
    )


def build_report_data(mmm: Any, config: ReportConfig) -> ReportData:
    """Build all report sections and metadata from a fitted MMM."""
    sections: OrderedDict[str, ReportSection] = OrderedDict()
    sections["model_overview"] = _section_model_overview(mmm)
    sections["diagnostics"] = _section_diagnostics(mmm, config)
    sections["posterior_predictive"] = _section_posterior_predictive(mmm, config)
    sections["component_contributions"] = _section_component_contributions(mmm, config)
    sections["roas"] = _section_roas(mmm, config)
    sections["saturation_curves"] = _section_saturation_curves(mmm, config)
    sensitivity_section = _section_sensitivity(mmm, config)
    if sensitivity_section is not None:
        sections["sensitivity"] = sensitivity_section

    return ReportData(
        metadata=_build_metadata(mmm),
        sections=sections,
        parity_matrix=PARITY_MATRIX,
    )
