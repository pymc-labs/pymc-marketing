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
"""
MMM Dashboard - Marketing Mix Model Visualization Dashboard
Built with Dash and Dash Mantine Components
"""

import itertools
import os
from datetime import datetime

import arviz as az
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
import yaml
from dash import Dash, Input, Output, State, dcc, html
from plotly.subplots import make_subplots
from scipy import stats

from pymc_marketing.mmm.builders.yaml import build_mmm_from_yaml

# ============================================================================
# Model Loading
# ============================================================================


def load_model_and_data(x_filename, y_filename, yaml_filename):
    """Load model and data with user-specified file names.

    Parameters
    ----------
    x_filename : str
        Name of X data file (without .csv extension)
    y_filename : str
        Name of Y data file (without .csv extension)
    yaml_filename : str
        Name of YAML config file (without .yml extension)

    Returns
    -------
    mmm : MMM model object
        Fitted marketing mix model
    y : np.ndarray
        Target variable values
    """
    print("Loading the yml file...")
    yaml_path = f"files/{yaml_filename}.yml"
    with open(yaml_path) as file:
        yml_content = yaml.safe_load(file)  # Parse YAML to dictionary
    print(yml_content)

    print("Loading data...")
    x_path = f"files/{x_filename}.csv"
    y_path = f"files/{y_filename}.csv"
    X = pd.read_csv(x_path, parse_dates=[yml_content["model"]["kwargs"]["date_column"]])
    y_df = pd.read_csv(y_path)
    y = y_df[yml_content["model"]["kwargs"]["target_column"]].values

    print("Building MMM from YAML configuration...")
    mmm = build_mmm_from_yaml(config_path=yaml_path)

    print("Sampling posterior predictive...")
    mmm.sample_posterior_predictive(X, extend_idata=True, combined=True)

    print("Model loaded successfully!")
    return mmm, y


# Initialize model and data as None (will be loaded after user input)
mmm = None
y = None

# These will be populated after model is loaded
all_posterior_vars = []
default_diagnostic_vars = []


# ============================================================================
# Visualization Functions
# ============================================================================


def create_posterior_predictive_plot():
    """Create posterior predictive plot with HDI intervals."""
    # Extract predictions
    y_pred = mmm.idata.posterior_predictive.y_original_scale

    # Calculate HDI intervals
    hdi_94 = az.hdi(y_pred, hdi_prob=0.94)
    hdi_50 = az.hdi(y_pred, hdi_prob=0.50)

    # Calculate mean prediction
    y_mean = y_pred.mean(dim=["chain", "draw"])

    # Get dates
    dates = hdi_94.date.values

    # Create the figure
    fig = go.Figure()

    # Add HDI 94% band
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=hdi_94["y_original_scale"].sel(hdi="higher").values,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            name="HDI 94% upper",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=hdi_94["y_original_scale"].sel(hdi="lower").values,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(0, 100, 80, 0.2)",
            showlegend=True,
            name="HDI 94%",
        )
    )

    # Add HDI 50% band
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=hdi_50["y_original_scale"].sel(hdi="higher").values,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            name="HDI 50% upper",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=hdi_50["y_original_scale"].sel(hdi="lower").values,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(0, 100, 80, 0.4)",
            showlegend=True,
            name="HDI 50%",
        )
    )

    # Add the mean line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=y_mean.values,
            mode="lines",
            line=dict(color="rgb(0, 100, 80)", width=2),
            name="Mean",
        )
    )

    # Add observed data
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=y,
            mode="markers",
            marker=dict(color="black", size=4),
            name="Observed",
        )
    )

    # Update layout
    fig.update_layout(
        title="Posterior Predictive Plot",
        xaxis_title="Date",
        yaxis_title="Sales",
        template="plotly_white",
        hovermode="x unified",
        height=600,
    )

    return fig


def create_decomposition_plot():
    """Create decomposition plot showing all contributions over time."""
    # Extract the contribution variables
    channel_contrib = mmm.idata.posterior.channel_contribution_original_scale.sum(
        dim="channel"
    )
    control_contrib = mmm.idata.posterior.control_contribution_original_scale.sum(
        dim="control"
    )
    trend_contrib = mmm.idata.posterior.trend_effect_contribution_original_scale
    seasonality_contrib = (
        mmm.idata.posterior.yearly_seasonality_contribution_original_scale
    )

    # Store in a dictionary for easier iteration
    contributions = {
        "Channel Effects": channel_contrib,
        "Control Variables": control_contrib,
        "Trend Effect": trend_contrib,
        "Yearly Seasonality": seasonality_contrib,
    }

    # Define colors for each contribution type
    colors = {
        "Channel Effects": "rgb(31, 119, 180)",  # blue
        "Control Variables": "rgb(255, 127, 14)",  # orange
        "Trend Effect": "rgb(44, 160, 44)",  # green
        "Yearly Seasonality": "rgb(214, 39, 40)",  # red
    }

    # Create figure
    fig = go.Figure()

    # Get dates
    dates = channel_contrib.date.values

    # Add traces for each variable
    for var_name, var_data in contributions.items():
        # Calculate HDI intervals
        hdi_94 = az.hdi(var_data, hdi_prob=0.94)

        # Calculate mean
        var_mean = var_data.mean(dim=["chain", "draw"])

        # Get the variable name from the dataset
        data_var_name = list(hdi_94.data_vars)[0]

        # Extract HDI bounds
        hdi_upper = hdi_94[data_var_name].sel(hdi="higher").values
        hdi_lower = hdi_94[data_var_name].sel(hdi="lower").values

        # Add HDI band (filled area)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([dates, dates[::-1]]),
                y=np.concatenate([hdi_upper, hdi_lower[::-1]]),
                fill="toself",
                fillcolor=colors[var_name]
                .replace("rgb", "rgba")
                .replace(")", ", 0.2)"),
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
                name=f"{var_name} 94% HDI",
            )
        )

        # Add mean line
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=var_mean.values,
                mode="lines",
                line=dict(color=colors[var_name], width=2),
                name=var_name,
            )
        )

    # Update layout
    fig.update_layout(
        title="Model Contributions Over Time (Original Scale) with 94% HDI",
        xaxis_title="Date",
        yaxis_title="Contribution",
        template="plotly_white",
        hovermode="x unified",
        height=600,
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


def create_channel_contributions_plot():
    """Create channel contributions plot with subplots for each channel."""
    # Extract the channel contributions
    channel_contributions = mmm.idata.posterior.channel_contribution_original_scale
    channels = channel_contributions.channel.values
    dates = channel_contributions.date.values

    # Calculate number of rows for subplots
    n_channels = len(channels)
    n_cols = 1
    n_rows = n_channels

    # Create subplots
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"{channel}" for channel in channels],
        vertical_spacing=0.2,
        horizontal_spacing=0.10,
    )

    # Color palette
    color_palette = [
        "rgb(31, 119, 180)",
        "rgb(255, 127, 14)",
        "rgb(44, 160, 44)",
        "rgb(214, 39, 40)",
        "rgb(148, 103, 189)",
        "rgb(140, 86, 75)",
    ]

    # Add traces for each channel
    for idx, channel in enumerate(channels):
        row = idx + 1
        col = 1

        # Select channel data
        channel_data = channel_contributions.sel(channel=channel)

        # Calculate HDI and mean
        hdi_94 = az.hdi(channel_data, hdi_prob=0.94)
        mean_contribution = channel_data.mean(dim=["chain", "draw"])

        # Get data variable name
        data_var_name = list(hdi_94.data_vars)[0]

        # Extract values
        hdi_upper = hdi_94[data_var_name].sel(hdi="higher").values
        hdi_lower = hdi_94[data_var_name].sel(hdi="lower").values

        # Choose color
        color = color_palette[idx % len(color_palette)]
        fill_color = color.replace("rgb", "rgba").replace(")", ", 0.2)")

        # Add HDI band
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([dates, dates[::-1]]),
                y=np.concatenate([hdi_upper, hdi_lower[::-1]]),
                fill="toself",
                fillcolor=fill_color,
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        # Add mean line
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=mean_contribution.values,
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    # Update layout
    fig.update_layout(
        title_text="Channel Contributions Over Time (Original Scale) with 94% HDI",
        template="plotly_white",
        height=300 * n_rows,
        showlegend=False,
    )

    # Update x and y axis labels
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Contribution")

    return fig


def create_roas_distribution_plot():
    """Create ROAS distribution plot for each channel."""
    # Calculate total ROAS
    roas_samples = mmm.idata.posterior.channel_contribution_original_scale.sum(
        dim="date"
    ) / mmm.idata.constant_data.channel_data.sum(dim="date")

    # Get channels
    channels = roas_samples.channel.values

    # Create subplots
    n_channels = len(channels)
    fig = make_subplots(
        rows=n_channels,
        cols=1,
        subplot_titles=[f"Channel: {channel}" for channel in channels],
        vertical_spacing=0.15,
    )

    # Color palette
    color_palette = [
        "rgba(31, 119, 180, 0.7)",
        "rgba(255, 127, 14, 0.7)",
        "rgba(44, 160, 44, 0.7)",
        "rgba(214, 39, 40, 0.7)",
    ]

    # Create distribution plot for each channel
    for idx, channel in enumerate(channels):
        # Get ROAS samples for this channel
        channel_roas = roas_samples.sel(channel=channel)

        # Flatten the samples
        samples = channel_roas.values.flatten()

        # Calculate HDI
        hdi_94 = az.hdi(channel_roas, hdi_prob=0.94)
        data_var_name = list(hdi_94.data_vars)[0]
        hdi_lower = float(hdi_94[data_var_name].sel(hdi="lower").values)
        hdi_upper = float(hdi_94[data_var_name].sel(hdi="higher").values)

        # Calculate mean
        mean_roas = float(samples.mean())

        # Create histogram
        fig.add_trace(
            go.Histogram(
                x=samples,
                name=channel,
                marker_color=color_palette[idx % len(color_palette)],
                opacity=0.7,
                nbinsx=50,
                histnorm="probability density",
                showlegend=False,
            ),
            row=idx + 1,
            col=1,
        )

        # Add vertical lines for mean and HDI
        y_max = 1

        # Mean line
        fig.add_trace(
            go.Scatter(
                x=[mean_roas, mean_roas],
                y=[0, y_max],
                mode="lines",
                line=dict(color="black", width=2, dash="dash"),
                name="Mean",
                showlegend=(idx == 0),
                hovertemplate=f"Mean: {mean_roas:.4f}<extra></extra>",
            ),
            row=idx + 1,
            col=1,
        )

        # HDI lines
        for hdi_val, name in [(hdi_lower, "HDI Lower"), (hdi_upper, "HDI Upper")]:
            fig.add_trace(
                go.Scatter(
                    x=[hdi_val, hdi_val],
                    y=[0, y_max],
                    mode="lines",
                    line=dict(color="red", width=1.5),
                    name="94% HDI" if idx == 0 and name == "HDI Lower" else None,
                    showlegend=(idx == 0 and name == "HDI Lower"),
                    hovertemplate=f"{name}: {hdi_val:.4f}<extra></extra>",
                ),
                row=idx + 1,
                col=1,
            )

        # Update x-axis label for bottom plot
        if idx == n_channels - 1:
            fig.update_xaxes(
                title_text="ROAS (Total Response / Total Input)", row=idx + 1, col=1
            )

    # Update layout
    fig.update_layout(
        title_text="Return on Input Posterior Distributions",
        template="plotly_white",
        height=400 * n_channels,
        showlegend=True,
    )

    fig.update_yaxes(title_text="Density")

    return fig


def create_saturation_curves_plot():
    """Create saturation curves plot for each channel."""
    # Sample curves
    curve = mmm.saturation.sample_curve(
        mmm.idata.posterior[["saturation_beta", "saturation_lam"]], max_value=2
    )

    channels = curve.channel.values
    n_channels = len(channels)
    n_cols = 2
    n_rows = int(np.ceil(n_channels / n_cols))

    # Create subplots
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"{channel}" for channel in channels],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    color_palette = [
        "rgb(31, 119, 180)",
        "rgb(255, 127, 14)",
        "rgb(44, 160, 44)",
        "rgb(214, 39, 40)",
    ]

    for idx, channel in enumerate(channels):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        # Select and scale
        curve_channel = curve.sel(channel=channel)
        curve_original_scale = (
            curve_channel * mmm.idata.constant_data.target_scale
        ).rename("saturation_curve")
        channel_scale = mmm.idata.constant_data.channel_scale.sel(channel=channel)
        x_original = curve_channel.coords["x"].values * float(channel_scale.values)

        # Calculate statistics
        curve_mean = curve_original_scale.mean(dim=["chain", "draw"])
        hdi_94 = az.hdi(curve_original_scale, hdi_prob=0.94)

        data_var_name = list(hdi_94.data_vars)[0]
        hdi_upper = hdi_94[data_var_name].sel(hdi="higher").values
        hdi_lower = hdi_94[data_var_name].sel(hdi="lower").values

        color = color_palette[idx % len(color_palette)]
        fill_color = color.replace("rgb", "rgba").replace(")", ", 0.2)")

        # HDI band
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x_original, x_original[::-1]]),
                y=np.concatenate([hdi_upper, hdi_lower[::-1]]),
                fill="toself",
                fillcolor=fill_color,
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        # Mean curve
        fig.add_trace(
            go.Scatter(
                x=x_original,
                y=curve_mean.values,
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        # Scatter points
        channel_data = mmm.idata.constant_data.channel_data.sel(channel=channel)
        channel_contrib = mmm.idata.posterior.channel_contribution_original_scale.sel(
            channel=channel
        ).mean(dim=["chain", "draw"])

        fig.add_trace(
            go.Scatter(
                x=channel_data.values,
                y=channel_contrib.values,
                mode="markers",
                marker=dict(color=color, size=4, opacity=0.6),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title_text="Saturation Curves (Original Scale) with 94% HDI",
        template="plotly_white",
        height=400 * n_rows,
        showlegend=False,
    )

    fig.update_xaxes(title_text="Channel Input")
    fig.update_yaxes(title_text="Contribution")

    return fig


def create_summary_table(selected_vars):
    """Generate a Mantine table from az.summary for selected variables."""
    if not selected_vars:
        return dmc.Text("No variables selected", c="dimmed", ta="center")

    # Get summary statistics
    summary_df = az.summary(mmm.idata, var_names=selected_vars)
    summary_df = summary_df.round(3)

    # Create table header
    header = dmc.TableThead(
        dmc.TableTr(
            [dmc.TableTh("Variable"), *[dmc.TableTh(col) for col in summary_df.columns]]
        )
    )

    # Create table rows
    rows = []
    for var_name, row_data in summary_df.iterrows():
        rows.append(
            dmc.TableTr(
                [
                    dmc.TableTd(str(var_name), style={"fontWeight": 500}),
                    *[dmc.TableTd(f"{val:.3f}") for val in row_data],
                ]
            )
        )

    body = dmc.TableTbody(rows)

    return dmc.Table(
        [header, body], striped=True, highlightOnHover=True, withTableBorder=True
    )


def create_contributions_dataset():
    """Create xarray dataset with all contributions combined."""
    # Combine all contributions into a single dataset using xr.concat
    contributions = xr.concat(
        [
            mmm.idata.posterior.control_contribution_original_scale.sum(dim="control")
            .expand_dims("component")
            .assign_coords(component=["control"]),
            mmm.idata.posterior.trend_effect_contribution_original_scale.expand_dims(
                "component"
            ).assign_coords(component=["trend"]),
            mmm.idata.posterior.yearly_seasonality_contribution_original_scale.expand_dims(
                "component"
            ).assign_coords(component=["seasonality"]),
            mmm.idata.posterior.intercept_contribution_original_scale.expand_dims(
                "component"
            )
            .assign_coords(component=["intercept"])
            .broadcast_like(
                mmm.idata.posterior.trend_effect_contribution_original_scale.expand_dims(
                    "component"
                )
            ),
            mmm.idata.posterior.channel_contribution_original_scale.sum(dim="channel")
            .expand_dims("component")
            .assign_coords(component=["media"]),
        ],
        dim="component",
    )

    return contributions


def create_contributions_table(df, row_limit):
    """Create a Mantine table from a contributions DataFrame."""
    # Limit rows for display
    display_df = df.head(row_limit)

    # Create table header
    header = dmc.TableThead(
        dmc.TableTr([dmc.TableTh(col) for col in display_df.columns])
    )

    # Create table rows
    rows = []
    for _, row_data in display_df.iterrows():
        cells = []
        for idx, val in enumerate(row_data):
            # First column (date) should be bold
            if idx == 0:
                cells.append(dmc.TableTd(str(val), style={"fontWeight": 500}))
            else:
                # Format numeric values
                cells.append(
                    dmc.TableTd(
                        f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
                    )
                )
        rows.append(dmc.TableTr(cells))

    body = dmc.TableTbody(rows)

    total_rows = len(df)

    # Add a caption showing how many rows are displayed
    caption = dmc.Text(
        f"Showing {len(display_df)} of {total_rows} rows",
        size="sm",
        c="dimmed",
        style={"marginTop": "10px", "marginBottom": "10px"},
    )

    return dmc.Stack(
        [
            dmc.Table(
                [header, body],
                striped=True,
                highlightOnHover=True,
                withTableBorder=True,
                horizontalSpacing="md",
            ),
            caption,
        ],
        gap="xs",
    )


def plot_trace_plotly(idata, var_names=None):
    """
    Replicate ArviZ's plot_trace using Plotly.

    Parameters
    ----------
    idata : arviz.InferenceData
        The inference data object containing posterior samples
    var_names : list of str, optional
        List of variable names to plot. If None, plots all variables.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The Plotly figure object
    """
    # Get posterior group
    posterior = idata.posterior

    # Get variable names if not provided
    if var_names is None:
        var_names = list(posterior.data_vars)

    # Filter variables and flatten multi-dimensional variables
    plot_vars = []
    for var in var_names:
        var_data = posterior[var]
        # Handle multi-dimensional variables
        if var_data.ndim == 2:  # Only chain and draw dimensions
            plot_vars.append((var, None))
        else:  # Has additional dimensions
            # Flatten additional dimensions
            coords = {
                dim: var_data[dim].values
                for dim in var_data.dims
                if dim not in ["chain", "draw"]
            }
            if coords:
                # Create all combinations of coordinates
                coord_names = list(coords.keys())
                coord_values = [coords[name] for name in coord_names]
                for combo in itertools.product(*coord_values):
                    selector = dict(zip(coord_names, combo))
                    label = f"{var}[{','.join(str(c) for c in combo)}]"
                    plot_vars.append((var, selector, label))
            else:
                plot_vars.append((var, None, var))

    n_vars = len(plot_vars)

    # Create subplots: 2 columns (distribution, trace), n_vars rows
    subplot_titles = []
    for item in plot_vars:
        var_label = item[2] if len(item) > 2 else item[0]
        subplot_titles.extend([f"{var_label} - Distribution", f"{var_label} - Trace"])

    fig = make_subplots(
        rows=n_vars,
        cols=2,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.05,
        vertical_spacing=0.15 if n_vars > 1 else 0.1,
    )

    # Color palette for chains
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    for idx, item in enumerate(plot_vars):
        var = item[0]
        selector = item[1] if len(item) > 1 else None
        var_label = item[2] if len(item) > 2 else item[0]
        row = idx + 1

        # Get the data
        var_data = posterior[var]
        if selector:
            var_data = var_data.sel(selector)

        # Extract chains
        n_chains = var_data.sizes["chain"]
        n_draws = var_data.sizes["draw"]

        # Plot distribution (left column)
        for chain in range(n_chains):
            chain_data = var_data.isel(chain=chain).values

            # Create KDE
            kde = stats.gaussian_kde(chain_data)
            x_range = np.linspace(chain_data.min(), chain_data.max(), 200)
            y_kde = kde(x_range)

            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_kde,
                    mode="lines",
                    name=f"Chain {chain}",
                    line=dict(color=colors[chain % len(colors)]),
                    showlegend=(idx == 0),  # Only show legend for first variable
                    legendgroup=f"chain{chain}",
                ),
                row=row,
                col=1,
            )

        # Plot trace (right column)
        for chain in range(n_chains):
            chain_data = var_data.isel(chain=chain).values

            fig.add_trace(
                go.Scatter(
                    x=np.arange(n_draws),
                    y=chain_data,
                    mode="lines",
                    name=f"Chain {chain}",
                    line=dict(color=colors[chain % len(colors)], width=1),
                    showlegend=False,
                    legendgroup=f"chain{chain}",
                ),
                row=row,
                col=2,
            )

        # Update axes labels
        fig.update_xaxes(title_text="Value", row=row, col=1)
        fig.update_xaxes(title_text="Draw", row=row, col=2)
        fig.update_yaxes(title_text="Density", row=row, col=1)
        fig.update_yaxes(title_text="Value", row=row, col=2)

    # Update layout
    height = max(500, n_vars * 350)
    fig.update_layout(
        height=height,
        title_text="Trace Plot",
        title_x=0.5,
        title_font_size=20,
        hovermode="closest",
        template="plotly_white",
    )

    return fig


# ============================================================================
# Generate all figures (deferred until model is loaded)
# ============================================================================

# These will be generated after model is loaded via callback
fig_posterior = None
fig_decomposition = None
fig_channels = None
fig_roas = None
fig_saturation = None


# ============================================================================
# Dash App Layout
# ============================================================================

# Initialize Dash app
app = Dash(__name__)

# Define app layout
app.layout = dmc.MantineProvider(
    theme={"colorScheme": "light", "primaryColor": "blue"},
    children=[
        # Store component to track model loading state
        dcc.Store(id="model-loaded-flag", data=False),
        dcc.Store(id="file-names-store", data={}),
        # Input Modal
        dmc.Modal(
            id="input-modal",
            opened=True,
            closeOnClickOutside=False,
            closeOnEscape=False,
            withCloseButton=False,
            title="Model Configuration",
            size="lg",
            children=[
                # Input Form (shown initially)
                html.Div(
                    id="input-form-container",
                    children=[
                        dmc.Stack(
                            gap="md",
                            children=[
                                dmc.Text(
                                    "Enter the file names (without extensions) for your model data. You can change these settings anytime by clicking the settings button (⚙️) in the header.",
                                    size="sm",
                                    c="dimmed",
                                ),
                                dmc.TextInput(
                                    id="x-file-input",
                                    label="X Data File Name",
                                    placeholder="x_data",
                                    description="CSV file containing feature data (exclude .csv extension)",
                                    required=True,
                                ),
                                dmc.TextInput(
                                    id="y-file-input",
                                    label="Y Data File Name",
                                    placeholder="y_data",
                                    description="CSV file containing target data (exclude .csv extension)",
                                    required=True,
                                ),
                                dmc.TextInput(
                                    id="yaml-file-input",
                                    label="YAML Config File Name",
                                    placeholder="builder",
                                    description="YAML configuration file (exclude .yml extension)",
                                    required=True,
                                ),
                                dmc.Alert(
                                    id="file-error-message",
                                    title="Error",
                                    color="red",
                                    children="",
                                    style={"display": "none"},
                                ),
                                dmc.Button(
                                    "Load Model",
                                    id="submit-files-button",
                                    fullWidth=True,
                                    color="blue",
                                    size="md",
                                ),
                            ],
                        )
                    ],
                ),
                # Loading Stepper (shown after validation)
                html.Div(
                    id="loading-stepper-container",
                    style={"display": "none", "position": "relative"},
                    children=[
                        dmc.Stepper(
                            id="loading-stepper",
                            active=0,
                            children=[
                                dmc.StepperStep(
                                    label="Set Input", description="Files validated"
                                ),
                                dmc.StepperStep(
                                    label="Loading Model",
                                    description="Building and sampling...",
                                ),
                                dmc.StepperStep(
                                    label="Complete", description="Model ready!"
                                ),
                            ],
                        ),
                        dmc.LoadingOverlay(
                            id="loading-overlay",
                            visible=False,
                            loaderProps={"type": "bars"},
                        ),
                    ],
                ),
            ],
        ),
        dmc.Container(
            fluid=True,
            style={"backgroundColor": "white", "minHeight": "100vh", "padding": "20px"},
            children=[
                # Header with Settings Button
                dmc.Group(
                    justify="space-between",
                    align="center",
                    style={"marginBottom": "30px"},
                    children=[
                        dmc.Title(
                            "Bayesian Media Mix Model Dashboard",
                            order=1,
                            style={
                                "color": "#1971c2",
                                "flex": 1,
                                "textAlign": "center",
                            },
                        ),
                        dmc.Tooltip(
                            label="Configure Model Settings",
                            position="bottom",
                            children=[
                                dmc.ActionIcon(
                                    id="settings-button",
                                    children=html.Span("⚙️", style={"fontSize": "16px"}),
                                    size="md",
                                    variant="subtle",
                                    color="gray",
                                    style={"position": "absolute", "right": "20px"},
                                )
                            ],
                        ),
                    ],
                ),
                # Divider line after header
                dmc.Divider(style={"marginBottom": "30px"}),
                # Tabs
                dmc.Tabs(
                    orientation="vertical",
                    value="figures",
                    children=[
                        dmc.TabsList(
                            [
                                dmc.TabsTab("🔍 Diagnostics", value="diagnostics"),
                                dmc.TabsTab("📊 Figures", value="figures"),
                                dmc.TabsTab("📋 Tabular Tables", value="tables"),
                            ]
                        ),
                        # Diagnostics Tab
                        dmc.TabsPanel(
                            value="diagnostics",
                            children=[
                                dmc.Container(
                                    fluid=True,
                                    style={"padding": "40px"},
                                    children=[
                                        # Title
                                        dmc.Title(
                                            "Model Diagnostics",
                                            order=2,
                                            style={
                                                "color": "#1971c2",
                                                "marginBottom": "20px",
                                            },
                                        ),
                                        # Description
                                        dmc.Text(
                                            "This section helps you assess the quality of your model's parameter estimates. "
                                            "The table below shows statistical summaries for key model parameters. "
                                            "By default, we display saturation and adstock parameters, which control how marketing effects "
                                            "decay over time and saturate with increased spending.",
                                            size="md",
                                            style={
                                                "marginBottom": "30px",
                                                "color": "#495057",
                                                "lineHeight": "1.6",
                                            },
                                        ),
                                        # What to look for
                                        dmc.Alert(
                                            title="What to Check:",
                                            color="green",
                                            radius="lg",
                                            variant="light",
                                            style={"marginBottom": "30px"},
                                            children=[
                                                dmc.List(
                                                    [
                                                        dmc.ListItem(
                                                            [
                                                                dmc.Text(
                                                                    [
                                                                        html.Strong(
                                                                            "r_hat"
                                                                        ),
                                                                        ": Should be close to 1.0 (ideally < 1.01). Values above 1.05 suggest the model hasn't converged properly.",
                                                                    ]
                                                                )
                                                            ]
                                                        ),
                                                        dmc.ListItem(
                                                            [
                                                                dmc.Text(
                                                                    [
                                                                        html.Strong(
                                                                            "ess_bulk & ess_tail"
                                                                        ),
                                                                        ": Effective sample size. Higher is better (ideally > 400). Low values mean less reliable estimates.",
                                                                    ]
                                                                )
                                                            ]
                                                        ),
                                                        dmc.ListItem(
                                                            [
                                                                dmc.Text(
                                                                    [
                                                                        html.Strong(
                                                                            "mean & sd"
                                                                        ),
                                                                        ": The average parameter value and its uncertainty. Large sd relative to mean indicates high uncertainty.",
                                                                    ]
                                                                )
                                                            ]
                                                        ),
                                                        dmc.ListItem(
                                                            [
                                                                dmc.Text(
                                                                    [
                                                                        html.Strong(
                                                                            "hdi_3% & hdi_97%"
                                                                        ),
                                                                        ": The 94% credible interval. This range contains the true value with 94% probability.",
                                                                    ]
                                                                )
                                                            ]
                                                        ),
                                                    ],
                                                    size="sm",
                                                    spacing="xs",
                                                )
                                            ],
                                        ),
                                        # Variable selector
                                        dmc.Stack(
                                            gap="md",
                                            children=[
                                                dmc.Title(
                                                    "Select Variables",
                                                    order=4,
                                                    style={"color": "#1971c2"},
                                                ),
                                                dmc.MultiSelect(
                                                    id="diagnostic-var-selector",
                                                    label="Choose posterior variables to display",
                                                    description="Select one or more variables to see their diagnostic statistics",
                                                    placeholder="Select variables...",
                                                    data=[],
                                                    value=[],
                                                    searchable=True,
                                                    clearable=True,
                                                    style={"marginBottom": "20px"},
                                                ),
                                            ],
                                        ),
                                        # Summary table (will be updated by callback)
                                        dmc.Stack(
                                            gap="md",
                                            style={"marginTop": "30px"},
                                            children=[
                                                dmc.Title(
                                                    "Summary Statistics",
                                                    order=4,
                                                    style={"color": "#1971c2"},
                                                ),
                                                html.Div(id="diagnostic-summary-table"),
                                            ],
                                        ),
                                        # Trace plot (will be updated by callback)
                                        dmc.Stack(
                                            gap="md",
                                            style={"marginTop": "40px"},
                                            children=[
                                                dmc.Title(
                                                    "Trace Plots",
                                                    order=4,
                                                    style={"color": "#1971c2"},
                                                ),
                                                dmc.Text(
                                                    "Trace plots help diagnose whether the MCMC chains have converged properly. "
                                                    "The left panels show the posterior distribution for each parameter (should be smooth and unimodal). "
                                                    "The right panels show how parameter values evolved during sampling. "
                                                    "Look for chains that overlap well (good mixing) and appear stationary without trends.",
                                                    size="sm",
                                                    style={
                                                        "marginBottom": "15px",
                                                        "color": "#495057",
                                                    },
                                                ),
                                                dmc.Paper(
                                                    shadow="sm",
                                                    p="md",
                                                    children=[
                                                        html.Div(
                                                            id="diagnostic-trace-plot"
                                                        )
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                )
                            ],
                        ),
                        # Figures Tab
                        dmc.TabsPanel(
                            value="figures",
                            children=[
                                dmc.Container(
                                    fluid=True,
                                    style={"padding": "20px"},
                                    children=[
                                        dmc.Title(
                                            "Model Visualizations",
                                            order=2,
                                            style={
                                                "color": "#1971c2",
                                                "marginBottom": "30px",
                                            },
                                        ),
                                        dmc.Text(
                                            "These plots show the model's predictions and decompositions."
                                            "Use them to understand how the model is performing and how each component contributes to the outcome."
                                            "Look for the relative importance of each component or specific marketing channels.",
                                            size="sm",
                                            style={
                                                "marginBottom": "15px",
                                                "color": "#495057",
                                                "minHeight": "60px",
                                            },
                                        ),
                                        # Row 1: Posterior Predictive and Decomposition
                                        dmc.Grid(
                                            gutter="lg",
                                            style={"marginBottom": "40px"},
                                            children=[
                                                # Posterior Predictive
                                                dmc.GridCol(
                                                    span=6,
                                                    children=[
                                                        dmc.Title(
                                                            "Posterior Predictive",
                                                            order=4,
                                                            style={
                                                                "color": "#1971c2",
                                                                "marginBottom": "10px",
                                                            },
                                                        ),
                                                        dmc.Text(
                                                            "This plot shows the model's predictions (mean and uncertainty intervals) compared to the actual observed data. "
                                                            "Check if the observed data points (black dots) fall within the uncertainty bands. "
                                                            "Good fit means most observations are within the 94% HDI.",
                                                            size="sm",
                                                            style={
                                                                "marginBottom": "15px",
                                                                "color": "#495057",
                                                                "minHeight": "60px",
                                                            },
                                                        ),
                                                        dmc.Paper(
                                                            shadow="sm",
                                                            p="md",
                                                            children=[
                                                                dcc.Graph(
                                                                    id="graph-posterior",
                                                                    style={
                                                                        "height": "600px"
                                                                    },
                                                                )
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                                # Decomposition
                                                dmc.GridCol(
                                                    span=6,
                                                    children=[
                                                        dmc.Title(
                                                            "Model Decomposition",
                                                            order=4,
                                                            style={
                                                                "color": "#1971c2",
                                                                "marginBottom": "10px",
                                                            },
                                                        ),
                                                        dmc.Text(
                                                            "This breaks down the total effect into components: channels, controls, trend, and seasonality. "
                                                            "Use this to understand which factors drive your outcomes over time. "
                                                            "Look for the relative importance of each component.",
                                                            size="sm",
                                                            style={
                                                                "marginBottom": "15px",
                                                                "color": "#495057",
                                                                "minHeight": "60px",
                                                            },
                                                        ),
                                                        dmc.Paper(
                                                            shadow="sm",
                                                            p="md",
                                                            children=[
                                                                dcc.Graph(
                                                                    id="graph-decomposition",
                                                                    style={
                                                                        "height": "600px"
                                                                    },
                                                                )
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        # Row 2: Channel Contributions and ROAS Distribution
                                        dmc.Grid(
                                            gutter="lg",
                                            style={"marginBottom": "40px"},
                                            children=[
                                                # Channel Contributions
                                                dmc.GridCol(
                                                    span=6,
                                                    children=[
                                                        dmc.Title(
                                                            "Channel Contributions Over Time",
                                                            order=4,
                                                            style={
                                                                "color": "#1971c2",
                                                                "marginBottom": "10px",
                                                            },
                                                        ),
                                                        dmc.Text(
                                                            "Individual contribution of each marketing channel over time. "
                                                            "This helps identify when each channel was most effective and how their impact varies. "
                                                            "Compare channels to see which ones drive more value.",
                                                            size="sm",
                                                            style={
                                                                "marginBottom": "15px",
                                                                "color": "#495057",
                                                                "minHeight": "60px",
                                                            },
                                                        ),
                                                        dmc.Paper(
                                                            shadow="sm",
                                                            p="md",
                                                            children=[
                                                                dcc.Graph(
                                                                    id="graph-channels",
                                                                    style={
                                                                        "height": "900px"
                                                                    },
                                                                )
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                                # ROAS Distribution
                                                dmc.GridCol(
                                                    span=6,
                                                    children=[
                                                        dmc.Title(
                                                            "Return on Input Distribution",
                                                            order=4,
                                                            style={
                                                                "color": "#1971c2",
                                                                "marginBottom": "10px",
                                                            },
                                                        ),
                                                        dmc.Text(
                                                            "Distribution of return on ad spend (ROAS) for each channel. "
                                                            "The black dashed line shows the average, and red lines mark the 94% credible interval. "
                                                            "Higher and narrower distributions indicate better and more certain returns.",
                                                            size="sm",
                                                            style={
                                                                "marginBottom": "15px",
                                                                "color": "#495057",
                                                                "minHeight": "60px",
                                                            },
                                                        ),
                                                        dmc.Paper(
                                                            shadow="sm",
                                                            p="md",
                                                            children=[
                                                                dcc.Graph(
                                                                    id="graph-roas",
                                                                    style={
                                                                        "height": "900px"
                                                                    },
                                                                )
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        # Row 3: Saturation Curves (full width)
                                        dmc.Stack(
                                            gap="md",
                                            style={"marginBottom": "40px"},
                                            children=[
                                                dmc.Title(
                                                    "Saturation Curves",
                                                    order=4,
                                                    style={
                                                        "color": "#1971c2",
                                                        "marginBottom": "10px",
                                                    },
                                                ),
                                                dmc.Text(
                                                    "These curves show how each channel's effectiveness changes with spending. "
                                                    "The curve flattens as you spend more (diminishing returns). "
                                                    "Dots represent actual data points. Use this to identify optimal spending levels.",
                                                    size="sm",
                                                    style={
                                                        "marginBottom": "15px",
                                                        "color": "#495057",
                                                        "minHeight": "60px",
                                                    },
                                                ),
                                                dmc.Paper(
                                                    shadow="sm",
                                                    p="md",
                                                    children=[
                                                        dcc.Graph(
                                                            id="graph-saturation",
                                                            style={"height": "800px"},
                                                        )
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                )
                            ],
                        ),
                        # Tables Tab
                        dmc.TabsPanel(
                            value="tables",
                            children=[
                                dmc.Container(
                                    fluid=True,
                                    style={"padding": "40px"},
                                    children=[
                                        # Main title
                                        dmc.Title(
                                            "Tabular Tables",
                                            order=2,
                                            style={
                                                "color": "#1971c2",
                                                "marginBottom": "20px",
                                            },
                                        ),
                                        # Introduction
                                        dmc.Text(
                                            "This section provides tabular views of model component contributions over time. "
                                            "You can control how many rows are displayed and download the complete datasets as CSV files for further analysis.",
                                            size="md",
                                            style={
                                                "marginBottom": "30px",
                                                "color": "#495057",
                                                "lineHeight": "1.6",
                                            },
                                        ),
                                        # Control inputs
                                        dmc.Grid(
                                            gutter="lg",
                                            style={"marginBottom": "30px"},
                                            children=[
                                                dmc.GridCol(
                                                    span=6,
                                                    children=[
                                                        dmc.NumberInput(
                                                            id="table-row-limit",
                                                            label="Number of Rows to Display",
                                                            description="Set how many rows to show in the tables below",
                                                            value=10,
                                                            min=1,
                                                            max=100,
                                                            step=1,
                                                            style={"width": "100%"},
                                                        )
                                                    ],
                                                ),
                                                dmc.GridCol(
                                                    span=6,
                                                    children=[
                                                        dmc.NumberInput(
                                                            id="table-quantile-value",
                                                            label="Quantile Value",
                                                            description="Set the quantile for the quantile contribution table (0.5 = median)",
                                                            value=0.5,
                                                            min=0.01,
                                                            max=0.99,
                                                            step=0.01,
                                                            decimalScale=2,
                                                            style={"width": "100%"},
                                                        )
                                                    ],
                                                ),
                                            ],
                                        ),
                                        # Mean Contributions Table Section
                                        dmc.Stack(
                                            gap="md",
                                            style={"marginBottom": "40px"},
                                            children=[
                                                dmc.Title(
                                                    "Mean Contribution by Component Over Time",
                                                    order=4,
                                                    style={"color": "#1971c2"},
                                                ),
                                                dmc.Text(
                                                    "This table shows the mean (average) contribution of each model component "
                                                    "(control variables, trend, seasonality, intercept, and media channels) across all posterior samples. "
                                                    "Each row represents a time point (date), and each column shows the expected contribution of that component "
                                                    "to the target variable in the original scale. Higher values indicate stronger positive effects on the outcome.",
                                                    size="sm",
                                                    style={
                                                        "marginBottom": "15px",
                                                        "color": "#495057",
                                                        "lineHeight": "1.6",
                                                    },
                                                ),
                                                dmc.Paper(
                                                    shadow="sm",
                                                    p="md",
                                                    children=[
                                                        html.Div(
                                                            id="mean-contributions-table"
                                                        )
                                                    ],
                                                ),
                                                dmc.Button(
                                                    "Download Complete Mean Contributions (CSV)",
                                                    id="download-mean-button",
                                                    variant="light",
                                                    color="blue",
                                                    style={"marginTop": "10px"},
                                                ),
                                                dcc.Download(id="download-mean-csv"),
                                            ],
                                        ),
                                        # Quantile Contributions Table Section
                                        dmc.Stack(
                                            gap="md",
                                            style={"marginBottom": "40px"},
                                            children=[
                                                dmc.Title(
                                                    "Quantile Contribution by Component Over Time",
                                                    order=4,
                                                    style={"color": "#1971c2"},
                                                ),
                                                dmc.Text(
                                                    "This table shows a specific quantile of the contribution distribution for each model component. "
                                                    "The quantile value can be adjusted using the input above. For example, a quantile of 0.5 shows the median, "
                                                    "0.05 shows the 5th percentile (lower bound of uncertainty), and 0.95 shows the 95th percentile (upper bound). "
                                                    "This helps understand the uncertainty in component contributions and identify best/worst case scenarios.",
                                                    size="sm",
                                                    style={
                                                        "marginBottom": "15px",
                                                        "color": "#495057",
                                                        "lineHeight": "1.6",
                                                    },
                                                ),
                                                dmc.Paper(
                                                    shadow="sm",
                                                    p="md",
                                                    children=[
                                                        html.Div(
                                                            id="quantile-contributions-table"
                                                        )
                                                    ],
                                                ),
                                                dmc.Button(
                                                    "Download Complete Quantile Contributions (CSV)",
                                                    id="download-quantile-button",
                                                    variant="light",
                                                    color="blue",
                                                    style={"marginTop": "10px"},
                                                ),
                                                dcc.Download(
                                                    id="download-quantile-csv"
                                                ),
                                            ],
                                        ),
                                    ],
                                )
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ============================================================================
# Callbacks
# ============================================================================


@app.callback(
    Output("input-modal", "opened", allow_duplicate=True),
    Output("input-form-container", "style", allow_duplicate=True),
    Output("loading-stepper-container", "style", allow_duplicate=True),
    Output("loading-stepper", "active", allow_duplicate=True),
    Output("file-error-message", "children", allow_duplicate=True),
    Output("file-error-message", "style", allow_duplicate=True),
    Input("settings-button", "n_clicks"),
    prevent_initial_call=True,
)
def open_settings_modal(n_clicks):
    """Open the configuration modal when settings button is clicked."""
    if n_clicks:
        # Reset modal to initial state (show input form, hide stepper)
        return True, {}, {"display": "none"}, 0, "", {"display": "none"}
    return False, {}, {"display": "none"}, 0, "", {"display": "none"}


@app.callback(
    Output("file-error-message", "children"),
    Output("file-error-message", "style"),
    Output("input-form-container", "style"),
    Output("loading-stepper-container", "style"),
    Output("loading-stepper", "active"),
    Output("file-names-store", "data"),
    Input("submit-files-button", "n_clicks"),
    State("x-file-input", "value"),
    State("y-file-input", "value"),
    State("yaml-file-input", "value"),
    prevent_initial_call=True,
)
def validate_files(n_clicks, x_file, y_file, yaml_file):
    """Validate that input files exist before loading."""
    if not n_clicks:
        return "", {"display": "none"}, {}, {"display": "none"}, 0, {}

    # Check if all inputs are provided
    if not x_file or not y_file or not yaml_file:
        error_msg = "Please fill in all file names."
        return error_msg, {"display": "block"}, {}, {"display": "none"}, 0, {}

    # Check if files exist
    files_to_check = [
        (f"files/{x_file}.csv", f"X data file '{x_file}.csv'"),
        (f"files/{y_file}.csv", f"Y data file '{y_file}.csv'"),
        (f"files/{yaml_file}.yml", f"YAML config file '{yaml_file}.yml'"),
    ]

    for file_path, file_desc in files_to_check:
        if not os.path.exists(file_path):
            error_msg = f"{file_desc} not found in files/ directory."
            return error_msg, {"display": "block"}, {}, {"display": "none"}, 0, {}

    # All files exist - hide input form, show stepper
    file_names = {"x": x_file, "y": y_file, "yaml": yaml_file}
    return (
        "",
        {"display": "none"},
        {"display": "none"},
        {"display": "block"},
        1,
        file_names,
    )


@app.callback(
    Output("loading-overlay", "visible"),
    Output("loading-stepper", "active", allow_duplicate=True),
    Output("input-modal", "opened"),
    Output("model-loaded-flag", "data"),
    Input("file-names-store", "data"),
    prevent_initial_call=True,
)
def load_model(file_names):
    """Load the model after files are validated."""
    global mmm, y, all_posterior_vars, default_diagnostic_vars
    global fig_posterior, fig_decomposition, fig_channels, fig_roas, fig_saturation

    if not file_names or not file_names.get("x"):
        return False, 1, True, False

    try:
        # Show loading overlay
        # Load model and data
        mmm, y = load_model_and_data(
            x_filename=file_names["x"],
            y_filename=file_names["y"],
            yaml_filename=file_names["yaml"],
        )

        # Extract posterior variable names
        all_posterior_vars = list(mmm.idata.posterior.data_vars.keys())

        # Default variables: those starting with saturation_ or adstock_
        default_diagnostic_vars = [
            var
            for var in all_posterior_vars
            if var.startswith("saturation_") or var.startswith("adstock_")
        ]

        # Generate all figures
        print("Generating visualizations...")
        fig_posterior = create_posterior_predictive_plot()
        fig_decomposition = create_decomposition_plot()
        fig_channels = create_channel_contributions_plot()
        fig_roas = create_roas_distribution_plot()
        fig_saturation = create_saturation_curves_plot()
        print("Visualizations created!")

        # Step 2: Complete
        return False, 2, False, True

    except Exception as e:
        print(f"Error loading model: {e}")
        # Keep modal open on error
        return False, 1, True, False


@app.callback(
    Output("graph-posterior", "figure"),
    Output("graph-decomposition", "figure"),
    Output("graph-channels", "figure"),
    Output("graph-roas", "figure"),
    Output("graph-saturation", "figure"),
    Output("diagnostic-var-selector", "data"),
    Output("diagnostic-var-selector", "value"),
    Input("model-loaded-flag", "data"),
)
def update_graphs_after_loading(model_loaded):
    """Update all graph figures after model is loaded."""
    if not model_loaded or mmm is None:
        # Return empty figures
        empty_fig = go.Figure()
        empty_fig.update_layout(
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                {
                    "text": "Model not loaded yet",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 20},
                }
            ],
        )
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, [], []

    # Return the global figures and update variable selector
    var_options = [{"value": var, "label": var} for var in all_posterior_vars]
    return (
        fig_posterior,
        fig_decomposition,
        fig_channels,
        fig_roas,
        fig_saturation,
        var_options,
        default_diagnostic_vars,
    )


@app.callback(
    Output("diagnostic-summary-table", "children"),
    Output("diagnostic-trace-plot", "children"),
    Input("diagnostic-var-selector", "value"),
    State("model-loaded-flag", "data"),
)
def update_diagnostic_visualizations(selected_vars, model_loaded):
    """Update the diagnostic summary table and trace plot based on selected variables."""
    if not model_loaded or mmm is None:
        return dmc.Text("Model not loaded yet", c="dimmed", ta="center"), dmc.Text(
            "Model not loaded yet", c="dimmed", ta="center"
        )

    # Create summary table
    table = create_summary_table(selected_vars)

    # Create trace plot
    if not selected_vars:
        trace_plot = dmc.Text("No variables selected", c="dimmed", ta="center")
    else:
        fig = plot_trace_plotly(mmm.idata, var_names=selected_vars)
        trace_plot = dcc.Graph(
            figure=fig, style={"height": f"{max(500, len(selected_vars) * 350)}px"}
        )

    return table, trace_plot


@app.callback(
    Output("mean-contributions-table", "children"),
    Output("quantile-contributions-table", "children"),
    Input("table-row-limit", "value"),
    Input("table-quantile-value", "value"),
    State("model-loaded-flag", "data"),
)
def update_contributions_tables(row_limit, quantile, model_loaded):
    """Update both contributions tables based on user inputs."""
    if not model_loaded or mmm is None:
        placeholder = dmc.Text("Model not loaded yet", c="dimmed", ta="center")
        return placeholder, placeholder

    # Create contributions dataset
    contributions = create_contributions_dataset()

    # Create mean contributions DataFrame
    mean_df = (
        contributions.stack(sample=["chain", "draw"])
        .mean(dim="sample")
        .to_pandas()
        .T.reset_index()
    )
    mean_table = create_contributions_table(mean_df, row_limit)

    # Create quantile contributions DataFrame
    quantile_df = (
        contributions.stack(sample=["chain", "draw"])
        .quantile([quantile], dim="sample")
        .isel(quantile=0)
        .to_pandas()
        .T.reset_index()
    )
    quantile_table = create_contributions_table(quantile_df, row_limit)

    return mean_table, quantile_table


@app.callback(
    Output("download-mean-csv", "data"),
    Input("download-mean-button", "n_clicks"),
    State("model-loaded-flag", "data"),
    prevent_initial_call=True,
)
def download_mean_contributions(n_clicks, model_loaded):
    """Download the complete mean contributions table as CSV."""
    if not model_loaded or mmm is None:
        return None

    # Create contributions dataset
    contributions = create_contributions_dataset()

    # Create mean contributions DataFrame
    mean_df = (
        contributions.stack(sample=["chain", "draw"])
        .mean(dim="sample")
        .to_pandas()
        .T.reset_index()
    )

    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mean_contributions_{timestamp}.csv"

    return dcc.send_data_frame(mean_df.to_csv, filename, index=False)


@app.callback(
    Output("download-quantile-csv", "data"),
    Input("download-quantile-button", "n_clicks"),
    State("table-quantile-value", "value"),
    State("model-loaded-flag", "data"),
    prevent_initial_call=True,
)
def download_quantile_contributions(n_clicks, quantile, model_loaded):
    """Download the complete quantile contributions table as CSV."""
    if not model_loaded or mmm is None:
        return None

    # Create contributions dataset
    contributions = create_contributions_dataset()

    # Create quantile contributions DataFrame
    quantile_df = (
        contributions.stack(sample=["chain", "draw"])
        .quantile([quantile], dim="sample")
        .isel(quantile=0)
        .to_pandas()
        .T.reset_index()
    )

    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"quantile_{quantile}_contributions_{timestamp}.csv"

    return dcc.send_data_frame(quantile_df.to_csv, filename, index=False)


# ============================================================================
# Run App
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Starting MMM Dashboard...")
    print("Open your browser and navigate to: http://127.0.0.1:8050")
    print("=" * 60 + "\n")
    app.run(debug=True, use_reloader=False)
