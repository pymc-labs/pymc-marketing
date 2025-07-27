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
"""Functions for plotting prior distributions."""

# Imports
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import preliz as pz
from scipy.stats import gaussian_kde

import streamlit as st


@st.cache_data  # 👈 Add the caching decorator, make app run faster
def get_distribution(distribution_name=pz.distributions, **params):
    """Retrieve and create a distribution instance from the PreliZ library.

    Parameters
    ----------
    distribution_name (str): The name of the distribution to create.
    **params: Variable length dict of parameters and values required by the distribution.

    Returns
    -------
    object: An instance of the requested distribution.

    """
    try:
        # Get the distribution class from preliz
        dist_class = getattr(pz, distribution_name)
        # Create an instance of the distribution with the provided parameters
        return dist_class(**params)
    except AttributeError:
        raise ValueError(f"Distribution '{distribution_name}' is not found in preliz.")
    except TypeError:
        raise ValueError(
            f"Incorrect parameters for the distribution '{distribution_name}'."
        )


def plot_prior_distribution(
    draws, nbins=100, opacity=0.1, title="Prior Distribution - Visualised"
):
    """Plot samples of a prior distribution as a histogram.

    It uses a KDE (Kernel Density Estimate) overlay and a violin plot along the top too
    with quartile values.

    Parameters
    ----------
    - draws: numpy array of samples from prior distribution.
    - nbins: int, the number of bins for the histogram.
    - opacity: float, the opacity level for the histogram bars.
    - title: str, the title of the plot.

    """
    # Create the histogram using Plotly Express
    fig = px.histogram(
        draws,
        x=draws,
        nbins=nbins,
        title=title,
        labels={"x": "Value"},
        histnorm="probability density",
        opacity=opacity,
        marginal="violin",
        color_discrete_sequence=["#0047AB"],
    )

    # Compute the KDE
    kde = gaussian_kde(draws)
    x_range = np.linspace(min(draws), max(draws), 500)
    kde_values = kde(x_range)

    # Add the KDE plot to the histogram figure
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=kde_values,
            mode="lines",
            name="KDE",
            line_color="#DA70D6",
            opacity=0.8,
        )
    )

    # Customize the layout
    fig.update_layout(xaxis_title="Value of Prior", yaxis_title="Density")

    # Return the plot
    return fig
