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
"""Streamlit page for fourier modes."""

import plotly.graph_objects as go
from pymc_extras.prior import Prior

import streamlit as st
from pymc_marketing.mmm import MonthlyFourier, YearlyFourier

# Constants
PLOT_HEIGHT = 500
PLOT_WIDTH = 1000

# -------------------------- TOP OF PAGE INFORMATION -------------------------

# Set browser / tab config
st.set_page_config(
    page_title="MMM App - Fourier Modes",
    page_icon="🧊",
)

# Give some context for what the page displays
st.title("Fourier Modes")

st.markdown(
    "This page demonstrates Fourier seasonality transformations for use \
        in MMM. Fourier seasonality relies on sine and cosine \
        functions to capture recurring patterns in the data, making it useful \
        for modeling periodic trends."
)

st.markdown("___The Fourier component takes the form:___")

# LaTeX string for Fourier seasonal component
fourier_formula = r"""
f(t) = \sum_{k=1}^{K} \Bigg[ a_k \cos\Big(\frac{2 \pi k t}{T}\Big)
      + b_k \sin\Big(\frac{2 \pi k t}{T}\Big) \Bigg]
"""
st.latex(fourier_formula)

st.markdown("""
**Where:**

- $t$ = time index (e.g., day, week, month)
- $T$ = period of the seasonality (e.g., 12 for monthly, 365 for yearly)
- $K$ = order of the Fourier series (number of sine/cosine pairs)
- $a_k, b_k$ = Fourier coefficients
""")

st.markdown(
    "🗒️ **Note:** \n \
- Yearly Fourier: A yearly seasonality with a period ($T$) of **_:red[365.25 days]_** \n \
- Monthly Fourier: A monthly seasonality with a period ($T$) of **_:red[365.25 / 12 days]_**"
)

st.divider()

# User inputs
st.subheader(":orange[User Inputs]")
# Slider for selecting the order
n_order = st.slider(
    "Fourier order $K$ (n_order)", min_value=1, max_value=20, value=6, step=1
)
# Slider for selecting the scale param
b = st.slider(
    "Laplace scale (__b__)", min_value=0.01, max_value=1.0, value=0.1, step=0.01
)

# Setup
prior = Prior("Laplace", mu=0, b=b, dims="fourier")

# Create tabs for plots
tab1, tab2 = st.tabs(["Yearly", "Monthly"])

# -------------------------- YEARLY SEASONALITY -------------------------
with tab1:
    st.subheader(":orange[Yearly Seasonality]")

    fourier = YearlyFourier(n_order=n_order, prior=prior)

    # Displayed in the APP
    parameters = fourier.sample_prior()
    curve = fourier.sample_curve(parameters)
    # Drop chain if it's always 1
    curve = curve.squeeze("chain")
    # Compute mean and quantiles across draws
    mean_trend = curve.mean("draw")
    # Grab the days for the x-axis
    days = curve.coords["day"].values

    # Build Plotly figure
    fig = go.Figure()

    # Mean line
    fig.add_trace(
        go.Scatter(
            x=days,
            y=mean_trend.values,
            mode="lines",
            line=dict(color="blue"),
            name="Mean trend",
        )
    )

    fig.update_layout(
        title="Yearly Fourier Trend",
        xaxis_title="Day",
        yaxis_title="Trend",
        height=PLOT_HEIGHT,
        width=PLOT_WIDTH,
    )

    st.plotly_chart(fig, use_container_width=True)

# -------------------------- MONTHLY SEASONALITY -------------------------
with tab2:
    st.subheader(":orange[Monthly Seasonality]")

    fourier = MonthlyFourier(n_order=n_order, prior=prior)

    # Displayed in the APP
    parameters = fourier.sample_prior()
    curve = fourier.sample_curve(parameters)
    # Drop chain if it's always 1
    curve = curve.squeeze("chain")
    # Compute mean and quantiles across draws
    mean_trend = curve.mean("draw")
    # Grab the days for the x-axis
    days = curve.coords["day"].values

    # Build Plotly figure
    fig = go.Figure()

    # Mean line
    fig.add_trace(
        go.Scatter(
            x=days,
            y=mean_trend.values,
            mode="lines",
            line=dict(color="blue"),
            name="Mean trend",
        )
    )

    fig.update_layout(
        title="Monthly Fourier Trend",
        xaxis_title="Day",
        yaxis_title="Trend",
        height=PLOT_HEIGHT,
        width=PLOT_WIDTH,
    )

    st.plotly_chart(fig, use_container_width=True)
