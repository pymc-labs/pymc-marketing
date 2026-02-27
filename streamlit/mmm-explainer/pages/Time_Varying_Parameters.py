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
"""Streamlit page for HSGP."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import streamlit as st
from pymc_marketing.mmm import HSGP

# Constants
PLOT_HEIGHT = 500
PLOT_WIDTH = 1000
SEED = sum(map(ord, "Out of the box GP"))
RNG = np.random.default_rng(SEED)

# -------------------------- TOP OF PAGE INFORMATION -------------------------

# Set browser / tab config
st.set_page_config(
    page_title="MMM App - HSGP",
    page_icon="🧊",
)

# Give some context for what the page displays
st.title("Time-Varying Parameters")
# TODO: Update this !
st.markdown(
    "In real-world scenarios, the effectiveness of marketing activities is not \
        static but varies over time due to factors like competitive actions, \
        and market dynamics. To account for this, we introduce a time-dependent \
        component into the MMM framework using a Gaussian Process, specifically a \
        [Hilbert Space GP](https://www.pymc.io/projects/docs/en/stable/api/gp/generated/pymc.gp.HSGP.html). \
        This allows us to capture the hidden latent temporal variation of the \
        marketing contributions. \
    "
)

st.markdown("""
    When `time_media_varying` is set to `True`, we capture a single latent \
        process that multiplies all channels. We assume all channels \
        share the same time-dependent fluctuations, contrasting with \
        implementations where each channel has an independent latent \
        process. The modified model can be represented as:
""")

tvp_media_formula = r"""
    y_{t} = \alpha + \lambda_{t} \cdot \sum_{m=1}^{M}\beta_{m}f(x_{m, t}) \ +
    \sum_{c=1}^{C}\gamma_{c}z_{c, t} + \varepsilon_{t},
"""
st.latex(tvp_media_formula)

st.markdown("""
    **Where:**

    $\\lambda_{t}$ is the time-varying component modeled as a latent process. This shared time-dependent \
        variation $\\lambda_{t}$ allows us to capture the overall temporal effects that influence all \
        media channels simultaneously.
""")

# Generate some data for the example
n = 52
X = np.arange(n)

st.divider()

# User inputs
st.subheader(":blue[User Inputs]")

# Sliders for params
ls = st.slider("Lengthscale (ls)", min_value=1, max_value=100, value=25, step=1)
eta = st.slider("Variance (eta)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
m = st.slider(
    "The number of basis vectors (m)", min_value=50, max_value=500, value=200, step=10
)
L = st.slider("Boundary condition (L)", min_value=50, max_value=500, value=150, step=10)

# Fixed parameters
dims = "time"
drop_first = False

# Collect kwargs
kwargs = dict(X=X, ls=ls, eta=eta, dims=dims, m=m, L=L, drop_first=drop_first)

hsgp = HSGP(**kwargs)

dates = pd.date_range("2022-01-01", periods=n, freq="W-MON")
coords = {"time": dates}


def sample_curve(hsgp):
    """Use to sample HSGP."""
    return hsgp.sample_prior(coords=coords, random_seed=RNG)["f"]


curve = sample_curve(hsgp).rename("False")
curve = curve.squeeze("chain")  # drop chain=1
time = curve.coords["time"].values

# Compute posterior mean and credible interval
mean_vals = curve.mean("draw")

fig = go.Figure()

# Mean line
fig.add_trace(
    go.Scatter(
        x=time, y=mean_vals.values, mode="lines", line=dict(color="blue"), name="Mean"
    )
)

fig.update_layout(
    title="Time-Dependent Variation",
    xaxis_title="Time",
    yaxis_title="Value",
    template="plotly_white",
)

st.plotly_chart(fig, use_container_width=True)
