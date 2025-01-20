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
"""Streamlit page for visualising priors."""

import prior_functions as pf

import streamlit as st

# Constants
SEED = 42
N_DRAWS = 50_000
# Specify the possible distributions and their paramaters you want to visualise
DISTRIBUTIONS_DICT = {
    "Beta": ["alpha", "beta"],
    "Bernoulli": ["p"],
    "Exponential": ["lam"],
    "Gamma": ["alpha", "beta"],
    "HalfNormal": ["sigma"],
    "LogNormal": ["mu", "sigma"],
    "Normal": ["mu", "sigma"],
    "Poisson": ["mu"],
    "StudentT": ["nu", "mu", "sigma"],
    "TruncatedNormal": ["mu", "sigma", "lower", "upper"],
    "Uniform": ["lower", "upper"],
    "Weibull": ["alpha", "beta"],
}
PLOT_HEIGHT = 500
PLOT_WIDTH = 1000

# -------------------------- TOP OF PAGE INFORMATION -------------------------

# Set browser / tab config
st.set_page_config(
    page_title="MMM App - Prior Distributions Transformations",
    page_icon="💎",
)

# Give some context for what the page displays
st.title("Bayesian Prior Distribution Demonstrator")

# -------------------------- VISUALISE PRIOR -------------------------

# Select the distribution to visualise
dist_name = st.selectbox(
    "Please select the distribution you would like to visualise:",
    options=DISTRIBUTIONS_DICT.keys(),
)
st.header(f":blue[{dist_name} Distribution]")  # header

# Variables need to be instantiated to avoid error where upper < lower
lower = None
upper = None

# Initialize parameters with None
params = {param: None for param in DISTRIBUTIONS_DICT[dist_name]}

# User inputs for distribution parameters
for param in params.keys():
    if param == "lower":
        params[param] = st.number_input(
            f"Please enter the value for {param.title()}:", key=param, value=0.0
        )
    elif param == "upper":
        params[param] = st.number_input(
            f"Please enter the value for {param.title()}:", key=param, value=2.0
        )
    elif param == "alpha":
        params[param] = st.number_input(
            f"Please enter the value for {param.title()}:",
            key=param,
            value=1.0,
            min_value=0.01,
        )
    elif param == "beta":
        params[param] = st.number_input(
            f"Please enter the value for {param.title()}:",
            key=param,
            value=1.0,
            min_value=0.01,
        )
    elif param == "sigma":
        params[param] = st.number_input(
            f"Please enter the value for {param.title()}:",
            key=param,
            value=1.0,
            min_value=0.01,
        )
    # Poisson mu must be > 0
    elif param == "mu" and dist_name == "Poisson":
        params[param] = st.number_input(
            f"Please enter the value for {param.title()}:",
            key=param,
            value=1.0,
            min_value=0.01,
        )
    elif param == "mu":
        params[param] = st.number_input(
            f"Please enter the value for {param.title()}:", key=param, value=0.0
        )
    elif param == "p":
        params[param] = st.number_input(
            f"Please enter the value for {param.title()}:",
            key=param,
            value=0.5,
            min_value=0.0,
            max_value=1.0,
        )
    elif param == "lam":
        params[param] = st.number_input(
            f"Please enter the value for {param.title()}:",
            key=param,
            value=1.0,
            min_value=0.01,
        )
    elif param == "nu":
        params[param] = st.number_input(
            f"Please enter the value for {param.title()}:",
            key=param,
            value=10.0,
            min_value=0.01,
        )


# Check to ensure lower < upper
if lower and lower >= upper:
    st.error("Error: Lower bound must be less than upper bound.")

## Create the selected distribution and sample from it
dist = pf.get_distribution(dist_name, **params)
draws = dist.rvs(N_DRAWS, random_state=SEED)


# Plot distribution
fig_root = pf.plot_prior_distribution(draws, title=f"{dist_name} Distribution Samples")
fig_root.update_layout(height=PLOT_HEIGHT, width=PLOT_WIDTH)
st.plotly_chart(fig_root, use_container_width=True)
