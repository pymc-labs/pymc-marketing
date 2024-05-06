#   Copyright 2024 The PyMC Labs Developers
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
# Import custom functions
import prior_functions as pf
import pymc as pm

import streamlit as st

# -------------------------- TOP OF PAGE INFORMATION -------------------------

# Set browser / tab config
st.set_page_config(
    page_title="MMM App - Prior Distributions Transformations",
    page_icon="💎",
)

# Give some context for what the page displays
st.title("Bayesian Prior Distribution Demonstrator")

# Use tabs for different distributions
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs(
    [
        "🔵 Uniform",
        "🟣 Normal",
        "🟠 HalfNormal",
        "🟢 Beta",
        "🔴 Gamma",
        "⚪ Poisson",
        "🌈 Bernoulli",
        "🔵 Exponential",
        "🟣 Weibull",
        "🟠 TruncatedNormal",
        "🟢 StudentT",
        "🔴 LogNormal",
    ]
)


# Set seed
seed = 42

# -------------------------- UNIFORM DISTRIBUTION -------------------------
with tab1:
    st.header(":blue[Uniform Distribution]")

    # User inputs for distribution parameters
    st.subheader(":blue[User Inputs]")
    lower = st.number_input("Lower Bound", value=0.0)
    upper = st.number_input("Upper Bound", value=1.0)
    # Check to ensure lower < upper
    if lower >= upper:
        st.error("Error: Lower bound must be less than upper bound.")
    else:
        # Draw samples
        samples = pf.draw_samples_from_prior(
            pm.Uniform, lower=lower, upper=upper, seed=seed
        )

        # Plot distribution
        fig_root = pf.plot_prior_distribution(
            samples, title="Uniform Distribution Samples"
        )
        fig_root.update_layout(height=500, width=1000)
        st.plotly_chart(fig_root, use_container_width=True)

# -------------------------- NORMAL DISTRIBUTION -------------------------
with tab2:
    st.header(":violet[Normal Distribution]")

    # User inputs for distribution parameters
    st.subheader(":violet[User Inputs]")
    mu = st.number_input("Mean (mu)", value=0.0)
    sigma = st.number_input("Standard Deviation (sigma)", value=1.0, min_value=0.01)

    # Draw samples
    samples = pf.draw_samples_from_prior(pm.Normal, mu=mu, sigma=sigma, seed=seed)

    # Plot distribution
    fig = pf.plot_prior_distribution(samples, title="Normal Distribution Samples")
    fig.update_layout(height=500, width=1000)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------- HALF-NORMAL DISTRIBUTION -------------------------
with tab3:
    st.header(":orange[HalfNormal Distribution]")

    st.subheader(":orange[User Inputs]")
    sigma = st.number_input(
        "Standard Deviation (sigma) for HalfNormal", value=1.0, min_value=0.01
    )

    # Draw samples
    samples = pf.draw_samples_from_prior(pm.HalfNormal, sigma=sigma, seed=seed)

    # Plot distribution
    fig = pf.plot_prior_distribution(samples, title="HalfNormal Distribution Samples")
    fig.update_layout(height=500, width=1000)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------- BETA DISTRIBUTION -------------------------
with tab4:
    st.header(":green[Beta Distribution]")

    st.subheader(":green[User Inputs]")
    alpha = st.number_input("Alpha", value=1.0, min_value=0.01)
    beta = st.number_input("Beta", value=1.0, min_value=0.01)

    # Draw samples
    samples = pf.draw_samples_from_prior(pm.Beta, alpha=alpha, beta=beta, seed=seed)

    # Plot distribution
    fig = pf.plot_prior_distribution(samples, title="Beta Distribution Samples")
    fig.update_layout(height=500, width=1000)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------- GAMMA DISTRIBUTION -------------------------
with tab5:
    st.header(":red[Gamma Distribution]")

    st.subheader(":red[User Inputs]")
    alpha = st.number_input("Alpha for Gamma", value=1.0, min_value=0.01)
    beta = st.number_input("Beta for Gamma (rate)", value=1.0, min_value=0.01)

    # Draw samples
    samples = pf.draw_samples_from_prior(pm.Gamma, alpha=alpha, beta=beta, seed=seed)

    # Plot distribution
    fig = pf.plot_prior_distribution(samples, title="Gamma Distribution Samples")
    fig.update_layout(height=500, width=1000)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------- POISSON DISTRIBUTION -------------------------
with tab6:
    st.header(":grey[Poisson Distribution]")

    st.subheader(":grey[User Inputs]")
    mu = st.number_input("Lambda (mu)", value=1.0, min_value=0.01)

    # Draw samples
    samples = pf.draw_samples_from_prior(pm.Poisson, mu=mu, seed=seed)

    # Plot distribution
    fig = pf.plot_prior_distribution(samples, title="Poisson Distribution Samples")
    fig.update_layout(height=500, width=1000)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------- BERNOULLI DISTRIBUTION -------------------------
with tab7:
    st.header(":rainbow[Bernoulli Distribution]")

    st.subheader(":rainbow[User Inputs]")
    p = st.number_input(
        "Probability of Success (p)", value=0.5, min_value=0.0, max_value=1.0
    )

    # Draw samples
    samples = pf.draw_samples_from_prior(pm.Bernoulli, p=p, seed=seed)

    # Plot distribution
    fig = pf.plot_prior_distribution(samples, title="Bernoulli Distribution Samples")
    fig.update_layout(height=500, width=1000)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------- EXPONENTIAL DISTRIBUTION -------------------------
with tab8:
    st.header(":blue[Exponential Distribution]")

    st.subheader(":blue[User Inputs]")
    lam = st.number_input("Rate (lambda)", value=1.0, min_value=0.01)

    # Draw samples
    samples = pf.draw_samples_from_prior(pm.Exponential, lam=lam, seed=seed)

    # Plot distribution
    fig = pf.plot_prior_distribution(samples, title="Exponential Distribution Samples")
    fig.update_layout(height=500, width=1000)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------- WEIBULL DISTRIBUTION ---------------------------
with tab9:
    st.header(":violet[Weibull Distribution]")

    st.subheader(":violet[User Inputs]")
    alpha = st.number_input("Shape (alpha)", value=1.5, min_value=0.01)
    beta = st.number_input("Scale (beta)", value=1.0, min_value=0.01)

    samples = pf.draw_samples_from_prior(pm.Weibull, alpha=alpha, beta=beta, seed=seed)
    fig = pf.plot_prior_distribution(samples, title="Weibull Distribution Samples")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------- TRUNCATED NORMAL DISTRIBUTION ------------------
with tab10:
    st.header(":orange[TruncatedNormal Distribution]")

    st.subheader(":orange[User Inputs]")
    mu = st.number_input("Mean (mu) for TruncatedNormal", value=0.0)
    sigma = st.number_input(
        "Standard Deviation (sigma) for TruncatedNormal", value=1.0, min_value=0.01
    )
    lower = st.number_input("Lower Bound for TruncatedNormal", value=0.0)
    upper = st.number_input("Upper Bound for TruncatedNormal", value=2.0)
    # Check to ensure lower < upper
    if lower >= upper:
        st.error("Error: Lower bound must be less than upper bound.")
    else:
        samples = pf.draw_samples_from_prior(
            pm.TruncatedNormal, mu=mu, sigma=sigma, lower=lower, upper=upper, seed=seed
        )
        fig = pf.plot_prior_distribution(
            samples, title="TruncatedNormal Distribution Samples"
        )
        st.plotly_chart(fig, use_container_width=True)

# -------------------------- STUDENT T DISTRIBUTION -------------------------
with tab11:
    st.header(":green[StudentT Distribution]")

    st.subheader(":green[User Inputs]")
    nu = st.number_input("Degrees of Freedom (nu)", value=10.0, min_value=0.01)
    mu = st.number_input("Location (mu)", value=0.0)
    sigma = st.number_input("Scale (sigma)", value=1.0, min_value=0.01)

    samples = pf.draw_samples_from_prior(
        pm.StudentT, nu=nu, mu=mu, sigma=sigma, seed=seed
    )
    fig = pf.plot_prior_distribution(samples, title="StudentT Distribution Samples")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------- LOGNORMAL DISTRIBUTION -------------------------
with tab12:
    st.header(":red[LogNormal Distribution]")

    st.subheader(":red[User Inputs]")
    mu = st.number_input("Mean (mu) for LogNormal", value=0.0)
    sigma = st.number_input(
        "Standard Deviation (sigma) for LogNormal", value=1.0, min_value=0.01
    )

    samples = pf.draw_samples_from_prior(pm.Lognormal, mu=mu, sigma=sigma, seed=seed)
    fig = pf.plot_prior_distribution(samples, title="LogNormal Distribution Samples")
    st.plotly_chart(fig, use_container_width=True)
