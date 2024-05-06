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
import adstock_saturation_functions as asf
import pandas as pd
import plotly.express as px

import streamlit as st

# -------------------------- TOP OF PAGE INFORMATION -------------------------

# Set browser / tab config
st.set_page_config(
    page_title="MMM App - Adstock Transformations",
    page_icon="🧊",
)

# Give some context for what the page displays
st.title("Adstock Transformations")
st.markdown(
    "This page demonstrates the effect of various adstock \
            transformations on a variable.  \nFor these examples, let's imagine \
            that we have _some variable that represents a quantity of a particlar_ \
            _advertising channel_.  \n\nFor example, this could be the number of impressions\
            we get from Facebook.  For an online channel such as this, we might expect the impact of these ads to be immediate: \
            \n   ___We see an ad on Facebook - we either click on it, or we don't.___  \n\
            \n :blue[So, at the start of our example (_Week 1_), \
            we could have the impact of **100 impressions from Facebook**.] \
            \n\n Alternatively, for a channel like TV, we may not expect the impact of those ads  \n \
            to come through immediately - there may be some delay. \
            \n\n :green[So, at the start of our example (_Week 1_), we may have the impact of **0 Gross Rating Points (TV viewership metric)**, \
            but 7 weeks later those TV ads might reach their full impact of **100 Gross Rating Points**.]\
            \n\n**_:violet[We will use this starting value of 100 for all of our adstock examples]_**. \
            "
)

st.markdown(
    "**Reminder:** \n \
- Geometric adstock transformations have **_:red[fixed decay]_**  \n\
- Weibull adstock transformations have **_:red[flexible decay]_**"
)

# Starting value for adstock
initial_impact = 100

# Separate the adstock transformations into 3 tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["Geometric", "Delayed Geometric", "Weibull CDF", "Weibull PDF"]
)

# -------------------------- GEOMETRIC ADSTOCK DISPLAY -------------------------
with tab1:
    st.header(":blue[Geometric Adstock Transformation]")
    st.divider()
    st.markdown(
        "___Geometric adstock is the simplest adstock function, it depends on a single parameter $\\beta > 0$ which represents the fixed-rate decay.___ \n \
                \n __The geometric adstock function takes the following form :__"
    )
    st.latex(r"""
        x_t^{\textrm{transf}} = x_t + \beta x_{t-1}^{\textrm{transf}}
        """)
    st.divider()
    st.markdown(
        "**Typical values for geometric adstock:** \n \
- TV: **:blue[0.3 - 0.8]** - _decays slowly_ \n \
- OOH/Print/Radio:  **:blue[0.1 - 0.4]** - _decays moderately_ \n \
- Digital:  **:blue[0.0 - 0.3]** - _decays quickly_ \n"
    )
    st.caption(
        ":link: [Values taken from Meta's Analyst's Guide to MMM](https://facebookexperimental.github.io/Robyn/docs/analysts-guide-to-MMM/#feature-engineering)"
    )

    # User inputs
    st.subheader(":blue[User Inputs]")
    num_periods = st.slider(
        "Number of weeks after impressions first received :alarm_clock:",
        1,
        100,
        20,
        key="Geometric",
    )
    # Let user choose decay rates to plot with
    decay_rate_1 = st.slider(":blue[Beta 1 : ]", 0.0, 1.0, 0.3)
    # Add up to 2 more lines if the user wants it
    st.markdown("**Would you like to show multiple (3) decay lines on the plot**")
    multi_plot = st.checkbox("Okay! :grin:")
    # Create a list of decay rates
    if multi_plot:
        # Let user choose additional decay rates to plot with
        decay_rate_2 = st.slider(":red[Beta 2 : ]", 0.0, 1.0, 0.6)
        decay_rate_3 = st.slider(":green[Beta 3: ]", 0.0, 1.0, 0.9)
        decay_rates = [decay_rate_1, decay_rate_2, decay_rate_3]
    else:
        decay_rates = [decay_rate_1]

    # Create df to store each adstock in
    all_adstocks = pd.DataFrame()
    # Iterate through decay rates and generate df of values to plot
    for i, beta in enumerate(decay_rates):
        # Get geometric adstock values, decayed over time
        adstock_df = pd.DataFrame(
            {
                "Week": range(1, (num_periods + 1)),
                ## Calculate adstock values
                "Adstock": asf.geometric_adstock_decay(
                    initial_impact, beta, num_periods
                ),
                ## Format adstock labels for neater plotting
                "Adstock Labels": [
                    f"{x:,.0f}"
                    for x in asf.geometric_adstock_decay(
                        initial_impact, beta, num_periods
                    )
                ],
                ## Create column to label each adstock
                "Beta": f"Beta {i + 1}",
            }
        )

        all_adstocks = pd.concat([all_adstocks, adstock_df])

    # Plot adstock values
    # Annotate the plot if user wants it
    st.markdown("**Would you like to show the adstock values directly on the plot?**")
    annotate = st.checkbox("Yes please! :pray:", key="Geometric Annotate")
    if annotate:
        fig = px.line(
            all_adstocks,
            x="Week",
            y="Adstock",
            text="Adstock Labels",
            markers=True,
            color="Beta",
            # Replaces default color mapping by value
            color_discrete_map={
                "Beta 1": "#636EFA",
                "Beta 2": "#EF553B",
                "Beta 3": "#00CC96",
            },
        )
        fig.update_traces(textposition="bottom left")
    else:
        fig = px.line(
            all_adstocks,
            x="Week",
            y="Adstock",
            markers=True,
            color="Beta",
            # Replaces default color mapping by value
            color_discrete_map={
                "Beta 1": "#636EFA",
                "Beta 2": "#EF553B",
                "Beta 3": "#00CC96",
            },
        )
    # Format plot
    fig.layout.height = 600
    fig.layout.width = 1000
    fig.update_layout(
        title_text="Geometric Adstock Decayed Over Weeks", title_font=dict(size=30)
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=False)

# -------------------------- DELAYED GEOMETRIC ADSTOCK DISPLAY -------------------------
with tab2:
    st.header(":red[Delayed Geometric Adstock Transformation]")
    st.divider()
    st.markdown(
        "___Delayed geometric adstock builds on geometric adstock___ \
                 ___by adding in a delay $\\theta$ before the maximum adstock is observed (this happens at week 0 for the plain geometric decay).___ \
                \n ___It also adds a maximum duration for the carryover/adstock  $L_{max}$,  such that adstock after this point is 0.___ \n \
                \n __The delayed geometric adstock function takes the following form :__"
    )
    st.latex(r"""
        x_t^{\textrm{transf}} = \sum_{i=0}^{L_{\max}-1} \left( \beta^{|i-\theta|} \cdot x_{t-i} \right) \\""")
    st.markdown(
        "- $x_t^{\\textrm{transf}}$ refers to the transformed value at time $t$ after applying the delayed adstock transformation"
    )
    st.markdown("- $\\beta$ is the retention rate of the ad effect")
    st.markdown("- $\\theta$ represents the delay before the peak effect occurs")
    st.markdown("- $L_{max}$ is the maximum duration of the carryover effect")
    st.divider()
    st.markdown(
        "**Typical values for geometric adstock:** \n \
- TV: **:blue[0.3 - 0.8]** - _decays slowly_ \n \
- OOH/Print/Radio:  **:blue[0.1 - 0.4]** - _decays moderately_ \n \
- Digital:  **:blue[0.0 - 0.3]** - _decays quickly_ \n"
    )
    st.caption(
        ":link: [Values taken from Meta's Analyst's Guide to MMM](https://facebookexperimental.github.io/Robyn/docs/analysts-guide-to-MMM/#feature-engineering)"
    )

    # User inputs
    st.subheader(":red[User Inputs]")
    max_lag = st.slider(
        "Number of weeks after impressions first received :alarm_clock: : ",
        1,
        100,
        30,
        key="Delayed Geometric",
    )
    max_peak = st.slider(
        ":red[Number of weeks after impressions first received that max impact occurs :thermometer: : ]",
        0,
        100,
        10,
        key="delayed_geom_L",
    )
    # Let user choose decay rates to plot with
    decay_rate_1 = st.slider(":red[Beta 1: ]", 0.0, 1.0, 0.5, key="delay_decay")

    # Add up to 2 more lines if the user wants it
    st.markdown("**Would you like to show multiple (3) decay lines on the plot**")
    multi_plot = st.checkbox("Okay! :grin:", key="Delay Geom Multi")
    # Create a list of decay rates
    if multi_plot:
        # Let user choose additional decay rates, lags and peaks to plot with
        decay_rate_2 = st.slider(":blue[Beta 2: ]", 0.0, 1.0, 0.6, key="delay_decay2")
        max_peak_2 = st.slider(
            ":blue[Number of weeks after impressions first received that max impact occurs :thermometer: :]",
            1,
            100,
            5,
            key="delayed_geom_L 2",
        )
        max_lag_2 = st.slider(
            ":blue[Number of weeks after impressions first received :alarm_clock: : ]",
            1,
            100,
            20,
            key="Delayed Geometric 2 ",
        )
        decay_rate_3 = st.slider(":green[Beta 3: ]", 0.0, 1.0, 0.9, key="delay_decay3")
        max_lag_3 = st.slider(
            ":green[Number of weeks after impressions first received :alarm_clock: : ]",
            1,
            100,
            20,
            key="Delayed Geometric 3 ",
        )
        max_peak_3 = st.slider(
            ":green[Number of weeks after impressions first received that max impact occurs :thermometer: :]",
            1,
            100,
            5,
            key="delayed_geom_L 3",
        )

        # Put in lists to iterate through later
        decay_rates = [decay_rate_1, decay_rate_2, decay_rate_3]
        lags = [max_lag, max_lag_2, max_lag_3]
        peaks = [max_peak, max_peak_2, max_peak_3]

    else:
        decay_rates = [decay_rate_1]
        lags = [max_lag]
        peaks = [max_peak]

    # Create df to store each adstock in
    all_adstocks = pd.DataFrame()
    # Iterate through decay rates and generate df of values to plot
    for i, beta in enumerate(decay_rates):
        # Get geometric adstock values, decayed over time
        adstock_df = pd.DataFrame(
            {
                "Week": range(1, (lags[i] + 1)),
                ## Calculate adstock values
                "Adstock": asf.delayed_geometric_decay(
                    impact=initial_impact, decay_factor=beta, theta=peaks[i], L=lags[i]
                ),
                ## Format adstock labels for neater plotting
                "Adstock Labels": [
                    f"{x:,.0f}"
                    for x in asf.delayed_geometric_decay(
                        initial_impact, beta, peaks[i], lags[i]
                    )
                ],
                ## Create column to label each adstock
                "Beta": f"Beta {i + 1}",
            }
        )

        all_adstocks = pd.concat([all_adstocks, adstock_df])

    # Plot adstock values
    # Annotate the plot if user wants it
    st.markdown("**Would you like to show the adstock values directly on the plot?**")
    annotate = st.checkbox("Yes please! :pray:", key="Delayed Geometric Annotate")
    if annotate:
        fig = px.line(
            all_adstocks,
            x="Week",
            y="Adstock",
            text="Adstock Labels",
            markers=True,
            color="Beta",
            # Replaces default color mapping by value
            color_discrete_map={
                "Beta 1": "#636EFA",
                "Beta 2": "#EF553B",
                "Beta 3": "#00CC96",
            },
        )
        fig.update_traces(textposition="bottom left")
    else:
        fig = px.line(
            all_adstocks,
            x="Week",
            y="Adstock",
            markers=True,
            color="Beta",
            # Replaces default color mapping by value
            color_discrete_map={
                "Beta 1": "#636EFA",
                "Beta 2": "#EF553B",
                "Beta 3": "#00CC96",
            },
        )
    # Format plot
    fig.layout.height = 600
    fig.layout.width = 1000
    fig.update_layout(
        title_text="Geometric Adstock Decayed Over Weeks", title_font=dict(size=30)
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=False)

# -------------------------- WEIBULL CDF ADSTOCK DISPLAY -------------------------
with tab3:
    st.header(":green[Weibull CDF Adstock Transformation]")
    st.divider()
    st.markdown(
        "___The Weibull CDF is a function depending on two variables, $k$ (known as the **shape**) and $\\lambda$ (known as the **scale**)___.  \n  \
                The idea is closely related to geometric adstock but with one important difference : the rate of decay (what we called $\\beta$ in the geometric adstock equation)  \
                 is no longer fixed. Instead it’s **time-dependent**. \
                \n \n **The Weibull CDF adstock function therefore takes the form :**"
    )
    st.latex(r"""
        x_t^{\textrm{transf}} = x_t + \beta_t x_{t-1}^{\textrm{transf}}""")
    st.markdown("- where $\\beta_t$ is now a function of time $t$")
    st.markdown(
        "**The Weibull CDF is actually used to build the $\\beta_t$’s, and it takes the form :**"
    )
    st.latex(r"""
             F_{k, \lambda}(t) = 1 - e^{-(\frac{t}{\lambda})^k}""")
    st.markdown("Then, $\\beta_t$ is computed as : ")
    st.latex(r"""
        \beta_t = 1 - F_{k,\lambda}(t)""")
    st.divider()
    # User inputs
    st.subheader(":green[User Inputs]")
    num_periods_2 = st.slider(
        "Number of weeks after impressions first received :alarm_clock: :",
        1,
        100,
        20,
        key="Weibull CDF Periods",
    )
    # Let user choose shape and scale parameters to compare two Weibull PDF decay curves simultaneously
    # Params for Line A
    shape_parameter_A = st.slider(
        ":triangular_ruler: :green[Shape $k$ of Line A]:",
        0.0,
        10.0,
        0.1,
        key="Weibull CDF Shape A",
    )
    scale_parameter_A = st.slider(
        r":green[Scale $\lambda$ of Line A]:", 0.0, 1.0, 0.1, key="Weibull CDF Scale A"
    )

    # Calculate weibull pdf adstock values, decayed over time for both sets of params
    adstock_series_A = asf.weibull_adstock_decay(
        initial_impact,
        shape_parameter_A,
        scale_parameter_A,
        num_periods_2,
        adstock_type="cdf",
        normalised=True,
    )

    # Create df of adstock values, to plot with
    adstock_df_A = pd.DataFrame(
        {
            "Week": range(1, (num_periods_2 + 1)),
            "Adstock": adstock_series_A,
            "Line": "Line A",
        }
    )

    # Create plotting df
    weibull_pdf_df = adstock_df_A.copy()

    # Plot 2nd line if user desires values
    st.markdown("**Would you like to add a second line to the plot?**")
    second_cdf = st.checkbox("Okay! :grin:", key="Add 2nd Weibull CDF")

    if second_cdf:
        # Params for Line B
        shape_parameter_B = st.slider(
            ":triangular_ruler: :red[Shape $k$ of Line B : ]",
            0.0,
            10.0,
            9.0,
            key="Weibull CDF Shape B",
        )
        scale_parameter_B = st.slider(
            r":red[Scale $\lambda$ of Line B : ]",
            0.0,
            1.0,
            0.5,
            key="Weibull CDF Scale B",
        )
        # Calculate weibull pdf adstock values, decayed over time for both sets of params
        adstock_series_B = asf.weibull_adstock_decay(
            initial_impact,
            shape_parameter_B,
            scale_parameter_B,
            num_periods_2,
            adstock_type="cdf",
            normalised=True,
        )

        # Create df of adstock values, to plot with
        adstock_df_B = pd.DataFrame(
            {
                "Week": range(1, (num_periods_2 + 1)),
                "Adstock": adstock_series_B,
                "Line": "Line B",
            }
        )
        # Create plotting df
        weibull_pdf_df = pd.concat([adstock_df_A, adstock_df_B])

    # Multiply by 100 to get back to scale of initial impact (100 FB impressions)
    weibull_pdf_df.Adstock = weibull_pdf_df.Adstock
    # Format adstock labels for neater plotting
    weibull_pdf_df["Adstock Labels"] = weibull_pdf_df.Adstock.map("{:,.0f}".format)

    # Plot adstock values
    # Annotate the plot if user wants it
    st.markdown("**Would you like to show the adstock values directly on the plot?**")
    annotate = st.checkbox("Yes please! :pray:", key="Weibull CDF Annotate")
    if annotate:
        fig = px.line(
            weibull_pdf_df,
            x="Week",
            y="Adstock",
            text="Adstock Labels",
            markers=True,
            color="Line",
            # Replaces default color mapping by value
            color_discrete_map={"Line A": "#636EFA", "Line B": "#EF553B"},
        )
        fig.update_traces(textposition="bottom left")
    else:
        fig = px.line(
            weibull_pdf_df,
            x="Week",
            y="Adstock",
            markers=True,
            color="Line",
            # Replaces default color mapping by value
            color_discrete_map={"Line A": "#636EFA", "Line B": "#EF553B"},
        )
    # Format plot
    fig.layout.height = 600
    fig.layout.width = 1000
    fig.update_layout(
        title_text="Weibull CDF Adstock Decayed Over Weeks", title_font=dict(size=30)
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=False)

# -------------------------- WEIBULL PDF ADSTOCK DISPLAY -------------------------
with tab4:
    st.header(":violet[Weibull PDF Adstock Transformation]")
    st.divider()
    st.markdown(
        "___The Weibull PDF is also a function depending on two variables, $k$ (shape) and $\\lambda$ (scale) \
                 and the same remarks for Weibull CDF apply to Weibull PDF.___ \
                \n The key difference is that Weibull PDF \
                 allows for lagged effects to be taken into account - the **time delay effect**. \
                \n \n **The Weibull PDF adstock function therefore takes the form :**"
    )
    st.latex(r"""
        x_t^{\textrm{transf}} = x_t + \beta_t x_{t-1}^{\textrm{transf}}""")
    st.markdown("- where $\\beta_t$ is now a function of time $t$")
    st.markdown(
        "**The Weibull PDF is actually used to build the $\\beta_t$’s, and it takes the form :**"
    )
    st.latex(r"""
             G_{k,\lambda}(t) = \frac{k}{\lambda}\Big(\frac{t}{\lambda} \Big)^{k-1}e^{-(\frac{t}{\lambda})^k}""")
    st.divider()

    # User inputs
    st.subheader(":violet[User Inputs]")
    num_periods_3 = st.slider(
        "Number of weeks after impressions first received :alarm_clock: : ",
        1,
        100,
        20,
        key="Weibull PDF Periods",
    )
    # Let user choose shape and scale parameters to compare two Weibull PDF decay curves simultaneously
    # Params for Line A
    shape_parameter_A = st.slider(
        ":triangular_ruler: :blue[Shape $k$ of Line A : ]",
        0.0,
        10.0,
        2.0,
        key="Weibull PDF Shape A",
    )
    scale_parameter_A = st.slider(
        r":blue[Scale $\lambda$ of Line A : ]", 0.0, 1.0, 0.5, key="Weibull PDF Scale A"
    )

    # Calculate weibull pdf adstock values, decayed over time for both sets of params
    adstock_series_A = asf.weibull_adstock_decay(
        initial_impact,
        shape_parameter_A,
        scale_parameter_A,
        num_periods_3,
        adstock_type="pdf",
        normalised=True,
    )

    # Create df of adstock values, to plot with
    adstock_df_A = pd.DataFrame(
        {
            "Week": range(1, (num_periods_3 + 1)),
            "Adstock": adstock_series_A,
            "Line": "Line A",
        }
    )

    # Create plotting df
    weibull_pdf_df = adstock_df_A.copy()

    # Plot 2nd line if user desires values
    st.markdown("**Would you like to add a second line to the plot?**")
    second_pdf = st.checkbox("Okay! :grin:", key="Add 2nd Weibull PDF")

    if second_pdf:
        # Params for Line B
        shape_parameter_B = st.slider(
            ":triangular_ruler: :red[Shape $k$ of Line B : ]",
            0.0,
            10.0,
            0.5,
            key="Weibull PDF Shape B",
        )
        scale_parameter_B = st.slider(
            r":red[Scale $\lambda$ of Line B : ]",
            0.0,
            1.0,
            0.01,
            key="Weibull PDF Scale B",
        )

        # Calculate weibull pdf adstock values, decayed over time for both sets of params
        adstock_series_B = asf.weibull_adstock_decay(
            initial_impact,
            shape_parameter_B,
            scale_parameter_B,
            num_periods_3,
            adstock_type="pdf",
            normalised=True,
        )

        # Create df of adstock values, to plot with
        adstock_df_B = pd.DataFrame(
            {
                "Week": range(1, (num_periods_3 + 1)),
                "Adstock": adstock_series_B,
                "Line": "Line B",
            }
        )
        # Create plotting df
        weibull_pdf_df = pd.concat([adstock_df_A, adstock_df_B])

    # Multiply by 100 to get back to scale of initial impact (100 FB impressions)
    weibull_pdf_df.Adstock = weibull_pdf_df.Adstock
    # Format adstock labels for neater plotting
    weibull_pdf_df["Adstock Labels"] = weibull_pdf_df.Adstock.map("{:,.0f}".format)

    # Plot adstock values
    # Annotate the plot if user wants it
    st.markdown("**Would you like to show the adstock values directly on the plot?**")
    annotate = st.checkbox("Yes please! :pray:", key="Weibull PDF Annotate")
    if annotate:
        fig = px.line(
            weibull_pdf_df,
            x="Week",
            y="Adstock",
            text="Adstock Labels",
            markers=True,
            color="Line",
            # Replaces default color mapping by value
            color_discrete_map={"Line A": "#636EFA", "Line B": "#EF553B"},
        )
        fig.update_traces(textposition="bottom left")
    else:
        fig = px.line(
            weibull_pdf_df,
            x="Week",
            y="Adstock",
            markers=True,
            color="Line",
            # Replaces default color mapping by value
            color_discrete_map={"Line A": "#636EFA", "Line B": "#EF553B"},
        )
    # Format plot
    fig.layout.height = 600
    fig.layout.width = 1000
    fig.update_layout(
        title_text="Weibull PDF Adstock Decayed Over Weeks", title_font=dict(size=30)
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=False)
