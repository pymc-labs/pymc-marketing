import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pymc_marketing.mmm.transformers import (
    delayed_adstock,
    geometric_adstock,
    logistic_saturation,
)

warnings.filterwarnings("ignore", category=FutureWarning)
FIGSIZE = (15, 8)


class DataGenerator:
    """# Data Generator
    Adapted from PYMC-Marketing Demo
    ## Models a Time-Series Equation:
        y_t = α + Σ_m [β_m*f(x_m_t)] + Σ_c [γ_c*z_c_t] + ε_t
    where:
        y_t     = Sales or KPI at time t
        α       = Baseline intercept and trend of dataset
        Σ_m     = Sum over media channels, m ∈ {M}.
        β_m     = Coef for media channel m
        x_c_t   = Spend on channel m at time t
        f       = Function of spend, often lagged and/or saturated (see below).
        Σ_c     = Sum over control columns, c ∈ {C}.
        γ_c     = Coef for control columns c
        z_c_t   = Control column c at time t. (Ex. Year, month, event-spike)
        ε_t     = Noise at time t

    ## Generate a dataset while being able to tune and toggle the following effects:
        - Intercept
        - Channel Spend
        - Lagging interest due to ad spend (common approaches include: adstock, carryover, convolution models)
        - Lagged effects (α and maybe θ)
        - Saturation effects (λ)
        - Overall trend
        - Event effects
        - Yearly seasonality (γ)
    Determine if the model is flexible enough to correctly identify which effects are or are not present.
    And how close to the true coefficients are recovered.

    ## Media Data and Channel Betas (X & β)
    Media data can be singular -- spend per channel _only_ --
    OR dual -- spend per channel and impressions per channel --.
    This generator allows the creation of dual-media-datasets, and dual sets drawn from a
    multi-variate normal distribution to model the so called "Halo Effect" or covariance
    between channel impressions in response to spend in other channels.

    ## Response Variable (y)
    The response variable can be singular (total sales over all channels (∑sales)),
        or multi-variate (∑sales, ∑impressions, ∑KPI_3).
    Basically y can be one or more KPIs. #TODO
    Some models will be able to incorporate the additional information.
    """

    def __init__(
        self,
        min_date: str = "2020-02-02",
        max_date: str = "2022-01-20",
        num_channels=2,
        beta_scale=3,
    ):
        seed: int = sum(map(ord, "six-mmm"))
        self.rng: np.random.Generator = np.random.default_rng(seed=seed)
        # create dataframe and channel columns
        self.add_daterange(min_date, max_date)
        self.add_channel_spends(num_channels, beta_scale)
        self.lag_fns = {"geometric": geometric_adstock, "adstock": delayed_adstock}
        # Switch these on if/when added to the signal
        self.is_lagged = False
        self.is_saturated = False
        self.is_trended_and_seasonal = False
        self.has_events = False
        self.has_intercept = False
        print(f"Number of observations: {self.n}")

    def add_daterange(self, min_date, max_date):
        """Set a time range for our data."""
        self.min_date = pd.to_datetime(min_date)
        self.max_date = pd.to_datetime(max_date)
        self.df = pd.DataFrame(
            data={
                "date_week": pd.date_range(start=min_date, end=max_date, freq="W-MON")
            }
        ).assign(
            year=lambda x: x["date_week"].dt.year,
            month=lambda x: x["date_week"].dt.month,
            dayofyear=lambda x: x["date_week"].dt.dayofyear,
        )
        self.n = self.df.shape[0]

    def add_channel_spends(self, num_channels=2, beta_scale=3):
        """2. Media Costs Data
        - Generate synthetic data from channel columns. We refer to
          it as the raw signal as it is going to be the input at the modeling phase. We
          expect the contribution of each channel to be different, based on the
          carryover and saturation parameters.
        Parameters
        ----------
        num_channels: int
            Number of media channels present in the dataset.
        beta_scale: float
            Mean (λ) of the exponential distribution the channel spend params are sampled from.
        """
        # Save these for benchmarking
        self.num_channels = num_channels
        self.betas = self.rng.exponential(beta_scale, size=num_channels)
        self.channel_spend_probs = self.rng.uniform(0.5, 1.0, num_channels)

        for c in range(num_channels):
            x = self.rng.uniform(low=0.0, high=1.0, size=self.n)
            self.df[f"x{c}"] = np.where(x > self.channel_spend_probs[c], x, 0)

    def plot_channel_spends(
        self,
        suffix: str = "",
        fig=None,
        ax=None,
        alpha: float = 1.0,
        title: str = "Media Costs Data",
        show: bool = True,
    ):
        """Creates a subplot for each channel.

        Parameters
        ----------
        suffix: str
            Channel suffix. I.e. x1_lag "_lag" would be the
            channel suffix.
        show: bool
            Can be turned off for overlays.
        Returns
        -------
        fig, ax: matplotlib plot handlers
            Used to add additional transform info to the base signals.
        """
        if ax is None:
            fig, ax = plt.subplots(
                nrows=self.num_channels,
                ncols=1,
                figsize=FIGSIZE,
                sharex=True,
                sharey=True,
                layout="constrained",
            )
        for c in range(self.num_channels):
            sns.lineplot(
                x="date_week",
                y=f"x{c}{suffix}",
                data=self.df,
                color=f"C{c}",
                label=f"x{c}{suffix}",
                alpha=alpha,
                ax=ax[c],
            )

        ax[c].set(xlabel="date")
        fig.suptitle(title, fontsize=16)
        if show:
            plt.show()
        return fig, ax

    def add_lag_effect(self, fn="geometric", max_lag: int = 8):
        """Effect Signal
        - Next, we pass the raw signal through the two transformations: first
          the geometric adstock (carryover effect) and then the logistic
          saturation. Note that we set the parameters ourselves, but we will
          recover them back from the model.
        - Let's start with the adstock transformation. We set the adstock
          parameter $0 < \alpha < 1$ to be $0.4$ and $0.2$ for $x_1$ and $x_2$
          respectively. We set a maximum lag effect of $8$ weeks.
        Parameters
        ----------
        fn: str
            'geometric' or 'adstock'. References a pymc_marketing transform.
        alpha : float, by default 0.0
            Retention rate of ad effect (AKA decay) . Must be between 0 and 1.
        theta : float, by default 0
            Delay of the peak effect. Must be between 0 and `l_max` - 1.
            Only used if fn is 'adstock'.
        l_max : int, by default 12
            Maximum duration of carryover effect.
        normalize : bool, by default False
            Whether to normalize the weights.
        """
        assert (
            fn in self.lag_fns.keys()
        ), f'Lag function must be in \
        ("geometric", "adstock"). Function unrecognized: {fn}'
        df = self.df
        self.is_lagged = True  # signals that adstock has been applied
        self.lag_fn = fn
        self.alphas = self.rng.beta(0.5, 0.5, self.num_channels)
        self.thetas = None
        if fn == "adstock":
            # trying to keep delays between 0 and 3 for the most part
            self.thetas = self.rng.binomial(max_lag - 1, 0.1, self.num_channels)
        self.max_lag = max_lag

        # Apply adstock transformation
        for c in range(self.num_channels):
            args = {
                "x": df[f"x{c}"].to_numpy(),
                "alpha": self.alphas[c],
                "l_max": max_lag,
                "normalize": True,
            }
            if fn == "geometric":
                fn = geometric_adstock
            else:
                fn = delayed_adstock
                assert (
                    self.thetas is not None
                ), "thetas must be specified to apply adstock."
                args["theta"] = self.thetas[c]
            df[f"x{c}_lag"] = fn(**args).eval().flatten()

    def plot_lag_effect(self):
        """Plots the raw channels compared to when their lag effects are applied."""
        assert self.is_lagged, "Invoke `add_lag_effect` before plotting it."
        fig, ax = self.plot_channel_spends(show=False)
        self.plot_channel_spends(
            suffix="_lag",
            fig=fig,
            ax=ax,
            title=f"{self.lag_fn.capitalize()} Effect",
            alpha=0.5,
        )

    def add_saturation_effect(self, lambdas_scale: float = 3.5, lambas_sd: float = 1):
        """Apply a logistic saturation transformation

        Note: column reference will be "x1_lag" if adstock has been applied
            or just "x1" if adstock was not opted for

        Parameters
        ----------
        lambdas_scale: float
            Mean of the normal distribution the lambdas are sampled from.
        lambdas_sd: float
            Standard deviation of the normal distribution the lambdas are sampled from.
        """
        self.is_saturated = True
        self.lambdas = self.rng.normal(lambdas_scale, lambas_sd, self.num_channels)

        for i, λ in enumerate(self.lambdas):
            col_ref = [f"x{i}", f"x{i}_lag"][self.is_lagged]
            col_name = col_ref + "_saturated"
            self.df[col_name] = logistic_saturation(
                x=self.df[col_ref].to_numpy(), lam=λ
            ).eval()

    def plot_saturation_effect(self):
        """Plots the raw channels compared to saturation effects are applied.
        If lagging has been applied, then the lagged signal is plotted in lieu of the raw signal.
        """
        assert self.is_saturated, "Apply add_saturation_effect() before plotting it."
        suffix = ""
        if self.is_lagged:
            suffix = "_lag"
        fig, ax = self.plot_channel_spends(suffix=suffix, show=False)
        suffix += "_saturated"
        self.plot_channel_spends(
            suffix=suffix, alpha=0.5, title="Saturation Effect", fig=fig, ax=ax
        )

    def add_trend_and_seasonality_effects(
        self, trend_scale=5, seasonality_scale=1, freq=52, degree=3
    ):
        """Generate trend and seasonal effect columns in the dataframe.

        Parameters
        ----------
        trend_scale: float
            Mean of exponential distribution that trend is sampled from.
        seasonality_scale: float
            Mean of exponential distribution that seasonality is sampled from.
        freq: int
            number of periods before a repeat
            daily - use 365
            weeky - use 52
        degree: int
            degree of the Fourier series to accommodate.
        """
        self.is_trended_and_seasonal = True
        df = self.df
        self.trend = self.rng.exponential(trend_scale)
        df["trend"] = np.linspace(start=0.0, stop=self.trend, num=self.n)

        df["seasonality"] = np.zeros(self.n)
        self.seasonality_coefs = list()
        for d in range(degree):
            (sin_coef, cos_coef) = self.rng.exponential(seasonality_scale, size=2)
            self.seasonality_coefs.append((sin_coef, cos_coef))
            sin_wave = -sin_coef * np.sin(d * 2 * np.pi * df.index / freq)
            cos_wave = cos_coef * np.cos(d * 2 * np.pi * df.index / freq)
            df["seasonality"] += 0.5 * (sin_wave + cos_wave)

    def plot_trend_and_seasonality_effects(self, show=True):
        """Plots trend and seasonality effects -- separately and composed."""
        df = self.df
        fig, ax = plt.subplots(figsize=FIGSIZE)
        plt.plot(df["date_week"], df["trend"], color="C3", label="trend", alpha=0.4)
        plt.plot(
            df["date_week"],
            df["seasonality"],
            color="C4",
            label="seasonality",
            alpha=0.5,
        )
        plt.plot(
            df["date_week"],
            df.seasonality + df.trend,
            color="C5",
            label="trend + seasonality",
        )
        plt.legend()
        plt.title("Trend & Seasonality Components")
        plt.xlabel("date")
        if show:
            plt.show()

    def add_event_effects(self, num_events=2, event_scale=1):
        """4. Control Variables
        We add events where there was a remarkable peak in our target variable. We
        assume they are independent an not seasonal (e.g. launch of a particular product).
        Parameters
        ----------
        num_events: int
            Number of events to add to control data.
        event_scale: float
            Mean of exponential distribution events were sampled from."""
        assert num_events <= self.n
        self.has_events = True
        self.num_events = num_events
        df = self.df
        self.events = pd.to_datetime(self.rng.choice(df.date_week, size=num_events))
        self.event_coefs = np.zeros(num_events)
        for i, e in enumerate(self.events):
            coef = self.rng.exponential(event_scale)
            df[f"event_{i}"] = coef * (df["date_week"] == e).astype(float)
            self.event_coefs[i] = coef

    def plot_events(self):
        """Plot event spikes."""
        if not self.has_events:
            print("Run .add_event_effects() before plotting them")
            return
        plt.figure(figsize=FIGSIZE)
        df = self.df
        for i in range(self.num_events):
            df[f"event_{i}"].plot(label=f"event_{i}")
        plt.legend()
        plt.show()

    def add_intercept(self, intercept_scale=3.0):
        """Samples an intercept from an exponential distribution."""
        self.has_intercept = True
        self.intercept = self.rng.exponential(intercept_scale)
        self.df["intercept"] = self.intercept  # broadcast

    def _get_channel_ref(self, channel_num):
        """Helper method which determines which lag and saturation effects have been applied.
        Then returns the corresponding column reference.
        Parameters
        ----------
        channel_num: int
            Channel number to create reference for."""
        c = channel_num
        ref = f"x{c}"
        if self.is_lagged:
            ref += "_lag"
        if self.is_saturated:
            ref += "_saturated"
        return ref

    def _get_channel_names(self):
        """Helper method which returns the names of the transformed channels"""
        for c in range(self.num_channels):
            yield self._get_channel_ref(c)

    def compile_target_variable(self, amplitude=1):
        """5. Target Variable
        - Finally, we define the target variable (sales) $y$. We assume it is a linear
          combination of the effect signal, the trend and the seasonal components,  events
          and an intercept depending on what's been selected thus far.
        - We also add some Gaussian noise.
        Parameters
        ----------
        amplitude: ink
            Sets the scale of the compiled sales signal."""
        assert len(np.array([amplitude])) == 1
        self.amplitude = amplitude

        df = self.df
        y = np.zeros(self.n)

        # baseline effects
        if self.has_intercept:
            y += df.intercept
        if self.is_trended_and_seasonal:
            y += df.trend
            y += df.seasonality
        df["baseline"] = y  # keep track of baseline effect

        if self.has_events:
            for i in range(len(self.events)):
                y += df[f"event_{i}"]
        df["baseline_and_events"] = y

        for c in range(self.num_channels):
            ref = self._get_channel_ref(c)
            y += self.betas[c] * df[ref]

        # Add Noise
        df["epsilon"] = self.rng.normal(loc=0.0, scale=0.25, size=self.n)
        y += df.epsilon
        y *= amplitude  # original conception

        df["y"] = pd.Series(y)

    def plot_baseline(self, show: bool = True):
        """Plots composite baseline effect vs y (compilation of all effects)."""
        df = self.df
        plt.figure(figsize=FIGSIZE)
        fillmin = min(df.y.min(), df.baseline.min())
        plt.fill_between(x=df.date_week, y1=fillmin, y2=df.y, color="C0", label="y")
        plt.fill_between(
            x=df.date_week,
            y1=fillmin,
            y2=df.baseline,
            color="C1",
            label="baseline",
            alpha=0.7,
        )
        plt.legend()
        if show:
            plt.show()

    def plot_channel_decomposition(self, show: bool = True):
        """Plots each channel independently vs y with the baseline and event effects filtered out.
        Note: seasonal effects are optionally left in."""
        df = self.df
        betas = self.betas
        plt.figure(figsize=FIGSIZE)
        (df.y - df.baseline_and_events).plot(
            alpha=0.5, ls=":", color="grey", label="y - baseline - events"
        )
        for i in range(self.num_channels):
            (betas[i] * df[f"x{i}_lag_saturated"]).plot(alpha=0.5, label=f"x{i}")
        plt.title("Channel Decomposition")
        plt.legend()
        if show:
            plt.show()

    def calc_contribution_share(self):
        """Returns - True relative contribution share of each channel (size == num_channels)."""
        df = self.df
        contribution_share = np.zeros(self.num_channels)
        denominator = 0
        for c in range(self.num_channels):
            channel_ref = self._get_channel_ref(c)
            contribution_share[c] = (self.betas[c] * df[channel_ref]).sum()
            denominator += contribution_share[c]
        contribution_share /= denominator
        print("channel contribution shares: ", contribution_share)
        self.contribution_share = contribution_share
        return contribution_share

    def calc_ROI(self):
        pass  # TODO

    def calc_campaign_effectiveness(self):
        """Returns the ratio of the total media mass (media coefs dot transformed channel columns)
        against the compiled (y) signal."""
        df = self.df
        channel_names = self._get_channel_names()
        total_channel_mass = np.einsum(
            "tc, c -> ", np.array(df[channel_names].values), self.betas
        )
        return total_channel_mass / df.y.sum()

    def plot_contributions(self):
        """Scatter of channel contribution estimate: (β * x_transform) over x_raw.
        As in PYMC-MMM demo."""
        df = self.df
        _, ax = plt.subplots(
            nrows=self.num_channels,
            ncols=1,
            figsize=FIGSIZE,
            sharex=True,
            sharey=False,
            layout="constrained",
        )
        for c in range(self.num_channels):
            channel_ref = self._get_channel_ref(c)
            sns.scatterplot(
                x=df[f"x{c}"],
                y=self.amplitude * self.betas[c] * df[channel_ref],
                color=f"C{c}",
                ax=ax[c],
            )
            ax[c].set(
                title=f"$x_{c}$ contribution",
                ylabel=f"$\\beta_{c} \cdot {channel_ref}",
                xlabel="x",
            )
        plt.show()

    def get_true_params(self):
        """Populate a dictionary with the answers to the modeling exercise.
        IE. Every parameter used to construct y.
        Gathering up all the parameter choices made into a single dictionary for
        convenience."""
        d = {
            "channel_spend_probs": self.channel_spend_probs,
            "betas": self.betas,
            "amplitude": self.amplitude,
        }
        if self.is_lagged:
            d["lag_fn"] = self.lag_fn
            d["alphas"] = self.alphas
            if self.lag_fn == "adstock":
                d["thetas"] = self.thetas
            # Not solving for max_lag! (Yet?)
        if self.is_saturated:
            d["lambdas"] = self.lambdas
        if self.is_trended_and_seasonal:
            d["trend"] = self.trend
            d["seasonality_coefs"] = self.seasonality_coefs
        if self.has_events:
            d["event_coefs"] = self.event_coefs
        if self.intercept is not None:
            d["intercept"] = self.intercept
        return d

    def get_param_descriptions(self):
        d = {
            "channel_spend_probs": "Instantaneous (every time step) probability of\
        spending on a channel.",
            "betas": "Per-channel ad spend effectiveness. A key parameter in\
             determining ROI!",
            "amplitude": "Scale of signal for model robustness. Usually 1.",
        }
        if self.is_lagged:
            d["lag_fn"] = "Ex. Adstock or carryover."
            d["alphas"] = "Retention rate of ad effect. Must be between 0 and 1."
            if self.lag_fn == "adstock":
                d[
                    "thetas"
                ] = """Delay of the peak effect. Must be between 0 and `l_max` - 1.
            Only used if fn is 'adstock'."""
        if self.is_saturated:
            d[
                "lambdas"
            ] = """Per-channel logistic saturation coefficients for the Hill
            transform. Approximated via a transform exponent (less than 1) in the carryover
             transform."""
        if self.is_trended_and_seasonal:
            d["trend"] = "Signal Slope."
            d[
                "seasonality_coefs"
            ] = "Coefficients of the Fourier seasonal components. [(sin1, cos1), (sin2, cos2)...]"
        if self.has_events:
            d[
                "event_coefs"
            ] = """Spikes where there was a remarkable peak in our target variable. We
        assume they are independent an not seasonal (e.g. launch of a particular product).
        Looks like: [0...0,1,0...0]."""
        if self.intercept is not None:
            d["intercept"] = "Signal offset."
        return d

    @staticmethod
    def generate_typical_mmm_dataset(
        min_date: str = "2018-04-01",
        max_date: str = "2021-09-01",
        num_channels: int = 3,
        is_lagged: bool = True,
        lag_fn: str = "adstock",
        is_saturated: bool = True,
        is_trended_and_seasoned: bool = True,
        has_events: bool = True,
        has_intercept: bool = True,
    ):
        """Convenience method for stochastically simulating a MMM dataset from sampled parameters.
        Can use via:
        '''
        from data_generator import DataGenerator
        dg = DataGenerator.generate_typical_mmm_dataset()
        '''
        """
        dg = DataGenerator(min_date, max_date, num_channels)
        assert lag_fn in dg.lag_fns.keys()
        if is_lagged:
            dg.add_lag_effect(fn=lag_fn)
        if is_saturated:
            dg.add_saturation_effect()
        if is_trended_and_seasoned:
            dg.add_trend_and_seasonality_effects()
        if has_events:
            dg.add_event_effects()
        if has_intercept:
            dg.add_intercept()
        dg.compile_target_variable()
        return dg
