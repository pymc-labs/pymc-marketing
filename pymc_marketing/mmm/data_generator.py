from pymc_marketing.mmm.transformers import (
    delayed_adstock,
    geometric_adstock,
    logistic_saturation,
    tanh_saturation)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)


class DataGenerator:
    ''' # Data Generator
    Adapted from PYMC-Marketing Demo
    ## Models a Time-Series Equation:
        y_t = α + Σ_m β_m*f(x_m_t) + Σ_c γ_c*z_c_t + ε_t
    Where:
        y_t = sales or KPI at time t
        α = baseline intercept and trend of dataset
        β_m = coef for channel m
        x_c_t = spend on channel m at time t
        f = function of spend, often lagged and/or saturated (see below).
        γ_c = coef for control columns c
        z_c_t = control column c at time t. (Ex. Year, month, event-spike)
        ε_t = noise at time t

    ## Generate a dataset while being able to toggle the following effects:
        - Lagging interest due to ad spend (common approaches include: adstock, carryover, convolution models)
        - Lagged effects (α and maybe θ)
        - Saturation effects (λ)
        - Overall trend
        - Event effects
        - Yearly seasonality
        - Intercept
    Determine if the model is flexible enough to correctly identify which effects are or are not present.
    And how close to the true coefficients are recovered.

    ## Media Data and Channel Betas (X & β)
    Media data can be singular -- spend per channel _only_ --
    OR dual -- spend per channel and impressions per channel --.
    This generator allows the creation of dual-media-datasets, and dual sets drawn from a
    multi-variate normal distribution to model the so called "Halo Effect" or covariance
    between channel impressions in response to spend in other channels.

    ## Response Variable (y)
    The response variable can be singular (total sales over all channels (∑sales)) or multiplicative (∑sales, ∑impressions, ∑KPI_3).
    Basically y can be one or more KPIs.
    Some models will be able to incorporate the additional information.
    ''';

    def __init__(
            self
            , min_date:str="2018-04-01"
            , max_date:str="2021-09-01"
            , num_channels=2
            , beta_scale=3
        ):
        seed: int = sum(map(ord, "six-mmm"))
        self.rng: np.random.Generator = np.random.default_rng(seed=seed)
        # create dataframe and channel columns
        self.add_daterange(min_date, max_date)
        self.add_channel_spends(num_channels, beta_scale)
        self.adstock_fns = {'geometric':geometric_adstock, 'delayed':delayed_adstock}
        # Switch these on if/when added to the signal
        self.is_lagged = False
        self.is_saturated = False
        self.is_trended_and_seasonal = False
        self.has_events = False
        self.has_intercept = False
        print(f"Number of observations: {self.n}")

    def add_daterange(self, min_date, max_date):
        ''' Set a time range for our data. '''
        self.min_date = pd.to_datetime(min_date)
        self.max_date = pd.to_datetime(max_date)
        self.df = pd.DataFrame(
                data={"date_week": pd.date_range(start=min_date, end=max_date, freq="W-MON")}
            ).assign(
                year=lambda x: x["date_week"].dt.year,
                month=lambda x: x["date_week"].dt.month,
                dayofyear=lambda x: x["date_week"].dt.dayofyear,
            )
        self.n = self.df.shape[0]

    def add_channel_spends(self, num_channels=2, beta_scale=3):
        '''2. Media Costs Data
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
        '''
        # Save these for benchmarking
        self.num_channels = num_channels
        self.betas = np.random.exponential(beta_scale, size=num_channels)
        self.channel_spend_probs = np.random.uniform(.5, 1.0, num_channels)

        for c in range(num_channels):
            x = self.rng.uniform(low=0.0, high=1.0, size=self.n)
            self.df[f"x{c}"] = np.where(x > self.channel_spend_probs[c], x, 0)


    def plot_channel_spends(
            self, suffix:str='', fig=None, ax=None, alpha:float=1.0
            , title:str="Media Costs Data", show:bool=True):
        '''
        Parameters
        suffix: str
            Channel suffix. I.e. x1_adstock "_adstock" would be the
            channel suffix.
        show: bool
            Can be turned off for overlays.'''
        if ax is None:
            fig, ax = plt.subplots(
                nrows=self.num_channels, ncols=1, figsize=(10, 7)
                , sharex=True, sharey=True, layout="constrained"
            )
        for c in range(self.num_channels):
            sns.lineplot(x="date_week", y=f"x{c}{suffix}", data=self.df
                         , color=f"C{c}", label=f"x{c}{suffix}", alpha=alpha, ax=ax[c])

        ax[c].set(xlabel="date")
        fig.suptitle(title, fontsize=16);
        if show: plt.show()
        return fig, ax


    def add_adstock_effect(self, fn='geometric', max_lag:int=8):
        ''' Effect Signal
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
            'geometric' or 'delayed'. References a pymc_marketing transform.
        alpha : float, by default 0.0
            Retention rate of ad effect (AKA decay) . Must be between 0 and 1.
        theta : float, by default 0
            Delay of the peak effect. Must be between 0 and `l_max` - 1.
            Only used if fn is 'delayed'.
        l_max : int, by default 12
            Maximum duration of carryover effect.
        normalize : bool, by default False
            Whether to normalize the weights.
        ''';
        assert fn in self.adstock_fns.keys()
        df = self.df
        self.is_lagged = True # signals that adstock has been applied
        self.lag_fn = fn
        self.alphas = np.random.beta(.5, .5, self.num_channels)
        self.thetas = None
        if fn == 'delayed':
            # trying to keep delays between 0 and 3 for the most part
            self.thetas = np.random.binomial(max_lag-1, .1, self.num_channels)
        self.max_lag = max_lag

        # Apply adstock transformation
        for c in range(self.num_channels):
            args = {
                  'x':df[f"x{c}"].to_numpy()
                , 'alpha':self.alphas[c]
                , 'l_max':max_lag
                , 'normalize':True
                }
            if fn == 'geometric':
                fn = geometric_adstock
            else:
                fn = delayed_adstock
                args['theta']=self.thetas[c]
            df[f"x{c}_adstock"] = fn(**args).eval().flatten()


    def plot_adstock_effect(self):
        '''Plots the raw channels compared to when their adstock effects are applied.'''
        assert self.is_lagged
        fig, ax = self.plot_channel_spends(show=False)
        self.plot_channel_spends(
                suffix="_adstock", fig=fig, ax=ax, title='Adstock Effect', alpha=.5)


    def add_saturation_effect(self, lambdas_scale:float=3.5, lambas_sd:float=1):
        ''' Apply a logistic saturation transformation
        Parameters
        ----------
        lambdas_scale: float
            Mean of the normal distribution the lambdas are sampled from.
        lambdas_sd: float
            Standard deviation of the normal distribution the lambdas are sampled from.
        ''';
        self.is_saturated = True
        # column reference will be "x1_adstock" if adstock has been applied
        # or just "x1" if adstock was not opted for
        lambdas = np.random.normal(lambdas_scale, lambas_sd, self.num_channels)
        self.lambdas = lambdas

        for i, λ in enumerate(lambdas):
            col_ref = [f"x{i}",f"x{i}_adstock"][self.is_lagged]
            col_name = col_ref+"_saturated"
            self.df[col_name] = logistic_saturation(x=self.df[col_ref].to_numpy(), lam=λ).eval()


    def plot_saturation_effect(self):
        '''Plots the raw channels compared to saturation effects are applied.
        If lagging has been applied, then the lagged signal is plotted in lieu of the raw signal.'''
        assert self.is_saturated
        if not self.is_lagged:
            fig, ax = self.plot_channel_spends(show=False)
            self.plot_channel_spends(
                    suffix="_saturated", title="Saturation Effect", fig=fig, ax=ax, alpha=.5)
        else:
            fig, ax = self.plot_channel_spends(suffix="_adstock", show=False)
            self.plot_channel_spends(
                    suffix="_adstock_saturated", alpha=.5, title="Saturation Effect",
                    fig=fig, ax=ax)


    def add_trend_and_seasonality_effects(self, trend_scale=5, seasonality_scale=3):
        ''' 3. Trend & Seasonal Components
        - Now we add synthetic trend and seasonal components to the effect signal.
        - Only a single degree of Forier seasonality is currently implemented.
        Parameters
        ----------
        trend_scale: float
            Mean of exponential distribution that trend is sampled from.
        seasonality_scale: float
            Mean of exponential distribution that seasonality is sampled from.
        '''
        self.is_trended_and_seasonal = True
        df = self.df
        self.trend = np.random.exponential(trend_scale)
        df["trend"] = np.linspace(start=0.0, stop=self.trend, num=self.n)

        # TODO Generalize first order seasonality to Forier degree n
        (self.sin_coef, self.cos_coef) = np.random.exponential(seasonality_scale, size=2)
        num_periods = len(df)
        freq = 52 # weeks per year
        df["cs"] = -self.sin_coef * np.sin(num_periods * 2 * np.pi * df.index / freq)
        df["cc"] =  self.cos_coef * np.cos(num_periods * 2 * np.pi * df.index / freq)
        df["seasonality"] = 0.5 * (df["cs"] + df["cc"])


    def plot_trend_and_seasonality_effects(self, show=True):
        '''Plots trend and seasonality effects -- separately and composed.'''
        df = self.df
        fig, ax = plt.subplots()
        sns.lineplot(x="date_week", y="trend", color="C3", label="trend", alpha=.4, data=df, ax=ax)
        sns.lineplot(x="date_week", y="seasonality", color="C4", label="seasonality",
                     alpha=.5, data=df, ax=ax)
        sns.lineplot(x=df["date_week"], y=df.seasonality+df.trend, color="C5",
                     label="trend + seasonality", ax=ax)
        ax.legend();
        ax.set(title="Trend & Seasonality Components", xlabel="date", ylabel=None);
        if show: plt.show()


    def add_event_effects(self, num_events=2, event_scale=1):
        ''' 4. Control Variables
        We add events where there was a remarkable peak in our target variable. We
        assume they are independent an not seasonal (e.g. launch of a particular product).
        Parameters
        ----------
        num_events: int
            Number of events to add to control data.
        event_scale: float
            Mean of exponential distribution events were sampled from. '''
        assert num_events <= self.n
        self.has_events = True
        self.num_events = num_events
        df = self.df
        self.events = pd.to_datetime(np.random.choice(df.date_week, size=num_events))
        self.event_coefs = np.zeros(num_events)
        for i, e in enumerate(self.events):
            coef = np.random.exponential(event_scale)
            df[f"event_{i}"] = coef*(df["date_week"] == e).astype(float)
            self.event_coefs[i] = coef

    def plot_events(self):
        '''Plot event spikes.'''
        if not self.has_events: print('Run .add_event_effects() before plotting them'); return
        df = self.df
        for i in range(self.num_events):
            df[f'event_{i}'].plot(label=f'event_{i}')
        plt.legend()
        plt.show()

    def add_intercept(self, intercept_scale=3.0):
        '''Samples an intercept from an exponential distribution.'''
        self.has_intercept = True
        self.intercept = np.random.exponential(intercept_scale)
        self.df['intercept'] = self.intercept # broadcast


    def _get_channel_ref(self, channel_num):
        '''Helper method which determines which lag and saturation effects have been applied.
        Then returns the corresponding column reference.
        Parameters
        ----------
        channel_num: int
            Channel number to create reference for. '''
        c = channel_num
        ref = f'x{c}'
        if self.is_lagged: ref+='_adstock'
        if self.is_saturated: ref+='_saturated'
        return ref


    def _get_channel_names(self):
        '''Helper method which returns the names of the transformed channels'''
        for c in range(self.num_channels):
            yield self._get_channel_ref(c)


    def compile_target_variable(self, amplitude=1):
        ''' 5. Target Variable
        - Finally, we define the target variable (sales) $y$. We assume it is a linear
          combination of the effect signal, the trend and the seasonal components,  events
          and an intercept depending on what's been selected thus far.
        - We also add some Gaussian noise.
        Parameters
        ----------
        amplitude: ink
            Sets the scale of the compiled sales signal. '''
        assert len(np.array([amplitude])) == 1
        self.amplitude = amplitude

        df = self.df
        y = np.zeros(self.n)

        # baseline effects
        if self.has_intercept: y += df.intercept
        if self.is_trended_and_seasonal:
            y += df.trend
            y += df.seasonality
        df['baseline'] = y # keep track of baseline effect

        if self.has_events:
            for i in range(len(self.events)):
                y += df[f'event_{i}']
        df['baseline_and_events'] = y

        for c in range(self.num_channels):
            ref = self._get_channel_ref(c)
            y += self.betas[c]*df[ref]

        # Add Noise
        df["epsilon"] = self.rng.normal(loc=0.0, scale=0.25, size=self.n)
        y += df.epsilon
        y *= amplitude # original conception

        df["y"] = pd.Series(y)


    def plot_baseline(self):
        '''Plots composite baseline effect vs y (compilation of all effects).'''
        df = self.df
        plt.fill_between(x=df.date_week, y1=df.y.min(), y2=df.y, color='C0', label='y')
        plt.fill_between(x=df.date_week, y1=df.baseline.min(), y2=df.baseline,
                         color='C1', label='baseline')
        plt.legend()
        plt.show()


    def plot_channel_decomposition(self):
        '''Plots each channel independently vs y with the baseline and event effects filtered out.
        Note: seasonal effects are optionally left in.'''
        df = self.df
        betas = self.betas
        (df.y-df.baseline_and_events).plot(
                alpha=.5, ls=':', color='grey', label='y - baseline - events')
        for i in range(self.num_channels):
            (betas[i]*df[f'x{i}_adstock_saturated']).plot(alpha=.5, label=f'x{i}')
        plt.title('Channel Decomposition')
        plt.legend()
        plt.show()


    def calc_contribution_share(self):
        '''Returns a tuple of:
        - True relative contribution share of each channel (size == num_channels).
        - True relative contribution share of _all_ channels vs baseline effect.'''
        df = self.df
        contribution_share = np.zeros(self.num_channels)
        denominator=0
        for c in range(self.num_channels):
            channel_ref = self._get_channel_ref(c)
            contribution_share[c] = (self.betas[c] * df[channel_ref]).sum()
            denominator += contribution_share[c]
        contribution_share /= denominator
        print('channel contribution shares: ', contribution_share)
        self.contribution_share = contribution_share
        return contribution_share


    def calc_campaign_effectiveness(self):
        '''Returns the ratio of the total media mass (media coefs dot transformed channel columns)
        against the compiled (y) signal.'''
        df = self.df
        channel_names = self._get_channel_names()
        total_channel_mass = np.einsum('tc, c -> ', df[channel_names].values, self.betas)
        return total_channel_mass/df.y.sum()


    def plot_contributions(self):
        '''Scatter of channel contribution estimate: (β * x_transform) over x_raw.
        As in PYMC-MMM demo.'''
        df = self.df
        fig, ax = plt.subplots(
            nrows=self.num_channels, ncols=1, figsize=(12, 8)
            , sharex=True, sharey=False, layout="constrained"
        )
        for c in range(self.num_channels):
            channel_ref = self.get_channel_ref(c)
            sns.scatterplot(
                x=df[f'x{c}'],
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
        '''Populate a dictionary with the answers to the modeling exercise.
        IE. Every parameter used to construct y.
        Gathering up all the parameter choices made into a single dictionary for
        convenience.'''
        d = { 'channel_spend_probs':self.channel_spend_probs
             , 'betas':self.betas
             , 'amplitude':self.amplitude}
        if self.is_lagged:
            d['lag_fn'] = self.lag_fn
            d['alphas'] = self.alphas
            if self.lag_fn == 'delayed': d['thetas'] = self.thetas
            # Not solving for max_lag! (Yet?)
        if self.is_saturated:
            d['lambdas'] = self.lambdas
        if self.is_trended_and_seasonal:
            d['trend'] = self.trend
            d['sin_coef'] = self.sin_coef
            d['cos_coef'] = self.cos_coef
        if self.has_events:
            d['event_coefs'] = self.event_coefs
        if self.intercept is not None:
            d['intercept'] = self.intercept
        return d


    def get_param_descriptions(self):
        d = { 'channel_spend_probs': 'Instantaneous (every time step) probability of\
        spending on a channel.'
             , 'betas': 'Per-channel ad spend effectiveness. The key parameter in\
             determining ROI!'
             , 'amplitude':'Scale of signal for model robustness. Usually 1.'
        }
        if self.is_lagged:
            d['lag_fn'] = 'Ex. Adstock or carryover.'
            d['alphas'] = 'Retention rate of ad effect. Must be between 0 and 1.'
            if self.lag_fn == 'delayed':
             d['thetas'] = '''Delay of the peak effect. Must be between 0 and `l_max` - 1.
            Only used if fn is 'delayed'.'''
        if self.is_saturated:
            d['lambdas'] = '''Per-channel logistic saturation coefficients for the Hill
            transform. Approximated via a transform exponent (less than 1) in the carryover
             transform.'''
        if self.is_trended_and_seasonal:
            d['trend'] = 'Signal Slope.'
            d['sin_coef'] = 'Sinusoidal effect size.'
            d['cos_coef'] = 'Cosinusoial effect size.'
        if self.has_events:
            d['event_coefs'] = '''Spikes where there was a remarkable peak in our target variable. We
        assume they are independent an not seasonal (e.g. launch of a particular product).
        Looks like: [0...0,1,0...0].'''
        if self.intercept is not None:
            d['intercept'] = 'Signal offset.'
        return d

    @staticmethod
    def generate_typical_mmm_dataset(
              min_date:str="2018-04-01"
            , max_date:str="2021-09-01"
            , num_channels:int=3
            , is_adstocked:bool=True
            , adstock_fn:str='delayed'
            , is_saturated:bool=True
            , is_trended_and_seasoned:bool=True
            , has_events:bool=True
            , has_intercept:bool=True):
        """Convenience method for stochastically simulating a MMM dataset from sampled parameters.
        Can use via:
        '''
        from data_generator import DataGenerator
        dg = DataGenerator.generate_typical_mmm_dataset()
        '''
        """
        dg = DataGenerator(min_date, max_date, num_channels)
        assert adstock_fn in dg.adstock_fns.keys()
        if is_adstocked: dg.add_adstock_effect(fn=adstock_fn)
        if is_saturated: dg.add_saturation_effect()
        if is_trended_and_seasoned: dg.add_trend_and_seasonality_effects()
        if has_events: dg.add_event_effects()
        if has_intercept: dg.add_intercept()
        dg.compile_target_variable()
        return dg
