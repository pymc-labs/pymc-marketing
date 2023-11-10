import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import xarray
from numpy import exp, log
from pymc.util import RandomState
from pytensor.compile import Mode, get_default_mode
from pytensor.graph import Constant, node_rewriter
from pytensor.scalar import Grad2F1Loop
from pytensor.tensor.elemwise import Elemwise
from scipy.special import betaln, gammaln, hyp2f1
from xarray_einstats.stats import logsumexp as xr_logsumexp

from pymc_marketing.clv.distributions import ParetoNBD
from pymc_marketing.clv.models.basic import CLVModel
from pymc_marketing.clv.utils import to_xarray


@node_rewriter([Elemwise])
def local_reduce_max_num_iters_hyp2f1_grad(fgraph, node):
    """Rewrite that reduces the maximum number of iterations in the hyp2f1 grad scalar loop.

    This is critical to get NUTS to converge in the beginning.
    Otherwise, it can take a long time to get started.
    """
    if not isinstance(node.op.scalar_op, Grad2F1Loop):
        return
    max_steps, *other_args = node.inputs

    # max_steps = switch(skip_loop, 0, 1e6) by default
    if max_steps.owner and max_steps.owner.op == pt.switch:
        cond, zero, max_steps_const = max_steps.owner.inputs
        if (isinstance(zero, Constant) and np.all(zero.data == 0)) and (
            isinstance(max_steps_const, Constant)
            and np.all(max_steps_const.data == 1e6)
        ):
            new_max_steps = pt.switch(cond, zero, np.array(int(1e5), dtype="int32"))
            return node.op.make_node(new_max_steps, *other_args).outputs


pytensor.compile.optdb["specialize"].register(
    "local_reduce_max_num_iters_hyp2f1_grad",
    local_reduce_max_num_iters_hyp2f1_grad,
    use_db_name_as_tag=False,  # Not included by default
)


class ParetoNBDModel(CLVModel):
    """Pareto Negative Binomial Distribution (Pareto/NBD) model for continuous, non-contractual customers,
    first introduced by Schmittlein, et al. [1]_, with additional derivations and predictive methods by
    Hardie & Fader [2]_ [3]_ [4]_.

    The Pareto/NBD model assumes the time duration a customer is active follows a Gamma distribution,
    and time between purchases is also Gamma-distributed while the customer is still active.

    This model requires data to be summarized by recency, frequency, and T for each customer,
    using `clv.rfm_summary()` or equivalent. Covariates impacting customer dropouts and transaction rates are optional.

    Parameters
    ----------
    data: pd.DataFrame
        DataFrame containing the following columns:
            * `frequency`: number of repeat purchases
            * `recency`: time between the first and the last purchase
            * `T`: time between the first purchase and the end of the observation period; model assumptions require T >= recency
            * `customer_id`: unique customer identifier
        Along with optional covariate columns.
    pr_covar_columns: list, optional
        List containing column names of covariates for customer purchase rates.
    dr_covar_columns: list, optional
        List containing column names of covariates for customer dropouts.
    model_config: dict, optional
        Dictionary containing model parameters:
            * `r_prior`: Shape parameter of time between purchases; defaults to `pymc.Weibull.dist(alpha=2, beta=1)`
            * `alpha_prior`: Scale parameter of time between purchases; defaults to `pymc.Weibull.dist(alpha=2, beta=10)`
            * `s_prior`: Shape parameter of time until dropout; defaults to `pymc.Weibull.dist(alpha=2, beta=1)`
            * `beta_prior`: Scale parameter of time until dropout; defaults to `pymc.Weibull.dist(alpha=2, beta=10)`
            * `alpha0_prior: Scale parameter of time between purchases if using covariates; defaults to `pymc.Weibull.dist(alpha=2, beta=10)`
            * `beta0_prior: Scale parameter of time until dropout if using covariates; ; defaults to `pymc.Weibull.dist(alpha=2, beta=10)`
            * `pr_prior`: Coefficients for purchase rate covariates; defaults to `pymc.Normal.dist(mu=0, sigma=1, n=len(tr_covar)`
            * `dr_prior`: Coefficients for dropout covariates; defaults to `pymc.Normal.dist(mu=0, sigma=1, n=len(dr_covar)`
        If not provided, the model will use default priors specified in the `default_model_config` class attribute.
    sampler_config: dict, optional
        Dictionary of sampler parameters. Defaults to None.

    Examples
    --------
        .. code-block:: python

            import pymc as pm
            from pymc_marketing.clv import ParetoNBDModel, rfm_summary

            rfm_df = rfm_summary(raw_data,'id_col_name','date_col_name')

            # Initialize model with customer data; `model_config` parameter is optional
            model = ParetoNBDModel(
                data=rfm_df,
                model_config={
                    "r_prior": pm.Weibull.dist(alpha=2,beta=1),
                    "alpha_prior": pm.Weibull.dist(alpha=2,beta=10),
                    "s_prior": pm.Weibull.dist(alpha=2,beta=1),
                    "beta_prior": pm.Weibull.dist(alpha=2,beta=10),
                },
            )

            # Fit model quickly to large datasets via the default Maximum a Posteriori method
            model.fit(fit_method='map')
            print(model.fit_summary())

            # Use 'mcmc' for more informative predictions and reliable performance on smaller datasets
            model.fit(fit_method='mcmc')
            print(model.fit_summary())

            # Predict number of purchases for customers over the next 10 time periods
            expected_purchases = model.expected_purchases(
                data=rfm_df,
                future_t=10,
            )

            # Predict probability of customer making 'n' purchases over 't' time periods
            # Data parameter is omitted here because predictions are ran on original dataset
            expected_num_purchases = model.expected_purchase_probability(
                n=[0, 1, 2, 3],
                future_t=[10,20,30,40],
            )

            new_data = pd.DataFrame(
                data = {
                "customer_id": [0, 1, 2, 3],
                "frequency": [5, 2, 1, 8],
                "recency": [7, 4, 2.5, 11],
                "T": [10, 8, 10, 22]
                }
            )

            # Predict probability customers will still be active in 'future_t' time periods
            probability_alive = model.expected_probability_alive(
                data=new_data,
                future_t=[0, 3, 6, 9],
            )

            # Predict number of purchases for a new customer over 't' time periods.
            expected_purchases_new_customer = model.expected_purchases_new_customer(
                t=[2, 5, 7, 10],
            )

    References
    ----------
    .. [1] David C. Schmittlein, Donald G. Morrison and Richard Colombo.
           "Counting Your Customers: Who Are They and What Will They Do Next".
           Management Science,Vol. 33, No. 1 (Jan., 1987), pp. 1-24.
    .. [2] Fader, Peter & G. S. Hardie, Bruce (2005).
           "A Note on Deriving the Pareto/NBD Model and Related Expressions".
           http://brucehardie.com/notes/009/pareto_nbd_derivations_2005-11-05.pdf
    .. [3] Fader, Peter & G. S. Hardie, Bruce (2014).
           "Additional Results for the Pareto/NBD Model".
           https://www.brucehardie.com/notes/015/additional_pareto_nbd_results.pdf
    .. [4] Fader, Peter & G. S. Hardie, Bruce (2014).
           "Deriving the Conditional PMF of the Pareto/NBD Model".
           https://www.brucehardie.com/notes/028/pareto_nbd_conditional_pmf.pdf
    .. [5] Fader, Peter & G. S. Hardie, Bruce (2007).
           "Incorporating Time-Invariant Covariates into the Pareto/NBD and BG/NBD Models".
           https://www.brucehardie.com/notes/019/time_invariant_covariates.pdf
    """

    _model_type = "Pareto/NBD"  # Pareto Negative-Binomial Distribution

    def __init__(
        self,
        data: pd.DataFrame,
        pr_covar_columns: Optional[List[str]] = None,
        dr_covar_columns: Optional[List[str]] = None,
        model_config: Optional[Dict] = None,
        sampler_config: Optional[Dict] = None,
    ):
        # TODO: Write a separate, more comprehensive RFM validation method in a future PR.
        # Perform validation checks on column names and customer identifiers
        self._validate_column_names(data, ["customer_id", "frequency", "recency", "T"])
        # TODO: self.data is persisted in idata object.
        #       Consider making the assignment optional as it can reduce saved model size considerably.
        self.data = data
        if len(np.unique(self.data["customer_id"])) != len(self.data["customer_id"]):
            raise ValueError("Customers must have unique ID labels.")
        self.coords = {"customer_id": self.data["customer_id"]}

        self.pr_covar_columns = pr_covar_columns
        self.dr_covar_columns = dr_covar_columns

        for _ in zip(
            [self.pr_covar_columns, self.dr_covar_columns],
            ["purchase_rate_covariates", "dropout_covariates"],
        ):
            if _[0] is not None:
                self._validate_column_names(data, _[0])
                # self.coords[_[1]] = _[0]  # type: ignore

        super().__init__(
            model_config=model_config,
            sampler_config=sampler_config,
        )

        self.r_prior = self._create_distribution(self.model_config["r_prior"])
        self.s_prior = self._create_distribution(self.model_config["s_prior"])

        priors = [self.r_prior, self.s_prior]

        if self.pr_covar_columns is None:
            self.alpha_prior = self._create_distribution(
                self.model_config["alpha_prior"]
            )
            priors.extend([self.alpha_prior])
        else:
            self.alpha0_prior = self._create_distribution(
                self.model_config["alpha0_prior"]
            )
            # TODO: Re-add coefficients when customer covariate coefficient priors are supported
            priors.extend([self.alpha0_prior])

        if self.dr_covar_columns is None:
            self.beta_prior = self._create_distribution(self.model_config["beta_prior"])
            priors.extend([self.beta_prior])
        else:
            self.beta0_prior = self._create_distribution(
                self.model_config["beta0_prior"]
            )
            # TODO: Re-add coefficients when customer covariate coefficient priors are supported
            priors.extend([self.beta0_prior])

        self._process_priors(*priors)

        # TODO: Add self.build_model() call here

    # TODO: _create_distributions changes required in clv/basic.py to support custom covariate coefficient distributions
    @property
    def default_model_config(self) -> Dict[str, Dict]:
        return {
            "r_prior": {"dist": "Weibull", "kwargs": {"alpha": 2, "beta": 1}},
            "alpha_prior": {"dist": "Weibull", "kwargs": {"alpha": 2, "beta": 10}},
            "s_prior": {"dist": "Weibull", "kwargs": {"alpha": 2, "beta": 1}},
            "beta_prior": {"dist": "Weibull", "kwargs": {"alpha": 2, "beta": 10}},
            "alpha0_prior": {"dist": "Weibull", "kwargs": {"alpha": 2, "beta": 10}},
            "beta0_prior": {"dist": "Weibull", "kwargs": {"alpha": 2, "beta": 10}},
            "dr_coeff": {"nu": 1, "dims": ("dropout_covariates",)},
            "pr_coeff": {"nu": 1, "dims": ("purchase_rate_covariates",)},
        }

    def build_model(  # type: ignore
        self,
    ) -> None:
        with pm.Model(coords=self.coords) as self.model:
            # purchase rate priors
            r = self.model.register_rv(self.r_prior, name="r")
            if self.pr_covar_columns is not None:
                alpha0 = self.model.register_rv(self.alpha0_prior, name="alpha0")
                # TODO: _create_distributions changes required in clv/basic.py to support custom distributions
                pr_coeff = pm.StudentT(
                    name="pr_coeff",
                    nu=self.model_config["pr_coeff"]["nu"],
                    shape=2,  # self.model_config["pr_coeff"]["dims"],
                )
                # TODO: coordinates must be resolved
                alpha = pm.Deterministic(
                    name="alpha",
                    var=alpha0
                    * pm.math.exp(
                        -pm.math.dot(self.data[self.pr_covar_columns], pr_coeff)
                    ),
                )
            else:
                alpha = self.model.register_rv(self.alpha_prior, name="alpha")

            # dropout priors
            s = self.model.register_rv(self.s_prior, name="s")
            if self.dr_covar_columns is not None:
                beta0 = self.model.register_rv(self.beta0_prior, name="beta0")
                # TODO: _create_distributions changes required in clv/basic.py to support custom distributions
                dr_coeff = pm.StudentT(
                    name="dr_coeff",
                    nu=self.model_config["dr_coeff"]["nu"],
                    shape=2,  # self.model_config["dr_coeff"]["dims"],
                )
                # TODO: coordinates must be resolved
                beta = pm.Deterministic(
                    name="beta",
                    var=beta0
                    * pm.math.exp(
                        -pm.math.dot(self.data[self.dr_covar_columns], dr_coeff)
                    ),
                )
            else:
                beta = self.model.register_rv(self.beta_prior, name="beta")

            ParetoNBD(
                name="likelihood",
                r=r,
                alpha=alpha,
                s=s,
                beta=beta,
                T=self.data["T"],
                observed=np.stack(
                    (self.data["recency"], self.data["frequency"]), axis=1
                ),
                dims="customer_id",
            )

    @staticmethod
    def _validate_column_names(data: pd.DataFrame, columns: List[str]):
        """Utility function to check fit data for required column names."""
        for name in columns:
            if name not in data.columns:
                err = f"{name} column is missing from data"
                raise KeyError(err)

    def _unload_params(
        self,
        data: Union[pd.DataFrame, None],
    ) -> Tuple[Any, ...]:
        """Utility function retrieving posterior parameters for predictive methods"""
        r = self.fit_result["r"]
        s = self.fit_result["s"]

        alpha, beta = self._process_covariates(data)

        return r, alpha, s, beta

    def _process_covariates(
        self,
        data: Union[pd.DataFrame, None],
    ) -> Tuple[Any, Any]:
        """
        Utility function to process covariates into model parameters.
        """

        if data is None:
            alpha = self.fit_result["alpha"]
            beta = self.fit_result["beta"]
        else:
            if self.pr_covar_columns is None:
                alpha = self.fit_result["alpha"]
            else:
                self._validate_column_names(data, self.pr_covar_columns)
                alpha0 = self.fit_result["alpha0"]
                pr_coeff = self.fit_result["pr_coeff"]
                alpha = alpha0 * np.exp(-np.dot(data[self.pr_covar_columns], pr_coeff))
                alpha = to_xarray("customer_id", alpha)
            if self.dr_covar_columns is None:
                beta = self.fit_result["beta"]
            else:
                self._validate_column_names(data, self.dr_covar_columns)
                beta0 = self.fit_result["beta0"]
                dr_coeff = self.fit_result["dr_coeff"]
                beta = beta0 * np.exp(-np.dot(data[self.dr_covar_columns], dr_coeff))
                beta = to_xarray("customer_id", beta)
        return alpha, beta

    def _process_customers(
        self,
        data: Union[pd.DataFrame, None],
    ) -> Tuple[xarray.DataArray, ...]:
        """Utility function assigning default customer arguments
        for predictive methods and converting to xarrays.
        """
        if data is None:
            data = self.data
        else:
            # TODO: How beneficial is standardizing column casing given covariate name specifications?
            # data.columns = data.columns.str.upper()
            self._validate_column_names(
                data, ["customer_id", "frequency", "recency", "T"]
            )

        return to_xarray(
            data["customer_id"], data["frequency"], data["recency"], data["T"]
        )

    @staticmethod
    def _logp(
        r: xarray.DataArray,
        alpha: xarray.DataArray,
        s: xarray.DataArray,
        beta: xarray.DataArray,
        x: xarray.DataArray,
        t_x: xarray.DataArray,
        T: xarray.DataArray,
    ) -> xarray.DataArray:
        """
        Utility function for using ParetoNBD log-likelihood in predictive methods.
        """
        # Add one dummy dimension to the right of the scalar parameters, so they broadcast with the `T` vector
        pareto_dist = ParetoNBD.dist(
            r=r.values[..., None],
            alpha=alpha.values[..., None],
            s=s.values[..., None],
            beta=beta.values[..., None],
            T=T.values,
        )
        values = np.vstack((t_x.values, x.values)).T
        # TODO: Instead of compiling this function everytime this method is called
        #  we could compile it once (with mutable inputs) and cache it for reuse with new inputs.
        loglike = pm.logp(pareto_dist, values).eval()
        return xarray.DataArray(data=loglike, dims=("chain", "draw", "customer_id"))

    def fit(self, fit_method: str = "map", **kwargs):  # type: ignore
        """Infer posteriors of model parameters to run predictions.

        Parameters
        ----------
        fit_method: str
            Method used to fit the model. Options are:
            * "map": Posterior point estimates via Maximum a Posteriori (default)
            * "mcmc": Full posterior distributions via No U-Turn Sampler (NUTS)
        kwargs:
            Other keyword arguments passed to the underlying PyMC routines
        """

        mode = get_default_mode()
        if fit_method == "mcmc":
            # Include rewrite in mode
            opt_qry = mode.provided_optimizer.including(
                "local_reduce_max_num_iters_hyp2f1_grad"
            )
            mode = Mode(linker=mode.linker, optimizer=opt_qry)

        with pytensor.config.change_flags(mode=mode, on_opt_error="raise"):
            # Suppress annoying warning
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    message="Optimization Warning: The Op hyp2f1 does not provide a C implementation. As well as being potentially slow, this also disables loop fusion.",
                    action="ignore",
                    category=UserWarning,
                )
                super().fit(fit_method, **kwargs)

        # TODO: return self or None?

    def expected_purchases(
        self,
        future_t: Union[float, np.ndarray, pd.Series],
        data: Optional[pd.DataFrame] = None,
    ) -> xarray.DataArray:
        """
        Given *recency*, *frequency*, and *T* for an individual customer, this method predicts the
        expected number of future purchases across *future_t* time periods.

        `data` parameter is not required if estimating probabilities for customers in model fit dataset.

        Adapted from equation (41) In Bruce Hardie's notes [2]_, and `lifetimes` package:
        https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/fitters/pareto_nbd_fitter.py#L242

        Parameters
        ----------
        future_t: array_like
            Number of time periods to predict expected purchases.
        data: pd.DataFrame
            Optional dataframe containing the following columns:
                * `frequency`: number of repeat purchases
                * `recency`: time between the first and the last purchase
                * `T`: time between the first purchase and the end of the observation period, model assumptions require T >= recency
                * `customer_id`: unique customer identifier

        References
        ----------
        .. [2] Fader, Peter & G. S. Hardie, Bruce (2005).
               "A Note on Deriving the Pareto/NBD Model and Related Expressions."
               http://brucehardie.com/notes/009/pareto_nbd_derivations_2005-11-05.pdf
        """

        x, t_x, T = self._process_customers(data)
        r, alpha, s, beta = self._unload_params(data)
        loglike = self._logp(r, alpha, s, beta, x, t_x, T)

        first_term = (
            gammaln(r + x)
            - gammaln(r)
            + r * log(alpha)
            + s * log(beta)
            - (r + x) * log(alpha + T)
            - s * log(beta + T)
        )
        second_term = log(r + x) + log(beta + T) - log(alpha + T)
        third_term = log(
            (1 - ((beta + T) / (beta + T + future_t)) ** (s - 1)) / (s - 1)
        )

        exp_purchases = exp(first_term + second_term + third_term - loglike)

        return exp_purchases.transpose(
            "chain", "draw", "customer_id", missing_dims="ignore"
        )

    def expected_purchases_new_customer(
        self,
        t: Union[np.ndarray, pd.Series],
    ) -> xarray.DataArray:
        """
        Expected number of purchases for a new customer across *t* time periods.

        Adapted from equation (27) in Bruce Hardie's notes [2]_, and `lifetimes` package:
        https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/fitters/pareto_nbd_fitter.py#L359

        Parameters
        ----------
        t: array_like
            Number of time periods over which to estimate purchases.

        References
        ----------
        .. [2] Fader, Peter & G. S. Hardie, Bruce (2005).
               "A Note on Deriving the Pareto/NBD Model and Related Expressions."
               http://brucehardie.com/notes/009/pareto_nbd_derivations_2005-11-05.pdf
        """

        t = np.asarray(t)

        r, alpha, s, beta = self._unload_params(self.data)
        first_term = r * beta / alpha / (s - 1)
        second_term = 1 - (beta / (beta + t)) ** (s - 1)

        return (first_term * second_term).transpose(
            "chain", "draw", "t", missing_dims="ignore"
        )

    def expected_probability_alive(
        self,
        future_t: Union[int, float] = 0,
        data: Optional[pd.DataFrame] = None,
    ) -> xarray.DataArray:
        """
        Compute the probability that a customer with history *frequency*, *recency*, and *T*
        is currently active. Can also estimate alive probability for *future_t* periods into the future.

        `data` parameter is not required if estimating probabilities for customers in model fit dataset.

        Adapted from equation (18) in Bruce Hardie's notes [3]_.

        Parameters
        ----------
        future_t: scalar
            Number of time periods in the future to estimate alive probability; defaults to 0.
        data: pd.DataFrame
            Optional dataframe containing the following columns:
                * `frequency`: number of repeat purchases
                * `recency`: time between the first and the last purchase
                * `T`: time between the first purchase and the end of the observation period, model assumptions require T >= recency
                * `customer_id`: unique customer identifier

        References
        ----------
        .. [3] Fader, Peter & G. S. Hardie, Bruce (2014).
               "Additional Results for the Pareto/NBD Model."
               https://www.brucehardie.com/notes/015/additional_pareto_nbd_results.pdf
        """

        x, t_x, T = self._process_customers(data)
        r, alpha, s, beta = self._unload_params(data)
        loglike = self._logp(r, alpha, s, beta, x, t_x, T)

        term1 = gammaln(r + x) - gammaln(r)
        term2 = r * log(alpha / (alpha + T))
        term3 = -x * log(alpha + T)
        term4 = s * log(beta / (beta + T + future_t))

        prob_alive = exp(term1 + term2 + term3 + term4 - loglike)

        return prob_alive.transpose(
            "chain", "draw", "customer_id", missing_dims="ignore"
        )

    def expected_purchase_probability(
        self,
        n_purchases: Union[int, np.ndarray, pd.Series],
        future_t: Union[float, np.ndarray, pd.Series],
        data: Optional[pd.DataFrame] = None,
    ) -> xarray.DataArray:
        """
        Estimate probability of *n_purchases* over *future_t* time periods,
        given an individual customer's current *frequency*, *recency*, and *T*.

        `data` parameter is not required if estimating probabilities for customers in model fit dataset.

        Adapted from equation (16) in Bruce Hardie's notes [4]_, and `lifetimes` package:
        https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/fitters/pareto_nbd_fitter.py#L388

        Parameters
        ----------
        n_purchases: int
            Number of purchases predicted.
        future_t: scalar
            Time periods over which the probability should be estimated.
        data: pd.DataFrame
            Optional dataframe containing the following columns:
                * `frequency`: number of repeat purchases
                * `recency`: time between the first and the last purchase
                * `T`: time between the first purchase and the end of the observation period, model assumptions require T >= recency
                * `customer_id`: unique customer identifier

        References
        ----------
        .. [4] Fader, Peter & G. S. Hardie, Bruce (2014).
               "Deriving the Conditional PMF of the Pareto/NBD Model."
               https://www.brucehardie.com/notes/028/pareto_nbd_conditional_pmf.pdf
        """

        x, t_x, T = self._process_customers(data)
        r, alpha, s, beta = self._unload_params(data)
        loglike = self._logp(r, alpha, s, beta, x, t_x, T)

        _alpha_less_than_beta = alpha < beta
        min_of_alpha_beta = xarray.where(_alpha_less_than_beta, alpha, beta)
        max_of_alpha_beta = xarray.where(_alpha_less_than_beta, beta, alpha)
        p = xarray.where(_alpha_less_than_beta, r + x + n_purchases, s + 1)

        abs_alpha_beta = max_of_alpha_beta - min_of_alpha_beta

        log_p_zero = (
            gammaln(r + x)
            + r * log(alpha)
            + s * log(beta)
            - (gammaln(r) + (r + x) * log(alpha + T) + s * log(beta + T) + loglike)
        )
        log_B_one = (
            gammaln(r + x + n_purchases)
            + r * log(alpha)
            + s * log(beta)
            - (
                gammaln(r)
                + (r + x + n_purchases) * log(alpha + T + future_t)
                + s * log(beta + T + future_t)
            )
        )
        log_B_two = (
            r * log(alpha)
            + s * log(beta)
            + gammaln(r + s + x)
            + betaln(r + x + n_purchases, s + 1)
            + log(
                hyp2f1(
                    r + s + x,
                    p,
                    r + s + x + n_purchases + 1,
                    abs_alpha_beta / (max_of_alpha_beta + T),
                )
            )
            - (gammaln(r) + gammaln(s) + (r + s + x) * log(max_of_alpha_beta + T))
        )

        def _log_B_three(i):
            return (
                r * log(alpha)
                + s * log(beta)
                + gammaln(r + s + x + i)
                + betaln(r + x + n_purchases, s + 1)
                + log(
                    hyp2f1(
                        r + s + x + i,
                        p,
                        r + s + x + n_purchases + 1,
                        abs_alpha_beta / (max_of_alpha_beta + T + future_t),
                    )
                )
                - (
                    gammaln(r)
                    + gammaln(s)
                    + (r + s + x + i) * log(max_of_alpha_beta + T + future_t)
                )
            )

        zeroth_term = (n_purchases == 0) * (1 - exp(log_p_zero))

        # ignore numerical errors when future_t <= 0,
        # this is an unusual edge case in practice, so refactoring is unwarranted
        with np.errstate(divide="ignore", invalid="ignore"):
            first_term = (
                n_purchases * log(future_t)
                - gammaln(n_purchases + 1)
                + log_B_one
                - loglike
            )
            second_term = log_B_two - loglike

            third_term = xr_logsumexp(
                xarray.concat(
                    [
                        i * log(future_t) - gammaln(i + 1) + _log_B_three(i) - loglike
                        for i in range(n_purchases + 1)
                    ],
                    dim="concat_dim_",
                ),
                dims="concat_dim_",
            )

        purchase_prob = zeroth_term + exp(
            xr_logsumexp(
                xarray.concat([first_term, second_term, third_term], dim="_concat_dim"),
                b=xarray.DataArray(data=[1, 1, -1], dims="_concat_dim"),
                dims="_concat_dim",
            )
        )

        if future_t <= 0:
            purchase_prob = purchase_prob.fillna(0)

        return purchase_prob.transpose(
            "chain", "draw", "customer_id", missing_dims="ignore"
        )

    def _distribution_new_customers(
        self,
        random_seed: Optional[RandomState] = None,
        T: Union[None, np.ndarray, pd.Series] = None,
        var_names: Sequence[str] = (
            "population_dropout",
            "population_purchase_rate",
            "customer_population",
        ),
    ) -> xarray.Dataset:
        """Utility function for posterior predictive sampling from dropout and purchase rate distributions."""
        if T is None:
            T = self.data["T"]

        # This is the shape if using fit_method="map"
        if self.fit_result.dims == {"chain": 1, "draw": 1}:
            shape_kwargs = {"shape": 1000}
        else:
            shape_kwargs = {}

        with pm.Model():
            # purchase rate priors
            r = pm.HalfFlat("r")
            alpha = pm.HalfFlat("alpha")

            # dropout priors
            s = pm.HalfFlat("s")
            beta = pm.HalfFlat("beta")

            pm.Gamma(
                "population_purchase_rate", alpha=r, beta=1 / alpha, **shape_kwargs
            )
            pm.Gamma("population_dropout", alpha=s, beta=1 / beta, **shape_kwargs)

            ParetoNBD(
                name="customer_population",
                r=r,
                alpha=alpha,
                s=s,
                beta=beta,
                T=T,
            )

            return pm.sample_posterior_predictive(
                self.fit_result,
                var_names=var_names,
                random_seed=random_seed,
            ).posterior_predictive

    def distribution_new_customer_dropout(
        self,
        random_seed: Optional[RandomState] = None,
    ) -> xarray.Dataset:
        """Sample from the Gamma distribution representing dropout times for new customers.

        This is the duration of time a new customer is active before churning, or dropping out.

        Parameters
        ----------
        random_seed: RandomState, optional
            Random state to use for sampling.

        Returns
        -------
        xr.Dataset
            Dataset containing the posterior samples for the population-level dropout rate.
        """
        return self._distribution_new_customers(
            random_seed=random_seed,
            T=None,
            var_names=["population_dropout"],
        )["population_dropout"]

    def distribution_new_customer_purchase_rate(
        self,
        random_seed: Optional[RandomState] = None,
    ) -> xarray.Dataset:
        """Sample from the Gamma distribution representing purchase rates for new customers.

        This is the purchase rate for a new customer and determines the time between
        purchases for any new customer.

        Parameters
        ----------
        random_seed : RandomState, optional
            Random state to use for sampling.

        Returns
        -------
        xr.Dataset
            Dataset containing the posterior samples for the population-level purchase rate.
        """
        return self._distribution_new_customers(
            random_seed=random_seed,
            T=None,
            var_names=["population_purchase_rate"],
        )["population_purchase_rate"]

    def distribution_customer_population(
        self,
        T: Union[None, np.ndarray, pd.Series] = None,
        random_seed: Optional[RandomState] = None,
    ) -> xarray.Dataset:
        """Pareto/NBD process representing purchases across the customer population.

        This is the distribution of purchase frequencies given 'T' observation periods for each customer.

        Parameters
        ----------
        T: array_like
            Number of observation periods for each customer. If not provided, T values from fit dataset will be used.
        random_seed : RandomState, optional
            Random state to use for sampling.

        Returns
        -------
        xr.Dataset
            Dataset containing the posterior samples for the customer population.
        """
        return self._distribution_new_customers(
            random_seed=random_seed,
            T=T,
            var_names=["customer_population"],
        )["customer_population"]
