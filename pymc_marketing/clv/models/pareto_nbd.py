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
    purchase_covariate_cols: list, optional
        List containing column names of covariates for customer purchase rates.
    dropout_covariate_cols: list, optional
        List containing column names of covariates for customer dropouts.
    model_config: dict, optional
        Dictionary containing model parameters:
            * `r_prior`: Shape parameter of time between purchases; defaults to `pymc.Weibull.dist(alpha=2, beta=1)`
            * `alpha_prior`: Scale parameter of time between purchases; defaults to `pymc.Weibull.dist(alpha=2, beta=10)`
            * `s_prior`: Shape parameter of time until dropout; defaults to `pymc.Weibull.dist(alpha=2, beta=1)`
            * `beta_prior`: Scale parameter of time until dropout; defaults to `pymc.Weibull.dist(alpha=2, beta=10)`
            * `alpha_scale_prior: Scale parameter of time between purchases if using covariates; defaults to `pymc.Weibull.dist(alpha=2, beta=10)`
            * `beta_scale_prior: Scale parameter of time until dropout if using covariates; ; defaults to `pymc.Weibull.dist(alpha=2, beta=10)`
            * `purchase_covariates_prior`: Coefficients for purchase rate covariates; defaults to `pymc.Normal.dist(mu=0, sigma=1, n=len(purchase_covariates_cols)`
            * `dropout_covariates_prior`: Coefficients for dropout covariates; defaults to `pymc.Normal.dist(mu=0, sigma=1, n=len(dropout_covariates_cols)`
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
        purchase_covariate_cols: Optional[List[str]] = None,
        dropout_covariate_cols: Optional[List[str]] = None,
        model_config: Optional[Dict] = None,
        sampler_config: Optional[Dict] = None,
    ):
        # TODO: Write a separate, more comprehensive RFM validation method in a future PR.
        # Perform validation checks on column names and customer identifiers
        self._validate_column_names(data, ["customer_id", "frequency", "recency", "T"])
        if len(np.unique(data["customer_id"])) != len(data["customer_id"]):
            raise ValueError("Customers must have unique ID labels.")

        # validate if data contains specified column names for covariates
        for covariates in [purchase_covariate_cols, dropout_covariate_cols]:
            if covariates is not None:
                self._validate_column_names(data, covariates)

        self.purchase_covariate_shape = (
            0
            if purchase_covariate_cols is None
            else np.array(purchase_covariate_cols).shape
        )
        self.dropout_covariate_shape = (
            0
            if dropout_covariate_cols is None
            else np.array(dropout_covariate_cols).shape
        )

        self.purchase_covariate_cols = purchase_covariate_cols
        self.dropout_covariate_cols = dropout_covariate_cols

        # TODO: self.data is persisted in idata object.
        #       Consider making the assignment optional as it can reduce saved model size considerably.
        super().__init__(
            data=data,
            model_config=model_config,
            sampler_config=sampler_config,
        )

        # Type declaration required for mypy
        self.data: pd.DataFrame

        self.coords = {"customer_id": self.data["customer_id"]}

        self.r_prior = self._create_distribution(self.model_config["r_prior"])
        self.s_prior = self._create_distribution(self.model_config["s_prior"])

        priors = [self.r_prior, self.s_prior]

        if self.purchase_covariate_cols is None:
            self.alpha_prior = self._create_distribution(
                self.model_config["alpha_prior"]
            )
            priors.extend([self.alpha_prior])
        else:
            self.coords["purchase_covariates"] = self.purchase_covariate_cols  # type: ignore
            self.alpha_scale_prior = self._create_distribution(
                self.model_config["alpha_scale_prior"]
            )
            pr_ndim = len(self.purchase_covariate_shape)  # type: ignore
            self.purchase_covariates_coeff = self._create_distribution(
                self.model_config["purchase_covariates_prior"], ndim=pr_ndim
            )
            priors.extend([self.alpha_scale_prior, self.purchase_covariates_coeff])

        if self.dropout_covariate_cols is None:
            self.beta_prior = self._create_distribution(self.model_config["beta_prior"])
            priors.extend([self.beta_prior])
        else:
            self.coords["dropout_covariates"] = self.dropout_covariate_cols  # type: ignore

            self.beta_scale_prior = self._create_distribution(
                self.model_config["beta_scale_prior"]
            )
            dr_ndim = len(self.purchase_covariate_shape)  # type: ignore
            self.dropout_covariates_coeff = self._create_distribution(
                self.model_config["dropout_covariates_prior"], ndim=dr_ndim
            )
            priors.extend([self.beta_scale_prior, self.dropout_covariates_coeff])

        self._process_priors(*priors)

        # TODO: Add self.build_model() call here

    @property
    def default_model_config(self) -> Dict[str, Dict]:
        return {
            "r_prior": {"dist": "Weibull", "kwargs": {"alpha": 2, "beta": 1}},
            "alpha_prior": {"dist": "Weibull", "kwargs": {"alpha": 2, "beta": 10}},
            "s_prior": {"dist": "Weibull", "kwargs": {"alpha": 2, "beta": 1}},
            "beta_prior": {"dist": "Weibull", "kwargs": {"alpha": 2, "beta": 10}},
            "alpha_scale_prior": {
                "dist": "Weibull",
                "kwargs": {"alpha": 2, "beta": 10},
            },
            "beta_scale_prior": {"dist": "Weibull", "kwargs": {"alpha": 2, "beta": 10}},
            "dropout_covariates_prior": {
                "dist": "StudentT",
                "kwargs": {"nu": 1, "shape": self.dropout_covariate_shape},
            },
            "purchase_covariates_prior": {
                "dist": "StudentT",
                "kwargs": {"nu": 1, "shape": self.purchase_covariate_shape},
            },
        }

    def build_model(  # type: ignore
        self,
    ) -> None:
        with pm.Model(coords=self.coords) as self.model:
            # purchase rate priors
            r = self.model.register_rv(self.r_prior, name="r")
            if self.purchase_covariate_cols is not None:
                alpha_scale = self.model.register_rv(
                    self.alpha_scale_prior, name="alpha_scale"
                )

                purchase_covariates_coeff = self.model.register_rv(
                    self.purchase_covariates_coeff,
                    name="purchase_covariates_coeff",
                    dims=("purchase_covariates",),
                )

                alpha = pm.Deterministic(
                    name="alpha",
                    var=alpha_scale
                    * pm.math.exp(
                        -pm.math.dot(
                            self.data[self.purchase_covariate_cols],
                            purchase_covariates_coeff,
                        )
                    ),
                    dims="customer_id",
                )
            else:
                alpha = self.model.register_rv(self.alpha_prior, name="alpha")

            # dropout priors
            s = self.model.register_rv(self.s_prior, name="s")
            if self.dropout_covariate_cols is not None:
                beta_scale = self.model.register_rv(
                    self.beta_scale_prior, name="beta_scale"
                )

                dropout_covariates_coeff = self.model.register_rv(
                    self.dropout_covariates_coeff,
                    name="dropout_covariates_coeff",
                    dims=("dropout_covariates",),
                )

                beta = pm.Deterministic(
                    name="beta",
                    var=beta_scale
                    * pm.math.exp(
                        -pm.math.dot(
                            self.data[self.dropout_covariate_cols],
                            dropout_covariates_coeff,
                        )
                    ),
                    dims="customer_id",
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
        Utility function to process covariates. If used, covariate column names are validated,
        and alpha and/or beta parameters replaced with covariate expressions.
        """

        # TODO: broadcasting needs to be resolved for out-of-sample data
        if data is None:
            data = self.data

        if self.purchase_covariate_cols is None:
            alpha = self.fit_result["alpha"]
        else:
            self._validate_column_names(data, self.purchase_covariate_cols)
            purchase_xarray = xarray.DataArray(
                data[self.purchase_covariate_cols],
                dims=["customer_id", "purchase_covariates"],
            )
            alpha_scale = self.fit_result["alpha_scale"]
            purchase_coeff = self.fit_result["purchase_covariates_coeff"]
            alpha = alpha_scale * np.exp(
                -xarray.dot(purchase_coeff, purchase_xarray, dims="purchase_covariates")
            )

        if self.dropout_covariate_cols is None:
            beta = self.fit_result["beta"]
        else:
            self._validate_column_names(data, self.dropout_covariate_cols)
            dropout_xarray = xarray.DataArray(
                data[self.dropout_covariate_cols],
                dims=["customer_id", "dropout_covariates"],
            )
            beta_scale = self.fit_result["beta_scale"]
            dropout_coeff = self.fit_result["dropout_covariates_coeff"]
            beta = beta_scale * np.exp(
                -xarray.dot(dropout_coeff, dropout_xarray, dims="dropout_covariates")
            )
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

    def _logp(
        self,
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
        # Add one dummy dimension to the right of every parameter so they broadcast with the `T` vector
        # If using covariates, this dimension is already included for alpha and beta
        if self.purchase_covariate_cols is None:
            alpha = alpha.values[..., None]
        else:
            alpha = alpha.values

        if self.dropout_covariate_cols is None:
            beta = beta.values[..., None]
        else:
            beta = beta.values

        pareto_dist = ParetoNBD.dist(
            r=r.values[..., None],
            alpha=alpha,
            s=s.values[..., None],
            beta=beta,
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

        # TODO: If requested, this is still possible from a what-if standpoint.
        if self.purchase_covariate_cols or self.purchase_covariate_cols is not None:
            raise NotImplementedError(
                "New customer estimates not currently supported with covariates"
            )

        t = np.asarray(t)

        # TODO: _unload_params requires modification for covariates w/ new customers
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

        # TODO: If requested, covariate support still possible
        if self.purchase_covariate_cols or self.dropout_covariate_cols is not None:
            raise NotImplementedError(
                "New customer estimates not currently supported with covariates"
            )

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
            if self.purchase_covariate_cols is not None:
                alpha_scale = pm.HalfFlat("alpha_scale")
                purchase_covariates_coeff = pm.Flat(
                    "purchase_covariates_coeff", shape=len(self.purchase_covariate_cols)
                )

                alpha = pm.Deterministic(
                    "alpha",
                    alpha_scale
                    * pm.math.exp(
                        -pm.math.dot(
                            self.data[self.purchase_covariate_cols],
                            purchase_covariates_coeff,
                        )
                    ),
                )
            else:
                alpha = pm.HalfFlat("alpha")

            # dropout priors
            s = pm.HalfFlat("s")
            if self.dropout_covariate_cols is not None:
                beta_scale = pm.HalfFlat("beta_scale")
                dropout_covariates_coeff = pm.Flat(
                    "dropout_covariates_coeff", shape=len(self.dropout_covariate_cols)
                )

                beta = pm.Deterministic(
                    "beta",
                    beta_scale
                    * pm.math.exp(
                        -pm.math.dot(
                            self.data[self.dropout_covariate_cols],
                            dropout_covariates_coeff,
                        )
                    ),
                )
            else:
                beta = pm.HalfFlat("beta")

            pm.Gamma("population_purchase_rate", alpha=r, beta=alpha, **shape_kwargs)
            pm.Gamma("population_dropout", alpha=s, beta=beta, **shape_kwargs)

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
