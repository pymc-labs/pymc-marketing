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

"""PyTensor utility functions."""

import arviz as az
import pandas as pd
import pytensor.tensor as pt
from arviz import InferenceData
from pymc import Model
from pytensor import clone_replace
from pytensor.graph import rewrite_graph, vectorize_graph
from pytensor.graph.basic import ancestors

try:
    from pymc.pytensorf import rvs_in_graph
except ImportError:
    from pymc.logprob.utils import rvs_in_graph


def extract_response_distribution(
    pymc_model: Model,
    idata: InferenceData,
    response_variable: str,
) -> pt.TensorVariable:
    """Extract the response distribution graph, conditioned on posterior parameters.

    Parameters
    ----------
    pymc_model : Model
        The PyMC model to extract the response distribution from.
    idata : InferenceData
        The inference data containing posterior samples.
    response_variable : str
        The name of the response variable to extract.

    Returns
    -------
    pt.TensorVariable
        The response distribution graph.

    Example
    -------
    `extract_response_distribution(model, idata, "channel_contribution")`
    returns a graph that computes `"channel_contribution"` as a function of both
    the newly introduced budgets and the posterior of model parameters.
    """
    # Convert InferenceData to a sample-major xarray
    posterior = az.extract(idata).transpose("sample", ...)  # type: ignore

    # The PyMC variable to extract
    response_var = pymc_model[response_variable]

    # Identify which free RVs are needed to compute `response_var`
    free_rvs = set(pymc_model.free_RVs)
    needed_rvs = [
        rv for rv in ancestors([response_var], blockers=free_rvs) if rv in free_rvs
    ]
    placeholder_replace_dict = {
        pymc_model[rv.name]: pt.tensor(
            name=rv.name, shape=rv.type.shape, dtype=rv.dtype
        )
        for rv in needed_rvs
    }

    [response_var] = clone_replace(
        [response_var],
        replace=placeholder_replace_dict,
    )

    if rvs_in_graph([response_var]):
        raise RuntimeError("RVs found in the extracted graph, this is likely a bug")

    # Cleanup graph
    response_var = rewrite_graph(response_var, include=("canonicalize", "ShapeOpt"))

    # Replace placeholders with actual posterior samples
    replace_dict = {}
    for placeholder in placeholder_replace_dict.values():
        replace_dict[placeholder] = pt.constant(
            posterior[placeholder.name].astype(placeholder.dtype),
            name=placeholder.name,
        )

    # Vectorize across samples
    response_distribution = vectorize_graph(response_var, replace=replace_dict)

    # Final cleanup
    response_distribution = rewrite_graph(
        response_distribution,
        include=(
            "useless",
            "local_eager_useless_unbatched_blockwise",
            "local_useless_unbatched_blockwise",
        ),
    )

    return response_distribution


class ModelSamplerEstimator:
    """Estimate computational characteristics of a PyMC model using JAX/NumPyro.

    This utility measures the average evaluation time of the model's logp and gradients
    and estimates the number of integrator steps taken by NUTS during warmup + sampling.
    It then compiles the information into a single-row pandas DataFrame with helpful
    metadata to guide planning and benchmarking.

    Parameters
    ----------
    tune : int, default 1000
        Number of warmup iterations to use when estimating NUTS steps.
    draws : int, default 1000
        Number of sampling iterations to use when estimating NUTS steps.
    chains : int, default 1
        Intended number of chains (metadata only; not used in JAX runs here).
    sequential_chains : int, default 1
        Number of chains expected to run sequentially on the target environment.
        Used to scale the wall-clock time estimate.
    seed : int | None, default None
        Random seed used for the step estimation runs.

    Examples
    --------
    .. code-block:: python

        est = ModelSamplerEstimator(
            tune=1000, draws=1000, chains=4, sequential_chains=1, seed=1
        )
        df = est.run(model)
        print(df)
    """

    def __init__(
        self,
        *,
        tune: int = 1000,
        draws: int = 1000,
        chains: int = 1,
        sequential_chains: int = 1,
        seed: int | None = None,
    ) -> None:
        self.tune = int(tune)
        self.draws = int(draws)
        self.chains = int(chains)
        self.sequential_chains = int(sequential_chains)
        self.seed = seed

    def estimate_model_eval_time(self, model: Model, n: int | None = None) -> float:
        """Estimate average evaluation time (seconds) of logp+dlogp using JAX.

        Parameters
        ----------
        model : Model
            PyMC model whose logp and gradients are jitted and evaluated.
        n : int | None, optional
            Number of repeated evaluations to average over. If ``None``, a value
            is chosen to take roughly 5 seconds in total for a stable estimate.

        Returns
        -------
        float
            Average evaluation time in seconds.
        """
        from time import perf_counter_ns

        import numpy as np

        try:
            import jax
            from pymc.sampling.jax import get_jaxified_logp
        except Exception as err:  # pragma: no cover - environment specific
            raise RuntimeError(
                "JAX backend is required for ModelSamplerEstimator."
            ) from err

        initial_point = list(model.initial_point().values())
        logp_fn = get_jaxified_logp(model)
        logp_dlogp_fn = jax.jit(jax.value_and_grad(logp_fn, argnums=0))
        logp_res, grad_res = logp_dlogp_fn(initial_point)
        for val in (logp_res, *grad_res):
            if not np.isfinite(val).all():
                raise RuntimeError(
                    "logp or gradients are not finite at the model initial point; the model may be misspecified"
                )

        if n is None:
            start = perf_counter_ns()
            jax.block_until_ready(logp_dlogp_fn(initial_point))
            end = perf_counter_ns()
            n = max(5, int(5e9 / max(end - start, 1)))

        start = perf_counter_ns()
        for _ in range(n):
            jax.block_until_ready(logp_dlogp_fn(initial_point))
        end = perf_counter_ns()
        eval_time = (end - start) / n * 1e-9
        return float(eval_time)

    def estimate_num_steps_sampling(
        self,
        model: Model,
        *,
        tune: int | None = None,
        draws: int | None = None,
        seed: int | None = None,
    ) -> int:
        """Estimate total number of NUTS steps during warmup + sampling using NumPyro.

        Parameters
        ----------
        model : Model
            PyMC model to estimate steps for using a JAX/NumPyro NUTS kernel.
        tune : int | None, optional
            Warmup iterations. Defaults to the estimator setting if ``None``.
        draws : int | None, optional
            Sampling iterations. Defaults to the estimator setting if ``None``.
        seed : int | None, optional
            Random seed for the JAX run. Defaults to the estimator setting if ``None``.

        Returns
        -------
        int
            Total number of leapfrog steps across warmup + sampling.
        """
        import numpy as np

        try:
            import jax
            from numpyro.infer import MCMC, NUTS
            from pymc.sampling.jax import get_jaxified_logp
        except Exception as err:  # pragma: no cover - environment specific
            raise RuntimeError(
                "JAX and NumPyro are required for ModelSamplerEstimator."
            ) from err

        num_warmup = int(self.tune if tune is None else tune)
        num_samples = int(self.draws if draws is None else draws)

        initial_point = list(model.initial_point().values())
        logp_fn = get_jaxified_logp(model, negative_logp=False)
        nuts_kernel = NUTS(
            potential_fn=logp_fn,
            target_accept_prob=0.8,
            adapt_step_size=True,
            adapt_mass_matrix=True,
            dense_mass=False,
        )

        mcmc = MCMC(
            nuts_kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=1,
            postprocess_fn=None,
            chain_method="sequential",
            progress_bar=False,
        )

        if seed is None:
            rng_seed = int(np.random.default_rng().integers(2**32))
        else:
            rng_seed = int(seed)

        tune_rng, sample_rng = jax.random.split(jax.random.PRNGKey(int(rng_seed)), 2)
        mcmc.warmup(
            tune_rng,
            init_params=initial_point,
            extra_fields=("num_steps",),
            collect_warmup=True,
        )
        warmup_steps = int(mcmc.get_extra_fields()["num_steps"].sum())
        mcmc.run(sample_rng, extra_fields=("num_steps",))
        sample_steps = int(mcmc.get_extra_fields()["num_steps"].sum())
        return int(warmup_steps + sample_steps)

    def run(self, model: Model) -> pd.DataFrame:
        """Execute the estimation pipeline and return a single-row DataFrame.

        Parameters
        ----------
        model : Model
            PyMC model to evaluate.

        Returns
        -------
        pandas.DataFrame
            Single-row DataFrame with columns including ``num_steps``, ``eval_time_seconds``,
            ``sequential_chains``, and estimated sampling wall-clock time in seconds,
            minutes, and hours, along with metadata such as ``tune``, ``draws``, ``chains``,
            ``seed``, ``timestamp``, and ``model_name``.

        Examples
        --------
        .. code-block:: python

            df = ModelSamplerEstimator().run(model)
            df[
                [
                    "num_steps",
                    "eval_time_seconds",
                    "estimated_sampling_time_minutes",
                ]
            ]
        """
        import time

        steps = self.estimate_num_steps_sampling(
            model, tune=self.tune, draws=self.draws, seed=self.seed
        )
        eval_time_s = self.estimate_model_eval_time(model)

        sampling_time_seconds = float(
            eval_time_s * steps * max(self.sequential_chains, 1)
        )
        data = {
            "model_name": getattr(model, "name", "PyMCModel"),
            "num_steps": int(steps),
            "eval_time_seconds": float(eval_time_s),
            "sequential_chains": int(self.sequential_chains),
            "estimated_sampling_time_seconds": sampling_time_seconds,
            "estimated_sampling_time_minutes": sampling_time_seconds / 60.0,
            "estimated_sampling_time_hours": sampling_time_seconds / 3600.0,
            "tune": int(self.tune),
            "draws": int(self.draws),
            "chains": int(self.chains),
            "seed": int(self.seed) if self.seed is not None else None,
            "timestamp": pd.Timestamp.utcfromtimestamp(int(time.time())),
        }
        df = pd.DataFrame([data])
        return df
