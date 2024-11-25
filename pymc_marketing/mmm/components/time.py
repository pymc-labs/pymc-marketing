# Imports
from typing import Optional, Any
import pymc as pm
import numpy as np
import pytensor.tensor as pt
from pymc.gp.cov import Covariance
from pymc_marketing.prior import Prior
from pymc_marketing.hsgp_kwargs import HSGPKwargs
from pymc_marketing.mmm.components.base import GaussianProcessBase

class HSGPTransformation(GaussianProcessBase):
    """Hilbert Space Gaussian Process (HSGP) Transformation."""
    def __init__(
        self,
        prefix: str = "gp",
        priors: Optional[dict[str, Prior]] = None,
        hsgp_kwargs: Optional[HSGPKwargs] = None,
        cov_func: Optional[Covariance] = None,
    ):
        # Ensure default priors are available before superclass init
        default_priors = {
            "eta": Prior("Exponential", lam=1),
            "lengthscale": Prior("InverseGamma", mu=5, sigma=5),
        }
        if priors is not None:
            default_priors.update(priors)

        super().__init__(prefix=prefix, priors=default_priors)
        self.hsgp_kwargs = hsgp_kwargs or HSGPKwargs()
        self.cov_func = cov_func

    def function(
        self,
        time_index: pt.TensorLike,
        eta: pt.TensorLike,
        lengthscale: pt.TensorLike,
        dims: tuple[str, ...],
        cov_func: Optional[Covariance] = None,
    ) -> pt.TensorVariable:
        """Define the HSGP logic with dimensionality."""
        model = pm.modelcontext(None)

        if self.hsgp_kwargs.L is None:
            self.hsgp_kwargs.L = float(time_index.eval().mean()) * 2

        cov_func = cov_func or self.cov_func
        if cov_func is None:
            cov_func = eta**2 * pm.gp.cov.Matern52(1, ls=lengthscale)

        model.add_coord("m", np.arange(self.hsgp_kwargs.m))

        gp = pm.gp.HSGP(
            m=[self.hsgp_kwargs.m], L=[self.hsgp_kwargs.L], cov_func=cov_func
        )
        phi, sqrt_psd = gp.prior_linearized(time_index[:, None] - time_index.mean())

        non_time_dims = [dim for dim in dims if dim != "time"]
        dims_shape = [len(model.coords[dim]) for dim in non_time_dims]
        num_time = time_index.shape[0]

        hsgp_coefs = pm.Normal(f"{self.prefix}_coefs", dims=(*non_time_dims, "m"))

        hsgp_coefs = pt.reshape(hsgp_coefs, (*dims_shape, self.hsgp_kwargs.m))
        sqrt_psd = sqrt_psd.reshape((1,) * len(non_time_dims) + (self.hsgp_kwargs.m,))

        broadcast_shape = (num_time,) + tuple(1 for _ in dims_shape) + (self.hsgp_kwargs.m,)
        phi = pt.broadcast_to(
            phi.reshape(broadcast_shape), (num_time, *dims_shape, self.hsgp_kwargs.m)
        )

        hsgp_coefs = pt.broadcast_to(hsgp_coefs, (1, *dims_shape, self.hsgp_kwargs.m))
        sqrt_psd = pt.broadcast_to(
            sqrt_psd, (1,) * len(dims_shape) + (self.hsgp_kwargs.m,)
        )

        f = pt.sum(phi * hsgp_coefs * sqrt_psd, axis=-1)

        f = f - f.mean(axis=0, keepdims=True) + 1

        return pm.Deterministic(f"{self.prefix}_output", f, dims=dims)