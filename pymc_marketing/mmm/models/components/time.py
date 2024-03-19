import pymc as pm
from typing import Union, Optional, Tuple
from scipy.stats import gamma, invgamma
import numpy as np

def find_c_and_m(
    lower: Union[int, float], upper: Union[int, float], mass: Union[int, float], N: int
) -> tuple[int, float]:
    """Find good values for `c` and `m`.

    Given a prior for the lengthscale, choose m and c such that the HSGP
    approx is accurate over the bulk of the prior.  Choose the smallest m and largest
    c for computational efficiency.
    """
    # Set fudge factor so approx is accurate outside of (lower, upper)
    fudge_factor = 1 + 2 * (1 - mass)

    S = N // 2
    c = 1.1

    # Increment c by 0.1, starting at minimum for Matern52
    while c < 4.1 * (upper / S):
        c += 0.1

    # Calculate m for given c
    m = 2.65 * c / (lower / S)

    m = int(np.round(fudge_factor * m))
    c = fudge_factor * c

    return m, c


def compute_beta_guess_for_gamma(alpha: Union[int, float], lower: Union[int, float], upper: Union[int, float]) -> Union[int, float]:
    """Compute a guess for the beta parameter of a gamma distribution.

    Assume that the mode is the midpoint between the lower and upper bounds.
    The mode is given by (alpha - 1) / beta if alpha > 1, and 1 otherwise,
    so beta can easily be solved for.
    """
    mode = (lower + upper) / 2
    return (alpha - 1) / mode if alpha > 1 else 1


def compute_beta_guess_for_inverse_gamma(
    alpha: Union[int, float], lower: Union[int, float], upper: Union[int, float]
) -> Optional[Union[int, float]]:
    """Compute a guess for the beta parameter of an inverse gamma distribution.

    Assume that the mode is the midpoint between the lower and upper bounds.
    The mode is given by beta / (alpha + 1) if alpha > 1, and does not exist otherwise,
    so beta can easily be solved for.
    """
    mode = (lower + upper) / 2
    return mode * (alpha + 1) if alpha > 1 else None


def compute_mass_within_bounds_for_gamma(
    alpha: Union[int, float], beta: Union[int, float], lower: Union[int, float], upper: Union[int, float]
) -> float:
    """Compute the mass for a gamma distribution.

    The mass is the probability that the lengthscale lies within the bounds.
    """

    def _mass_below(x):
        return gamma.cdf(x, a=alpha, scale=1 / beta)
    
    # plot the inverse gamma distribution with bounds visualized
    # import matplotlib.pyplot as plt
    # x = np.linspace(0, lower + upper, 100)
    # plt.plot(x, gamma.pdf(x, a=alpha, scale=1/beta))
    # plt.axvline(lower, color='r')
    # plt.axvline(upper, color='r')
    # plt.show()

    return _mass_below(upper) - _mass_below(lower)


def compute_mass_within_bounds_for_inverse_gamma(
    alpha: Union[int, float], beta: Union[int, float], lower: Union[int, float], upper: Union[int, float]
) -> float:
    """Compute the mass for an inverse gamma distribution.

    The mass is the probability that the lengthscale lies within the bounds.
    """

    def _mass_below(x):
        return invgamma.cdf(x, a=alpha, scale=beta)
    
    # plot the inverse gamma distribution with bounds visualized
    # import matplotlib.pyplot as plt
    # x = np.linspace(0, lower + upper, 100)
    # plt.plot(x, invgamma.pdf(x, a=alpha, scale=beta))
    # plt.axvline(lower, color='r')
    # plt.axvline(upper, color='r')
    # plt.show()

    return _mass_below(upper) - _mass_below(lower)

class HSGPTVP:
    def __init__(
        self,
        name: str,
        m: int,
        c: int,
        cov: pm.gp.cov.Prod,
        dims: Union[Tuple[str, str], str],
        size: tuple[int, int],
        model: Optional[pm.Model] = None,
    ) -> pm.Deterministic:
        """
        Time Varying Prior, based on the HSGP.

        Args:
            name : The name of the deterministic variable.
            m : The number of basis functions to use in the HSGP.
            c : The number of basis functions to use in the HSGP.
            cov : The covariance function to use in the HSGP.
            dims : The dimensions of the deterministic variable.
            size : The size of the deterministic variable.
        """
        self.name = name
        self.m = m
        self.c = c
        self.cov = cov
        self.dims = dims
        self.size = size
        self.model = model

    def build(self) -> pm.Deterministic:
        """Create linearlized TVP based on the HSGP."""
        with pm.modelcontext(self.model):
            # Set HSGP and write in linearized form
            gp = pm.gp.HSGP(m=[self.m], c=self.c, cov_func=self.cov)

            # Define the GP inputs
            Xs = np.arange(0, self.size[0])[:, None] - self.size[0] // 2
            phi, sqrt_psd = gp.prior_linearized(Xs=Xs)

            # Define the HSGP coefficients
            size = (self.size[1], self.m) if self.size[1] > 1 else self.m
            hsgp_coeffs = pm.TruncatedNormal(
                f"_hsgp_coeffs_{self.name}", mu=0.5, sigma=0.5, lower=0, size=size
            )

            # Return the deterministic variable
            return pm.Deterministic(
                self.name,
                (phi @ (hsgp_coeffs * sqrt_psd).T),
                dims=self.dims,
            )