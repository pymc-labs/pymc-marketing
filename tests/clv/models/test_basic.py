import numpy as np
import pymc as pm
import pytest
from arviz import from_dict

from pymc_marketing.clv.models.basic import CLVModel


class CLVModelTest(CLVModel):
    _model_type = "CLVModelTest"

    def __init__(self):
        super().__init__()

    @property
    def default_model_config(self):
        pass

    @property
    def default_sampler_config(self):
        pass


@pytest.fixture(scope="module")
def posterior():
    # Create a random numpy array for posterior samples
    posterior_samples = np.random.randn(
        4, 100, 2
    )  # shape convention: (chain, draw, *shape)

    # Create a dictionary for posterior
    posterior_dict = {"theta": posterior_samples}
    return from_dict(posterior=posterior_dict)


class TestCLVModel:

    # The rest of the tests need to be moved to subclasses, as basic.py is not a standalone class

    def test_check_prior_ndim(self):
        prior = pm.Normal.dist(shape=(5,))  # ndim = 1
        with pytest.raises(
            ValueError, match="must be have 0 ndims, but it has 1 ndims"
        ):
            # Default ndim=0
            CLVModel._check_prior_ndim(prior)
        CLVModel._check_prior_ndim(prior, ndim=1)
        with pytest.raises(
            ValueError, match="must be have 2 ndims, but it has 1 ndims"
        ):
            CLVModel._check_prior_ndim(prior, ndim=2)

    def test_process_priors(self):
        prior1 = pm.Normal.dist()
        prior2 = pm.HalfNormal.dist()

        ret_prior1, ret_prior2 = CLVModel._process_priors(prior1, prior2)

        assert ret_prior1 is prior1
        assert ret_prior2 is prior2
        assert ret_prior1.str_repr() == "Normal(0, 1)"
        assert ret_prior2.str_repr() == "HalfNormal(0, 1)"

        with pytest.raises(ValueError, match="Prior variables must be unique"):
            CLVModel._process_priors(prior1, prior2, prior1)
