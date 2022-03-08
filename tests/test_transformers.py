import numpy as np
import pytest

from pymmmc.transformers import geometric_adstock


def test_geometric_adsstock_alpha_zero():
    x = np.ones(shape=(100))
    y = geometric_adstock(x=x, alpha=0.0)
    np.testing.assert_array_equal(x, y.eval())


def test_geometric_adsstock_x_zero():
    x = np.zeros(shape=(100))
    y = geometric_adstock(x=x, alpha=0.2)
    np.testing.assert_array_equal(x, y.eval())


@pytest.mark.parametrize(
    "x,alpha,l_max",
    [
        (np.ones(shape=(100)), 0.0, 10),
        (np.ones(shape=(100)), 0.0, 100),
        (np.zeros(shape=(100)), 0.2, 5),
        (np.ones(shape=(100)), 0.5, 7),
        (np.linspace(start=0.0, stop=1.0, num=50), 0.8, 3),
        (np.linspace(start=0.0, stop=1.0, num=50), 0.8, 50),
    ],
)
def test_geometric_adsstock_alpha_non_zero(x, alpha, l_max):
    y = geometric_adstock(x=x, alpha=alpha, l_max=l_max)
    y_np = y.eval()
    assert y_np[0] == x[0]
    assert y_np[1] == x[1] + alpha * x[0]
    assert y_np[2] == x[2] + alpha * x[1] + (alpha**2) * x[0]
