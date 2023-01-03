import numpy as np
import pytensor
import pytest
from pytensor import tensor as pt
from pytensor.tensor.shape import specify_broadcastable

from pymc_marketing.mmm.utils import generate_fourier_modes, params_broadcast_shapes


@pytest.mark.parametrize(
    "periods, n_order, expected_shape",
    [
        (np.linspace(start=0.0, stop=1.0, num=50), 10, (50, 10 * 2)),
        (np.linspace(start=-1.0, stop=1.0, num=70), 9, (70, 9 * 2)),
        (np.ones(shape=1), 1, (1, 1 * 2)),
    ],
)
def test_fourier_modes_shape(periods, n_order, expected_shape):
    fourier_modes_df = generate_fourier_modes(periods=periods, n_order=n_order)
    assert fourier_modes_df.shape == expected_shape


@pytest.mark.parametrize(
    "periods, n_order",
    [
        (np.linspace(start=0.0, stop=1.0, num=50), 10),
        (np.linspace(start=-1.0, stop=1.0, num=70), 9),
        (np.ones(shape=1), 1),
    ],
)
def test_fourier_modes_range(periods, n_order):
    fourier_modes_df = generate_fourier_modes(periods=periods, n_order=n_order)
    assert fourier_modes_df.min().min() >= -1.0
    assert fourier_modes_df.max().max() <= 1.0


@pytest.mark.parametrize(
    "periods, n_order",
    [
        (np.linspace(start=-1.0, stop=1.0, num=100), 10),
        (np.linspace(start=-10.0, stop=2.0, num=170), 60),
        (np.linspace(start=-15, stop=5.0, num=160), 20),
    ],
)
def test_fourier_modes_frequency_integer_range(periods, n_order):
    fourier_modes_df = generate_fourier_modes(periods=periods, n_order=n_order)
    assert (fourier_modes_df.filter(regex="sin").mean(axis=0) < 1e-10).all()
    assert (fourier_modes_df.filter(regex="cos").iloc[:-1].mean(axis=0) < 1e-10).all()
    assert not fourier_modes_df[fourier_modes_df > 0].empty
    assert not fourier_modes_df[fourier_modes_df < 0].empty
    assert not fourier_modes_df[fourier_modes_df == 0].empty
    assert not fourier_modes_df[fourier_modes_df == 1].empty


@pytest.mark.parametrize(
    "periods, n_order",
    [
        (np.linspace(start=0.0, stop=1.0, num=100), 10),
        (np.linspace(start=0.0, stop=2.0, num=170), 60),
        (np.linspace(start=0.0, stop=5.0, num=160), 20),
        (np.linspace(start=-9.0, stop=1.0, num=100), 10),
        (np.linspace(start=-80.0, stop=2.0, num=170), 60),
        (np.linspace(start=-100.0, stop=-5.0, num=160), 20),
    ],
)
def test_fourier_modes_pythagoras(periods, n_order):
    fourier_modes_df = generate_fourier_modes(periods=periods, n_order=n_order)
    for i in range(1, n_order + 1):
        norm = (
            fourier_modes_df[f"sin_order_{i}"] ** 2
            + fourier_modes_df[f"cos_order_{i}"] ** 2
        )
        assert abs(norm - 1).all() < 1e-10


@pytest.mark.parametrize("n_order", [0, -1, -100])
def test_bad_order(n_order):
    with pytest.raises(ValueError, match="n_order must be greater than or equal to 1"):
        generate_fourier_modes(
            periods=np.linspace(start=0.0, stop=1.0, num=50), n_order=n_order
        )


class TestParamsBroadcastShapes:
    def test_numpy(self):
        ndims_params = [0, 0]

        a = np.empty(3)
        b = np.empty(())
        param_shapes = [a.shape, b.shape]
        broadcastable_patterns = [
            tuple([s == 1 for s in a.shape]),
            tuple([s == 1 for s in b.shape]),
        ]
        res, bcast = params_broadcast_shapes(
            param_shapes, ndims_params, broadcastable_patterns
        )
        assert np.array_equal(res[0], [3])
        assert np.array_equal(res[1], [3])
        assert bcast == [(False,), (False,)]

        ndims_params = [1, 2]

        a = np.empty(3)
        b = np.empty((2, 3, 3))
        param_shapes = [a.shape, b.shape]
        broadcastable_patterns = [
            tuple([s == 1 for s in a.shape]),
            tuple([s == 1 for s in b.shape]),
        ]
        res, bcast = params_broadcast_shapes(
            param_shapes, ndims_params, broadcastable_patterns
        )
        assert np.array_equal(res[0], [2, 3])
        assert np.array_equal(res[1], [2, 3, 3])
        assert bcast == [(False,) * 2, (False,) * 3]

        a = np.empty(3)
        b = np.empty((1, 3, 3))
        param_shapes = [a.shape, b.shape]
        broadcastable_patterns = [
            tuple([s == 1 for s in a.shape]),
            tuple([s == 1 for s in b.shape]),
        ]
        res, bcast = params_broadcast_shapes(
            param_shapes, ndims_params, broadcastable_patterns
        )
        assert np.array_equal(res[0], [1, 3])
        assert np.array_equal(res[1], [1, 3, 3])
        assert bcast == [(True, False), (True, False, False)]

        a = np.empty((2, 1, 3))
        b = np.empty((2, 3, 3))
        param_shapes = [a.shape, b.shape]
        broadcastable_patterns = [
            tuple([s == 1 for s in a.shape]),
            tuple([s == 1 for s in b.shape]),
        ]
        res, bcast = params_broadcast_shapes(
            param_shapes, ndims_params, broadcastable_patterns
        )
        assert np.array_equal(res[0], [2, 2, 3])
        assert np.array_equal(res[1], [2, 2, 3, 3])
        assert bcast == [(False,) * 3, (False,) * 4]

        with pytest.raises(
            ValueError,
            match="Cannot broadcast shapes 4 and 5 together",
        ):
            a = np.empty((4, 3))
            b = np.empty((5, 3, 3))
            param_shapes = [a.shape, b.shape]
            broadcastable_patterns = [
                tuple([s == 1 for s in a.shape]),
                tuple([s == 1 for s in b.shape]),
            ]
            res, bcast = params_broadcast_shapes(
                param_shapes, ndims_params, broadcastable_patterns
            )
        with pytest.raises(
            ValueError,
            match="Shape along axis 0 in the 1 supplied param_shape was tagged as not broadcastable",
        ):
            a = np.empty((1, 3))
            b = np.empty((5, 3, 3))
            param_shapes = [a.shape, b.shape]
            broadcastable_patterns = [(False,) * 2, tuple([s == 1 for s in b.shape])]
            res, bcast = params_broadcast_shapes(
                param_shapes, ndims_params, broadcastable_patterns
            )

    def test_pytensor_concrete(self):
        ndims_params = [0, 0]

        a = pt.as_tensor_variable(np.empty(3))
        b = pt.as_tensor_variable(np.empty(()))
        param_shapes = [a.shape, b.shape]
        broadcastable_patterns = [a.broadcastable, b.broadcastable]
        res, bcast = params_broadcast_shapes(
            param_shapes, ndims_params, broadcastable_patterns
        )
        assert np.array_equal(res[0], [3])
        assert np.array_equal(res[1], [3])
        assert bcast == [(False,), (False,)]

        ndims_params = [1, 2]

        a = pt.as_tensor_variable(np.empty(3))
        b = pt.as_tensor_variable(np.empty((2, 3, 3)))
        param_shapes = [a.shape, b.shape]
        broadcastable_patterns = [a.broadcastable, b.broadcastable]
        res, bcast = params_broadcast_shapes(
            param_shapes, ndims_params, broadcastable_patterns
        )
        assert np.array_equal(res[0], [2, 3])
        assert np.array_equal(res[1], [2, 3, 3])
        assert bcast == [(False,) * 2, (False,) * 3]

        a = pt.as_tensor_variable(np.empty(3))
        b = pt.as_tensor_variable(np.empty((1, 3, 3)))
        param_shapes = [a.shape, b.shape]
        broadcastable_patterns = [a.broadcastable, b.broadcastable]
        res, bcast = params_broadcast_shapes(
            param_shapes, ndims_params, broadcastable_patterns
        )
        assert np.array_equal(res[0], [1, 3])
        assert np.array_equal(res[1], [1, 3, 3])
        assert bcast == [(True, False), (True, False, False)]

        a = pt.as_tensor_variable(np.empty((2, 1, 3)))
        b = pt.as_tensor_variable(np.empty((2, 3, 3)))
        param_shapes = [a.shape, b.shape]
        broadcastable_patterns = [a.broadcastable, b.broadcastable]
        res, bcast = params_broadcast_shapes(
            param_shapes, ndims_params, broadcastable_patterns
        )
        assert np.array_equal(res[0], [2, 2, 3])
        assert np.array_equal(res[1], [2, 2, 3, 3])
        assert bcast == [(False,) * 3, (False,) * 4]

        with pytest.raises(
            ValueError,
            match="Cannot broadcast shapes 4 and 5 together",
        ):
            a = pt.as_tensor_variable(np.empty((4, 3)))
            b = pt.as_tensor_variable(np.empty((5, 3, 3)))
            param_shapes = [a.shape, b.shape]
            broadcastable_patterns = [a.broadcastable, b.broadcastable]
            res, bcast = params_broadcast_shapes(
                param_shapes, ndims_params, broadcastable_patterns
            )

    def test_pytensor_symbolic(self):
        ndims_params = [0, 0]

        a = pt.dvector()
        b = pt.dscalar()
        param_shapes = [a.shape, b.shape]
        broadcastable_patterns = [a.broadcastable, b.broadcastable]
        res, bcast = params_broadcast_shapes(
            param_shapes, ndims_params, broadcastable_patterns
        )
        assert bcast == [(False,), (False,)]
        c = pt.broadcast_to(a, res[0])
        d = pt.broadcast_to(b, res[1])
        f = pytensor.function([a, b], [c, d])
        cv, dv = f(np.empty(3), np.empty(()))
        assert np.array_equal(cv.shape, [3])
        assert np.array_equal(dv.shape, [3])

        ndims_params = [1, 2]

        a = pt.dvector()
        b = pt.dtensor3()
        param_shapes = [a.shape, b.shape]
        broadcastable_patterns = [a.broadcastable, b.broadcastable]
        res, bcast = params_broadcast_shapes(
            param_shapes, ndims_params, broadcastable_patterns
        )
        assert bcast == [(False,) * 2, (False,) * 3]
        c = pt.broadcast_to(a, res[0])
        d = pt.broadcast_to(b, res[1])
        f = pytensor.function([a, b], [c, d])
        cv, dv = f(np.empty(3), np.empty((2, 3, 3)))
        assert np.array_equal(cv.shape, [2, 3])
        assert np.array_equal(dv.shape, [2, 3, 3])

        a = pt.dtensor3()
        b = pt.dtensor3()
        a = specify_broadcastable(a, 1)
        param_shapes = [a.shape, b.shape]
        broadcastable_patterns = [a.broadcastable, b.broadcastable]
        res, bcast = params_broadcast_shapes(
            param_shapes, ndims_params, broadcastable_patterns
        )
        assert bcast == [(False,) * 3, (False,) * 4]
        c = pt.broadcast_to(a, res[0])
        d = pt.broadcast_to(b, res[1])
        f = pytensor.function([a, b], [c, d])
        cv, dv = f(np.empty((2, 1, 3)), np.empty((2, 3, 3)))
        assert np.array_equal(cv.shape, [2, 2, 3])
        assert np.array_equal(dv.shape, [2, 2, 3, 3])

        a = pt.dtensor3()
        b = pt.dtensor3()
        param_shapes = [a.shape, b.shape]
        broadcastable_patterns = [a.broadcastable, b.broadcastable]
        res, bcast = params_broadcast_shapes(
            param_shapes, ndims_params, broadcastable_patterns
        )
        assert bcast == [(False,) * 3, (False,) * 4]
        c = pt.broadcast_to(a, res[0])
        d = pt.broadcast_to(b, res[1])
        f = pytensor.function([a, b], [c, d])
        with pytest.raises(
            AssertionError,
            match=(
                "Shape along axis 0 in the 1 supplied param_shape was tagged as "
                "not broadcastable and it was not exactly equal to the other supplied "
                "param_shapes."
            ),
        ):
            cv, dv = f(np.empty((2, 1, 3)), np.empty((2, 3, 3)))

        a = pt.dmatrix()
        b = pt.dtensor3()
        param_shapes = [a.shape, b.shape]
        broadcastable_patterns = [a.broadcastable, b.broadcastable]
        res, bcast = params_broadcast_shapes(
            param_shapes, ndims_params, broadcastable_patterns
        )
        assert bcast == [(False,) * 2, (False,) * 3]
        c = pt.broadcast_to(a, res[0])
        d = pt.broadcast_to(b, res[1])
        f = pytensor.function([a, b], [c, d])
        with pytest.raises(
            AssertionError,
            match=(
                "Shape along axis 0 in the 1 supplied param_shape was tagged as "
                "not broadcastable and it was not exactly equal to the other supplied "
                "param_shapes."
            ),
        ):
            cv, dv = f(np.empty((4, 3)), np.empty((5, 3, 3)))
