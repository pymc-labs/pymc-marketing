#   Copyright 2022 - 2026 The PyMC Labs Developers
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
import numpy as np
import pymc as pm
import pymc.dims as pmd
import pytensor.tensor as pt
import pytest
import scipy.stats as st
from pymc import Model, draw
from pymc.dims.distributions.transforms import ZeroSumTransform
from pymc.distributions.shape_utils import change_dist_size
from pymc.logprob.utils import ParameterValueError
from pymc.model.fgraph import clone_model
from pymc.model.transform.optimization import freeze_dims_and_data
from pymc.testing import assert_support_point_is_expected
from pytensor.xtensor import as_xtensor

from pymc_marketing.mmm.distributions import (
    DimWeightedZeroSumNormal,
    DimWeightedZeroSumTransform,
    WeightedZeroSumNormal,
    WeightedZeroSumTransform,
)

WEIGHTS = np.array([0.9, 0.05, 0.03, 0.02])


class TestWeightedZeroSumNormal:
    WEIGHTS = WEIGHTS

    @staticmethod
    def direction(w):
        return w / np.linalg.norm(w)

    @staticmethod
    def reference_logp(x, sigma, w):
        """Logp of an (n-1)-dim isotropic MvNormal evaluated on the isometric coordinates."""
        z = WeightedZeroSumTransform(w).forward(pt.as_tensor(x)).eval()
        n = len(w)
        sigma = np.broadcast_to(sigma, z.shape[:-1])
        return np.array(
            [
                st.multivariate_normal(np.zeros(n - 1), s**2 * np.eye(n - 1)).logpdf(zi)
                for s, zi in zip(sigma.ravel(), z.reshape(-1, n - 1), strict=True)
            ]
        ).reshape(z.shape[:-1])

    def test_constraint_and_covariance(self):
        w = self.WEIGHTS
        u = self.direction(w)
        dist = WeightedZeroSumNormal.dist(weights=w, sigma=2.0)
        draws = pm.draw(dist, draws=20_000, random_seed=964)

        assert np.abs(draws @ u).max() < 1e-12
        expected_cov = 4.0 * (np.eye(len(w)) - np.outer(u, u))
        np.testing.assert_allclose(np.cov(draws.T), expected_cov, atol=0.15)
        # marginal variances are sigma**2 * (1 - u_i**2), documented in the docstring
        np.testing.assert_allclose(draws.var(axis=0), 4.0 * (1 - u**2), atol=0.1)

    @pytest.mark.parametrize("sigma", [1.0, 2.5, np.array([1.0, 2.0, 3.0])])
    def test_logp_matches_mvnormal_reference(self, sigma):
        w = self.WEIGHTS
        rng = np.random.default_rng(207)
        z = rng.normal(size=(3, len(w) - 1))
        x = WeightedZeroSumTransform(w).backward(pt.as_tensor(z)).eval()

        dist = WeightedZeroSumNormal.dist(weights=w, sigma=sigma)
        np.testing.assert_allclose(
            pm.logp(dist, x).eval(), self.reference_logp(x, sigma, w)
        )

    def test_equal_weights_match_zerosumnormal(self):
        n = 5
        rng = np.random.default_rng(207)
        x = rng.normal(size=(3, n))
        x = x - x.mean(axis=-1, keepdims=True)

        zsn = pm.ZeroSumNormal.dist(shape=(n,))
        wzsn = WeightedZeroSumNormal.dist(weights=np.full(n, 1.0 / n))
        np.testing.assert_allclose(pm.logp(zsn, x).eval(), pm.logp(wzsn, x).eval())

    def test_default_transform_bijective(self):
        w = self.WEIGHTS
        u = self.direction(w)
        transform = WeightedZeroSumTransform(w)
        rng = np.random.default_rng(6)
        z = rng.normal(size=(7, len(w) - 1))

        x = transform.backward(pt.as_tensor(z)).eval()
        np.testing.assert_allclose(x @ u, 0.0, atol=1e-12)
        np.testing.assert_allclose(
            transform.forward(pt.as_tensor(x)).eval(), z, atol=1e-12
        )
        # isometry
        np.testing.assert_allclose(
            np.linalg.norm(x, axis=-1), np.linalg.norm(z, axis=-1), atol=1e-12
        )
        ljd = transform.log_jac_det(
            pt.as_tensor(z)
        ).eval()  # pymc passes the unconstrained value
        assert ljd.shape == (7,)
        np.testing.assert_array_equal(ljd, 0.0)

    def test_transform_reads_weights_from_rv_inputs(self):
        # The default transform is created without weights and takes them
        # from the RV inputs it is handed
        w = self.WEIGHTS
        with pm.Model() as model:
            x = WeightedZeroSumNormal("x", weights=w)
        transform = model.rvs_to_transforms[x]
        assert transform.weights is None
        z = np.array([0.3, -0.2, 0.1])
        expected = WeightedZeroSumTransform(w).backward(pt.as_tensor(z)).eval()
        np.testing.assert_allclose(
            transform.backward(pt.as_tensor(z), *x.owner.inputs).eval(), expected
        )
        with pytest.raises(ValueError, match="needs weights"):
            WeightedZeroSumTransform().backward(pt.as_tensor(z))

    def test_logp_rejects_unconstrained_value(self):
        w = self.WEIGHTS
        dist = WeightedZeroSumNormal.dist(weights=w)
        with pytest.raises(ParameterValueError):
            pm.logp(dist, np.ones(len(w))).eval()

    def test_weights_must_be_vector(self):
        with pytest.raises(ValueError, match="1-d"):
            WeightedZeroSumNormal.dist(weights=np.ones((2, 2)))

    @pytest.mark.parametrize(
        "weights", [[1.0, -1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
    )
    def test_constant_weights_must_be_positive(self, weights):
        with pytest.raises(ValueError, match="strictly positive"):
            WeightedZeroSumNormal.dist(weights=weights)

    def test_symbolic_weights_must_be_positive(self):
        weights = pt.vector("w", shape=(3,))
        dist = WeightedZeroSumNormal.dist(weights=weights)
        logp = pm.logp(dist, np.zeros(3))
        assert np.isfinite(logp.eval({weights: np.array([1.0, 2.0, 3.0])}))
        with pytest.raises(ParameterValueError, match="weights > 0"):
            logp.eval({weights: np.array([1.0, 0.0, 3.0])})

    def test_shape_must_match_weights(self):
        w = self.WEIGHTS
        assert WeightedZeroSumNormal.dist(weights=w, shape=(5, 4)).type.shape == (5, 4)
        with pytest.raises(ValueError, match="length of weights does not match"):
            WeightedZeroSumNormal.dist(weights=w, shape=(5, 7))
        with pytest.raises(ValueError, match="length of weights does not match"):
            WeightedZeroSumNormal.dist(weights=w, support_shape=3)

    def test_dims_must_match_weights(self):
        w = self.WEIGHTS
        with pm.Model(coords={"a": range(2), "b": range(4)}) as model:
            x = WeightedZeroSumNormal("x", weights=w, dims=("a", "b"))
        assert x.type.shape[-1] == 4
        assert model.initial_point()["x_weighted_zerosum__"].shape == (2, 3)

        # Model dim lengths are shared variables, so the mismatch is caught at runtime
        with pm.Model(coords={"a": range(2), "b": range(5)}) as model:
            WeightedZeroSumNormal("x", weights=w, dims=("a", "b"))
        with pytest.raises(AssertionError, match="length of weights does not match"):
            model.initial_point()

    def test_symbolic_weights_length_checked_at_runtime(self):
        with pm.Model(coords={"b": range(5)}) as model:
            WeightedZeroSumNormal("x", weights=pm.Data("w", self.WEIGHTS), dims="b")
        with pytest.raises(AssertionError, match="length of weights does not match"):
            model.initial_point()

    def test_observed(self):
        w = self.WEIGHTS
        rng = np.random.default_rng(3)
        z = rng.normal(size=(3, len(w) - 1))
        x = WeightedZeroSumTransform(w).backward(pt.as_tensor(z)).eval()
        with pm.Model() as model:
            WeightedZeroSumNormal("x", weights=w, observed=x)
        np.testing.assert_allclose(
            model.compile_logp(sum=False)({})[0], self.reference_logp(x, 1.0, w)
        )

    def test_support_point(self):
        with pm.Model() as model:
            WeightedZeroSumNormal("x", weights=self.WEIGHTS, shape=(3, 4))
        assert_support_point_is_expected(model, np.zeros((3, 4)))

    def test_change_dist_size(self):
        base_dist = WeightedZeroSumNormal.dist(weights=self.WEIGHTS)

        new_dist = change_dist_size(base_dist, new_size=(4, 3))
        assert tuple(new_dist.shape.eval()) == (4, 3, len(self.WEIGHTS))

        new_dist = change_dist_size(base_dist, new_size=(5,), expand=True)
        assert tuple(new_dist.shape.eval()) == (5, len(self.WEIGHTS))

        u = self.direction(self.WEIGHTS)
        draws = pm.draw(new_dist, random_seed=42)
        assert np.abs(draws @ u).max() < 1e-12

    def test_batched_sigma_draws(self):
        sigma = np.array([1.0, 2.0, 3.0])
        dist = WeightedZeroSumNormal.dist(sigma=sigma, weights=self.WEIGHTS)
        draws = pm.draw(dist, draws=5000, random_seed=910)
        assert draws.shape == (5000, 3, len(self.WEIGHTS))

        u = self.direction(self.WEIGHTS)
        assert np.abs(draws @ u).max() < 1e-12
        # variance of the freest component scales with sigma**2
        np.testing.assert_allclose(
            draws[..., -1].var(axis=0), sigma**2 * (1 - u[-1] ** 2), rtol=0.1
        )

    @pytest.mark.parametrize("batch", [3, 4])
    def test_batched_sigma_logp(self, batch):
        # batch == n is the case where a misaligned sigma would silently give wrong values
        w = self.WEIGHTS
        sigma = np.arange(1.0, batch + 1)
        with pm.Model() as model:
            x = WeightedZeroSumNormal("x", sigma=sigma, weights=w)
        assert x.type.shape == (batch, 4)
        assert model.logp(sum=False)[0].type.shape == (batch,)
        assert model.logp(sum=False, jacobian=False)[0].type.shape == (batch,)

        rng = np.random.default_rng(11)
        z = rng.normal(size=(batch, len(w) - 1))
        x_val = WeightedZeroSumTransform(w).backward(pt.as_tensor(z)).eval()
        np.testing.assert_allclose(
            model.compile_logp(sum=False)({"x_weighted_zerosum__": z})[0],
            self.reference_logp(x_val, sigma, w),
        )

    def test_data_weights(self):
        w = self.WEIGHTS
        with pm.Model() as model:
            weights = pm.Data("w", w)
            x = WeightedZeroSumNormal("x", weights=weights)

        # forward is used by initial_point, backward by the logp
        assert model.initial_point()["x_weighted_zerosum__"].shape == (3,)
        rng = np.random.default_rng(8)
        z = rng.normal(size=3)
        logp_fn = model.compile_logp()
        np.testing.assert_allclose(
            logp_fn({"x_weighted_zerosum__": z}),
            self.reference_logp(
                WeightedZeroSumTransform(w).backward(pt.as_tensor(z)).eval(), 1.0, w
            ).sum(),
        )

        # set_data changes the constraint the transform maps onto
        new_w = np.array([1.0, 1.0, 1.0, 5.0])
        with model:
            pm.set_data({"w": new_w})
        transform = model.rvs_to_transforms[x]
        x_val = transform.backward(pt.as_tensor(z), *x.owner.inputs).eval()
        np.testing.assert_allclose(x_val @ self.direction(new_w), 0.0, atol=1e-12)
        np.testing.assert_allclose(
            logp_fn({"x_weighted_zerosum__": z}),
            self.reference_logp(x_val, 1.0, new_w).sum(),
        )

        with model:
            idata = pm.sample(
                draws=10,
                tune=10,
                chains=1,
                progressbar=False,
                random_seed=6,
                compute_convergence_checks=False,
            )
        post = idata.posterior["x"].values.reshape(-1, len(w))
        assert np.abs(post @ self.direction(new_w)).max() < 1e-12

    def test_rv_weights(self):
        with pm.Model() as model:
            w = pm.Dirichlet("w", a=np.ones(4))
            x = WeightedZeroSumNormal("x", weights=w)

        w_val = np.array([0.4, 0.3, 0.2, 0.1])
        z = np.array([0.5, -0.3, 0.2])
        # The transform reads the weights from the RV inputs, here the Dirichlet itself
        transform = model.rvs_to_transforms[x]
        x_val = transform.backward(pt.as_tensor(z), *x.owner.inputs).eval({w: w_val})
        np.testing.assert_allclose(x_val @ self.direction(w_val), 0.0, atol=1e-12)

        # In the logp graph the weights are the Dirichlet's value variable
        point = {
            "w_simplex__": model.rvs_to_transforms[w]
            .forward(pt.as_tensor(w_val))
            .eval(),
            "x_weighted_zerosum__": z,
        }
        _w_logp, x_logp = model.compile_logp(sum=False)(point)
        np.testing.assert_allclose(x_logp, self.reference_logp(x_val, 1.0, w_val).sum())

    @pytest.mark.parametrize(
        "model_fn", ["clone_model", "freeze_dims_and_data", "copy"]
    )
    def test_model_transforms_with_data_weights(self, model_fn):
        with pm.Model() as model:
            WeightedZeroSumNormal("x", weights=pm.Data("w", self.WEIGHTS))

        if model_fn == "clone_model":
            new_model = clone_model(model)
        elif model_fn == "freeze_dims_and_data":
            new_model = freeze_dims_and_data(model)
        else:
            new_model = model.copy()

        point = {"x_weighted_zerosum__": np.array([0.3, -0.1, 0.2])}
        np.testing.assert_allclose(
            new_model.compile_logp()(point), model.compile_logp()(point)
        )


WEIGHTS = np.array([0.9, 0.05, 0.03, 0.02])


def test_dim_weighted_zerosumnormal():
    w = WEIGHTS
    coords = {"a": range(3), "b": range(4)}
    with Model(coords=coords) as model:
        DimWeightedZeroSumNormal("x", weights=w, core_dims=("b",), dims=("a", "b"))
        DimWeightedZeroSumNormal(
            "y", sigma=3, weights=w, core_dims=("b",), dims=("a", "b")
        )

    with Model(coords=coords) as reference_model:
        WeightedZeroSumNormal("x", weights=w, dims=("a", "b"))
        WeightedZeroSumNormal("y", sigma=3, weights=w, dims=("a", "b"))

    for name in ("x", "y"):
        draws = draw(model[name], draws=200, random_seed=1)
        ref_draws = draw(reference_model[name], draws=200, random_seed=1)
        assert draws.shape == ref_draws.shape == (200, 3, 4)
        np.testing.assert_allclose(draws @ (w / np.linalg.norm(w)), 0.0, atol=1e-12)
        np.testing.assert_allclose(draws.std(), ref_draws.std(), rtol=0.1)


def test_dim_weighted_zerosumnormal_logp_matches_regular():
    w = WEIGHTS
    coords = {"a": range(3), "b": range(4)}
    with Model(coords=coords) as model:
        DimWeightedZeroSumNormal(
            "x", sigma=2.0, weights=w, core_dims="b", dims=("a", "b")
        )
    with Model(coords=coords) as ref_model:
        WeightedZeroSumNormal("x", sigma=2.0, weights=w, dims=("a", "b"))

    rng = np.random.default_rng(3)
    point = {"x_weighted_zerosum__": rng.normal(size=(3, 3))}
    np.testing.assert_allclose(
        model.compile_logp(sum=False)(point), ref_model.compile_logp(sum=False)(point)
    )


def test_dim_weighted_zerosumnormal_matches_zerosumnormal_with_equal_weights():
    n = 4
    coords = {"b": range(n)}
    with Model(coords=coords) as model:
        x = DimWeightedZeroSumNormal("x", weights=np.full(n, 1.0 / n), core_dims=("b",))
    with Model(coords=coords) as ref_model:
        ref_x = pmd.ZeroSumNormal("x", core_dims=("b",))

    # Same unconstrained point maps to the same constrained value and logp
    rng = np.random.default_rng(4)
    z = rng.normal(size=n - 1)
    x_val = model.rvs_to_transforms[x].backward(
        as_xtensor(z, dims=("b",)), *x.owner.inputs
    )
    ref_x_val = ref_model.rvs_to_transforms[ref_x].backward(as_xtensor(z, dims=("b",)))
    np.testing.assert_allclose(x_val.eval(), ref_x_val.eval())
    np.testing.assert_allclose(
        model.compile_logp()({"x_weighted_zerosum__": z}),
        ref_model.compile_logp()({"x_zerosum__": z}),
    )


def test_dim_weighted_zerosumnormal_batch_sigma():
    coords = {"a": range(3), "b": range(4)}
    sigma = np.array([1, 2, 3.0])
    with Model(coords=coords) as model:
        x = DimWeightedZeroSumNormal(
            "x",
            sigma=as_xtensor(sigma, dims=("a",)),
            weights=WEIGHTS,
            core_dims=("b",),
        )
    assert x.type.dims == ("a", "b")

    with Model(coords=coords) as ref_model:
        WeightedZeroSumNormal("x", sigma=sigma, weights=WEIGHTS, dims=("a", "b"))

    assert model.logp(sum=False)[0].type.shape == (3,)
    rng = np.random.default_rng(5)
    point = {"x_weighted_zerosum__": rng.normal(size=(3, 3))}
    np.testing.assert_allclose(
        model.compile_logp(sum=False)(point),
        ref_model.compile_logp(sum=False)(point),
    )


def test_dim_weighted_zerosumnormal_requires_constant_weights():
    # like pymc's dims IntervalTransform, the dims flavour keeps its weights
    # as constants; symbolic weights are the tensor API's job
    coords = {"b": range(4)}
    with Model(coords=coords):
        data_weights = pmd.Data("w", WEIGHTS, dims="b")
        with pytest.raises(NotImplementedError, match="constant weights"):
            DimWeightedZeroSumNormal("x", weights=data_weights, core_dims="b")
        rv_weights = pmd.Dirichlet(
            "d", a=as_xtensor(np.ones(4), dims=("b",)), core_dims="b"
        )
        with pytest.raises(NotImplementedError, match="constant weights"):
            DimWeightedZeroSumNormal("y", weights=rv_weights, core_dims="b")
    with pytest.raises(NotImplementedError, match="constant weights"):
        DimWeightedZeroSumTransform("b", pt.vector("w", shape=(4,)))


def test_dim_weighted_zerosumnormal_constant_tensor_weights():
    # constants of either API are accepted
    with Model(coords={"b": range(4)}) as model:
        x = DimWeightedZeroSumNormal("x", weights=pt.as_tensor(WEIGHTS), core_dims="b")
        y = DimWeightedZeroSumNormal(
            "y", weights=as_xtensor(WEIGHTS, dims=("b",)), core_dims="b"
        )
    assert np.isfinite(model.compile_logp()(model.initial_point()))
    for var in (x, y):
        draws = draw(var, draws=50, random_seed=1)
        np.testing.assert_allclose(
            draws @ (WEIGHTS / np.linalg.norm(WEIGHTS)), 0.0, atol=1e-12
        )


@pytest.mark.parametrize("model_fn", ["clone_model", "freeze_dims_and_data"])
def test_dim_weighted_zerosumnormal_model_transforms(model_fn):
    with Model(coords={"b": range(4)}) as model:
        DimWeightedZeroSumNormal("x", weights=WEIGHTS, core_dims="b")

    new_model = (
        clone_model(model) if model_fn == "clone_model" else freeze_dims_and_data(model)
    )
    point = {"x_weighted_zerosum__": np.array([0.3, -0.1, 0.2])}
    np.testing.assert_allclose(
        new_model.compile_logp()(point), model.compile_logp()(point)
    )


def test_dim_weighted_zerosumnormal_errors():
    w = WEIGHTS
    with pytest.raises(ValueError, match="requires weights"):
        DimWeightedZeroSumNormal.dist(core_dims="b", dim_lengths={})
    with pytest.raises(ValueError, match="exactly one core_dims"):
        DimWeightedZeroSumNormal.dist(weights=w, core_dims=("a", "b"), dim_lengths={})
    with pytest.raises(ValueError, match="exactly one core_dims"):
        DimWeightedZeroSumNormal.dist(weights=w, dim_lengths={})
    with pytest.raises(ValueError, match="weights must have dims"):
        DimWeightedZeroSumNormal.dist(
            weights=as_xtensor(w, dims=("c",)), core_dims="b", dim_lengths={}
        )
    with pytest.raises(ValueError, match="strictly positive"):
        DimWeightedZeroSumNormal.dist(
            weights=[1.0, -1.0, 1.0], core_dims="b", dim_lengths={}
        )
    with pytest.raises(ValueError, match="invalid core dimensions"):
        DimWeightedZeroSumNormal.dist(
            sigma=as_xtensor(np.ones(4), dims=("b",)),
            weights=w,
            core_dims="b",
            dim_lengths={},
        )
    # Model dim lengths are shared variables, so a mismatch is caught at runtime
    with Model(coords={"b": range(5)}) as model:
        DimWeightedZeroSumNormal("x", weights=w, core_dims="b")
    with pytest.raises(AssertionError, match="length of weights does not match"):
        model.compile_logp()(model.initial_point())


TRANSFORM_WEIGHTS = [
    np.array([0.70, 0.20, 0.08, 0.02]),
    np.array([0.90, 0.05, 0.03, 0.02]),
    np.array([0.25, 0.25, 0.25, 0.25]),
    np.array([0.5, 0.5]),
]


@pytest.mark.parametrize("w", TRANSFORM_WEIGHTS)
def test_dim_weighted_zerosum_roundtrip_and_constraint(w):
    rng = np.random.default_rng(2026)
    n = len(w)
    transform = DimWeightedZeroSumTransform(dim="a", weights=w)

    z_np = rng.normal(size=(11, n - 1))
    z = as_xtensor(z_np, dims=("batch", "a"))
    x = transform.backward(z)
    x_np = x.transpose("batch", "a").eval()

    u = w / np.linalg.norm(w)
    np.testing.assert_allclose(x_np @ u, 0.0, atol=1e-12)

    z_back = transform.forward(x).transpose("batch", "a").eval()
    np.testing.assert_allclose(z_back, z_np, atol=1e-12)

    # isometry => log_jac_det == 0
    np.testing.assert_allclose(
        np.linalg.norm(x_np, axis=-1), np.linalg.norm(z_np, axis=-1), atol=1e-12
    )
    ljd = transform.log_jac_det(x).eval()
    np.testing.assert_allclose(ljd, 0.0, atol=1e-12)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_dim_weighted_zerosum_equal_weights_matches_zerosum(n):
    rng = np.random.default_rng(5)
    weighted = DimWeightedZeroSumTransform(dim="a", weights=np.full(n, 1.0 / n))
    uniform = ZeroSumTransform(dims=("a",))

    z = as_xtensor(rng.normal(size=(7, n - 1)), dims=("batch", "a"))
    x_w = weighted.backward(z).transpose("batch", "a").eval()
    x_u = uniform.backward(z).transpose("batch", "a").eval()
    np.testing.assert_allclose(x_w, x_u, atol=1e-12)

    x = as_xtensor(x_w, dims=("batch", "a"))
    np.testing.assert_allclose(
        weighted.forward(x).transpose("batch", "a").eval(),
        uniform.forward(x).transpose("batch", "a").eval(),
        atol=1e-12,
    )


def test_dim_weighted_zerosum_invalid_weights():
    with pytest.raises(ValueError, match="strictly positive"):
        DimWeightedZeroSumTransform(dim="a", weights=np.array([0.5, 0.0, 0.5]))
    with pytest.raises(ValueError, match="1-d"):
        DimWeightedZeroSumTransform(dim="a", weights=np.ones((2, 2)))


def test_dim_weighted_zerosum_transform_ignores_rv_inputs():
    # the dims transform's map is fixed at construction: RV inputs are ignored
    transform = DimWeightedZeroSumTransform("a", np.array([0.5, 0.3, 0.2]))
    z = as_xtensor(np.array([0.1, -0.2]), dims=("a",))
    np.testing.assert_allclose(
        transform.backward(z, "ignored").eval(), transform.backward(z).eval()
    )
