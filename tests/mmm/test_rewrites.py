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
"""Tests for :mod:`pymc_marketing.mmm.rewrites`.

Uses graph comparison strategies from the PyTensor test suite:
inspecting the compiled function graph for expected ops.
"""

import numpy as np
import pytensor.tensor as pt
import pytest
from pytensor.compile import Mode, function
from pytensor.tensor.math import Dot, Sum

from pymc_marketing.mmm.rewrites import create_sampling_mode

# -- helpers ----------------------------------------------------------------


def _has_op(fn, op_cls):
    """Check if any node in the compiled function is an instance of *op_cls*."""
    return any(isinstance(node.op, op_cls) for node in fn.maker.fgraph.toposort())


def _count_op(fn, op_cls):
    """Count nodes matching *op_cls* in the compiled function."""
    return sum(1 for node in fn.maker.fgraph.toposort() if isinstance(node.op, op_cls))


# -- graph structure tests --------------------------------------------------


class TestGraphTransformation:
    """Verify the rewrite changes the graph structure: Dot replaces Sum."""

    @pytest.mark.parametrize(
        "shape",
        [
            (10, 3),
            (52, 8),
            (156, 20),
            (10, 5, 3),
            (52, 5, 8),
            (156, 10, 15),
        ],
        ids=lambda s: f"{'x'.join(str(d) for d in s)}",
    )
    def test_dot_appears_sum_disappears(self, shape):
        """After rewrite: Dot is present, Sum over the target axis is gone."""
        ndim = len(shape)
        X = pt.tensor3("X") if ndim == 3 else pt.matrix("X")
        w = pt.dvector("w")
        expr = (X * w).sum(axis=-1)

        mode = create_sampling_mode()
        fn = function([X, w], expr, mode=mode)

        # Dot must appear
        assert _has_op(fn, Dot), "Rewrite should introduce a Dot op in the graph"

        # Sum must decrease relative to default mode
        fn_default = function([X, w], expr)
        assert _count_op(fn, Sum) < _count_op(fn_default, Sum), (
            "Rewrite should reduce Sum count"
        )

    def test_default_mode_unchanged(self):
        """Without the rewrite, the graph preserves Sum/Mul."""
        X = pt.matrix("X")
        w = pt.dvector("w")
        expr = (X * w).sum(axis=-1)

        fn = function([X, w], expr)

        assert _has_op(fn, Sum), "Expected Sum in default-compiled graph"
        assert not _has_op(fn, Dot), "Did not expect Dot in default-compiled graph"

    def test_output_node_is_dot_with_rewrite(self):
        """The output-producing node should be a Dot under the rewrite."""
        X = pt.matrix("X")
        w = pt.dvector("w")
        expr = (X * w).sum(axis=-1)

        mode = create_sampling_mode()
        fn = function([X, w], expr, mode=mode)

        # The output variable's owner op must NOT be a Sum
        out = fn.maker.fgraph.outputs[0]
        assert not isinstance(out.owner.op, Sum), (
            f"Output should not be produced by Sum, got {out.owner.op}"
        )

    def test_output_node_is_sum_without_rewrite(self):
        """Without the rewrite, the output-producing node is Sum."""
        X = pt.matrix("X")
        w = pt.dvector("w")
        expr = (X * w).sum(axis=-1)

        fn = function([X, w], expr)
        out = fn.maker.fgraph.outputs[0]
        assert isinstance(out.owner.op, Sum), (
            f"Output should be produced by Sum, got {out.owner.op}"
        )


# -- numerical correctness --------------------------------------------------


class TestNumericalEquivalence:
    """Verify the transformed graph produces identical numerical results."""

    def test_2d(self):
        T, C = 10, 4
        X = pt.matrix("X")
        w = pt.dvector("w")
        expr = (X * w).sum(axis=-1)

        mode = create_sampling_mode()
        fn = function([X, w], expr, mode=mode)

        xv = np.random.default_rng(42).standard_normal((T, C)).astype("float64")
        wv = np.random.default_rng(43).standard_normal(C).astype("float64")
        result = fn(xv, wv)

        expected = (xv * wv).sum(axis=-1)
        np.testing.assert_allclose(result, expected)

    def test_3d(self):
        T, G, C = 10, 5, 4
        X = pt.tensor3("X")
        w = pt.dvector("w")
        expr = (X * w).sum(axis=-1)

        mode = create_sampling_mode()
        fn = function([X, w], expr, mode=mode)

        xv = np.random.default_rng(42).standard_normal((T, G, C)).astype("float64")
        wv = np.random.default_rng(43).standard_normal(C).astype("float64")
        result = fn(xv, wv)

        expected = (xv * wv).sum(axis=-1)
        np.testing.assert_allclose(result, expected)

    @pytest.mark.parametrize(
        "shape",
        [(52, 8), (156, 20), (52, 5, 8)],
    )
    def test_larger_shapes(self, shape):
        """Numerical equivalence holds for larger realistic MMM shapes."""
        ndim = len(shape)
        X = pt.tensor3("X") if ndim == 3 else pt.matrix("X")
        w = pt.dvector("w")
        expr = (X * w).sum(axis=-1)

        mode = create_sampling_mode()
        fn = function([X, w], expr, mode=mode)

        rng = np.random.default_rng(42)
        xv = rng.standard_normal(shape).astype("float64")
        wv = rng.standard_normal(shape[-1]).astype("float64")
        result = fn(xv, wv)

        expected = (xv * wv).sum(axis=-1)
        np.testing.assert_allclose(result, expected)


# -- rewrite conditions -----------------------------------------------------


class TestRewriteConditions:
    """Verify rewrite only fires for the intended patterns."""

    def test_skips_sum_all_axes(self):
        """Sum over all axes (axis=None) is skipped."""
        X = pt.matrix("X")
        w = pt.dvector("w")
        expr = (X * w).sum()

        mode = create_sampling_mode()
        fn = function([X, w], expr, mode=mode)

        assert not _has_op(fn, Dot), "Dot should not appear for sum over all axes"

        xv = np.random.randn(10, 4).astype("float64")
        wv = np.random.randn(4).astype("float64")
        result = fn(xv, wv)
        expected = (xv * wv).sum()
        np.testing.assert_allclose(result, expected)

    def test_skips_when_no_expand_dims(self):
        """Neither Mul input is an expanded vector — skip."""
        X = pt.matrix("X")
        Y = pt.matrix("Y")
        expr = (X * Y).sum(axis=-1)

        mode = create_sampling_mode()
        fn = function([X, Y], expr, mode=mode)

        assert not _has_op(fn, Dot)

    def test_works_across_linkers(self):
        """Rewrite works with CVM and Numba linkers."""
        T, C = 10, 4
        X = pt.matrix("X")
        w = pt.dvector("w")
        expr = (X * w).sum(axis=-1)

        xv = np.random.randn(T, C).astype("float64")
        wv = np.random.randn(C).astype("float64")
        expected = (xv * wv).sum(axis=-1)

        for linker in ["cvm", "numba"]:
            base = Mode(linker=linker, optimizer="fast_run")
            mode = create_sampling_mode(base_mode=base)
            try:
                fn = function([X, w], expr, mode=mode)
                result = fn(xv, wv)
                assert np.allclose(result, expected), f"Failed with linker={linker}"
                assert _has_op(fn, Dot), f"Dot should appear with linker={linker}"
            except Exception as e:
                pytest.fail(f"Linker {linker} failed: {e}")


# -- integration with MMM ---------------------------------------------------


class TestIntegrationMMM:
    """End-to-end verification with a real MMM model."""

    def test_mmm_model_samples_correctly(self):
        """Full MMM.fit pipeline: sample and compute_deterministics."""
        import pandas as pd

        from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

        rng = np.random.default_rng(42)
        dates = pd.date_range("2023-01-01", periods=26, freq="W-MON")
        X = pd.DataFrame(
            {
                "date": dates,
                "ch1": rng.standard_normal(26) * 100,
                "ch2": rng.standard_normal(26) * 50,
            }
        )
        y = pd.Series(rng.standard_normal(26) + 5, name="y")

        mmm = MMM(
            date_column="date",
            channel_columns=["ch1", "ch2"],
            target_column="y",
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )
        idata = mmm.fit(
            X=X,
            y=y,
            draws=50,
            tune=50,
            chains=1,
            progressbar=False,
        )

        assert "channel_contribution" in idata["/posterior"]
        assert "intercept_contribution" in idata["/posterior"]
        assert idata["/posterior"]["channel_contribution"].shape[-1] == 2

    def test_mmm_rewrite_property(self):
        """MMM advertises the correct rewrite name."""
        from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

        mmm = MMM(
            date_column="date",
            channel_columns=["ch1"],
            target_column="y",
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )
        assert mmm._get_sampling_rewrites() == ["local_sum_mul_to_dot"]
