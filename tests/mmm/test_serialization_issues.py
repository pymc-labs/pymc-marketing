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
"""Tests verifying known serialization issues documented in the serialization overhaul design plan.

Ticket: [#2087](https://github.com/pymc-labs/pymc-marketing/issues/2087)
THIS FILE SHOULD NOT BE MERGED TO THE MAIN REPO.
IT IS ONLY USED FOR TESTING THE SERIALIZATION OVERHAUL DESIGN PLAN.


Each test asserts the CORRECT behavior that should hold once the underlying
bug is fixed. Until then, the tests fail — demonstrating the issue exists.

Issues verified (FAILING — demonstrating bugs):
1. Prior equality after roundtrip with PyTensor expression parameters
2. HSGP/HSGPPeriodic/SoftPlusHSGP dims tuple→list mismatch after JSON roundtrip
3. EventAdditiveEffect deserialization raises ValueError
4. build_from_idata silently drops MuEffects that fail to deserialize
5. Custom MuEffect subclass not in deserialization registry
6. LinearTrendEffect model_dump(mode="json") fails on Prior fields
6b. LinearTrendEffect linear_trend_first_date lost through serialization roundtrip
7. Default _serialize_mu_effect fails for custom MuEffects with complex fields
"""

import json

import numpy as np
import pandas as pd
from pymc_extras.prior import Prior

from pymc_marketing.mmm.additive_effect import (
    LinearTrendEffect,
    MuEffect,
)
from pymc_marketing.mmm.hsgp import (
    HSGP,
    HSGPPeriodic,
    SoftPlusHSGP,
    create_eta_prior,
    hsgp_from_dict,
)
from pymc_marketing.mmm.linear_trend import LinearTrend
from pymc_marketing.mmm.multidimensional import (
    _deserialize_mu_effect,
    _serialize_mu_effect,
)


class TestPriorTensorParamRoundtrip:
    """Issue: Prior with PyTensor tensor params loses equality after roundtrip.

    create_eta_prior() returns Prior("Exponential", lam=-pt.log(mass)/upper)
    where lam is a TensorVariable. After to_dict() → from_dict(), the param
    becomes a plain float. The Priors are numerically equivalent but not equal
    because TensorVariable != float in Prior.__eq__.
    """

    def test_prior_with_tensor_params_roundtrips_with_equality(self):
        """Prior with tensor params should be equal after to_dict/from_dict roundtrip."""
        original = create_eta_prior(mass=0.05, upper=1.0)

        d = original.to_dict()
        roundtripped = Prior.from_dict(d)

        assert original == roundtripped


class TestHSGPDimsRoundtrip:
    """Issue: HSGP dims field changes from tuple to list after JSON roundtrip.

    JSON has no tuple type, so tuple → JSON array → list. After
    hsgp.to_dict() → json.dumps() → json.loads() → HSGP.from_dict(),
    the dims field is a list instead of a tuple, causing dict comparison to fail.
    """

    def test_hsgp_roundtrip_preserves_dims_type(self):
        """HSGP.to_dict() roundtrip through JSON should preserve dims as tuple."""
        X = np.arange(52)
        original = HSGP.parameterize_from_data(X, dims="time")

        serialized = {**original.to_dict(), "hsgp_class": "HSGP"}
        json_str = json.dumps(serialized)
        loaded = hsgp_from_dict(json.loads(json_str))

        orig_dict = original.to_dict()
        loaded_dict = loaded.to_dict()
        assert orig_dict == loaded_dict

    def test_hsgp_roundtrip_dims_is_tuple(self):
        """After deserialization, dims should be a tuple (not list)."""
        X = np.arange(52)
        original = HSGP.parameterize_from_data(X, dims="time")
        assert isinstance(original.dims, tuple)

        serialized = {**original.to_dict(), "hsgp_class": "HSGP"}
        json_str = json.dumps(serialized)
        loaded = hsgp_from_dict(json.loads(json_str))

        assert isinstance(loaded.dims, tuple)


class TestHSGPPeriodicDimsRoundtrip:
    """Issue: HSGPPeriodic dims also changes from tuple to list after JSON roundtrip."""

    def test_hsgp_periodic_roundtrip_preserves_dims(self):
        """HSGPPeriodic roundtrip through JSON should preserve dims as tuple."""
        scale = Prior("HalfNormal", sigma=1)
        ls = Prior("InverseGamma", alpha=2, beta=1)
        original = HSGPPeriodic(scale=scale, ls=ls, m=20, period=52, dims="time")

        serialized = {**original.to_dict(), "hsgp_class": "HSGPPeriodic"}
        json_str = json.dumps(serialized)
        loaded = hsgp_from_dict(json.loads(json_str))

        orig_dict = original.to_dict()
        loaded_dict = loaded.to_dict()
        assert orig_dict == loaded_dict


class TestSoftPlusHSGPRoundtrip:
    """Issue: SoftPlusHSGP inherits HSGP's dims tuple→list roundtrip bug."""

    def test_softplus_hsgp_roundtrip_preserves_dims(self):
        """SoftPlusHSGP roundtrip through JSON should preserve dims as tuple."""
        X = np.arange(52)
        original = SoftPlusHSGP.parameterize_from_data(X, dims="time")

        serialized = {**original.to_dict(), "hsgp_class": "SoftPlusHSGP"}
        json_str = json.dumps(serialized)
        loaded = hsgp_from_dict(json.loads(json_str))

        orig_dict = original.to_dict()
        loaded_dict = loaded.to_dict()
        assert orig_dict == loaded_dict


class TestEventAdditiveEffectDeserialization:
    """Issue: EventAdditiveEffect deserialization intentionally raises ValueError
    because df_events DataFrame is not stored in the serialized form.
    """

    def test_event_additive_effect_deserializes_successfully(self):
        """EventAdditiveEffect should deserialize without error."""
        effect_data = {
            "class": "EventAdditiveEffect",
            "prefix": "events",
            "reference_date": "2024-01-01",
            "date_dim_name": "date",
            "event_names": ["event1", "event2"],
        }

        effect = _deserialize_mu_effect(effect_data)
        assert effect is not None


class TestBuildFromIdataSilentFailures:
    """Issue: build_from_idata catches all MuEffect deserialization errors with a
    broad except Exception and only warns — model loads but with missing effects.

    The correct behavior is to raise an explicit error so users know their model
    loaded incompletely, rather than silently dropping effects.
    """

    def test_mu_effect_deserialization_failure_propagates_as_error(self):
        """Failed MuEffect deserialization should propagate as an error, not be caught."""

        effect_data_list = [
            {
                "class": "EventAdditiveEffect",
                "prefix": "events",
                "event_names": ["event1"],
            }
        ]

        # Simulate what build_from_idata does (lines 3293-3304 in multidimensional.py):
        #   for effect_data in mu_effects_data:
        #       try:
        #           effect = _deserialize_mu_effect(effect_data)
        #           self.mu_effects.append(effect)
        #       except Exception as e:
        #           warnings.warn(f"Could not deserialize mu_effect: {e}")
        #
        # The bug: the broad except catches ALL errors and only warns.
        # Correct behavior: deserialization failures should raise.

        mu_effects = []
        caught_warnings = []
        for effect_data in effect_data_list:
            try:
                effect = _deserialize_mu_effect(effect_data)
                mu_effects.append(effect)
            except Exception as e:
                caught_warnings.append(str(e))

        # After this loop, effects are silently dropped.
        # The model loads with 0 effects when it should have 1.
        assert len(mu_effects) == 1, (
            f"Expected 1 mu_effect but got {len(mu_effects)}. "
            f"Errors silently caught: {caught_warnings}"
        )


class TestCustomMuEffectDeserialization:
    """Issue: Custom MuEffect subclasses have no registry entry in
    _MUEFFECT_DESERIALIZERS, so they cannot be deserialized after save.
    """

    def test_custom_mu_effect_can_be_deserialized(self):
        """A user-defined MuEffect subclass should be deserializable after save."""

        class MyCustomEffect(MuEffect):
            my_param: float = 1.0

            def create_data(self, mmm):
                pass

            def create_effect(self, mmm):
                pass

            def set_data(self, mmm, model, X):
                pass

        effect = MyCustomEffect(my_param=42.0)
        serialized = _serialize_mu_effect(effect)
        assert serialized["class"] == "MyCustomEffect"

        deserialized = _deserialize_mu_effect(serialized)
        assert isinstance(deserialized, MyCustomEffect)
        assert deserialized.my_param == 42.0


class TestLinearTrendEffectSerialization:
    """Issue: LinearTrendEffect uses model_config={"extra": "allow"} for
    linear_trend_first_date, and model_dump(mode="json") fails because
    the trend field contains Prior objects that Pydantic can't serialize.

    Sub-issue (1): linear_trend_first_date is a runtime-only attribute not
    preserved through serialization — it's re-derived in create_data() during
    the normal load flow, but this is fragile.

    Sub-issue (2): model_dump(mode="json") raises PydanticSerializationError
    because LinearTrend contains Prior fields.
    """

    def test_linear_trend_effect_model_dump_json_succeeds(self):
        """LinearTrendEffect.model_dump(mode='json') should work even with Priors in trend."""
        effect = LinearTrendEffect(
            trend=LinearTrend(n_changepoints=5),
            prefix="trend",
        )
        # Simulate runtime state being set (as in create_data)
        effect.linear_trend_first_date = pd.Timestamp("2024-01-01")

        d = effect.model_dump(mode="json")
        json_str = json.dumps(d)
        assert json_str is not None

    def test_linear_trend_first_date_preserved_through_serialization(self):
        """linear_trend_first_date should survive serialization roundtrip."""
        effect = LinearTrendEffect(
            trend=LinearTrend(n_changepoints=5),
            prefix="trend",
        )
        effect.linear_trend_first_date = pd.Timestamp("2024-01-01")

        serialized = _serialize_mu_effect(effect)
        deserialized = _deserialize_mu_effect(serialized)

        assert hasattr(deserialized, "linear_trend_first_date")
        assert deserialized.linear_trend_first_date == pd.Timestamp("2024-01-01")


class TestDefaultSerializerWithPriorFields:
    """Issue: The singledispatch default _serialize_mu_effect calls
    model_dump(mode="json"), which fails for any MuEffect subclass
    containing non-trivially-serializable fields. Built-in types avoid
    this via custom singledispatch handlers, but custom user types hit it.

    LinearTrendEffect's custom handler works around this by separately
    serializing the trend field. A custom MuEffect that embeds a
    LinearTrend or Prior (via InstanceOf) has no such handler.
    """

    def test_custom_mu_effect_with_complex_field_serializes(self):
        """A MuEffect subclass with a LinearTrend field should be serializable via default handler."""
        from pydantic import InstanceOf

        class EffectWithTrend(MuEffect):
            trend: InstanceOf[LinearTrend] = LinearTrend(n_changepoints=3)

            def create_data(self, mmm):
                pass

            def create_effect(self, mmm):
                pass

            def set_data(self, mmm, model, X):
                pass

        effect = EffectWithTrend()
        serialized = _serialize_mu_effect(effect)

        assert serialized["class"] == "EffectWithTrend"
        assert "trend" in serialized


class TestSerializationFixesThroughRegistry:
    """Verify that known serialization issues are fixed via the TypeRegistry."""

    def test_hsgp_dims_preserved_as_tuple(self):
        """Issue: HSGP dims tuple->list after JSON roundtrip."""
        from pymc_marketing.serialization import registry

        original = HSGP(m=10, L=1.5, eta=1.0, ls=1.0, dims=("time", "geo"))
        data = registry.serialize(original)
        json_str = json.dumps(data)
        data_back = json.loads(json_str)
        restored = registry.deserialize(data_back)
        assert isinstance(restored.dims, tuple)
        assert restored.dims == ("time", "geo")

    def test_custom_mu_effect_roundtrips(self):
        """Issue: Custom MuEffects not in _MUEFFECT_DESERIALIZERS."""
        from pymc_marketing.serialization import registry

        class MyCustomEffect(MuEffect):
            param: float = 1.0

            def create_data(self, mmm):
                pass

            def create_effect(self, mmm):
                pass

            def set_data(self, mmm, model, X):
                pass

        original = MyCustomEffect(param=42.0)
        data = registry.serialize(original)
        restored = registry.deserialize(data)
        assert isinstance(restored, MyCustomEffect)
        assert restored.param == 42.0

    def test_linear_trend_effect_serializes_without_error(self):
        """Issue: LinearTrendEffect model_dump fails with Prior fields."""
        from pymc_marketing.serialization import registry

        effect = LinearTrendEffect(trend=LinearTrend(n_changepoints=5), prefix="trend")
        data = registry.serialize(effect)
        assert "__type__" in data
        restored = registry.deserialize(data)
        assert isinstance(restored, LinearTrendEffect)

    def test_deferred_factory_replaces_tensor_priors(self):
        """Issue: HSGP Prior equality breaks after roundtrip."""
        from pymc_marketing.serialization import DeferredFactory, registry

        deferred_eta = DeferredFactory(
            factory="pymc_marketing.mmm.hsgp.create_eta_prior",
            kwargs={"upper": 5.0, "mass": 0.95},
        )
        hsgp = HSGP(m=10, L=1.5, eta=deferred_eta, ls=1.0, dims="time")
        data = registry.serialize(hsgp)
        restored = registry.deserialize(data)
        assert isinstance(restored.eta, DeferredFactory)
        resolved = restored.eta.resolve()
        assert resolved is not None
