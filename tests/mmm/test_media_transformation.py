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
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import pytest

from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.components.adstock import DelayedAdstock
from pymc_marketing.mmm.components.saturation import TanhSaturation
from pymc_marketing.mmm.media_transformation import (
    MediaConfig,
    MediaConfigList,
    MediaTransformation,
)
from pymc_marketing.mmm.transformers import ConvMode


@pytest.fixture
def rng() -> np.random.Generator:
    seed = sum(map(ord, "Different Media Transformations"))
    return np.random.default_rng(seed)


@pytest.fixture
def media_data(rng) -> pd.DataFrame:
    n_dates = 10
    dates = pd.date_range(start="2021-01-01", periods=n_dates, freq="W-MON")

    columns = ["TV", "Radio", "Facebook", "Instagram", "YouTube", "TikTok"]

    data_dist = pm.HalfNormal.dist(sigma=5, shape=(n_dates, len(columns)))
    data = pm.draw(data_dist, random_seed=rng)

    return pd.DataFrame(data, index=dates, columns=columns)


@pytest.fixture
def create_media_config_list():
    def create(online_dims, offline_dims) -> MediaConfigList:
        return MediaConfigList(
            [
                MediaConfig(
                    name="online",
                    columns=["Facebook", "Instagram", "YouTube", "TikTok"],
                    media_transformation=MediaTransformation(
                        adstock=GeometricAdstock(
                            l_max=10,
                        ).set_dims_for_all_priors(online_dims),
                        saturation=LogisticSaturation().set_dims_for_all_priors(
                            online_dims
                        ),
                        adstock_first=True,
                        dims=online_dims,
                    ),
                ),
                MediaConfig(
                    name="offline",
                    columns=["TV", "Radio"],
                    media_transformation=MediaTransformation(
                        adstock=GeometricAdstock(
                            l_max=10,
                        ).set_dims_for_all_priors(offline_dims),
                        saturation=LogisticSaturation().set_dims_for_all_priors(
                            offline_dims
                        ),
                        adstock_first=False,
                        dims=offline_dims,
                    ),
                ),
            ]
        )

    return create


@pytest.fixture
def media_configs(create_media_config_list) -> MediaConfigList:
    return create_media_config_list(online_dims=(), offline_dims=())


def test_get_media_values(media_configs: MediaConfigList) -> None:
    assert media_configs.media_values == [
        "Facebook",
        "Instagram",
        "YouTube",
        "TikTok",
        "TV",
        "Radio",
    ]


@pytest.mark.parametrize("online_dims", [(), ("online",)], ids=["scalar", "vector"])
@pytest.mark.parametrize("offline_dims", [(), ("offline",)], ids=["scalar", "vector"])
def test_apply_media_transformation(
    online_dims,
    offline_dims,
    media_data: pd.DataFrame,
    create_media_config_list,
    media_configs: MediaConfigList,
) -> None:
    media_configs = create_media_config_list(
        online_dims=online_dims, offline_dims=offline_dims
    )
    media_columns = media_configs.media_values
    coords = {
        "date": media_data.index,
        "media": media_columns,
    }

    with pm.Model(coords=coords) as model:
        data = pmd.Data(
            "media_data",
            media_data.loc[:, media_columns].to_numpy(),
            dims=("date", "media"),
        )
        transformed_media_data = media_configs(
            data,
        ).eval()

    assert transformed_media_data.shape == (media_data.shape[0], len(media_columns))

    expected_free_RVs = {
        "online_adstock_alpha",
        "online_saturation_lam",
        "online_saturation_beta",
        "offline_saturation_lam",
        "offline_saturation_beta",
        "offline_adstock_alpha",
    }
    free_RVs = {rv.name for rv in model.free_RVs}

    assert free_RVs == expected_free_RVs

    actual_dims = model.named_vars_to_dims
    for rv in expected_free_RVs:
        expected_dims = offline_dims if rv.startswith("offline") else online_dims
        assert actual_dims[rv] == expected_dims


def test_media_transformation_deserialize() -> None:
    adstock = GeometricAdstock(l_max=10)
    saturation = LogisticSaturation()
    data = {
        "adstock": adstock.to_dict(),
        "saturation": saturation.to_dict(),
        "adstock_first": True,
    }

    media_transformation = MediaTransformation.from_dict(data)
    assert isinstance(media_transformation, MediaTransformation)


def test_media_config_list_deserialize() -> None:
    adstock = GeometricAdstock(l_max=10)
    saturation = LogisticSaturation()
    data = [
        {
            "name": "online",
            "columns": ["Facebook", "Instagram", "YouTube", "TikTok"],
            "media_transformation": {
                "adstock": adstock.to_dict(),
                "saturation": saturation.to_dict(),
                "adstock_first": True,
            },
        }
    ]

    media_config_list = MediaConfigList.from_dict(data)
    assert isinstance(media_config_list, MediaConfigList)


def test_media_transformation_round_trip() -> None:
    adstock = GeometricAdstock(l_max=10)
    saturation = LogisticSaturation()
    media_transformation = MediaTransformation(
        adstock=adstock,
        saturation=saturation,
        adstock_first=True,
        dims="media",
    )

    data = media_transformation.to_dict()

    assert data == {
        "__type__": f"{MediaTransformation.__module__}.{MediaTransformation.__qualname__}",
        "adstock": adstock.to_dict(),
        "saturation": saturation.to_dict(),
        "adstock_first": True,
        "dims": ("media",),
    }
    recovered = MediaTransformation.from_dict(data)
    assert type(recovered.adstock) is GeometricAdstock
    assert recovered.adstock.l_max == 10
    assert type(recovered.saturation) is LogisticSaturation
    assert recovered.adstock_first is True
    assert recovered.dims == ("media",)
    assert recovered == media_transformation


@pytest.mark.parametrize(
    "adstock_dims, saturation_dims",
    [
        ((), "media"),
        ("media", ()),
        ("media", "media"),
    ],
)
def test_incompatible_dims_raise(adstock_dims, saturation_dims) -> None:
    adstock = GeometricAdstock(l_max=10).set_dims_for_all_priors(adstock_dims)
    saturation = LogisticSaturation().set_dims_for_all_priors(saturation_dims)
    with pytest.raises(ValueError):
        MediaTransformation(
            adstock=adstock,
            saturation=saturation,
            adstock_first=True,
            dims=(),
        )


class TestMediaTransformationTypeRegistry:
    """Tests for TypeRegistry-based round-trip serialization of Media classes."""

    def test_media_transformation_roundtrip_all_parameters(self):
        from pymc_extras.prior import Prior

        from pymc_marketing.serialization import registry

        adstock = GeometricAdstock(
            l_max=8,
            normalize=False,
            mode=ConvMode.Before,
            prefix="custom_adstock",
            priors={"alpha": Prior("Beta", alpha=2.0, beta=5.0)},
        )
        saturation = LogisticSaturation(
            prefix="custom_sat",
            priors={
                "lam": Prior("Gamma", alpha=2, beta=2),
                "beta": Prior("HalfNormal", sigma=3),
            },
        )
        original = MediaTransformation(
            adstock=adstock,
            saturation=saturation,
            adstock_first=False,
            dims=("channel",),
        )
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is MediaTransformation
        assert type(restored.adstock) is GeometricAdstock
        assert type(restored.saturation) is LogisticSaturation
        assert restored.adstock_first is False
        assert restored.dims == ("channel",)
        assert restored.adstock.l_max == 8
        assert restored.adstock.normalize is False
        assert restored.adstock.mode == ConvMode.Before
        assert restored.adstock.prefix == "custom_adstock"
        assert restored.saturation.prefix == "custom_sat"
        assert restored == original

    def test_media_config_roundtrip_all_parameters(self):
        from pymc_marketing.serialization import registry

        mt = MediaTransformation(
            adstock=GeometricAdstock(l_max=6),
            saturation=LogisticSaturation(),
            adstock_first=True,
        )
        original = MediaConfig(
            name="online_channels",
            columns=["tv", "radio", "digital"],
            media_transformation=mt,
        )
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is MediaConfig
        assert restored.name == "online_channels"
        assert restored.columns == ["tv", "radio", "digital"]
        assert type(restored.media_transformation.adstock) is GeometricAdstock
        assert restored.media_transformation.adstock.l_max == 6
        assert restored == original

    def test_media_config_list_roundtrip_multiple_configs(self):
        from pymc_marketing.serialization import registry

        mt1 = MediaTransformation(
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
            adstock_first=True,
        )
        mt2 = MediaTransformation(
            adstock=DelayedAdstock(l_max=6),
            saturation=TanhSaturation(),
            adstock_first=False,
        )
        mc1 = MediaConfig(
            name="online", columns=["tv", "radio"], media_transformation=mt1
        )
        mc2 = MediaConfig(name="offline", columns=["print"], media_transformation=mt2)
        original = MediaConfigList([mc1, mc2])
        data = registry.serialize(original)
        restored = registry.deserialize(data)

        assert type(restored) is MediaConfigList
        assert len(restored.media_configs) == 2
        assert restored.media_configs[0].name == "online"
        assert restored.media_configs[0].columns == ["tv", "radio"]
        assert restored.media_configs[1].name == "offline"
        assert (
            type(restored.media_configs[1].media_transformation.adstock)
            is DelayedAdstock
        )
        assert restored == original
