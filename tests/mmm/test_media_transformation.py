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
from copy import deepcopy

import numpy as np
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import pytest
import xarray as xr
from pymc_extras.prior import Prior

from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.media_transformation import (
    MediaConfig,
    MediaConfigList,
    MediaTransformation,
)
from pymc_marketing.mmm.transformers import ConvMode
from pymc_marketing.serialization import SerializationError, serialization


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
            data, core_dim="date", media_dim="media"
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


def test_media_config_list_dim_order_independent(
    media_data: pd.DataFrame,
    create_media_config_list,
) -> None:
    """MediaConfigList works regardless of the dimension order of x."""
    media_configs = create_media_config_list(online_dims=(), offline_dims=())
    media_columns = media_configs.media_values
    n_dates = len(media_data.index)
    coords = {
        "date": media_data.index,
        "media": media_columns,
    }

    with pm.Model(coords=coords):
        data_media_first = pmd.Data(
            "media_data",
            media_data.loc[:, media_columns].to_numpy().T,
            dims=("media", "date"),
        )
        result = media_configs(
            data_media_first, core_dim="date", media_dim="media"
        ).eval()

    assert result.shape == (n_dates, len(media_columns))


def test_media_transformation_from_dict_rejects_v0_lookup_name() -> None:
    """Regression test for issue #2430: v0 saves without __type__ are rejected."""
    data = {
        "adstock": {"lookup_name": "geometric", "l_max": 10},
        "saturation": {"lookup_name": "logistic"},
        "adstock_first": True,
    }

    with pytest.raises(SerializationError, match="__type__"):
        MediaTransformation.from_dict(data)


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


@pytest.mark.parametrize(
    "adstock_first, first_name, second_name",
    [(True, "adstock", "saturation"), (False, "saturation", "adstock")],
)
def test_reassigned_transformations_are_applied(
    adstock_first, first_name, second_name
) -> None:
    """Assigning ``adstock`` / ``saturation`` after construction changes what is applied, in order."""
    media_transformation = MediaTransformation(
        adstock=GeometricAdstock(l_max=2, prefix="old_adstock"),
        saturation=LogisticSaturation(prefix="old_saturation"),
        adstock_first=adstock_first,
    )
    media_transformation.adstock = GeometricAdstock(l_max=2, prefix="new_adstock")
    media_transformation.saturation = LogisticSaturation(prefix="new_saturation")

    with pm.Model(coords={"date": range(5)}) as model:
        media_transformation(pmd.Data("x", np.ones(5), dims="date"), dim="date")

    assert {rv.name for rv in model.free_RVs} == {
        "new_adstock_alpha",
        "new_saturation_lam",
        "new_saturation_beta",
    }
    assert media_transformation.first is getattr(media_transformation, first_name)
    assert media_transformation.second is getattr(media_transformation, second_name)


class TestMediaTransformationRoundtrips:
    def test_full_media_config_list_all_parameters(self):
        from pymc_marketing.mmm.components.adstock import (
            DelayedAdstock,
        )
        from pymc_marketing.mmm.components.saturation import (
            TanhSaturation,
        )

        mt1 = MediaTransformation(
            adstock=GeometricAdstock(
                l_max=8,
                normalize=False,
                mode=ConvMode.Before,
                prefix="geo_adstock",
                priors={"alpha": Prior("Beta", alpha=2.0, beta=5.0)},
            ),
            saturation=LogisticSaturation(
                prefix="log_sat",
                priors={
                    "lam": Prior("Gamma", alpha=2, beta=2),
                    "beta": Prior("HalfNormal", sigma=3),
                },
            ),
            adstock_first=False,
            dims=("channel",),
        )
        mt2 = MediaTransformation(
            adstock=DelayedAdstock(l_max=6),
            saturation=TanhSaturation(),
            adstock_first=True,
        )
        mc1 = MediaConfig(
            name="online", columns=["tv", "radio"], media_transformation=mt1
        )
        mc2 = MediaConfig(name="offline", columns=["print"], media_transformation=mt2)
        original = MediaConfigList([mc1, mc2])
        data = serialization.serialize(original)
        restored = serialization.deserialize(data)

        assert type(restored) is MediaConfigList
        assert len(restored.media_configs) == 2

        r1 = restored.media_configs[0]
        assert r1.name == "online"
        assert r1.columns == ["tv", "radio"]
        assert type(r1.media_transformation.adstock) is GeometricAdstock
        assert r1.media_transformation.adstock.l_max == 8
        assert r1.media_transformation.adstock.normalize is False
        assert r1.media_transformation.adstock_first is False

        r2 = restored.media_configs[1]
        assert r2.name == "offline"
        assert type(r2.media_transformation.adstock) is DelayedAdstock

        assert restored == original


@pytest.mark.parametrize(
    "type_key",
    [
        "pymc_marketing.mmm.media_transformation.MediaTransformation",
        "pymc_marketing.mmm.media_transformation.MediaConfig",
        "pymc_marketing.mmm.media_transformation.MediaConfigList",
    ],
    ids=lambda s: s.rsplit(".", 1)[-1],
)
def test_media_type_registered(type_key):
    assert type_key in serialization._registry, f"{type_key} not registered"


@pytest.mark.parametrize("adstock_first", [True, False])
def test_media_config_list_preserves_config_across_models(adstock_first):
    transformation = MediaTransformation(
        adstock=GeometricAdstock(l_max=2).set_dims_for_all_priors("online"),
        saturation=LogisticSaturation().set_dims_for_all_priors("online"),
        adstock_first=adstock_first,
        dims=("online",),
    )
    configs = MediaConfigList([MediaConfig("online", ["a", "b"], transformation)])
    before = deepcopy(configs.to_dict())
    for _ in range(2):
        with pm.Model(coords={"date": range(5), "media": ["a", "b"]}) as model:
            configs(
                pmd.Data("x", np.ones((5, 2)), dims=("date", "media")),
                core_dim="date",
                media_dim="media",
            )
        assert {rv.name for rv in model.free_RVs} == {
            "online_adstock_alpha",
            "online_saturation_lam",
            "online_saturation_beta",
        }
        assert configs.to_dict() == before
        assert configs == MediaConfigList.from_dict(before)


@pytest.mark.parametrize("adstock_first", [True, False])
def test_media_config_list_preserves_shared_transformation(adstock_first):
    transformation = MediaTransformation(
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        adstock_first=adstock_first,
    )
    configs = MediaConfigList(
        [
            MediaConfig("online", ["a", "b"], transformation),
            MediaConfig("offline", ["c"], transformation),
        ]
    )
    before = deepcopy(configs.to_dict())
    for _ in range(2):
        with pm.Model(coords={"date": range(5), "media": ["a", "b", "c"]}) as model:
            configs(
                pmd.Data("x", np.ones((5, 3)), dims=("date", "media")),
                core_dim="date",
                media_dim="media",
            )
        assert {rv.name for rv in model.free_RVs} == {
            f"{group}_{parameter}"
            for group in ["online", "offline"]
            for parameter in ["adstock_alpha", "saturation_lam", "saturation_beta"]
        }
        assert configs.to_dict() == before
        assert configs == MediaConfigList.from_dict(before)


@pytest.mark.parametrize("parameter", ["hyperprior", "tied", "data"])
def test_media_config_list_preserves_model_connections(parameter):
    with pm.Model(coords={"date": range(5), "media": ["a", "b", "c"]}) as model:
        if parameter == "hyperprior":
            shared = pmd.HalfNormal("global_scale", 1.0)
        elif parameter == "tied":
            shared = pmd.Beta("tied_alpha", alpha=2.0, beta=2.0)
        else:
            shared = pmd.Data("fixed_alpha", np.array(0.2))

        transformation = MediaTransformation(
            GeometricAdstock(
                l_max=2, priors={"alpha": 0.4 if parameter == "hyperprior" else shared}
            ),
            LogisticSaturation(
                priors={
                    "beta": 1.0,
                    "lam": Prior("HalfNormal", sigma=shared)
                    if parameter == "hyperprior"
                    else 1.0,
                }
            ),
            adstock_first=True,
        )
        configs = MediaConfigList(
            [
                MediaConfig(name, columns, transformation)
                for name, columns in [("online", ["a", "b"]), ("offline", ["c"])]
            ]
        )
        out = configs(
            pmd.Data(
                "x", np.linspace(0.1, 1.5, 15).reshape(5, 3), dims=("date", "media")
            ),
            core_dim="date",
            media_dim="media",
        )
        pmd.Deterministic("transformed", out, dims=("date", "media"))
        pmd.Normal(
            "y",
            out.sum("media"),
            1.0,
            observed=pmd.Data("y_obs", np.full(5, 2.0), dims="date"),
        )

    assert np.isfinite(model.compile_logp()(model.initial_point()))
    assert transformation.adstock.prefix == "adstock"
    assert transformation.saturation.prefix == "saturation"

    def draw(model, seed=41):
        with model:
            return pm.sample_prior_predictive(
                draws=3, var_names=["transformed"], random_seed=seed
            ).prior["transformed"]

    if parameter == "data":
        before = draw(model)
        with model:
            pm.set_data({"fixed_alpha": np.array(0.8)})
        after = draw(model)
    else:
        low, high = (0.1, 2.0) if parameter == "hyperprior" else (0.2, 0.8)
        low_model = pm.do(model, {shared.name: np.array(low)})
        high_model = pm.do(model, {shared.name: np.array(high)})
        before, after = draw(low_model), draw(high_model)
        if parameter == "tied":
            # All transformation parameters are fixed after the intervention.
            xr.testing.assert_allclose(draw(low_model, 1), draw(low_model, 2))

    # The same model variable must drive both media groups.
    for config in configs:
        assert not np.allclose(
            before.sel(media=config.columns), after.sel(media=config.columns)
        )


def test_prefixed_components_use_each_groups_posterior():
    configs = MediaConfigList(
        [
            MediaConfig(
                name,
                columns,
                MediaTransformation(
                    GeometricAdstock(l_max=2, prefix="carry").set_dims_for_all_priors(
                        name
                    ),
                    LogisticSaturation(prefix="response").set_dims_for_all_priors(name),
                    adstock_first=True,
                    dims=(name,),
                ),
            )
            for name, columns in [("online", ["a", "b"]), ("offline", ["c"])]
        ]
    )
    before = deepcopy(configs.to_dict())
    with pm.Model(coords={"date": range(5), "media": ["a", "b", "c"]}):
        configs(
            pmd.Data("x", np.ones((5, 3)), dims=("date", "media")),
            core_dim="date",
            media_dim="media",
        )
        posterior = pm.sample_prior_predictive(draws=20, random_seed=0).prior

    for config in configs:
        transformation = config.prefixed_media_transformation()
        for component in [transformation.adstock, transformation.saturation]:
            curves = [
                component.sample_curve(posterior, random_seed=seed, progressbar=False)
                for seed in (1, 2)
            ]
            xr.testing.assert_allclose(*curves)
            for dim in ["chain", "draw", config.name]:
                xr.testing.assert_equal(curves[0].coords[dim], posterior.coords[dim])
    assert configs.to_dict() == before
