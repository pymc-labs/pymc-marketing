#   Copyright 2022 - 2025 The PyMC Labs Developers
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
"""Module for applying media transformations to media data.

Examples
--------
Create a media transformation for online and offline media channels:

.. code-block:: python

    from pymc_marketing.mmm import (
        GeometricAdstock,
        HillSaturation,
        MediaTransformation,
        MichaelisMentenSaturation,
    )

    # Shared media transformation for all offline media channels
    offline_media_transform = MediaTransformation(
        adstock=GeometricAdstock(l_max=15),
        saturation=HillSaturation(),
        adstock_first=True,
    )
    # Shared media transformation for all online media channels
    online_media_transform = MediaTransformation(
            adstock=GeometricAdstock(l_max=10),
            saturation=MichaelisMentenSaturation(),
            adstock_first=False,
        ),
    )

Create a combined media configuration for offline and online media channels:

.. code-block:: python

    from pymc_marketing.mmm import (
        MediaConfig,
        MediaConfigList,
    )

    media_configs = MediaConfigList(
        [
            MediaConfig(
                name="offline",
                columns=["TV", "Radio"],
                media_transformation=offline_media_transform,
            ),
            MediaConfig(
                name="online",
                columns=["Facebook", "Instagram", "YouTube", "TikTok"],
                media_transformation=online_media_transform,
            ),
        ]
    )


Apply the media transformation to media data in PyMC model:

.. code-block:: python

    import pymc as pm
    import pandas as pd

    df: pd.DataFrame = ...


    media_columns = media_configs.media_values

    coords = {
        "date": df["week"],
        "media": media_columns,
    }
    with pm.Model(coords=coords) as model:
        media_data = pm.Data(
            "media_data", df.loc[:, media_columns].to_numpy(), dims=("date", "media")
        )
        transformed_media_data = media_configs(media_data)

"""

from __future__ import annotations

from typing import cast

import pymc as pm
import pytensor.tensor as pt
from pydantic import (
    BaseModel,
    ConfigDict,
    RootModel,
    computed_field,
    field_serializer,
    model_validator,
)
from pymc.distributions.shape_utils import Dims
from pymc_extras.deserialize import deserialize, register_deserialization

from pymc_marketing.mmm.components.adstock import (
    AdstockTransformation,
)
from pymc_marketing.mmm.components.saturation import (
    SaturationTransformation,
)


class MediaTransformation(BaseModel):
    """Wrapper for applying adstock and saturation transformation to media data.

    Parameters
    ----------
    adstock : AdstockTransformation
        The adstock transformation to apply.
    saturation : SaturationTransformation
        The saturation transformation to apply.
    adstock_first : bool
        Flag to apply the adstock transformation first.
    dims : Dims
        The dimensions of the parameters.

    Attributes
    ----------
    first : AdstockTransformation | SaturationTransformation
        The first transformation to apply.
    second : AdstockTransformation | SaturationTransformation
        The second transformation to apply.

    """

    adstock: AdstockTransformation
    saturation: SaturationTransformation
    adstock_first: bool
    dims: Dims | None = None
    model_config = ConfigDict(extra="forbid")

    @field_serializer("adstock", when_used="json")
    def serialize_adstock(self, value: AdstockTransformation) -> dict:
        """Serialize AdstockTransformation to dict for JSON mode."""
        return value.to_dict()

    @field_serializer("saturation", when_used="json")
    def serialize_saturation(self, value: SaturationTransformation) -> dict:
        """Serialize SaturationTransformation to dict for JSON mode."""
        return value.to_dict()

    @computed_field
    def first(self) -> AdstockTransformation | SaturationTransformation:
        """First transformation to apply based on adstock_first flag."""
        return self.adstock if self.adstock_first else self.saturation

    @computed_field
    def second(self) -> AdstockTransformation | SaturationTransformation:
        """Second transformation to apply based on adstock_first flag."""
        return self.saturation if self.adstock_first else self.adstock

    @model_validator(mode="after")
    def _post_init(self):
        """Validate dims and ensure compatibility."""
        if isinstance(self.dims, str):
            self.dims = (self.dims,)

        self.dims = self.dims or ()

        self._check_compatible_dims()
        return self

    def _check_compatible_dims(self):
        self.dims = cast(Dims, self.dims)

        if not set(self.adstock.combined_dims).issubset(self.dims):
            raise ValueError(
                f"Adstock dimensions {self.adstock.combined_dims} are not a subset of {self.dims}"
            )

        if not set(self.saturation.combined_dims).issubset(self.dims):
            raise ValueError(
                f"Saturation dimensions {self.saturation.combined_dims} are not a subset of {self.dims}"
            )

    def __call__(self, x):
        """Apply adstock and saturation transformation to media data.

        Parameters
        ----------
        x : pt.TensorLike
            The media data to transform.
        dim : str
            The dimension of the parameters.

        Returns
        -------
        pt.TensorVariable
            The transformed media data.

        Examples
        --------
        Apply the media transformation to media data:

        .. code-block:: python

            from pymc_marketing.mmm import (
                GeometricAdstock,
                HillSaturation,
                MediaTransformation,
            )

            media_data = ...

            media_transformation = MediaTransformation(
                adstock=GeometricAdstock(l_max=15),
                saturation=HillSaturation(),
                adstock_first=True,
            )

            coords = {
                "date": ...,
                "media": ...,
            }
            with pm.Model(coords=coords) as model:
                transformed_media_data = media_transformation(
                    media_data,
                    dim="media",
                )

        """
        return self.second.apply(self.first.apply(x, self.dims), self.dims)

    def to_dict(self) -> dict:
        """Convert the media transformation to a dictionary.

        Returns
        -------
        dict
            The media transformation as a dictionary.

        """
        return {
            "adstock": self.adstock.to_dict(),
            "saturation": self.saturation.to_dict(),
            "adstock_first": self.adstock_first,
            "dims": self.dims,
        }

    @classmethod
    def from_dict(cls, data) -> MediaTransformation:
        """Create a media transformation from a dictionary.

        Parameters
        ----------
        data : dict
            The data to create the media transformation from.

        Returns
        -------
        MediaTransformation
            The media transformation created from the dictionary.

        """
        # Defensively deserialize Transformation fields if they are dicts
        inner_data = data.copy()
        if "adstock" in inner_data and isinstance(inner_data["adstock"], dict):
            inner_data["adstock"] = deserialize(inner_data["adstock"])
        if "saturation" in inner_data and isinstance(inner_data["saturation"], dict):
            inner_data["saturation"] = deserialize(inner_data["saturation"])

        return cls.model_validate(inner_data)


def _is_media_transformation(data):
    return (
        isinstance(data, dict)
        and "adstock" in data
        and "saturation" in data
        and "adstock_first" in data
    )


register_deserialization(
    is_type=_is_media_transformation,
    deserialize=MediaTransformation.from_dict,
)


class MediaConfig(BaseModel):
    """Configuration for a media transformation to certain media channels.

    Parameters
    ----------
    name : str
        The name of the media transformation and prefix of all media variables.
    columns : list[str]
        The media channels to apply the transformation to.
    media_transformation : MediaTransformation
        The media transformation to apply to the media channels.

    """

    name: str
    columns: list[str]
    media_transformation: MediaTransformation
    model_config = ConfigDict(extra="forbid")

    @field_serializer("media_transformation", when_used="json")
    def serialize_media_transformation(self, value: MediaTransformation) -> dict:
        """Serialize MediaTransformation to dict for JSON mode."""
        return value.to_dict()

    def to_dict(self) -> dict:
        """Convert the media configuration to a dictionary.

        Returns
        -------
        dict
            The media configuration as a dictionary.

        """
        return {
            "name": self.name,
            "columns": self.columns,
            "media_transformation": self.media_transformation.to_dict(),
        }

    @classmethod
    def from_dict(cls, data) -> MediaConfig:
        """Create a media configuration from a dictionary.

        Parameters
        ----------
        data : dict
            The data to create the media configuration from.

        Returns
        -------
        MediaConfig
            The media configuration created from the dictionary.

        """
        # Defensively deserialize nested MediaTransformation if it's a dict
        inner_data = data.copy()
        if "media_transformation" in inner_data and isinstance(
            inner_data["media_transformation"], dict
        ):
            inner_data["media_transformation"] = deserialize(
                inner_data["media_transformation"]
            )

        return cls.model_validate(inner_data)


def _is_media_config(data):
    return (
        isinstance(data, dict)
        and "name" in data
        and "columns" in data
        and "media_transformation" in data
        and _is_media_transformation(data["media_transformation"])
    )


class MediaConfigList(RootModel):
    """Wrapper for a list of media configurations to apply to media data.

    Parameters
    ----------
    media_configs : list[MediaConfig]
        The media configurations to apply to the media data.


    Examples
    --------
    Different order of media transformations for online and offline media channels:

    .. code-block:: python

        from pymc_marketing.mmm import (
            GeometricAdstock,
            LogisticSaturation,
            MediaTransformation,
            MediaConfig,
            MediaConfigList,
        )

        online = MediaConfig(
            name="online",
            columns=["Facebook", "Instagram", "YouTube", "TikTok"],
            media_transformation=MediaTransformation(
                adstock=GeometricAdstock(l_max=10).set_dims_for_all_priors("online"),
                saturation=LogisticSaturation().set_dims_for_all_priors("online"),
                adstock_first=True,
            ),
        )

        offline = MediaConfig(
            name="offline",
            columns=["TV", "Radio"],
            media_transformation=MediaTransformation(
                adstock=GeometricAdstock(
                    l_max=10,
                ).set_dims_for_all_priors("offline"),
                saturation=LogisticSaturation().set_dims_for_all_priors("offline"),
                adstock_first=False,
            ),
        )

        media_configs = MediaConfigList([online, offline])

    """

    root: list[MediaConfig]

    @property
    def media_configs(self) -> list[MediaConfig]:
        """Backward-compatible property for accessing the root list."""
        return self.root

    @field_serializer("root", when_used="json")
    def serialize_media_configs(self, value: list[MediaConfig]) -> list[dict]:
        """Serialize list of MediaConfigs to list of dicts for JSON mode."""
        return [config.to_dict() for config in value]

    def __eq__(self, other) -> bool:
        """Check if the media configuration lists are equal.

        Parameters
        ----------
        other : MediaConfigList
            The other media configuration list to compare.

        Returns
        -------
        bool
            True if the media configuration lists are equal, False otherwise.

        """
        return self.media_configs == other.media_configs

    def __getitem__(self, key: int) -> MediaConfig:
        """Get the media configuration at the specified index.

        Parameters
        ----------
        key : int
            The index of the media configuration to get.

        Returns
        -------
        MediaConfig
            The media configuration at the specified index.

        """
        return self.media_configs[key]

    @property
    def media_values(self) -> list[str]:
        """Get the media values from the media configurations.

        Returns
        -------
        list[str]
            The media values from the media configurations in the order they appear.

        """
        result = []
        for config in self.media_configs:
            result.extend(config.columns)
        return result

    def to_dict(self) -> list[dict]:
        """Convert the media configuration list to a dictionary.

        Returns
        -------
        list[dict]
            The media configuration list as a dictionary.

        """
        return [config.to_dict() for config in self.media_configs]

    @classmethod
    def from_dict(cls, data: list[dict]) -> MediaConfigList:
        """Create a media configuration list from a dictionary.

        Parameters
        ----------
        data : list[dict]
            The data to create the media configuration list from.

        Returns
        -------
        MediaConfigList
            The media configuration list created from the dictionary.

        """
        # Defensively deserialize each MediaConfig if needed
        media_configs = []
        for config_data in data:
            if isinstance(config_data, dict):
                media_configs.append(MediaConfig.from_dict(config_data))
            else:
                media_configs.append(config_data)

        return cls(media_configs)

    def __call__(self, x) -> pt.TensorVariable:
        """Apply media transformation to media data.

        Assumes that the columns in the data correspond to the media channels
        in the media_configs.

        Parameters
        ----------
        x : pt.TensorLike
            The media data to transform.

        Returns
        -------
        pt.TensorVariable
            The transformed media data.

        """
        model = pm.modelcontext(None)

        transformed_data = []
        start_idx = 0
        for config in self.media_configs:
            config.media_transformation.dims = config.name

            model.add_coord(config.name, config.columns)
            end_idx = start_idx + len(config.columns)

            media_data = x[:, start_idx:end_idx]

            adstock = config.media_transformation.adstock
            saturation = config.media_transformation.saturation
            adstock.prefix = f"{config.name}_{adstock.prefix}"
            saturation.prefix = f"{config.name}_{saturation.prefix}"

            media_transformation_data = config.media_transformation(
                media_data,
            )
            transformed_data.append(media_transformation_data)

            start_idx = end_idx

        return pt.concatenate(transformed_data, axis=1)


def _is_media_config_list(data):
    return isinstance(data, list) and all(_is_media_config(config) for config in data)


register_deserialization(
    is_type=_is_media_config_list,
    deserialize=MediaConfigList.from_dict,
)
