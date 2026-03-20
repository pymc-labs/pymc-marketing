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

    media_configs: MediaConfigList(
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
        transformed_media_data = media_configs(
            media_data, core_dim="date", media_dim="media"
        )

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pymc as pm
import pytensor.xtensor as ptx
from pymc.distributions.shape_utils import Dims
from pytensor.xtensor.type import XTensorVariable

from pymc_marketing.mmm.components.adstock import (
    AdstockTransformation,
    adstock_from_dict,
)
from pymc_marketing.mmm.components.saturation import (
    SaturationTransformation,
    saturation_from_dict,
)
from pymc_marketing.serialization import registry


@registry.register
@dataclass
class MediaTransformation:
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

    def __post_init__(self):
        """Set the first and second transformations based on the adstock_first flag."""
        self.first, self.second = (
            (self.adstock, self.saturation)
            if self.adstock_first
            else (self.saturation, self.adstock)
        )
        if isinstance(self.dims, str):
            self.dims = (self.dims,)

        self.dims = self.dims or ()

        self._check_compatible_dims()

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

    def __call__(self, x, dim: str | None = None):
        """Apply adstock and saturation transformation to media data.

        Parameters
        ----------
        x : XTensorLike
            The media data to transform.

        Returns
        -------
        XTensorVariable
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
        return self.second.apply(self.first.apply(x, core_dim=dim), core_dim=dim)

    def to_dict(self) -> dict:
        """Convert the media transformation to a dictionary.

        Returns
        -------
        dict
            The media transformation as a dictionary.

        """
        return {
            "__type__": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
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
        adstock_data = data["adstock"]
        saturation_data = data["saturation"]

        if "__type__" in adstock_data:
            adstock = registry.deserialize(adstock_data)
        else:
            adstock = adstock_from_dict(adstock_data)

        if "__type__" in saturation_data:
            saturation = registry.deserialize(saturation_data)
        else:
            saturation = saturation_from_dict(saturation_data)

        return cls(
            adstock=adstock,
            saturation=saturation,
            adstock_first=data["adstock_first"],
            dims=data.get("dims"),
        )


@registry.register
@dataclass
class MediaConfig:
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

    def to_dict(self) -> dict:
        """Convert the media configuration to a dictionary.

        Returns
        -------
        dict
            The media configuration as a dictionary.

        """
        return {
            "__type__": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
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
        return cls(
            name=data["name"],
            columns=data["columns"],
            media_transformation=MediaTransformation.from_dict(
                data["media_transformation"]
            ),
        )


@registry.register
class MediaConfigList:
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

    def __init__(self, media_configs: list[MediaConfig]) -> None:
        self.media_configs = media_configs

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

    def to_dict(self) -> dict:
        """Convert the media configuration list to a dictionary.

        Returns
        -------
        dict
            The media configuration list as a dictionary with ``__type__`` key.

        """
        return {
            "__type__": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "media_configs": [config.to_dict() for config in self.media_configs],
        }

    @classmethod
    def from_dict(cls, data: dict | list) -> MediaConfigList:
        """Create a media configuration list from a dictionary.

        Parameters
        ----------
        data : dict | list
            The data to create the media configuration list from. Supports
            both the new dict format (with ``__type__``) and legacy list format.

        Returns
        -------
        MediaConfigList
            The media configuration list created from the dictionary.

        """
        if isinstance(data, list):
            return cls([MediaConfig.from_dict(config) for config in data])
        configs = data.get("media_configs", [])
        return cls([MediaConfig.from_dict(config) for config in configs])

    def __call__(self, x, *, core_dim: str, media_dim: str) -> XTensorVariable:
        """Apply media transformation to media data.

        Assumes that the columns in the data correspond to the media channels
        in the media_configs.

        Parameters
        ----------
        x : XTensorLike
            The media data to transform.
        core_dim : str
            The dimension along which to apply the transformation (e.g. "date").
        media_dim : str
            The dimension indexing media channels (e.g. "channel" or "media").

        Returns
        -------
        XTensorVariable
            The transformed media data.

        """
        x = ptx.as_xtensor(x)
        model = pm.modelcontext(None)

        transformed_data = []
        start_idx = 0
        for config in self.media_configs:
            config.media_transformation.dims = config.name

            model.add_coord(config.name, config.columns)
            end_idx = start_idx + len(config.columns)

            media_data = x.isel({media_dim: slice(start_idx, end_idx)}).rename(
                {media_dim: config.name}
            )

            adstock = config.media_transformation.adstock
            saturation = config.media_transformation.saturation
            adstock.prefix = f"{config.name}_{adstock.prefix}"
            saturation.prefix = f"{config.name}_{saturation.prefix}"

            media_transformation_data = config.media_transformation(
                media_data, dim=core_dim
            ).rename({config.name: media_dim})
            transformed_data.append(media_transformation_data)

            start_idx = end_idx

        return ptx.concat(transformed_data, dim=media_dim).transpose(
            core_dim, media_dim
        )
