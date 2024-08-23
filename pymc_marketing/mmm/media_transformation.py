#   Copyright 2024 The PyMC Labs Developers
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

    offline_media_transform = MediaTransformation(
        adstock=GeometricAdstock(l_max=15),
        saturation=HillSaturation(),
        adstock_first=True,
    )
    online_media_transform = MediaTransformation(
            adstock=GeometricAdstock(l_max=10),
            saturation=MichaelisMentenSaturation(),
            adstock_first=False,
        ),
    )

Create a media configurations for offline and online media channels:

.. code-block:: python

    from pymc_marketing.mmm import (
        MediaConfig,
        MediaConfigs,
    )

    media_configs: MediaConfigs = [
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


Apply the media transformation to media data in PyMC model:

.. code-block:: python

    import pymc as pm
    import pandas as pd

    from pymc_marketing.mmm import get_media_values

    df: pd.DataFrame = ...

    media_columns = get_media_values(media_configs)

    coords = {
        "date": df["week"],
        "media": media_columns,
    }
    with pm.Model(coords=coords) as model:
        media_data = pm.Data(
            "media_data",
            df.loc[:, media_columns].to_numpy(),
            dims=("date", "media")
        )
        transformed_media_data = apply_media_transformation(
            media_data,
            media_configs,
        )

"""

from dataclasses import dataclass

import pymc as pm
import pytensor.tensor as pt

from pymc_marketing.mmm.components.adstock import (
    AdstockTransformation,
)
from pymc_marketing.mmm.components.saturation import SaturationTransformation


@dataclass
class MediaTransformation:
    """Wrapper for applying adstock and saturation transformation to media data."""

    adstock: AdstockTransformation
    saturation: SaturationTransformation
    adstock_first: bool

    def __post_init__(self):
        """Set the first and second transformations based on the adstock_first flag."""
        self.first, self.second = (
            (self.adstock, self.saturation)
            if self.adstock_first
            else (self.saturation, self.adstock)
        )

    def __call__(self, x, dim):
        """Apply adstock and saturation transformation to media data.

        Parameters
        ----------
        x : pt.TensorLike
            The media data to transform.
        dim : str
            The dimension to apply the transformations to.

        Returns
        -------
        pt.TensorVariable
            The transformed media data.

        """
        return self.second.apply(self.first.apply(x, dim), dim)


@dataclass
class MediaConfig:
    """Configuration for a media transformation to certain media channels."""

    name: str
    columns: list[str]
    media_transformation: MediaTransformation


MediaConfigs = list[MediaConfig]


def apply_media_transformation(
    data: pt.TensorLike,
    media_configs: MediaConfigs,
    model: pm.Model | None = None,
) -> pt.TensorVariable:
    """Apply media transformation to media data.

    Assumes that the columns in the data correspond to the media channels in the media_configs.

    Parameters
    ----------
    data : pt.TensorLike
        The media data to transform.
    media_configs : MediaConfigs
        The media configurations to apply.
    model : pm.Model, optional
        The PyMC model to add the media coordinates to. Defaults to model in context.

    Returns
    -------
    pt.TensorVariable
        The transformed media data.

    """
    current_model: pm.Model = pm.modelcontext(model)

    transformed_data = []
    start_idx = 0
    for config in media_configs:
        current_model.add_coord(config.name, config.columns)
        end_idx = start_idx + len(config.columns)

        media_data = data[:, start_idx:end_idx]

        adstock = config.media_transformation.adstock
        saturation = config.media_transformation.saturation

        adstock.prefix = f"{config.name}_{adstock.prefix}"
        saturation.prefix = f"{config.name}_{saturation.prefix}"

        media_transformation_data = config.media_transformation(
            media_data, dim=config.name
        )
        transformed_data.append(media_transformation_data)

        start_idx = end_idx

    return pt.concatenate(transformed_data, axis=1)


def get_media_values(media_configs: MediaConfigs) -> list[str]:
    """Get the media values from the media configurations.

    Parameters
    ----------
    media_configs : MediaConfigs
        The media configurations to extract the media values from.

    Returns
    -------
    list[str]
        The media values from the media configurations in the order they appear.

    """
    media_values = []
    for config in media_configs:
        media_values.extend(config.columns)
    return media_values
