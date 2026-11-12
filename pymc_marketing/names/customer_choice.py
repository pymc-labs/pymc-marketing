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
"""Names the discrete choice (logit) models share.

Covers :class:`~pymc_marketing.customer_choice.mnl_logit.MNLogit`,
:class:`~pymc_marketing.customer_choice.nested_logit.NestedLogit`,
:class:`~pymc_marketing.customer_choice.mixed_logit.MixedLogit`,
:class:`~pymc_marketing.customer_choice.consideration_set_logit.ConsiderationSetMixedLogit`
and their shared data mixin. Names that only one model creates and reads stay in
that model's module.

Each name is a ``Final`` string attribute, so ``mypy`` reports a typo as
``attr-defined`` and the runtime value stays a plain ``str``. The values are
public API: saved models store them, so they must not change.
"""

from typing import Final


class Dim:
    """Dimension names of the logit models."""

    ALTS: Final = "alts"
    OBS: Final = "obs"
    ALT_COVARIATES: Final = "alt_covariates"
    FIXED_COVARIATES: Final = "fixed_covariates"
    COVARIATES: Final = "covariates"
    INDIVIDUALS: Final = "individuals"


class Var:
    """Model variable and data container names of the logit models.

    The fixed-covariate data container is ``F`` in ``MNLogit`` and ``W`` in the
    nested, mixed and consideration-set models. Unifying them would rename a
    saved variable, so both names stay.
    """

    P: Final = "p"
    U: Final = "U"
    LIKELIHOOD: Final = "likelihood"
    X: Final = "X"
    Y: Final = "y"
    F: Final = "F"
    W: Final = "W"
