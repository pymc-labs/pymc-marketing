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

Each name is a member of a :class:`~pymc_marketing.names.Name` enum, so ``mypy``
reports a typo as ``attr-defined``; see :class:`~pymc_marketing.names.Name` for
how members behave when saved and loaded. The values are public API: saved
models store them, so they must not change.
"""

from enum import unique

from pymc_marketing.names import Name


@unique
class Dim(Name):
    """Dimension names of the logit models."""

    ALTS = "alts"
    OBS = "obs"
    ALT_COVARIATES = "alt_covariates"
    FIXED_COVARIATES = "fixed_covariates"
    COVARIATES = "covariates"
    INDIVIDUALS = "individuals"


@unique
class Var(Name):
    """Model variable and data container names of the logit models.

    The fixed-covariate data container is ``F`` in ``MNLogit`` and ``W`` in the
    nested, mixed and consideration-set models. Unifying them would rename a
    saved variable, so both names stay.
    """

    P = "p"
    U = "U"
    LIKELIHOOD = "likelihood"
    X = "X"
    Y = "y"
    F = "F"
    W = "W"
