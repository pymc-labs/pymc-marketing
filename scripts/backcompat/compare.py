from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from pymc_extras.prior import Prior

from pymc_marketing.model_builder import ModelIO

from .mock_pymc import mock_sampling
from .models import get_model_definition

# Add backcompat directory to sys.path to ensure relative imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Set environment variable for mock sampling
os.environ["PYMC_MARKETING_BACKCOMPAT_MOCK_SAMPLING"] = "1"

LOGGER = logging.getLogger(__name__)


def replace_incompatible_priors_in_config(model_config: dict) -> dict:
    """Replace HalfFlat and Flat priors with mock-compatible alternatives.

    This is necessary because Prior objects cache the distribution class,
    so even if we replace pm.HalfFlat with pm.HalfNormal at the module level,
    existing Prior objects still reference the original class.
    """
    modified_config = {}
    for param_name, prior_config in model_config.items():
        if isinstance(prior_config, Prior):
            if prior_config.distribution == "HalfFlat":
                # Replace HalfFlat with HalfNormal for mock sampling
                modified_config[param_name] = Prior("HalfNormal", sigma=10)
                LOGGER.debug(f"Replaced {param_name}: HalfFlat -> HalfNormal")
            elif prior_config.distribution == "Flat":
                # Replace Flat with Normal for mock sampling
                modified_config[param_name] = Prior("Normal", mu=0, sigma=10)
                LOGGER.debug(f"Replaced {param_name}: Flat -> Normal")
            else:
                modified_config[param_name] = prior_config
        else:
            modified_config[param_name] = prior_config
    return modified_config


class CompatibilityError(RuntimeError): ...


def _compare_posterior_structures(baseline: ModelIO, candidate: ModelIO) -> None:
    base_idata = baseline.idata
    cand_idata = candidate.idata

    if base_idata is None or cand_idata is None:
        raise CompatibilityError("Missing inference data for comparison.")

    base_post = base_idata.posterior
    cand_post = cand_idata.posterior

    if base_post.sizes != cand_post.sizes:
        raise CompatibilityError(
            f"Posterior size mismatch. baseline={base_post.sizes}, candidate={cand_post.sizes}"
        )

    base_vars = set(base_post.data_vars)
    cand_vars = set(cand_post.data_vars)
    if base_vars != cand_vars:
        raise CompatibilityError(
            f"Posterior vars mismatch. baseline={sorted(base_vars)}, candidate={sorted(cand_vars)}"
        )

    for var in base_vars:
        base_dims = base_post[var].dims
        cand_dims = cand_post[var].dims
        if base_dims != cand_dims:
            raise CompatibilityError(
                f"Variable '{var}' dims changed: baseline={base_dims}, candidate={cand_dims}"
            )


def compare_baseline(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    model_def = get_model_definition(manifest["model"])
    baseline_path = Path(manifest["model_path"])

    # Load baseline model with mock sampling context to handle HalfFlat/Flat priors
    with mock_sampling():
        baseline_model = model_def.builder_cls.load(str(baseline_path))
        # Replace incompatible priors in the loaded model's config
        baseline_model.model_config = replace_incompatible_priors_in_config(
            baseline_model.model_config
        )

    build_kwargs = model_def.build_args_fn()
    fit_kwargs = model_def.fit_args_fn() if model_def.fit_args_fn else {}
    fit_data_kwargs = (
        model_def.fit_data_fn(build_kwargs) if model_def.fit_data_fn else {}
    )
    candidate = model_def.builder_fn(**build_kwargs)
    # Replace incompatible priors in candidate's config as well
    candidate.model_config = replace_incompatible_priors_in_config(
        candidate.model_config
    )

    with mock_sampling():
        candidate.fit(
            random_seed=model_def.fit_seed,
            **model_def.sampler_kwargs,
            **fit_kwargs,
            **fit_data_kwargs,
        )

    if candidate.version != baseline_model.version:
        raise CompatibilityError("Model incompatible, increase the model version")

    _compare_posterior_structures(baseline_model, candidate)

    LOGGER.info("Model %s is compatible with baseline", model_def.name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare current branch against baseline"
    )
    parser.add_argument("manifest", type=Path, help="Path to baseline manifest.json")
    args = parser.parse_args()

    try:
        os.environ["PYMC_MARKETING_BACKCOMPAT_MOCK_SAMPLING"] = "1"
        compare_baseline(args.manifest)
        print("Comparison passed")
    except CompatibilityError as exc:
        LOGGER.exception("Compatibility check failed")
        raise SystemExit(exc)
    finally:
        del os.environ["PYMC_MARKETING_BACKCOMPAT_MOCK_SAMPLING"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
