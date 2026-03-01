from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from scripts.backcompat.mock_pymc import mock_sampling
from scripts.backcompat.models import get_model_definition

# Add backcompat directory to sys.path to ensure relative imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Set environment variable for mock sampling
os.environ["PYMC_MARKETING_BACKCOMPAT_MOCK_SAMPLING"] = "1"

LOGGER = logging.getLogger(__name__)


class CaptureError(Exception):
    pass


def build_baseline(model_name: str, output_dir: Path) -> Path:
    model_def = get_model_definition(model_name)
    build_kwargs = model_def.build_args_fn()
    fit_kwargs = model_def.fit_args_fn() if model_def.fit_args_fn else {}
    fit_data_kwargs = (
        model_def.fit_data_fn(build_kwargs) if model_def.fit_data_fn else build_kwargs
    )

    with mock_sampling():
        mmm = model_def.builder_fn(**build_kwargs)
        mmm.fit(
            random_seed=model_def.fit_seed,
            **model_def.sampler_kwargs,
            **fit_kwargs,
            **fit_data_kwargs,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{model_def.name}.nc"
    mmm.save(str(model_path))

    metadata = {
        "model": model_def.name,
        "version": mmm.version,
        "model_path": str(model_path),
        "sampler_config": mmm.sampler_config,
        "note": "Sampling is mocked via pymc.testing.mock_sample for speed.",
    }
    metadata_path = output_dir / "manifest.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture baseline models")
    parser.add_argument("model", help="Model name", default="basic_mmm", nargs="?")
    parser.add_argument(
        "output",
        help="Directory to place artifacts",
        default="scripts/backcompat/baselines/basic_mmm",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    capture_path = build_baseline(args.model, output_dir)
    LOGGER.info("Saved baseline for %s to %s", args.model, capture_path)
    print(f"Baseline manifest: {capture_path}")


if __name__ == "__main__":
    main()
