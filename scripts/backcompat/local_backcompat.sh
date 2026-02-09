#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(git rev-parse --show-toplevel)
ENV_NAME="pymc-marketing-dev"
WORKTREE_DIR="/tmp/backcompat-main"

# Default models that pass backwards compatibility tests
# Excluded models (pre-existing issues):
#   - mixed_logit: Serialization not fully supported (MixedLogit requires 4 positional args)
DEFAULT_MODELS=("basic_mmm" "beta_geo" "beta_geo_beta_binom" "gamma_gamma" "modified_beta_geo" "pareto_nbd" "shifted_beta_geo")

# Determine conda/mamba installation base
if command -v micromamba >/dev/null 2>&1; then
  MAMBA_BASE="${MAMBA_ROOT_PREFIX:-${HOME}/micromamba}"
elif command -v mamba >/dev/null 2>&1; then
  MAMBA_BASE=$(mamba info --base 2>/dev/null || echo "${HOME}/mambaforge")
elif command -v conda >/dev/null 2>&1; then
  MAMBA_BASE=$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")
else
  echo "conda, mamba, or micromamba is required for this script" >&2
  exit 1
fi

# Try common locations for python
PYTHON_PATHS=(
  "${MAMBA_BASE}/envs/${ENV_NAME}/bin/python"
  "${HOME}/mamba/envs/${ENV_NAME}/bin/python"
  "${HOME}/mambaforge/envs/${ENV_NAME}/bin/python"
  "${HOME}/miniconda3/envs/${ENV_NAME}/bin/python"
  "${HOME}/anaconda3/envs/${ENV_NAME}/bin/python"
)

PYTHON_PATH=""
for path in "${PYTHON_PATHS[@]}"; do
  if [[ -x "${path}" ]]; then
    PYTHON_PATH="${path}"
    break
  fi
done

if [[ -z "${PYTHON_PATH}" ]]; then
  echo "Could not find python executable for environment ${ENV_NAME}" >&2
  echo "Tried the following locations:" >&2
  printf '  %s\n' "${PYTHON_PATHS[@]}" >&2
  exit 1
fi

# Function to list available models
list_models() {
  cd "${ROOT_DIR}" && "${PYTHON_PATH}" -c "
from scripts.backcompat.models import MODEL_REGISTRY
for name in sorted(MODEL_REGISTRY.keys()):
    print(f'  - {name}')
" 2>/dev/null || echo "  (Could not auto-discover models)"
}

# Function to show usage
show_usage() {
  cat << EOF
Usage: $0 [model1 model2 ...]

Test backwards compatibility of PyMC Marketing models.

Options:
  No arguments       Test all passing models (${#DEFAULT_MODELS[@]} models)
  model1 model2 ...  Test specific models only

Available models (auto-discovered):
$(list_models)

Examples:
  $0                      # Test all passing models
  $0 gamma_gamma          # Test single model (fast iteration)
  $0 pareto_nbd beta_geo  # Test multiple specific models

EOF
}

# Handle --help or -h
if [[ $# -eq 1 && ("$1" == "--help" || "$1" == "-h") ]]; then
  show_usage
  exit 0
fi

# Determine which models to test
if [[ $# -gt 0 ]]; then
  MODELS=("$@")
  echo "Testing specific models: ${MODELS[*]}"
else
  MODELS=("${DEFAULT_MODELS[@]}")
  echo "Testing all passing models: ${MODELS[*]}"
fi

echo "Using python: ${PYTHON_PATH}"
echo ""

function run_capture() {
  local model=$1
  local outdir=$2
  local workdir=$3
  echo "Capturing ${model} to ${outdir}"
  cd "${workdir}" && "${PYTHON_PATH}" -m scripts.backcompat.capture "${model}" "${outdir}"
}

function run_compare() {
  local model=$1
  local manifest=$2
  echo "Comparing ${model} against ${manifest}"
  cd "${ROOT_DIR}" && "${PYTHON_PATH}" -m scripts.backcompat.compare "${manifest}"
}

git fetch origin main --depth=1
git worktree remove -f "${WORKTREE_DIR}" >/dev/null 2>&1 || true
rm -rf "${WORKTREE_DIR}"
git worktree add "${WORKTREE_DIR}" origin/main

for model in "${MODELS[@]}"; do
  main_dir="/tmp/backcompat/main/${model}"
  head_dir="/tmp/backcompat/head/${model}"

  rm -rf "${main_dir}" "${head_dir}"
  mkdir -p "${main_dir}" "${head_dir}"

  run_capture "${model}" "${main_dir}" "${WORKTREE_DIR}"
  run_capture "${model}" "${head_dir}" "${ROOT_DIR}"

  run_compare "${model}" "${main_dir}/manifest.json"
done

git worktree remove -f "${WORKTREE_DIR}"
rm -rf "/tmp/backcompat/main" "/tmp/backcompat/head"

echo ""
echo "✅ All ${#MODELS[@]} models passed backwards compatibility tests:"
for model in "${MODELS[@]}"; do
  echo "  - ${model}"
done
