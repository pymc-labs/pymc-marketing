#!/usr/bin/env bash
# Convenience wrapper for testing a single model
# This is just a shortcut to local_backcompat.sh with a single model argument

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Forward all arguments to local_backcompat.sh
exec "${SCRIPT_DIR}/local_backcompat.sh" "$@"
