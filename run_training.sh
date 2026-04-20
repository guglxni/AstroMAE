#!/usr/bin/env bash
# Convenience wrapper for running AstroMAE training.
# Accepts any arguments that `astromae train` accepts.

set -euo pipefail

# If conda is available and an `astromae` env exists, activate it.
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE=$(conda info --base)
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx astromae; then
        conda activate astromae
    fi
fi

CONFIG="${CONFIG:-configs/default.yaml}"
exec astromae train --config "$CONFIG" "$@"
