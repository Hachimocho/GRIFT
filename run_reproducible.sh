#!/bin/bash

# Wrapper script to run test_hierarchical.py with reproducibility settings

# --- Configuration ---
DEFAULT_SEED=42
CUBLAS_CONFIG=":4096:8"
VISIBLE_DEVICES=0
SCRIPT_NAME="test_hierarchical.py"

# --- Parse Arguments for Seed ---
SEED=${DEFAULT_SEED} # Default seed

# Iterate through arguments to find --seed
# We use a loop that consumes arguments so we can check the one after --seed
args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)
            if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
                SEED="$2"
                echo "Found --seed argument, using ${SEED} for PYTHONHASHSEED."
                # Add both --seed and its value to the args to pass to python
                args+=("$1" "$2") 
                shift # past argument
                shift # past value
            else
                echo "Error: --seed option requires a numeric argument." >&2
                # Still pass --seed to python, let it handle the error/default
                args+=("$1") 
                shift # past argument
            fi
            ;;
        *)
            # Preserve other arguments
            args+=("$1")
            shift # past argument
            ;;
    esac
done

# Restore arguments for passing to python script
set -- "${args[@]}"

# --- Environment Setup ---
echo "Setting PYTHONHASHSEED=${SEED}"
export PYTHONHASHSEED=${SEED}

echo "Setting CUBLAS_WORKSPACE_CONFIG=${CUBLAS_CONFIG}"
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_CONFIG}

echo "Setting CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES}"
export CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES}

# --- Get Script Directory ---
# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYTHON_SCRIPT_PATH="${SCRIPT_DIR}/${SCRIPT_NAME}"

# --- Execution ---
echo "Executing: python ${PYTHON_SCRIPT_PATH} $@"
echo "----------------------------------------"

# Execute the python script with all arguments passed to this bash script
python "${PYTHON_SCRIPT_PATH}" "$@"

# Exit with the python script's exit code
exit $?
