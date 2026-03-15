#!/bin/bash
set -e

###############################################################################
# COMBO - End-to-End Training Pipeline
#
# This script reproduces the inpainting VDM model (modl-100.pt) from scratch.
#
# Pipeline:
#   Step 0: Set up conda environment
#   Step 1: Generate training/test data via TDW simulator
#   Step 2: Preprocess text embeddings (T5-XXL)
#   Step 3: Train inpainting diffusion model (100K steps)
#
# Prerequisites:
#   - conda installed
#   - TDW simulator accessible (DISPLAY set for step 1)
#   - GPU(s) available
#
# Usage:
#   bash train_all.sh              # Run all steps (including env setup)
#   bash train_all.sh --step 1     # Skip env setup (env already exists)
#   bash train_all.sh --step 2     # Skip data generation
#   bash train_all.sh --step 3     # Skip data gen + preprocess (train only)
#   bash train_all.sh --debug      # Quick sanity check with tiny batch
#
# Output:
#   <RESULTS_DIR>/modl-100.pt      # Final inpainting model checkpoint (step 100K)
###############################################################################

# ─── Configuration ───────────────────────────────────────────────────────────
CONDA_ENV="combo"
COMBO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TDW_MACO_DIR="${COMBO_ROOT}/tdw_maco"
FLOW_DIR="${COMBO_ROOT}/AVDC/flowdiffusion"
RESULTS_DIR="${FLOW_DIR}/../results/tdw_maco_inpainting"

TRAIN_EPISODES=600
TEST_EPISODES=50
MAX_EPISODE=9999
TDW_PORT=12078
LM_ID="google/t5-v1_1-xxl"

START_STEP=0
DEBUG_FLAG=""

# ─── Parse arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --step)   START_STEP="$2"; shift 2 ;;
        --debug)  DEBUG_FLAG="--debug"; shift ;;
        *)        echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ─── Helpers ─────────────────────────────────────────────────────────────────
log()  { echo ""; echo "========== $1 =========="; echo "$(date)"; }
run()  { echo ">>> $@"; conda run -n ${CONDA_ENV} --no-capture-output "$@"; }

generate_random_episodes() {
    shuf -i 0-${MAX_EPISODE} -n $1 | tr '\n' ' '
}

###############################################################################
# Step 0: Set up conda environment
###############################################################################
if [ "$START_STEP" -le 0 ]; then
    log "Step 0/3: Setting up conda environment '${CONDA_ENV}'"

    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        echo "ERROR: conda not found. Please install Anaconda/Miniconda first."
        echo "  https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi

    # Create env if it doesn't exist
    if conda env list | grep -q "^${CONDA_ENV} "; then
        echo "Environment '${CONDA_ENV}' already exists. Updating..."
        conda env update -n ${CONDA_ENV} -f "${COMBO_ROOT}/environment.yml" --prune
    else
        echo "Creating environment '${CONDA_ENV}'..."
        conda env create -f "${COMBO_ROOT}/environment.yml"
    fi

    echo "Environment '${CONDA_ENV}' ready."
    echo ""
    echo "Verifying key packages:"
    conda run -n ${CONDA_ENV} python -c "
import torch
print(f'  Python:       {__import__(\"sys\").version.split()[0]}')
print(f'  PyTorch:      {torch.__version__}')
print(f'  CUDA:         {torch.cuda.is_available()} (devices: {torch.cuda.device_count()})')
print(f'  Transformers: {__import__(\"transformers\").__version__}')
print(f'  Accelerate:   {__import__(\"accelerate\").__version__}')
print(f'  TDW:          {__import__(\"tdw\").__version__}')
"
fi

###############################################################################
# Step 1: Generate training & test data via TDW simulator
###############################################################################
if [ "$START_STEP" -le 1 ]; then
    log "Step 1/3: Generating training & test data"

    export PYTHONPATH="${TDW_MACO_DIR}:${PYTHONPATH}"
    pkill -f -9 "port\ ${TDW_PORT}" 2>/dev/null || true
    sleep 2

    # --- Training data (cook, 600 episodes) ---
    TRAIN_EPS=$(generate_random_episodes ${TRAIN_EPISODES})
    echo "Training episodes (${TRAIN_EPISODES}): ${TRAIN_EPS}"
    mkdir -p "${TDW_MACO_DIR}/train_data/cook_basic"

    DISPLAY=:1 run python "${TDW_MACO_DIR}/challenge.py" \
        --task cook \
        --output_dir "${TDW_MACO_DIR}/train_data/cook_basic" \
        --data_path train.json \
        --data_prefix dataset/ \
        --agents_algo cook_plan_agent cook_plan_agent \
        --eval_episodes ${TRAIN_EPS} \
        --screen_size 336 \
        --port ${TDW_PORT}

    pkill -f -9 "port\ ${TDW_PORT}" 2>/dev/null || true
    sleep 2

    # --- Test data (cook, 50 episodes) ---
    TEST_EPS=$(generate_random_episodes ${TEST_EPISODES})
    echo "Test episodes (${TEST_EPISODES}): ${TEST_EPS}"
    mkdir -p "${TDW_MACO_DIR}/test_data/cook_basic"

    DISPLAY=:1 run python "${TDW_MACO_DIR}/challenge.py" \
        --task cook \
        --output_dir "${TDW_MACO_DIR}/test_data/cook_basic" \
        --data_path test.json \
        --data_prefix dataset/ \
        --agents_algo cook_plan_agent cook_plan_agent \
        --eval_episodes ${TEST_EPS} \
        --screen_size 336 \
        --port ${TDW_PORT}

    pkill -f -9 "port\ ${TDW_PORT}" 2>/dev/null || true

    TRAIN_COUNT=$(ls "${TDW_MACO_DIR}/train_data/cook_basic" 2>/dev/null | wc -l)
    TEST_COUNT=$(ls "${TDW_MACO_DIR}/test_data/cook_basic" 2>/dev/null | wc -l)
    echo "Data generated: train=${TRAIN_COUNT}, test=${TEST_COUNT}"
fi

###############################################################################
# Step 2: Preprocess text embeddings with T5-XXL
###############################################################################
if [ "$START_STEP" -le 2 ]; then
    log "Step 2/3: Preprocessing text embeddings (${LM_ID})"

    run python "${FLOW_DIR}/train_maco.py" \
        --mode preprocess \
        --inpainting \
        --lm_id "${LM_ID}" \
        ${DEBUG_FLAG}

    echo "Text embeddings cached."
fi

###############################################################################
# Step 3: Train inpainting diffusion model (100K steps)
###############################################################################
if [ "$START_STEP" -le 3 ]; then
    log "Step 3/3: Training inpainting model"

    echo "GPU status:"
    nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader,nounits 2>/dev/null || true

    mkdir -p "${RESULTS_DIR}"

    # Using accelerate for multi-GPU support
    conda run -n ${CONDA_ENV} --no-capture-output \
        accelerate launch "${FLOW_DIR}/train_maco.py" \
            --mode train \
            --inpainting \
            --save_milestone \
            --lm_id "${LM_ID}" \
            --result_dir "${RESULTS_DIR}" \
            ${DEBUG_FLAG}

    echo "Training complete."
fi

###############################################################################
# Done
###############################################################################
log "Pipeline finished"

OUTPUT_FILE="${RESULTS_DIR}/modl-100.pt"
if [ -f "${OUTPUT_FILE}" ]; then
    echo "Output model: ${OUTPUT_FILE} ($(du -h "${OUTPUT_FILE}" | cut -f1))"
else
    echo "WARNING: ${OUTPUT_FILE} not found."
    echo "Check ${RESULTS_DIR}/ for available checkpoints:"
    ls -lh "${RESULTS_DIR}"/modl-*.pt 2>/dev/null || echo "  (none)"
    ls -lh "${RESULTS_DIR}"/model_recent.pt 2>/dev/null || true
fi

echo ""
echo "Done: $(date)"
