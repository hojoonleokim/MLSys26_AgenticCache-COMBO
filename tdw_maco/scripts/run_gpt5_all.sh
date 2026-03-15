#!/bin/bash

# Unified GPT-5 experiment script — runs all 3 model variants
# Usage: 
#   ./run_gpt5_all.sh cook 0 10    # cook task, episode 0, 10 runs
#   ./run_gpt5_all.sh game 0 5     # game task, episode 0-4, 5 runs

TASK=${1:-cook}

if [ "$TASK" == "cook" ]; then
    DEFAULT_START_ID=2
    DEFAULT_NUM_RUNS=18
elif [ "$TASK" == "game" ]; then
    DEFAULT_START_ID=1
    DEFAULT_NUM_RUNS=9
else
    echo "Error: TASK must be 'cook' or 'game'"
    exit 1
fi

START_ID=$DEFAULT_START_ID
NUM_RUNS=$DEFAULT_NUM_RUNS

if [ -n "$2" ]; then
    if [[ "$2" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        START_ID="${BASH_REMATCH[1]}"
        END_ID="${BASH_REMATCH[2]}"
        if [ "$END_ID" -lt "$START_ID" ]; then
            echo "Error: invalid range '$2'"
            exit 1
        fi
        NUM_RUNS=$((END_ID - START_ID + 1))
    else
        START_ID="$2"
        if [ -n "$3" ]; then
            NUM_RUNS="$3"
        fi
    fi
fi

port=12077

# Auto-detect git branch name for output path
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

# Models to run (dated versions)
MODELS=("gpt-5-2025-08-07" "gpt-5-mini-2025-08-07" "gpt-5-nano-2025-08-07")
SHORT_NAMES=("gpt5" "gpt5-mini" "gpt5-nano")

# Set agents based on task
if [ "$TASK" == "cook" ]; then
    AGENTS="combo_agent combo_agent"
elif [ "$TASK" == "game" ]; then
    AGENTS="combo_agent combo_agent combo_agent combo_agent"
else
    echo "Error: TASK must be 'cook' or 'game'"
    exit 1
fi

echo "========================================"
echo "Branch: $BRANCH"
echo "Task: $TASK"
echo "Start Episode: $START_ID"
echo "Num Episodes: $NUM_RUNS"
echo "Models: ${MODELS[*]}"
echo "Results → results/$BRANCH/$TASK/{model}/"
echo "Prompt/Token log → results/$BRANCH/$TASK/{model}/$TASK/{full_model}/"
echo "========================================"

for idx in "${!MODELS[@]}"; do
    MODEL="${MODELS[$idx]}"
    SHORT="${SHORT_NAMES[$idx]}"

    echo ""
    echo "----------------------------------------"
    echo "Running model: $MODEL ($SHORT)"
    echo "----------------------------------------"

    pkill -f -9 "port\ $port" 2>/dev/null
    sleep 2

    DISPLAY=:1 python3 challenge.py \
        --port $port \
        --experiment_name "$BRANCH" \
        --task $TASK \
        --run_id "$SHORT" \
        --data_prefix dataset/ \
        --data_path test.json \
        --agents_algo $AGENTS \
        --screen_size 336 \
        --start_id $START_ID \
        --num_runs $NUM_RUNS \
        --max_steps 60 \
        --only_propose \
        --lm_source openai \
        --proposer_lm_id "$MODEL"

    pkill -f -9 "port\ $port" 2>/dev/null
    echo "Done: $MODEL"
done

echo ""
echo "========================================"
echo "All models completed!"
echo "========================================"
