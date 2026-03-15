#!/bin/bash

# Simple Multi-GPU Training for MACO Inpainting
# Usage: ./run_simple.sh [additional_args...]

echo "🚀 Starting MACO Inpainting Multi-GPU Training..."

# Check if accelerate is installed
if ! command -v accelerate &> /dev/null; then
    echo "❌ Error: accelerate not found. Please install it with: pip install accelerate"
    exit 1
fi

# Show GPU status
echo "📊 GPU Status:"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader,nounits

# Run training with accelerate (automatically uses all available GPUs)
echo "🏃 Running: accelerate launch train_maco.py --mode train --inpainting $@"
accelerate launch train_maco.py --mode train --inpainting --save_milestone "$@"

echo "✅ Training completed!"
