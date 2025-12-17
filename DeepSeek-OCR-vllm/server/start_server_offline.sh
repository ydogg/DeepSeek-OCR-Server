#!/bin/bash

# Start the DeepSeek OCR server in OFFLINE mode
echo "Starting DeepSeek OCR server in OFFLINE mode..."

# Set environment variables
export VLLM_USE_V1=0
export VLLM_USE_MODELSCOPE=True
export CUDA_VISIBLE_DEVICES=0
export ONLINE_OCR_MODE=false

# Start the server with offline mode entry point
python -m server.main_offline