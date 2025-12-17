#!/bin/bash

# Start the DeepSeek OCR server in ONLINE mode
echo "Starting DeepSeek OCR server in ONLINE mode..."

# Set environment variables
export VLLM_USE_V1=0
export VLLM_USE_MODELSCOPE=True
export CUDA_VISIBLE_DEVICES=0
export ONLINE_OCR_MODE=true

# Start the server with online mode entry point
python -m server.main_online