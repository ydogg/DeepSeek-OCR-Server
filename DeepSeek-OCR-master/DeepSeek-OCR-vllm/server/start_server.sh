#!/bin/bash

# Start the DeepSeek OCR server
echo "Starting DeepSeek OCR server..."

# Set environment variables
export VLLM_USE_V1=0
export VLLM_USE_MODELSCOPE=True
export CUDA_VISIBLE_DEVICES=0

# Start the server
python -m server.main
