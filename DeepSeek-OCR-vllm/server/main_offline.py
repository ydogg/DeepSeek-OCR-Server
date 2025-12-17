#!/usr/bin/env python3
"""
Main entry point for DeepSeek OCR Server - Offline Mode
This file contains all imports and code needed for offline mode,
including heavy dependencies like transformers and torch.
"""

import os
import sys
import base64
from contextlib import asynccontextmanager
from typing import List
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import heavy dependencies needed for offline mode
from deepseek_ocr import DeepseekOCRForCausalLM
from vllm.model_executor.models.registry import ModelRegistry

# Import configuration
from config_loader import SERVER_CONFIG, COMMON_CONFIG

# Import data models
from server.schemas.models import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponseChoice,
    ChatCompletionResponse,
    OCRRequest,
    OCRImageRequest,
    ContentText
)

# Register the model (required for offline mode)
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

# Import functions from api module
from server.api import (
    dowith_ocr_request,
    rematch_image,
    extract_coordinates_and_label,
    draw_bounding_boxes,
    analyze_extracted_images,
    enhance_vl_output,
    call_vl_model
)

# Import load_image_from_base64 function (should be lightweight)
from server.core.utils import load_image_from_base64

# Import only the offline processor instance
from server.core.instances import get_offline_processor

# Use the offline processor for this mode
processor = get_offline_processor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown events"""
    print("[OCR Main] Starting DeepSeek OCR Server in OFFLINE mode")
    print(f"[OCR Main] Server configuration - Address: {SERVER_CONFIG.address}, Port: {SERVER_CONFIG.port}")
    print(f"[OCR Main] Online OCR Mode: {SERVER_CONFIG.online_ocr_mode}")
    
    # Start the offline processor (heavy initialization)
    processor.start_workers()
    print("[OCR Main] Offline processor started")
    
    yield
    
    # Shutdown event
    print("[OCR Main] Shutting down DeepSeek OCR Server")
    processor.stop_workers()
    print("[OCR Main] Offline processor stopped")

# Create FastAPI app with lifespan
app = FastAPI(
    title="DeepSeek OCR API",
    description="DeepSeek OCR API for document analysis and conversion",
    version="1.0.0",
    lifespan=lifespan
)

# Import routes after app creation to avoid circular imports
# Note: These functions are now defined in api.py
from server.api import chat_completions, ocr_endpoint, health_check

# Add routes
app.add_api_route("/v1/chat/completions", chat_completions, methods=["POST"])
app.add_api_route("/v1/ocr", ocr_endpoint, methods=["POST"])
app.add_api_route("/health", health_check, methods=["GET"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.main_offline:app",
        host=SERVER_CONFIG.address,
        port=SERVER_CONFIG.port,
        workers=1
    )