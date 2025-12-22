#!/usr/bin/env python3
"""
Main entry point for DeepSeek OCR Server - Online Mode
This file contains only the imports and code needed for online mode,
avoiding heavy dependencies like transformers and torch.
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

# Import only configuration (lightweight)
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

# Import functions from api module (these should not depend on heavy packages)
from server.api import (
    dowith_ocr_request,
    rematch_image,
    extract_coordinates_and_label,
    draw_bounding_boxes,
    analyze_extracted_images,
    enhance_vl_output,
    enhance_text_format,
    call_vl_model
)

# Import load_image_from_base64 function (should be lightweight)
from server.core.utils import load_image_from_base64

# Import only the online processor instance
from server.core.instances import get_online_processor

# Use the online processor for this mode
processor = get_online_processor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown events"""
    print("[OCR Main] Starting DeepSeek OCR Server in ONLINE mode")
    print(f"[OCR Main] Server configuration - Address: {SERVER_CONFIG.address}, Port: {SERVER_CONFIG.port}")
    print(f"[OCR Main] Online OCR Mode: {SERVER_CONFIG.online_ocr_mode}")
    print(f"[OCR Main] Online OCR Base URL: {SERVER_CONFIG.online_ocr_base_url}")
    print(f"[OCR Main] VL Model Base URL: {SERVER_CONFIG.vl_model_base_url}")
    
    # Start the online processor (no heavy initialization needed)
    processor.start_workers()
    print("[OCR Main] Online processor started")
    
    yield
    
    # Shutdown event
    print("[OCR Main] Shutting down DeepSeek OCR Server")
    processor.stop_workers()
    print("[OCR Main] Online processor stopped")

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
        "server.main_online:app",
        host=SERVER_CONFIG.address,
        port=SERVER_CONFIG.port,
        workers=1
    )