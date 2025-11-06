import asyncio
import io
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import List

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image

from deepseek_ocr import DeepseekOCRForCausalLM
from vllm.model_executor.models.registry import ModelRegistry

from server.schemas.models import (
    ChatMessage, 
    ChatCompletionRequest, 
    ChatCompletionResponseChoice, 
    ChatCompletionResponse, 
    OCRRequest
)
from server.core.processor import OCRProcessor, load_image_from_base64
from server.core.utils import clean_ref_tags
from server.config import ADDRESS, PORT, CLEAN_REF_TAGS


# Register the model
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

# Global processor
processor = OCRProcessor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown events"""
    # Startup event
    processor.start_workers()
    yield
    # Shutdown event
    processor.stop_workers()

app = FastAPI(
    title="DeepSeek OCR API", 
    description="OpenAI compatible API for DeepSeek OCR", 
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """OpenAI compatible chat completion endpoint"""
    # Extract image from messages (assuming it's in the first user message)
    image_data = None
    text_prompt = None
    
    for message in request.messages:
        if message.role == "user":
            content = message.content
            # Check if content contains base64 image data
            if "data:image/" in content:
                # Extract base64 image data
                start = content.find("base64,") + 7
                end = content.find('"', start)
                if end == -1:
                    end = len(content)
                image_data = content[start:end]
            else:
                text_prompt = content
    
    if image_data is None:
        raise HTTPException(status_code=400, detail="No image data found in request")
    
    try:
        # Load image
        image = load_image_from_base64(image_data)
        
        # Create request
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        ocr_request = OCRRequest(request_id, image, text_prompt)
        
        # Add request to queue
        processor.submit_request(ocr_request)
        
        # Wait for result
        result = await processor.wait_for_result(request_id)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=f"Error processing image: {result['error']}")
        
        # Clean ref and det tags if enabled
        final_result = result["result"]
        if CLEAN_REF_TAGS:
            final_result = clean_ref_tags(final_result)
        
        # Create response
        choice = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content=final_result),
            finish_reason="stop"
        )
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=request.model,
            choices=[choice]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/v1/images/ocr")
async def ocr_image(
    image: UploadFile = File(...),
    prompt: str = Form(None)
):
    """Simple OCR endpoint that accepts image file upload"""
    try:
        # Read image file
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Create request
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        ocr_request = OCRRequest(request_id, img, prompt)
        
        # Add request to queue
        processor.submit_request(ocr_request)
        
        # Wait for result
        result = await processor.wait_for_result(request_id)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=f"Error processing image: {result['error']}")
        
        # Clean ref and det tags if enabled
        final_result = result["result"]
        if CLEAN_REF_TAGS:
            final_result = clean_ref_tags(final_result)
            
        return JSONResponse(content={"result": final_result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    is_healthy = processor.health_check()
    workers_status = [worker.is_alive() for worker in processor.workers]
    return {
        "status": "healthy" if is_healthy else "unhealthy", 
        "workers": len(processor.workers),
        "workers_status": workers_status
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=ADDRESS, port=PORT)