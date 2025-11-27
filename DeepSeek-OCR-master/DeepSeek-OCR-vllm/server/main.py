import os
import sys
import time
import uuid
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from deepseek_ocr import DeepseekOCRForCausalLM
from vllm.model_executor.models.registry import ModelRegistry

from server.schemas.models import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponseChoice,
    ChatCompletionResponse,
    ImageRequest
)
from server.core.processor import OCRProcessor, load_image_from_base64
from server.config import DEFAULT_OCR_PROMPT

# Import services
from server.services.ocr_service import process_ocr_request, extract_image_from_openai_request, create_mock_ocr_request

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules from the main project
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor
from config import OCR_MODEL_PATH, INPUT_PATH, OUTPUT_PATH, OCR_PROMPT, CROP_MODE


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
    description="""DeepSeek OCR API provides optical character recognition services with multiple processing levels:
    
### Basic Usage
- Send base64 encoded images to `/v1/ocr` endpoint
- Get OCR results in markdown format with bounding box information

### Multi-level Results
- **Raw**: Basic OCR without any post-processing
- **Clean**: OCR with reference tag cleaning
- **Image Clean**: Full processing with image extraction, bounding box analysis, and VL model analysis

### OpenAI Compatible Endpoint
- Use `/v1/chat/completions` with the same format as OpenAI's API
- Automatically processes images with image_clean level

### Result Levels
- `level=raw`: Basic OCR result
- `level=clean`: OCR result with cleaned reference tags
- `level=image_clean`: Full processing with image analysis (default)

### Temporary Files
Temporary files are stored with timestamp format: `/tmp/ocr_YYYYMMDDHHMMSS_req-xxxxxxxxxxxx/`
""",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """OpenAI compatible chat completion endpoint - wrapper for /v1/ocr with image_clean level"""
    print("[OCR Main] Received OpenAI compatible chat completion request")

    # Extract image from messages (assuming it's in the first user message)
    image_data, text_prompt = extract_image_from_openai_request(request.messages)

    if image_data is None:
        print("[OCR Main] No image data found in request")
        raise HTTPException(status_code=400, detail="No image data found in request")

    # Use provided text prompt or default from config
    prompt = text_prompt if text_prompt is not None else DEFAULT_OCR_PROMPT
    print(f"[OCR Main] Using prompt length: {len(prompt) if prompt else 0}")

    # Create a mock OCRImageRequest with image_clean level
    mock_request = create_mock_ocr_request(level="image_clean", prompt=prompt)
    
    # Call the common OCR processing function with image_clean level
    try:
        # Load image
        print("[OCR Main] Loading image from base64 data")
        image = load_image_from_base64(image_data)
        
        # Use the common OCR processing function
        result = await process_ocr_request(image_data, image, prompt, mock_request)
        
        if "error" in result:
            print(f"[OCR Main] OCR processing failed: {result['error']}")
            raise HTTPException(status_code=500, detail=f"Error processing image: {result['error']}")
        
        final_result = result["result"]
        
        # Create response
        choice = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content=str(final_result)),
            finish_reason="stop"
        )

        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=request.model,
            choices=[choice]
        )

        print(f"[OCR Main] OpenAI compatible response created with result length: {len(final_result)}")
        return response

    except Exception as e:
        print(f"[OCR Main] Exception in OCR processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/v1/ocr")
async def ocr_image(request: ImageRequest):
    """
    Unified OCR endpoint that accepts base64 encoded image in JSON format.
    All features can be enabled through request parameters:
    - Basic OCR: Always enabled
    - Result level: level="raw", "clean" (default), or "image_clean"

    Temporary files are stored with timestamp format: /tmp/ocr_YYYYMMDDHHMMSS_req-xxxxxxxxxxxx/
    """
    # Create a mock request object with default level
    class MockRequest:
        def __init__(self):
            self.level = "clean"  # Default level
            self.prompt = request.prompt

    mock_request = MockRequest()

    try:
        print(f"[OCR Main] Received OCR request with level: {mock_request.level}")

        # Decode base64 image
        print("[OCR Main] Decoding base64 image")
        image_data = request.image
        img = load_image_from_base64(image_data)

        # Use provided prompt or default from config
        prompt = request.prompt if request.prompt is not None else DEFAULT_OCR_PROMPT
        print(f"[OCR Main] Using prompt length: {len(prompt) if prompt else 0}")

        # Process OCR request
        result = await process_ocr_request(image_data, img, prompt, mock_request)

        if "error" in result:
            print(f"[OCR Main] OCR processing failed: {result['error']}")
            raise HTTPException(status_code=500, detail=f"Error processing image: {result['error']}")

        print(f"[OCR Main] Returning response with result length: {len(result['result'])}")
        return JSONResponse(content=result)
    except Exception as e:
        print(f"[OCR Main] Exception in OCR processing: {str(e)}")
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

def rematch_image(text, request_output_path=None):
    """
    Extract Markdown image references from processed text and return image paths

    Args:
        text: Processed text with Markdown image references
        request_output_path: Path to the request output directory

    Returns:
        tuple: (all_matches, image_paths)
            - all_matches: List of all Markdown image references found
            - image_paths: List of full paths to the actual image files
    """
    # Pattern to match ONLY OCR detected image references with special prefix
    # Simplified pattern that works correctly
    pattern = r'images/(ocr_detected_\d+\.jpg)'
    matches = re.findall(pattern, text)

    # If request_output_path is provided, construct full image paths
    image_paths = []
    if request_output_path:
        for match in matches:
            full_path = os.path.join(request_output_path, f"images/{match}")
            image_paths.append(full_path)
    else:
        # Return relative paths if no base path provided
        image_paths = [f"images/{match}" for match in matches]

    # Return full matches (with ![]()) and image paths
    full_matches = [f'![](images/{match})' for match in matches]
    return full_matches, image_paths

def extract_coordinates_and_label(ref_text, image_width, image_height):
    """Extract coordinates and label from ref text"""
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception:
        return None

    return (label_type, cor_list)

def draw_bounding_boxes(image, refs, output_path):
    """
    Draw bounding boxes on image based on refs

    Args:
        image: PIL Image object
        refs: List of reference objects with bounding box information
        output_path: Path to save the output image
    """
    draw = ImageDraw.Draw(image)
    
    # Try to load a better font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()

    for ref in refs:
        # Extract coordinates and label
        result = extract_coordinates_and_label(ref, image.width, image.height)
        if result is None:
            continue
            
        label_type, cor_list = result
        
        # Convert coordinates to pixel values
        points = []
        for i in range(0, len(cor_list), 2):
            x = int(cor_list[i] * image.width)
            y = int(cor_list[i+1] * image.height)
            points.append((x, y))
        
        # Draw bounding box
        if len(points) == 4:  # Rectangle
            draw.line(points + [points[0]], fill="red", width=2)
        else:  # Polygon
            draw.line(points + [points[0]], fill="red", width=2)
        
        # Draw label
        if points:
            draw.text(points[0], label_type, fill="red", font=font)
    
    # Save image
    image.save(output_path)