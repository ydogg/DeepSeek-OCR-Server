"""
OCR Service Module
Handles the core OCR processing logic, separating it from the API endpoint definitions.
"""
import base64
import datetime
import io
import os
import sys
import uuid
from typing import Dict, Any

# Add parent directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
sys.path.extend([parent_dir, project_root])

from PIL import Image
from fastapi import HTTPException
from io import BytesIO

from server.core.processor import OCRRequest, load_image_from_base64
from server.core.utils import clean_ref_tags
from server.core.ocr_utils import process_bounding_boxes, analyze_extracted_images
from server.config import DEFAULT_OCR_PROMPT


# Import the global processor instance from main module
# We need to import it locally to avoid circular imports
def get_processor():
    """Get the global processor instance"""
    import server.main
    return server.main.processor


async def process_ocr_request(image_data: str, img: Image.Image, prompt: str, request) -> Dict[str, Any]:
    """
    Common OCR processing function for both endpoints

    Args:
        image_data: Base64 encoded image data
        img: PIL Image object
        prompt: OCR prompt
        request: Request object with level attribute

    Returns:
        Dict containing result and request_id or error
    """
    try:
        # Get the global processor instance
        processor = get_processor()

        # Create request
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        ocr_request = OCRRequest(request_id, img, prompt)
        print(f"[OCR Service] Created OCR request with ID: {request_id}")

        # Add request to queue
        print("[OCR Service] Submitting request to processor queue")
        processor.submit_request(ocr_request)

        # Wait for result
        print("[OCR Service] Waiting for OCR result")
        result = await processor.wait_for_result(request_id)

        if result["status"] == "error":
            print(f"[OCR Service] OCR processing failed: {result['error']}")
            return {"error": result['error']}

        # Store raw result
        raw_result = result["result"]
        print(f"[OCR Service] OCR processing completed, result length: {len(raw_result)}")

        # Initialize variables early
        processed_result = ''
        vl_analyzed_result = ''
        final_result = ''
        request_output_path = None
        timestamp = None

        # If any enhanced features are requested, process them in the correct order:
        # 1. OCR解析 (already done)
        # 2. 图像提取（用于image_clean模式）
        # 3. VL分析（仅用于image_clean模式）
        # 4. 根据level确定返回结果
        # Note: We always create temporary files for processing
        if request.level == "raw":
            final_result = raw_result

        elif request.level == "clean":
            final_result = clean_ref_tags(raw_result)  # Clean from raw result

        elif request.level == "image_clean":
            # Create a temporary directory for this request with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            request_output_path = f"/tmp/ocr_{timestamp}_{request_id}"

            # Ensure directory exists
            os.makedirs(request_output_path, exist_ok=True)
            os.makedirs(f"{request_output_path}/images", exist_ok=True)

            # Save original result for processing
            with open(f"{request_output_path}/result_ori.mmd", "w", encoding="utf-8") as f:
                f.write(raw_result)

            # Step 2: 边界框分析 (Bounding box analysis)
            # Only do this if needed for drawing boxes or analyzing images
            print(f"[OCR Service] Performing bounding box analysis for request {request_id}")
            processed_result = process_bounding_boxes(
                request, img, raw_result, request_output_path, request_id
            )
            print(f"[OCR Service] Bounding box analysis completed for request {request_id}")

            # Save processed result
            with open(f"{request_output_path}/result_boxing.mmd", "w", encoding="utf-8") as f:
                f.write(processed_result)

            # Step 3: VL分析（仅用于image_clean模式）
            print(f"[OCR Service] Starting VL analysis for request {request_id}")
            vl_analyzed_result = await analyze_extracted_images(processed_result, f"{timestamp}_{request_id}")
            print(f"[OCR Service] VL analysis completed for request {request_id}")
            # Use VL analyzed result as the final result
            final_result = vl_analyzed_result

            # Save VL analyzed result
            with open(f"{request_output_path}/result_vl.mmd", "w", encoding="utf-8") as f:
                f.write(vl_analyzed_result)
        else:
            # No matching level, just return raw result
            final_result = raw_result

        return {
            "result": final_result,
            "request_id": request_id
        }
    except Exception as e:
        print(f"[OCR Service] Exception in OCR processing: {str(e)}")
        return {"error": str(e)}


def extract_image_from_openai_request(messages) -> tuple:
    """
    Extract image data and text prompt from OpenAI compatible request messages
    
    Args:
        messages: List of message objects from OpenAI request
        
    Returns:
        Tuple of (image_data, text_prompt)
    """
    image_data = None
    text_prompt = None

    for message in messages:
        if message.role == "user":
            # New format: content is an array of objects
            for item in message.content:
                if item.type == "image_url":
                    url = item.image_url.url
                    if url.startswith("data:image/"):
                        # Extract base64 image data
                        start = url.find("base64,") + 7
                        image_data = url[start:]
                        print("[OCR Service] Found base64 image data in request")
                elif item.type == "text":
                    text_prompt = item.text

    return image_data, text_prompt


def create_mock_ocr_request(level: str = "image_clean", prompt: str = None):
    """
    Create a mock OCR request object for use with the common processing function
    
    Args:
        level: Processing level (raw, clean, image_clean)
        prompt: OCR prompt
        
    Returns:
        Mock request object
    """
    class MockOCRRequest:
        def __init__(self, level, prompt):
            self.level = level
            self.prompt = prompt
    
    return MockOCRRequest(level, prompt)