#!/usr/bin/env python3
"""
API处理模块，包含OCR核心处理、图像处理和VL分析功能
"""

import os
import re
import uuid
import base64
import io
import datetime
from typing import Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from server.schemas.models import OCRRequest
from server.core.utils import clean_ref_tags, load_image_from_base64
from common.text_processing import convert_image_tags_to_md
from config_loader import SERVER_CONFIG, COMMON_CONFIG

# Import processor from instances module to use the same instance
# Note: This import is deprecated, use processor parameter in functions instead
# This is kept for backward compatibility with existing code
try:
    from server.core.instances import online_processor as processor
except ImportError:
    try:
        from server.core.instances import offline_processor as processor
    except ImportError:
        processor = None

# Import shared functions from common module
from common.image_processing import (
    rematch_image, extract_coordinates_and_label,
    draw_bounding_boxes, process_bounding_boxes
)
from common.text_processing import re_match


async def dowith_ocr_request(image: Image.Image, prompt: str = None, request_id: str = None, level: str = "md_text", processor=None) -> dict:
    """
    Common OCR processing method that works with both online and offline processors.
    Handles all levels of processing: raw, md_image, md_text, and md_merged.

    Args:
        image: PIL Image to process
        prompt: OCR prompt to use (defaults to OCR_PROMPT if None)
        request_id: Request ID (generates new one if None)
        level: Processing level - "raw", "md_image", "md_text", or "md_merged"
        processor: Processor instance to use (defaults to global processor if None)

    Returns:
        dict: OCR result with status and result/error
    """
    # Use provided prompt or default from config
    if prompt is None:
        prompt = COMMON_CONFIG.ocr_prompt

    # Generate request ID if not provided
    if request_id is None:
        request_id = f"req-{uuid.uuid4().hex[:12]}"

    # Use provided processor or fall back to global processor
    if processor is None:
        from server.core.instances import processor as global_processor
        processor = global_processor

    print(f"[OCR Common] Processing OCR request with ID: {request_id}, level: {level}")
    print(f"[OCR Common] Request details - Image size: {image.size if image else 'Unknown'}, Prompt length: {len(prompt) if prompt else 0}")

    try:
        # For online mode, we want to avoid multiple OCR calls
        # So we'll always get the raw result first, then process locally
        raw_result = None

        # Only call OCR API if we don't already have the raw result
        if raw_result is None:
            # Step 1: Basic OCR processing
            # Create request
            ocr_request = OCRRequest(request_id, image, prompt)
            print(f"[OCR Common] Created OCR request with ID: {request_id}")

            # Submit request to processor (works for both online and offline)
            print("[OCR Common] Submitting request to processor")
            processor.submit_request(ocr_request)

            # Wait for result (works for both online and offline)
            print("[OCR Common] Waiting for OCR result")
            result = await processor.wait_for_result(request_id)

            if result["status"] == "error":
                print(f"[OCR Common] OCR processing failed: {result['error']}")
                return result

            # Store raw result
            raw_result = result["result"]
            print(f"[OCR Common] OCR processing completed, result length: {len(raw_result)}")

        # Initialize variables
        processed_result = ''
        vl_analyzed_result = ''
        final_result = ''
        request_output_path = None
        timestamp = None

        # Step 2: Process based on level
        print(f"[OCR Common] Processing level: {level}")
        if level == "raw":
            print(f"[OCR Common] Returning raw result, length: {len(raw_result)}")
            final_result = raw_result

        elif level == "md_image":
            print(f"[OCR Common] Cleaning result but keeping image tags, raw length: {len(raw_result)}")
            final_result = clean_ref_tags(raw_result, keep_image_tags=True)  # Clean from raw result but keep image tags
            final_result = convert_image_tags_to_md(final_result)  # Convert image tags to Markdown format
            print(f"[OCR Common] Cleaned result with image tags, length: {len(final_result)}")

        elif level == "md_text":
            print(f"[OCR Common] Cleaning result, raw length: {len(raw_result)}")
            final_result = clean_ref_tags(raw_result)  # Clean from raw result
            print(f"[OCR Common] Cleaned result, length: {len(final_result)}")

        elif level == "md_merged":
            print(f"[OCR Common] Processing md_merged level, raw result length: {len(raw_result)}")
            # Create a temporary directory for this request with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            request_output_path = f"/tmp/ocr_{timestamp}_{request_id}"

            # Ensure directory exists
            os.makedirs(request_output_path, exist_ok=True)
            os.makedirs(f"{request_output_path}/images", exist_ok=True)

            # Save original result for processing
            with open(f"{request_output_path}/result_ori.md", "w", encoding="utf-8") as f:
                f.write(raw_result)

            # Step 3: Bounding box analysis
            # Only do this if needed for drawing boxes or analyzing images
            print(f"[OCR Common] Performing bounding box analysis for request {request_id}")
            # Create a mock request object for process_bounding_boxes
            class MockRequest:
                def __init__(self):
                    pass
            mock_request = MockRequest()
            processed_result = process_bounding_boxes(
                mock_request, image, raw_result, request_output_path, request_id
            )
            print(f"[OCR Common] Bounding box analysis completed for request {request_id}")

            # Save processed result
            with open(f"{request_output_path}/result_boxing.md", "w", encoding="utf-8") as f:
                f.write(processed_result)

            # Step 4: VL analysis (only for md_merged mode)
            print(f"[OCR Common] Starting VL analysis for request {request_id}")
            vl_analyzed_result = await analyze_extracted_images(processed_result, f"{timestamp}_{request_id}")
            print(f"[OCR Common] VL analysis completed for request {request_id}")
            # Use VL analyzed result as the final result
            final_result = vl_analyzed_result

            # Save VL analyzed result
            with open(f"{request_output_path}/result_vl.md", "w", encoding="utf-8") as f:
                f.write(vl_analyzed_result)
        else:
            # No matching level, just return raw result
            print(f"[OCR Common] Unknown level '{level}', returning raw result, length: {len(raw_result)}")
            final_result = raw_result

        # Step 5: Format enhancement (applied to all levels except raw)
        if level != "raw":
            print(f"[OCR Common] Starting format enhancement for request {request_id}")
            enhanced_result = await enhance_text_format(final_result)
            print(f"[OCR Common] Format enhancement completed for request {request_id}")

            # Save enhanced result if we have an output path
            if request_output_path:
                with open(f"{request_output_path}/result_enhanced.md", "w", encoding="utf-8") as f:
                    f.write(enhanced_result)

            # Use enhanced result as final result
            final_result = enhanced_result
        else:
            print(f"[OCR Common] Skipping format enhancement for raw level")

        print(f"[OCR Common] Returning enhanced result with length: {len(final_result)}")
        return {
            "status": "success",
            "result": final_result,
            "request_id": request_id
        }

    except Exception as e:
        print(f"[OCR Common] Exception in OCR processing: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
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
        output_path: Path to save extracted images, format /tmp/ocr_YYYYMMDDHHMMSS_req-xxxxxxxxxxxx
    """
    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.load_default()
    except IOError:
        font = ImageFont.load_default()

    img_idx = 0

    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result

                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))
                color_a = color + (20, )

                for points in points_list:
                    x1, y1, x2, y2 = points

                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)
                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    if label_type == 'image':
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped.save(f"{output_path}/images/ocr_detected_{img_idx}.jpg")
                        except Exception as e:
                            pass
                        img_idx += 1

                    try:
                        if label_type == 'title':
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)

                        text_x = x1
                        text_y = max(0, y1 - 15)

                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height],
                                    fill=(255, 255, 255, 30))

                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except:
                        pass
        except:
            continue
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def process_bounding_boxes(request: 'OCRImageRequest', img: Image.Image, raw_result: str,
                           request_output_path: str, request_id: str) -> str:
    """
    Process bounding boxes drawing and related features

    Args:
        request: OCR request with parameters
        img: Input image
        raw_result: Raw OCR result
        request_output_path: Path to save processed files
        request_id: Request ID

    Returns:
        str: Processed result with image references replaced
    """
    # Extract matches
    matches_ref, matches_images, matches_other = re_match(raw_result)

    # Draw bounding boxes
    result_image = draw_bounding_boxes(img.copy(), matches_ref, request_output_path)
    result_image.save(f"{request_output_path}/result_with_boxes.jpg")

    # Process image matches with special prefix for OCR detected images
    processed_result = raw_result
    for idx, a_match_image in enumerate(matches_images):
        # Handle both string and tuple formats
        if isinstance(a_match_image, tuple):
            # If it's a tuple, use the first element (the full match string)
            match_string = a_match_image[0]
        else:
            # If it's already a string, use it directly
            match_string = a_match_image
        processed_result = processed_result.replace(match_string, f'![](images/ocr_detected_{idx}.jpg)\n')

    # Process other matches
    for idx, a_match_other in enumerate(matches_other):
        # Handle both string and tuple formats
        if isinstance(a_match_other, tuple):
            # If it's a tuple, use the first element (the full match string)
            match_string = a_match_other[0]
        else:
            # If it's already a string, use it directly
            match_string = a_match_other
        processed_result = processed_result.replace(match_string, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')

    return processed_result


async def analyze_extracted_images(ocr_result: str, request_id_with_timestamp: str):
    """
    Analyze extracted images using VL model and replace them in the OCR result

    Args:
        ocr_result: The OCR result text
        request_id_with_timestamp: Request ID with timestamp in format YYYYMMDDHHMMSS_req-xxxxxxxxxxxx
    """
    print(f"[VL Analysis] Starting image analysis for request {request_id_with_timestamp}")

    # Extract image matches from OCR result using the new rematch_image function
    # Construct the request output path to get full image paths
    request_output_path = f"/tmp/ocr_{request_id_with_timestamp}"
    matches_images, image_paths = rematch_image(ocr_result, request_output_path)

    # If no image references found, return original result
    if not matches_images:
        print("[VL Analysis] No images found in OCR result, returning original result")
        return ocr_result

    # Process each extracted image
    processed_result = ocr_result
    print(f"[VL Analysis] Starting analysis of {len(matches_images)} images")

    for idx, (a_match_image, image_path) in enumerate(zip(matches_images, image_paths)):
        try:
            # Check if the image file exists
            if os.path.exists(image_path):
                # Convert image to base64
                with open(image_path, "rb") as image_file:
                    image_data = image_file.read()
                    image_base64 = base64.b64encode(image_data).decode('utf-8')

                    # Validate base64 data
                    try:
                        # Try to decode and validate the base64 data
                        decoded_data = base64.b64decode(image_base64)
                        if len(decoded_data) != len(image_data):
                            continue
                    except Exception:
                        continue

                    # Call VL model API to analyze the image
                    print(f"[VL Analysis] Calling VL model for image {idx+1}/{len(matches_images)}")
                    analysis_result = await call_vl_model(image_base64)

                    if analysis_result and analysis_result != "Image analysis failed":
                        # Apply enhancement to the analysis result
                        print(f"[VL Analysis] Applying enhancement to analysis result")
                        enhanced_result = await enhance_vl_output(analysis_result)

                        # Check if enhancement was successful
                        if enhanced_result and enhanced_result != "Output enhancement failed":
                            # Replace the image reference with the analysis result, adding descriptive text
                            # If enhancement was successful, use the enhanced result
                            replacement_text = f"\n[Image Analysis Result: The original image at this location contained the following content]\n{enhanced_result}\n"
                        else:
                            # If enhancement failed, use the original analysis result
                            replacement_text = f"\n[Image Analysis Result: The original image at this location contained the following content]\n{analysis_result}\n"

                        #print(f"[VL Analysis] Image {idx+1}", replacement_text)
                        processed_result = processed_result.replace(a_match_image, replacement_text)
                        print(f"[VL Analysis] Image {idx+1} reference replaced with analysis result")
                    else:
                        print(f"[VL Analysis] Image {idx+1} analysis failed or returned no result")
            else:
                print(f"[VL Analysis] Image {idx+1} file not found at: {image_path}")
                # If image file not found, keep the original image reference
        except Exception as e:
            print(f"[VL Analysis] Error processing image {idx+1}: {str(e)}")
            # If processing fails, keep the original image reference
            pass

    print(f"[VL Analysis] Completed analysis, final result length: {len(processed_result)} characters")
    return processed_result


async def enhance_vl_output(text_content: str):
    """
    Call the LLM API to enhance the output quality using OpenAI client
    """
    try:
        print(f"[VL Enhancement] Starting LLM call for output enhancement")

        # Import OpenAI client
        from openai import AsyncOpenAI

        # Initialize OpenAI client with separate LLM configuration
        client = AsyncOpenAI(
            base_url=SERVER_CONFIG.enhancement_llm_base_url,
            api_key=SERVER_CONFIG.enhancement_llm_api_key or "sk-test"  # Use test key if none provided
        )

        # Prepare the request with code block formatting for better handling
        messages = [
            {
                "role": "user",
                "content": f"{SERVER_CONFIG.vl_model_enhancement_prompt}\n\n```\n{text_content}\n```"
            }
        ]

        print(f"[VL Enhancement] Request messages prepared")
        print(f"[VL Enhancement] Model: {SERVER_CONFIG.enhancement_llm_model_name}")

        # Make the API call using OpenAI client
        print(f"[VL Enhancement] Making API call to {SERVER_CONFIG.enhancement_llm_base_url}")
        response = await client.chat.completions.create(
            model=SERVER_CONFIG.enhancement_llm_model_name,
            messages=messages,
            max_tokens=4096  # Reduce max_tokens to a more reasonable value
        )

        print(f"[VL Enhancement] API call completed successfully")

        # Process the response
        if response and response.choices:
            enhanced_text = response.choices[0].message.content
            print(f"[VL Enhancement] Enhanced text length: {len(enhanced_text) if enhanced_text else 0}")
            return enhanced_text
        else:
            print(f"[VL Enhancement] No response or choices in response")
            return "Output enhancement failed"

    except Exception as e:
        print(f"[VL Enhancement] Exception in enhance_vl_output: {str(e)}")
        # Add more detailed error information
        import traceback
        print(f"[VL Enhancement] Full traceback: {traceback.format_exc()}")
        return "Output enhancement failed"


async def enhance_text_format(text_content: str):
    """
    Call the LLM API to enhance text format and quality using OpenAI client
    """
    try:
        print(f"[Format Enhancement] Starting LLM call for text format enhancement")

        # Check if format enhancement is enabled
        if not hasattr(SERVER_CONFIG, 'format_enhancement_enabled') or not SERVER_CONFIG.format_enhancement_enabled:
            print(f"[Format Enhancement] Format enhancement is disabled, returning original text")
            return text_content

        # Import OpenAI client
        from openai import AsyncOpenAI

        # Initialize OpenAI client with format enhancement configuration
        client = AsyncOpenAI(
            base_url=SERVER_CONFIG.format_enhancement_base_url,
            api_key=SERVER_CONFIG.format_enhancement_api_key or "sk-test"  # Use test key if none provided
        )

        # Prepare the request with format enhancement prompt
        format_prompt = getattr(SERVER_CONFIG, 'format_enhancement_prompt',
            "请对以下OCR识别的文本内容进行格式优化和增强，修正识别错误，优化文档结构。")

        full_prompt = f"{format_prompt}\n\n```\n{text_content}\n```"

        messages = [
            {
                "role": "user",
                "content": full_prompt
            }
        ]

        model_name = getattr(SERVER_CONFIG, 'format_enhancement_model_name', 'qwen3-coder')
        print(f"[Format Enhancement] Request messages prepared")
        print(f"[Format Enhancement] Model: {model_name}")
        print(f"[Format Enhancement] Original text length: {len(text_content)}")

        # Make the API call using OpenAI client
        base_url = getattr(SERVER_CONFIG, 'format_enhancement_base_url', 'http://localhost:8000/v1')
        print(f"[Format Enhancement] Making API call to {base_url}")
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=8192,  # Allow more tokens for format enhancement
            temperature=0.1  # Lower temperature for more consistent formatting
        )

        print(f"[Format Enhancement] API call completed successfully")

        # Process the response
        if response and response.choices:
            enhanced_text = response.choices[0].message.content
            if enhanced_text:
                # Clean up the response - remove any extra formatting
                enhanced_text = enhanced_text.strip()
                # Remove potential code block markers if present
                if enhanced_text.startswith('```'):
                    lines = enhanced_text.split('\n')
                    if len(lines) > 1:
                        enhanced_text = '\n'.join(lines[1:-1]) if enhanced_text.endswith('```') else '\n'.join(lines[1:])
                enhanced_text = enhanced_text.strip()

                print(f"[Format Enhancement] Enhanced text length: {len(enhanced_text)}")
                return enhanced_text
            else:
                print(f"[Format Enhancement] Empty response received")
                return text_content  # Return original if enhancement failed
        else:
            print(f"[Format Enhancement] No response or choices in response")
            return text_content  # Return original if enhancement failed

    except Exception as e:
        print(f"[Format Enhancement] Exception in enhance_text_format: {str(e)}")
        # Add more detailed error information
        import traceback
        print(f"[Format Enhancement] Full traceback: {traceback.format_exc()}")
        print(f"[Format Enhancement] Returning original text due to enhancement failure")
        return text_content  # Return original text if enhancement failed


async def call_vl_model(image_base64: str):
    """
    Call the VL model API to analyze an image using OpenAI client
    """
    try:
        print(f"[VL Analysis] Starting VL model call with OpenAI client")

        # Import OpenAI client
        from openai import AsyncOpenAI

        # Initialize OpenAI client
        client = AsyncOpenAI(
            base_url=SERVER_CONFIG.vl_model_base_url,
            api_key=SERVER_CONFIG.vl_model_api_key or "sk-test"  # Use test key if none provided
        )

        # Prepare the request
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": SERVER_CONFIG.vl_model_analysis_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]

        print(f"[VL Analysis] Request messages prepared")
        print(f"[VL Analysis] Model: {SERVER_CONFIG.vl_model_name}")

        # Make the API call using OpenAI client
        print(f"[VL Analysis] Making API call to {SERVER_CONFIG.vl_model_base_url}")
        response = await client.chat.completions.create(
            model=SERVER_CONFIG.vl_model_name,
            messages=messages,
            #max_tokens=16384
            max_tokens=8100
        )

        print(f"[VL Analysis] API call completed successfully")

        # Process the response
        if response and response.choices:
            analysis_text = response.choices[0].message.content
            print(f"[VL Analysis] Analysis text length: {len(analysis_text) if analysis_text else 0}")
            return analysis_text
        else:
            print(f"[VL Analysis] No response or choices in response")
            return "Image analysis failed"

    except Exception as e:
        print(f"[VL Analysis] Exception in call_vl_model: {str(e)}")
        return "Image analysis failed"


# API endpoint functions (moved from main.py)
import time
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from server.schemas.models import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponseChoice,
    ChatCompletionResponse,
    OCRImageRequest
)
from server.core.utils import clean_ref_tags
from config_loader import COMMON_CONFIG

# Note: These functions are designed to be registered as routes in the main entry point files
# They should not use @app decorators directly to avoid circular imports

async def chat_completions(request: ChatCompletionRequest, processor=None):
    """OpenAI compatible chat completion endpoint"""
    print("[OCR Main] Received OpenAI compatible chat completion request")
    print("[OCR Main] Request details - OpenAI compatible endpoint")

    # Use provided processor or fall back to global processor
    if processor is None:
        from server.core.instances import processor as global_processor
        processor = global_processor

    # Extract image from messages (assuming it's in the first user message)
    image_data = None
    text_prompt = None

    for message in request.messages:
        if message.role == "user":
            # New format: content is an array of objects
            for item in message.content:
                if item.type == "image_url":
                    url = item.image_url.url
                    if url.startswith("data:image/"):
                        # Extract base64 image data
                        start = url.find("base64,") + 7
                        image_data = url[start:]
                        print("[OCR Main] Found base64 image data in request")
                elif item.type == "text":
                    text_prompt = item.text

    if image_data is None:
        print("[OCR Main] No image data found in request")
        raise HTTPException(status_code=400, detail="No image data found in request")

    # Use provided text prompt or default from config
    prompt = text_prompt if text_prompt is not None else COMMON_CONFIG.ocr_prompt
    print(f"[OCR Main] Using prompt length: {len(prompt) if prompt else 0}")

    try:
        # Load image
        print("[OCR Main] Loading image from base64 data")
        image = load_image_from_base64(image_data)
        print(f"[OCR Main] Image loaded, size: {image.size if image else 'Unknown'}")

        # Process OCR using common method with raw level (no additional processing)
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        print(f"[OCR Main] Processing request ID: {request_id} with level: raw (OpenAI compatible)")
        result = await dowith_ocr_request(image, prompt, request_id, "raw", processor)

        if result["status"] == "error":
            print(f"[OCR Main] OCR processing failed: {result['error']}")
            raise HTTPException(status_code=500, detail=f"Error processing image: {result['error']}")

        # Clean ref and det tags (always for OpenAI compatible endpoint)
        final_result = result["result"]
        final_result = clean_ref_tags(final_result)
        print(f"[OCR Main] OCR processing completed, cleaned result length: {len(final_result)}")

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
        print(f"[OCR Main] Returning OpenAI compatible response")
        return response
    except Exception as e:
        print(f"[OCR Main] Exception in OpenAI compatible endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


async def ocr_endpoint(request: OCRImageRequest, processor=None):
    """
    Unified OCR endpoint that accepts base64 encoded image in JSON format.
    All features can be enabled through request parameters:
    - Basic OCR: Always enabled
    - Result level: level="raw", "md_image", "md_text" (default), or "md_merged"

    Temporary files are stored with timestamp format: /tmp/ocr_YYYYMMDDHHMMSS_req-xxxxxxxxxxxx/
"""
    try:
        print(f"[OCR Main] Received OCR request with level: {request.level}")
        print(f"[OCR Main] Request details - Level: {request.level}, Prompt provided: {request.prompt is not None}")

        # Use provided processor or fall back to global processor
        if processor is None:
            from server.core.instances import processor as global_processor
            processor = global_processor

        # Decode base64 image
        print("[OCR Main] Decoding base64 image")
        img = load_image_from_base64(request.image)
        print(f"[OCR Main] Image decoded, size: {img.size if img else 'Unknown'}")

        # Use provided prompt or default from config
        prompt = request.prompt if request.prompt is not None else COMMON_CONFIG.ocr_prompt
        print(f"[OCR Main] Using prompt length: {len(prompt) if prompt else 0}")

        # Process OCR using common method with specified level
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        print(f"[OCR Main] Processing request ID: {request_id} with level: {request.level}")
        result = await dowith_ocr_request(img, prompt, request_id, request.level, processor)

        if result["status"] == "error":
            print(f"[OCR Main] OCR processing failed: {result['error']}")
            raise HTTPException(status_code=500, detail=f"Error processing image: {result['error']}")

        response_data = {
            "result": result["result"],
            "request_id": request_id
        }
        print(f"[OCR Main] Returning response with result length: {len(result['result'])}")
        return JSONResponse(content=response_data)
    except Exception as e:
        print(f"[OCR Main] Exception in OCR processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


async def health_check(processor=None):
    """Health check endpoint"""
    # Use provided processor or fall back to global processor
    if processor is None:
        from server.core.instances import processor as global_processor
        processor = global_processor

    is_healthy = processor.health_check()

    # Check if processor has workers (offline mode) or not (online mode)
    if hasattr(processor, 'workers'):
        workers_status = [worker.is_alive() for worker in processor.workers]
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "workers": len(processor.workers),
            "workers_status": workers_status
        }
    else:
        # Online mode - no workers
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "workers": 0,
            "workers_status": []
        }

