#!/usr/bin/env python3
"""
API处理模块，包含OCR核心处理、图像处理和VL分析功能
"""

import os
import re
import uuid
import base64
import datetime
from typing import Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from server.schemas.models import OCRRequest
from server.core.utils import clean_ref_tags
from server.config import VL_MODEL_BASE_URL, VL_MODEL_API_KEY, VL_MODEL_NAME, VL_MODEL_ANALYSIS_PROMPT, ENHANCEMENT_LLM_BASE_URL, ENHANCEMENT_LLM_MODEL_NAME, ENHANCEMENT_LLM_API_KEY, VL_MODEL_ENHANCEMENT_PROMPT, DEFAULT_OCR_PROMPT

# Import processor from instances module to use the same instance
from server.core.instances import processor

# Import shared functions from common module
from common.image_processing import (
    rematch_image, extract_coordinates_and_label,
    draw_bounding_boxes, process_bounding_boxes
)
from common.text_processing import re_match


async def dowith_ocr_request(image: Image.Image, prompt: str = None, request_id: str = None, level: str = "clean") -> dict:
    """
    Common OCR processing method that works with both online and offline processors.
    Handles all three levels of processing: raw, clean, and image_clean.

    Args:
        image: PIL Image to process
        prompt: OCR prompt to use (defaults to DEFAULT_OCR_PROMPT if None)
        request_id: Request ID (generates new one if None)
        level: Processing level - "raw", "clean", or "image_clean"

    Returns:
        dict: OCR result with status and result/error
    """
    # Use provided prompt or default from config
    if prompt is None:
        prompt = DEFAULT_OCR_PROMPT

    # Generate request ID if not provided
    if request_id is None:
        request_id = f"req-{uuid.uuid4().hex[:12]}"

    print(f"[OCR Common] Processing OCR request with ID: {request_id}, level: {level}")
    print(f"[OCR Common] Using prompt length: {len(prompt) if prompt else 0}")

    try:
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
        if level == "raw":
            final_result = raw_result

        elif level == "clean":
            final_result = clean_ref_tags(raw_result)  # Clean from raw result

        elif level == "image_clean":
            # Create a temporary directory for this request with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            request_output_path = f"/tmp/ocr_{timestamp}_{request_id}"

            # Ensure directory exists
            os.makedirs(request_output_path, exist_ok=True)
            os.makedirs(f"{request_output_path}/images", exist_ok=True)

            # Save original result for processing
            with open(f"{request_output_path}/result_ori.mmd", "w", encoding="utf-8") as f:
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
            with open(f"{request_output_path}/result_boxing.mmd", "w", encoding="utf-8") as f:
                f.write(processed_result)

            # Step 4: VL analysis (only for image_clean mode)
            print(f"[OCR Common] Starting VL analysis for request {request_id}")
            vl_analyzed_result = await analyze_extracted_images(processed_result, f"{timestamp}_{request_id}")
            print(f"[OCR Common] VL analysis completed for request {request_id}")
            # Use VL analyzed result as the final result
            final_result = vl_analyzed_result

            # Save VL analyzed result
            with open(f"{request_output_path}/result_vl.mmd", "w", encoding="utf-8") as f:
                f.write(vl_analyzed_result)
        else:
            # No matching level, just return raw result
            final_result = raw_result

        print(f"[OCR Common] Returning result with length: {len(final_result)}")
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
        processed_result = processed_result.replace(a_match_image, f'![](images/ocr_detected_{idx}.jpg)\n')

    # Process other matches
    for idx, a_match_other in enumerate(matches_other):
        processed_result = processed_result.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')

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
            base_url=ENHANCEMENT_LLM_BASE_URL,
            api_key=ENHANCEMENT_LLM_API_KEY or "sk-test"  # Use test key if none provided
        )

        # Prepare the request with code block formatting for better handling
        messages = [
            {
                "role": "user",
                "content": f"{VL_MODEL_ENHANCEMENT_PROMPT}\n\n```\n{text_content}\n```"
            }
        ]

        print(f"[VL Enhancement] Request messages prepared")
        print(f"[VL Enhancement] Model: {ENHANCEMENT_LLM_MODEL_NAME}")

        # Make the API call using OpenAI client
        print(f"[VL Enhancement] Making API call to {ENHANCEMENT_LLM_BASE_URL}")
        response = await client.chat.completions.create(
            model=ENHANCEMENT_LLM_MODEL_NAME,
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
            base_url=VL_MODEL_BASE_URL,
            api_key=VL_MODEL_API_KEY or "sk-test"  # Use test key if none provided
        )

        # Prepare the request
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": VL_MODEL_ANALYSIS_PROMPT
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
        print(f"[VL Analysis] Model: {VL_MODEL_NAME}")

        # Make the API call using OpenAI client
        print(f"[VL Analysis] Making API call to {VL_MODEL_BASE_URL}")
        response = await client.chat.completions.create(
            model=VL_MODEL_NAME,
            messages=messages,
            max_tokens=16384
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

