"""
Common image analysis functions
Shared between server and batch implementations
"""

import os
import base64
import asyncio
import io
from openai import AsyncOpenAI
from PIL import Image

# Import configuration
from config_loader import BATCH_CONFIG




def analyze_extracted_images_sync(ocr_result: str, request_id_with_timestamp: str, request_output_path: str):
    """
    Analyze extracted images using VL model and replace them in the OCR result
    This is a synchronous version for batch processing

    Args:
        ocr_result: The OCR result text
        request_id_with_timestamp: Request ID with timestamp in format YYYYMMDDHHMMSS_req-xxxxxxxxxxxx
        request_output_path: Path to the request output directory
    """
    from common.image_processing import rematch_image
    
    print(f"[VL Analysis] Starting image analysis for request {request_id_with_timestamp}")

    # Extract image matches from OCR result using the rematch_image function
    matches_images, image_paths = rematch_image(ocr_result, request_output_path)

    # If no image references found, return original result
    if not matches_images:
        print("[VL Analysis] No images found in OCR result, returning original result")
        return ocr_result

    # Process each extracted image
    processed_result = ocr_result
    print(f"[VL Analysis] Starting analysis of {len(matches_images)} images")

    # Process images sequentially (synchronous version)
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

                    # Call VL model API to analyze the image (synchronous version)
                    print(f"[VL Analysis] Calling VL model for image {idx+1}/{len(matches_images)}")
                    analysis_result = call_vl_model_sync(image_base64)

                    if analysis_result and analysis_result != "Image analysis failed":
                        # Apply enhancement to the analysis result (synchronous version)
                        print(f"[VL Analysis] Applying enhancement to analysis result")
                        enhanced_result = enhance_vl_output_sync(analysis_result)

                        # Check if enhancement was successful
                        if enhanced_result and enhanced_result != "Output enhancement failed":
                            # Replace the image reference with the analysis result, adding descriptive text
                            # If enhancement was successful, use the enhanced result
                            replacement_text = f"\n[Image Analysis Result: The original image at this location contained the following content]\n{enhanced_result}\n"
                        else:
                            # If enhancement failed, use the original analysis result
                            replacement_text = f"\n[Image Analysis Result: The original image at this location contained the following content]\n{analysis_result}\n"

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


async def analyze_extracted_images(ocr_result: str, request_id_with_timestamp: str):
    """
    Analyze extracted images using VL model and replace them in the OCR result
    This is an asynchronous version for server processing

    Args:
        ocr_result: The OCR result text
        request_id_with_timestamp: Request ID with timestamp in format YYYYMMDDHHMMSS_req-xxxxxxxxxxxx
    """
    from common.image_processing import rematch_image
    
    print(f"[VL Analysis] Starting image analysis for request {request_id_with_timestamp}")

    # Extract image matches from OCR result using the rematch_image function
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


def enhance_vl_output_sync(text_content: str):
    """
    Call the LLM API to enhance the output quality using synchronous OpenAI client
    """
    try:
        print(f"[VL Enhancement] Starting LLM call for output enhancement")

        # Make the API call using OpenAI client (synchronous version)
        import openai
        print(f"[VL Enhancement] Making API call to {BATCH_CONFIG.enhancement_llm_base_url}")

        # Use synchronous client instead of async
        sync_client = openai.OpenAI(
            base_url=BATCH_CONFIG.enhancement_llm_base_url,
            api_key=BATCH_CONFIG.enhancement_llm_api_key
        )

        # Prepare the request with code block formatting for better handling
        messages = [
            {
                "role": "user",
                "content": f"{BATCH_CONFIG.vl_model_enhancement_prompt}\n\n```\n{text_content}\n```"
            }
        ]

        print(f"[VL Enhancement] Request messages prepared")
        print(f"[VL Enhancement] Model: {BATCH_CONFIG.enhancement_llm_model_name}")

        response = sync_client.chat.completions.create(
            model=BATCH_CONFIG.enhancement_llm_model_name,
            messages=messages,
            max_tokens=4096
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


async def enhance_vl_output(text_content: str):
    """
    Call the LLM API to enhance the output quality using OpenAI client
    """
    try:
        print(f"[VL Enhancement] Starting LLM call for output enhancement")

        # Initialize OpenAI client with separate LLM configuration
        client = AsyncOpenAI(
            base_url=BATCH_CONFIG.enhancement_llm_base_url,
            api_key=BATCH_CONFIG.enhancement_llm_api_key
        )

        # Prepare the request with code block formatting for better handling
        messages = [
            {
                "role": "user",
                "content": f"{BATCH_CONFIG.vl_model_enhancement_prompt}\n\n```\n{text_content}\n```"
            }
        ]

        print(f"[VL Enhancement] Request messages prepared")
        print(f"[VL Enhancement] Model: {BATCH_CONFIG.enhancement_llm_model_name}")

        # Make the API call using OpenAI client
        print(f"[VL Enhancement] Making API call to {BATCH_CONFIG.enhancement_llm_base_url}")
        response = await client.chat.completions.create(
            model=BATCH_CONFIG.enhancement_llm_model_name,
            messages=messages,
            max_tokens=4096
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


def call_vl_model_sync(image_base64: str):
    """
    Call the VL model API to analyze an image using synchronous OpenAI client
    """
    try:
        print(f"[VL Analysis] Starting VL model call with OpenAI client")

        # Initialize OpenAI client
        import openai
        client = openai.OpenAI(
            base_url=BATCH_CONFIG.vl_model_base_url,
            api_key=BATCH_CONFIG.vl_model_api_key
        )

        # Prepare the request
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": BATCH_CONFIG.vl_model_analysis_prompt
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
        print(f"[VL Analysis] Model: {BATCH_CONFIG.vl_model_name}")

        # Make the API call using OpenAI client (synchronous version)
        print(f"[VL Analysis] Making API call to {BATCH_CONFIG.vl_model_base_url}")
        response = client.chat.completions.create(
            model=BATCH_CONFIG.vl_model_name,
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


async def call_vl_model(image_base64: str):
    """
    Call the VL model API to analyze an image using OpenAI client
    """
    try:
        print(f"[VL Analysis] Starting VL model call with OpenAI client")

        # Initialize OpenAI client
        client = AsyncOpenAI(
            base_url=BATCH_CONFIG.vl_model_base_url,
            api_key=BATCH_CONFIG.vl_model_api_key
        )

        # Prepare the request
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": BATCH_CONFIG.vl_model_analysis_prompt
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
        print(f"[VL Analysis] Model: {BATCH_CONFIG.vl_model_name}")

        # Make the API call using OpenAI client
        print(f"[VL Analysis] Making API call to {BATCH_CONFIG.vl_model_base_url}")
        response = await client.chat.completions.create(
            model=BATCH_CONFIG.vl_model_name,
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