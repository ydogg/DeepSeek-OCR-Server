"""
OCR Processing Utilities
Contains utility functions for OCR processing, bounding box analysis, and image handling.
"""
import base64
import os
import re
from typing import Tuple, List
from PIL import Image, ImageDraw, ImageFont

from server.core.utils import re_match, clean_ref_tags


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
    return image


def process_bounding_boxes(request, img: Image.Image, raw_result: str,
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
    result_image = draw_bounding_boxes(img.copy(), matches_ref, f"{request_output_path}/result_with_boxes.jpg")

    # Process image matches with special prefix for OCR detected images
    processed_result = raw_result
    for idx, a_match_image in enumerate(matches_images):
        processed_result = processed_result.replace(a_match_image, f'![](images/ocr_detected_{idx}.jpg)\n')

    # Process other matches
    for idx, a_match_other in enumerate(matches_other):
        processed_result = processed_result.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')

    return processed_result


async def call_vl_model(image_base64: str):
    """
    Call VL model API to analyze an image
    
    Args:
        image_base64: Base64 encoded image data
        
    Returns:
        str: Analysis result or error message
    """
    # This function should be implemented based on your VL model API
    # For now, we'll return a placeholder
    return "Image analysis result placeholder"


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
                        # Replace the image reference with the analysis result, adding descriptive text
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