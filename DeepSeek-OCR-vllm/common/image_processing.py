"""
Common image processing functions
Shared between server and batch implementations
"""

import os
import re
import uuid
import base64
import datetime
from typing import Optional, Tuple, List
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from common.text_processing import re_match, clean_ref_tags


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
        # Parse the ref text to extract label type and coordinates
        # Format: <|ref|>label_type<|/ref|><|det|>[[x1, y1, x2, y2], ...]<|/det|>
        ref_pattern = r'<\|ref\|>(.*?)<\|/ref\|>'
        det_pattern = r'<\|det\|>(.*?)<\|/det\|>'

        ref_match = re.search(ref_pattern, ref_text)
        det_match = re.search(det_pattern, ref_text)

        if not ref_match or not det_match:
            return None

        label_type = ref_match.group(1)
        cor_list = eval(det_match.group(1))
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
            # Extract the full match string from the tuple
            if isinstance(ref, tuple) and len(ref) > 0:
                ref_text = ref[0]
            else:
                ref_text = ref

            result = extract_coordinates_and_label(ref_text, image_width, image_height)
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
                            save_path = f"{output_path}/images/ocr_detected_{img_idx}.jpg"
                            cropped.save(save_path)
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