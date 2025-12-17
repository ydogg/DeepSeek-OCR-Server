import re
import os
import base64
import io
from PIL import Image

# Import from common module to maintain compatibility
from common.text_processing import clean_formula, re_match, clean_ref_tags


def load_image_from_base64(image_str: str) -> Image.Image:
    """Load PIL Image from base64 string"""
    image_data = base64.b64decode(image_str)
    image = Image.open(io.BytesIO(image_data))
    return image.convert('RGB')


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str