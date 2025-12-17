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