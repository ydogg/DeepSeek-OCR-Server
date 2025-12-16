"""
Tokenizer configuration for DeepSeek OCR
"""

from config_loader import COMMON_CONFIG
from transformers import AutoTokenizer

# Create tokenizer
TOKENIZER = AutoTokenizer.from_pretrained(COMMON_CONFIG.ocr_model_path, trust_remote_code=True)