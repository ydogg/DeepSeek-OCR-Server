"""
Batch processing configuration
"""

import os
from config import *
from common.config_utils import get_ocr_config, get_processing_config


# Batch-specific configuration
DEFAULT_INPUT_DIR = "/home/ai/deepseek-ocr/images"
DEFAULT_OUTPUT_DIR = "/home/ai/deepseek-ocr/output"

# Batch processing stages
STAGE_OCR = "raw"
STAGE_MD_IMAGE = "md_image"
STAGE_MD_TEXT = "md_text"
STAGE_MD_MERGED = "md_merged"
STAGE_ALL = "all"

# Default stages to process
DEFAULT_STAGES = [STAGE_OCR, STAGE_MD_IMAGE, STAGE_MD_TEXT]

# Batch size for processing
DEFAULT_BATCH_SIZE = 10

# Whether to enable image analysis in md_merged stage
ENABLE_IMAGE_ANALYSIS = True

# VL model configuration for image analysis (copied from server/config.py)
VL_MODEL_BASE_URL = "http://172.28.71.194:8000/v1"  # Default to local server
VL_MODEL_API_KEY = "test"  # API key for the VL model if needed
VL_MODEL_NAME = "gpt-5-chat"  # Model name for the VL model
VL_MODEL_ANALYSIS_PROMPT = """Analyze this image and provide a detailed description.
If it's a chart or graph, describe the data, trends, and key insights.
If it's a table, extract and present the data in a structured format.
If it's a diagram or flowchart, explain the process or relationships shown.
If it's a photograph, describe the content and context."""  # Prompt for VL model analysis

# LLM model configuration for output enhancement (copied from server/config.py)
ENHANCEMENT_LLM_BASE_URL = "http://llm.necsoft.jn.com.cn:8000/v1"
ENHANCEMENT_LLM_MODEL_NAME = "qwen3-coder"
ENHANCEMENT_LLM_API_KEY = "test"
VL_MODEL_ENHANCEMENT_PROMPT = "如果文字内容描述了流程图等UML风格的内容，将这些文字转为mermaid格式，其他无关文字直接保留即可。"

# Default OCR prompt (copied from server/config.py)
DEFAULT_OCR_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."


def get_batch_config():
    """Get batch-specific configuration"""
    return {
        'input_dir': DEFAULT_INPUT_DIR,
        'output_dir': DEFAULT_OUTPUT_DIR,
        'stages': DEFAULT_STAGES,
        'batch_size': DEFAULT_BATCH_SIZE,
        'enable_image_analysis': ENABLE_IMAGE_ANALYSIS,
        'vl_model_base_url': VL_MODEL_BASE_URL,
        'vl_model_api_key': VL_MODEL_API_KEY,
        'vl_model_name': VL_MODEL_NAME,
        'vl_model_analysis_prompt': VL_MODEL_ANALYSIS_PROMPT,
        'enhancement_llm_base_url': ENHANCEMENT_LLM_BASE_URL,
        'enhancement_llm_model_name': ENHANCEMENT_LLM_MODEL_NAME,
        'enhancement_llm_api_key': ENHANCEMENT_LLM_API_KEY,
        'vl_model_enhancement_prompt': VL_MODEL_ENHANCEMENT_PROMPT,
        'default_ocr_prompt': DEFAULT_OCR_PROMPT
    }