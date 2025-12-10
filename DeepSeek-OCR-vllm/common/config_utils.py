"""
Common configuration utilities
Shared between server and batch implementations
"""

from config import *


def get_ocr_config():
    """Get OCR-related configuration"""
    return {
        'model_path': OCR_MODEL_PATH,
        'prompt': OCR_PROMPT,
        'base_size': BASE_SIZE,
        'image_size': IMAGE_SIZE,
        'crop_mode': CROP_MODE,
        'min_crops': MIN_CROPS,
        'max_crops': MAX_CROPS
    }


def get_processing_config():
    """Get processing-related configuration"""
    return {
        'max_concurrency': MAX_CONCURRENCY,
        'num_workers': NUM_WORKERS,
        'print_num_vis_tokens': PRINT_NUM_VIS_TOKENS,
        'skip_repeat': SKIP_REPEAT
    }


def get_server_config():
    """Get server-specific configuration"""
    try:
        from server_config import (
            ADDRESS, PORT, MAX_WORKER_THREADS, ONLINE_OCR_MODE,
            ONLINE_OCR_BASE_URL, ONLINE_OCR_MODEL_NAME, ONLINE_OCR_API_KEY,
            VL_MODEL_BASE_URL, VL_MODEL_API_KEY, VL_MODEL_NAME,
            VL_MODEL_ANALYSIS_PROMPT, ENHANCEMENT_LLM_BASE_URL,
            ENHANCEMENT_LLM_MODEL_NAME, ENHANCEMENT_LLM_API_KEY,
            VL_MODEL_ENHANCEMENT_PROMPT, OCR_PROMPT
        )
        return {
            'address': ADDRESS,
            'port': PORT,
            'max_worker_threads': MAX_WORKER_THREADS,
            'online_ocr_mode': ONLINE_OCR_MODE,
            'online_ocr_base_url': ONLINE_OCR_BASE_URL,
            'online_ocr_model_name': ONLINE_OCR_MODEL_NAME,
            'online_ocr_api_key': ONLINE_OCR_API_KEY,
            'vl_model_base_url': VL_MODEL_BASE_URL,
            'vl_model_api_key': VL_MODEL_API_KEY,
            'vl_model_name': VL_MODEL_NAME,
            'vl_model_analysis_prompt': VL_MODEL_ANALYSIS_PROMPT,
            'enhancement_llm_base_url': ENHANCEMENT_LLM_BASE_URL,
            'enhancement_llm_model_name': ENHANCEMENT_LLM_MODEL_NAME,
            'enhancement_llm_api_key': ENHANCEMENT_LLM_API_KEY,
            'vl_model_enhancement_prompt': VL_MODEL_ENHANCEMENT_PROMPT,
            'default_ocr_prompt': OCR_PROMPT
        }
    except ImportError:
        # If server_config is not available, return None or default values
        return None