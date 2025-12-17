"""
Common configuration utilities
Shared between server and batch implementations
"""

from config_loader import COMMON_CONFIG


def get_ocr_config():
    """Get OCR-related configuration"""
    return {
        'model_path': COMMON_CONFIG.ocr_model_path,
        'prompt': COMMON_CONFIG.ocr_prompt,
        'base_size': COMMON_CONFIG.base_size,
        'image_size': COMMON_CONFIG.image_size,
        'crop_mode': COMMON_CONFIG.crop_mode,
        'min_crops': COMMON_CONFIG.min_crops,
        'max_crops': COMMON_CONFIG.max_crops
    }


def get_processing_config():
    """Get processing-related configuration"""
    return {
        'max_concurrency': COMMON_CONFIG.max_concurrency,
        'num_workers': COMMON_CONFIG.num_workers,
        'print_num_vis_tokens': COMMON_CONFIG.print_num_vis_tokens,
        'skip_repeat': COMMON_CONFIG.skip_repeat
    }


def get_server_config():
    """Get server-specific configuration"""
    try:
        from config_loader import SERVER_CONFIG
        return {
            'address': SERVER_CONFIG.address,
            'port': SERVER_CONFIG.port,
            'max_worker_threads': SERVER_CONFIG.max_worker_threads,
            'online_ocr_mode': SERVER_CONFIG.online_ocr_mode,
            'online_ocr_base_url': SERVER_CONFIG.online_ocr_base_url,
            'online_ocr_model_name': SERVER_CONFIG.online_ocr_model_name,
            'online_ocr_api_key': SERVER_CONFIG.online_ocr_api_key,
            'vl_model_base_url': SERVER_CONFIG.vl_model_base_url,
            'vl_model_api_key': SERVER_CONFIG.vl_model_api_key,
            'vl_model_name': SERVER_CONFIG.vl_model_name,
            'vl_model_analysis_prompt': SERVER_CONFIG.vl_model_analysis_prompt,
            'enhancement_llm_base_url': SERVER_CONFIG.enhancement_llm_base_url,
            'enhancement_llm_model_name': SERVER_CONFIG.enhancement_llm_model_name,
            'enhancement_llm_api_key': SERVER_CONFIG.enhancement_llm_api_key,
            'vl_model_enhancement_prompt': SERVER_CONFIG.vl_model_enhancement_prompt,
            'default_ocr_prompt': SERVER_CONFIG.ocr_prompt
        }
    except ImportError:
        # If server_config is not available, return None or default values
        return None