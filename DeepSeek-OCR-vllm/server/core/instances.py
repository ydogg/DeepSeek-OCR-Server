"""
Shared instances module for DeepSeek OCR Server
This module contains shared instances that can be imported by other modules
to avoid circular imports.

Note: This module is designed to be imported by specific entry points:
- main_online.py imports only online_processor
- main_offline.py imports only offline_processor
- The processor selection is done in the entry point files
"""

import os

# Lazy import functions to avoid loading heavy dependencies at module import time
def get_offline_processor():
    """Lazy import of offline processor to avoid loading heavy dependencies"""
    global _offline_processor
    if _offline_processor is None:
        try:
            # Import offline processor only when needed
            from .processor import OCRProcessor
            _offline_processor = OCRProcessor()
        except ImportError as e:
            # In online mode, offline processor may not be available
            print(f"[Instances] Warning: Could not import offline processor: {e}")
            _offline_processor = None
    return _offline_processor

def get_online_processor():
    """Lazy import of online processor"""
    global _online_processor
    if _online_processor is None:
        try:
            # Import online processor only when needed
            from .online_processor import OnlineOCRProcessor
            _online_processor = OnlineOCRProcessor()
        except ImportError as e:
            # This should not happen, but handle gracefully
            print(f"[Instances] Error: Could not import online processor: {e}")
            _online_processor = None
    return _online_processor

# Initialize processor variables as None
_offline_processor = None
_online_processor = None

# Global processor selection based on ONLINE_OCR_MODE environment variable
# This maintains backward compatibility with the original main.py
def _get_selected_processor():
    """Get the appropriate processor based on ONLINE_OCR_MODE environment variable"""
    online_mode = os.getenv('ONLINE_OCR_MODE', 'false').lower() == 'true'
    if online_mode:
        # In online mode, don't try to import offline processor to avoid heavy dependencies
        return get_online_processor()
    else:
        return get_offline_processor()

# Global processor instance for backward compatibility
# Note: This is deprecated and should not be used in new code
# Use processor parameters in functions instead
processor = _get_selected_processor()

# Note: The processor selection is now done in the entry point files
# This module just provides the available processors