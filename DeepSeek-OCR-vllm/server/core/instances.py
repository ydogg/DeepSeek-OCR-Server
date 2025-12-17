"""
Shared instances module for DeepSeek OCR Server
This module contains shared instances that can be imported by other modules
to avoid circular imports.

Note: This module is designed to be imported by specific entry points:
- main_online.py imports only online_processor
- main_offline.py imports only offline_processor
- The processor selection is done in the entry point files
"""

try:
    # Try to import offline processor (may fail in online mode if heavy dependencies are not installed)
    from .processor import OCRProcessor
    offline_processor = OCRProcessor()
except ImportError as e:
    # In online mode, offline processor may not be available
    print(f"[Instances] Warning: Could not import offline processor: {e}")
    offline_processor = None

try:
    # Import online processor (should always be available)
    from .online_processor import OnlineOCRProcessor
    online_processor = OnlineOCRProcessor()
except ImportError as e:
    # This should not happen, but handle gracefully
    print(f"[Instances] Error: Could not import online processor: {e}")
    online_processor = None

# Note: The processor selection is now done in the entry point files
# This module just provides the available processors