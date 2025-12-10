"""
Shared instances module for DeepSeek OCR Server
This module contains shared instances that can be imported by other modules
to avoid circular imports.
"""

from .processor import OCRProcessor
from .online_processor import OnlineOCRProcessor
from server_config import ONLINE_OCR_MODE

# Global processors
offline_processor = OCRProcessor()
online_processor = OnlineOCRProcessor()

# Select processor based on configuration
processor = online_processor if ONLINE_OCR_MODE else offline_processor