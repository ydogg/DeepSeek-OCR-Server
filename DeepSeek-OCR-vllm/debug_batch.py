#!/usr/bin/env python3
"""
Debug script for batch processing
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

print("Creating BatchProcessor...")
from batch.batch_processor import BatchProcessor

print("Initializing BatchProcessor...")
processor = BatchProcessor()

print("BatchProcessor initialized successfully!")
print("OCR config:", processor.ocr_config)
print("Processing config:", processor.processing_config)