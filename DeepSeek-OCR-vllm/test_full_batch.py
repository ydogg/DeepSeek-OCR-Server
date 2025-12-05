#!/usr/bin/env python3
"""
Test script for full batch processing flow
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

print("Running batch processing...")
processor.process(
    input_dir="/tmp/test_batch_input",
    output_dir="/tmp/test_batch_output",
    stages=["ocr", "clean"]
)

print("Batch processing completed!")