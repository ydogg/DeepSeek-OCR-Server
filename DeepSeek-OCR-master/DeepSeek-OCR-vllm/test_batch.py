#!/usr/bin/env python3
"""
Test script for batch processing
"""

import sys
import os
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from batch.batch_processor import BatchProcessor


def main():
    parser = argparse.ArgumentParser(description="Test DeepSeek OCR Batch Processing")
    parser.add_argument(
        "--input-dir", 
        required=True,
        help="Input directory containing images to process"
    )
    parser.add_argument(
        "--output-dir", 
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--stages", 
        default="ocr,clean",
        help="Processing stages to run (comma-separated): ocr, clean, image_clean, or all (default: ocr,clean)"
    )
    
    args = parser.parse_args()
    
    # Parse stages
    if args.stages == "all":
        stages = ["all"]
    else:
        stages = args.stages.split(',')
        # Validate stages
        valid_stages = ["ocr", "clean", "image_clean", "all"]
        for stage in stages:
            if stage not in valid_stages:
                print(f"Error: Invalid stage '{stage}'. Valid stages are: {', '.join(valid_stages)}")
                sys.exit(1)
    
    print(f"Starting batch processing...")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Stages: {stages}")
    
    # Create and run batch processor
    processor = BatchProcessor()
    processor.process(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        stages=stages
    )
    
    print("Batch processing completed!")


if __name__ == "__main__":
    main()