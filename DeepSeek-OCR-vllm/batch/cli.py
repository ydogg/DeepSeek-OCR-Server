"""
Command-line interface for batch processing
"""

import argparse
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from batch.batch_processor import BatchProcessor
from batch.config import STAGE_OCR, STAGE_MD_TEXT, STAGE_MD_MERGED, STAGE_ALL


def main():
    parser = argparse.ArgumentParser(description="DeepSeek OCR Batch Processing")
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
        default="raw,md_text",
        help=f"Processing stages to run (comma-separated): {STAGE_OCR}, {STAGE_MD_TEXT}, {STAGE_MD_MERGED}, or {STAGE_ALL} (default: raw,md_text)"
    )
    
    args = parser.parse_args()
    
    # Parse stages
    if args.stages == STAGE_ALL:
        stages = [STAGE_ALL]
    else:
        stages = args.stages.split(',')
        # Validate stages
        valid_stages = [STAGE_OCR, STAGE_MD_TEXT, STAGE_MD_MERGED, STAGE_ALL]
        for stage in stages:
            if stage not in valid_stages:
                print(f"Error: Invalid stage '{stage}'. Valid stages are: {', '.join(valid_stages)}")
                sys.exit(1)
    
    # Create and run batch processor
    processor = BatchProcessor()
    processor.process(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        stages=stages
    )


if __name__ == "__main__":
    main()