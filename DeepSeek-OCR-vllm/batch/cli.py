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
# Import configurations
from config_loader import BATCH_CONFIG

# Define constants directly from configuration
STAGE_OCR = 'raw'
STAGE_MD_IMAGE = 'md_image'
STAGE_MD_TEXT = 'md_text'
STAGE_MD_MERGED = 'md_merged'
STAGE_ALL = 'all'


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
        default="raw,md_image,md_text",
        help=f"Processing stages to run (comma-separated): {STAGE_OCR}, {STAGE_MD_IMAGE}, {STAGE_MD_TEXT}, {STAGE_MD_MERGED}, or {STAGE_ALL} (default: raw,md_image,md_text)"
    )

    # First check for common typos in sys.argv before parsing
    if "--stage" in sys.argv:
        print("Error: Did you mean '--stages' instead of '--stage'?")
        sys.exit(1)

    # Parse known args to catch unrecognized arguments
    args, unknown_args = parser.parse_known_args()

    # Check for unrecognized arguments
    if unknown_args:
        print(f"Error: Unrecognized arguments: {' '.join(unknown_args)}")
        parser.print_help()
        sys.exit(1)

    # Parse stages
    if args.stages == STAGE_ALL:
        stages = [STAGE_ALL]
    else:
        stages = args.stages.split(',')
        # Validate stages
        valid_stages = [STAGE_OCR, STAGE_MD_IMAGE, STAGE_MD_TEXT, STAGE_MD_MERGED, STAGE_ALL]
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