#!/usr/bin/env python3
"""
Test script for batch processing via command line interface
"""

import sys
import os
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser(description="Test DeepSeek OCR Batch Processing via CLI")
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
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Construct the command to run the batch CLI
    cmd = [
        sys.executable, 
        "-m", 
        "batch.cli",
        "--input-dir", args.input_dir,
        "--output-dir", args.output_dir,
        "--stages", args.stages
    ]
    
    print(f"Running batch processing command:")
    print(" ".join(cmd))
    print("-" * 50)
    
    # Run the command
    try:
        result = subprocess.run(cmd, cwd=project_root, check=True, text=True, 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print("Batch processing completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error running batch processing:")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()