"""
Batch processor for DeepSeek OCR
Implements three-stage batch processing:
1. OCR processing using vLLM batch capabilities
2. md_text processing using shared text processing functions
3. md_merged processing using shared image processing functions
"""

import os
import glob
import asyncio
import threading
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from PIL import Image

# 设置环境变量（与run_dpsk_ocr_eval_batch.py保持一致）
os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# 针对特定CUDA版本的处理
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"

from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry

from deepseek_ocr import DeepseekOCRForCausalLM
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor

# Import shared functions
from common.text_processing import clean_ref_tags, convert_image_tags_to_md
from common.image_processing import rematch_image, draw_bounding_boxes, process_bounding_boxes
from common.image_analysis import analyze_extracted_images_sync
from common.config_utils import get_ocr_config, get_processing_config

# Import configurations
from config_loader import BATCH_CONFIG, COMMON_CONFIG

# Define constants directly from configuration
STAGE_OCR = 'raw'
STAGE_MD_IMAGE = 'md_image'
STAGE_MD_TEXT = 'md_text'
STAGE_MD_MERGED = 'md_merged'
STAGE_ALL = 'all'

# Default OCR prompt
DEFAULT_OCR_PROMPT = COMMON_CONFIG.ocr_prompt

# Import required modules for image analysis
import base64
import datetime
import uuid
import asyncio
from openai import AsyncOpenAI


class BatchProcessor:
    def __init__(self):
        self.batch_config = BATCH_CONFIG
        self.ocr_config = get_ocr_config()
        self.processing_config = get_processing_config()

        # Initialize vLLM engine for OCR processing
        self._initialize_vllm_engine()

    def _initialize_vllm_engine(self):
        """Initialize vLLM engine for batch OCR processing"""
        try:
            print(f"Initializing model with path: {self.ocr_config['model_path']}")
            ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

            engine_args = {
                "model": self.ocr_config['model_path'],
                "hf_overrides": {"architectures": ["DeepseekOCRForCausalLM"]},
                "block_size": 256,
                "enforce_eager": False,
                "trust_remote_code": True,
                "max_model_len": 8192,
                "swap_space": 0,
                "max_num_seqs": self.processing_config['max_concurrency'],
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
            }

            print("Engine args:", engine_args)

            logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=40, window_size=90, whitelist_token_ids={128821, 128822})]

            self.sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=8192,
                logits_processors=logits_processors,
                skip_special_tokens=False,
            )

            print("Initializing LLM engine...")
            self.llm = LLM(**engine_args)
            print("LLM engine initialized successfully")
        except Exception as e:
            print(f"Failed to initialize model: {e}")
            raise

    def process_ocr_batch(self, image_paths):
        """First stage: OCR batch processing using vLLM"""
        print(f"Processing OCR for {len(image_paths)} images...")

        # Load images
        images = []
        for image_path in image_paths:
            image = Image.open(image_path).convert('RGB')
            images.append(image)

        # Prepare batch inputs using the same logic as run_dpsk_ocr_eval_batch.py
        batch_inputs = []
        for image in tqdm(images, desc="Pre-processing images"):
            prompt = self.ocr_config['prompt']
            cache_item = {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": DeepseekOCRProcessor().tokenize_with_images(
                        images=[image],
                        bos=True,
                        eos=True,
                        cropping=self.ocr_config['crop_mode']
                    )
                },
            }
            batch_inputs.append(cache_item)

        # Process batch with vLLM
        outputs_list = self.llm.generate(batch_inputs, sampling_params=self.sampling_params)

        # Extract results
        raw_results = []
        for output in outputs_list:
            raw_results.append(output.outputs[0].text)

        return raw_results

    def process_clean_batch(self, raw_results):
        """Second stage: md_text batch processing (remove all tags)"""
        print(f"Processing md_text for {len(raw_results)} results...")

        clean_results = []
        for result in tqdm(raw_results, desc="Cleaning results"):
            clean_result = clean_ref_tags(result)
            clean_results.append(clean_result)

        return clean_results

    def process_md_image_batch(self, raw_results, image_paths):
        """New stage: md_image batch processing (keep image tags)"""
        print(f"Processing md_image for {len(raw_results)} results...")

        md_image_results = []
        for idx, (result, image_path) in enumerate(tqdm(zip(raw_results, image_paths),
                                                       total=len(raw_results),
                                                       desc="Processing md_image results")):
            # First clean non-image tags
            md_image_result = clean_ref_tags(result, keep_image_tags=True)
            # Then convert image tags to Markdown format
            md_image_result = convert_image_tags_to_md(md_image_result)
            md_image_results.append(md_image_result)

        return md_image_results

    def _copy_processed_images_for_md_image(self, image_paths):
        """Copy processed images from processed_images directory to md_image directories"""
        processed_images_dir = os.path.join(self.batch_config['output_dir'], "processed_images")
        md_image_dir = os.path.join(self.batch_config['output_dir'], STAGE_MD_IMAGE)

        if not os.path.exists(processed_images_dir):
            print("No processed_images directory found, skipping image copying")
            return

        if not os.path.exists(md_image_dir):
            print("No md_image directory found, skipping image copying")
            return

        print("Copying processed images to md_image directories...")

        for image_path in tqdm(image_paths, desc="Copying images"):
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)

            # Source directory (from processed_images)
            source_request_dir = os.path.join(processed_images_dir, name)
            source_images_dir = os.path.join(source_request_dir, "images")

            # Destination directory (in md_image stage)
            dest_request_dir = os.path.join(md_image_dir, f"{name}_{STAGE_MD_IMAGE}")
            dest_images_dir = os.path.join(dest_request_dir, "images")

            # Check if source directory exists
            if os.path.exists(source_images_dir):
                # Create destination directory if it doesn't exist
                os.makedirs(dest_images_dir, exist_ok=True)

                # Copy all images from source to destination
                for image_file in os.listdir(source_images_dir):
                    source_image_path = os.path.join(source_images_dir, image_file)
                    dest_image_path = os.path.join(dest_images_dir, image_file)

                    # Copy the image file
                    try:
                        from shutil import copy2
                        copy2(source_image_path, dest_image_path)
                    except Exception as e:
                        print(f"Warning: Failed to copy {source_image_path} to {dest_image_path}: {e}")
            else:
                print(f"Warning: Source images directory not found for {name}")

    def process_md_merged_batch(self, raw_results, image_paths):
        """Third stage: md_merged batch processing"""
        print(f"Processing md_merged for {len(raw_results)} results...")

        image_clean_results = []

        # Create output directory for processed images
        processed_images_dir = os.path.join(self.batch_config['output_dir'], "processed_images")
        os.makedirs(processed_images_dir, exist_ok=True)

        for idx, (raw_result, image_path) in enumerate(tqdm(zip(raw_results, image_paths),
                                                           total=len(raw_results),
                                                           desc="Processing md_merged")):
            # Create request-specific output directory
            filename = os.path.basename(image_path)
            name, _ = os.path.splitext(filename)
            request_output_path = os.path.join(processed_images_dir, name)
            os.makedirs(request_output_path, exist_ok=True)
            os.makedirs(os.path.join(request_output_path, "images"), exist_ok=True)

            # Load image
            image = Image.open(image_path).convert('RGB')

            # Process bounding boxes (similar to server implementation)
            processed_result = process_bounding_boxes(
                None, image, raw_result, request_output_path, f"batch-{idx}"
            )

            # Perform image analysis if enabled
            if self.batch_config.get('enable_image_analysis', False):
                # Generate timestamp and request ID for image analysis
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                request_id_with_timestamp = f"{timestamp}_batch-{idx}"
                
                # Analyze extracted images using the shared function
                analyzed_result = analyze_extracted_images_sync(processed_result, request_id_with_timestamp, request_output_path)
                image_clean_results.append(analyzed_result)
            else:
                # Just use the processed result with bounding boxes
                image_clean_results.append(processed_result)

        return image_clean_results

    def save_results(self, results, image_paths, stage):
        """Save results to output directory with simplified structure"""
        output_dir = os.path.join(self.batch_config['output_dir'], stage)
        os.makedirs(output_dir, exist_ok=True)

        for result, image_path in zip(results, image_paths):
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)

            # Only create subdirectories for md_image stage (which contains images)
            if stage == STAGE_MD_IMAGE:
                # Create request-specific directory for md_image (contains images)
                request_output_path = os.path.join(output_dir, f"{name}_{stage}")
                os.makedirs(request_output_path, exist_ok=True)
                os.makedirs(os.path.join(request_output_path, "images"), exist_ok=True)
                output_path = os.path.join(request_output_path, "result_md_image.md")
            else:
                # For other stages, save directly in the stage directory
                if stage == STAGE_OCR:
                    output_filename = "result_ori.md"
                elif stage == STAGE_MD_TEXT:
                    output_filename = "result_md_text.md"
                elif stage == STAGE_MD_MERGED:
                    output_filename = "result_vl.md"
                else:
                    output_filename = f"result_{stage}.md"

                output_path = os.path.join(output_dir, f"{name}_{stage}.md")

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)

        print(f"Saved {len(results)} {stage} results to {output_dir}")

    def process(self, input_dir=None, output_dir=None, stages=None):
        """Main processing function"""
        # Use provided parameters or defaults
        if input_dir is None:
            input_dir = self.batch_config['input_dir']
        if output_dir is None:
            output_dir = self.batch_config['output_dir']
        if stages is None:
            stages = self.batch_config['stages']

        # Update config
        self.batch_config['input_dir'] = input_dir
        self.batch_config['output_dir'] = output_dir

        # Create main output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get image paths (recursive)
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_paths = []
        for ext in image_extensions:
            # Use recursive glob to find images in subdirectories
            image_paths.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))

        if not image_paths:
            print(f"No images found in {input_dir}")
            return

        print(f"Found {len(image_paths)} images to process")

        # Process in stages
        raw_results = None
        md_image_results = None
        clean_results = None
        image_clean_results = None

        # Stage 1: OCR processing
        if STAGE_OCR in stages or STAGE_ALL in stages:
            # Check if results already exist
            existing_results = self.load_existing_results(image_paths, STAGE_OCR)
            if existing_results and all(result != "" for result in existing_results):
                print("Found existing raw results, skipping OCR processing...")
                raw_results = existing_results
            else:
                raw_results = self.process_ocr_batch(image_paths)
                self.save_results(raw_results, image_paths, STAGE_OCR)

        # Stage 2: md_image processing
        if STAGE_MD_IMAGE in stages or STAGE_ALL in stages:
            if raw_results is None:
                # Load existing raw results if OCR stage wasn't run
                raw_results = self.load_existing_results(image_paths, STAGE_OCR)

            if raw_results:
                md_image_results = self.process_md_image_batch(raw_results, image_paths)
                self.save_results(md_image_results, image_paths, STAGE_MD_IMAGE)

                # Copy processed images to md_image directories for proper display
                self._copy_processed_images_for_md_image(image_paths)

        # Stage 3: md_text processing
        if STAGE_MD_TEXT in stages or STAGE_ALL in stages:
            if raw_results is None:
                # Load existing raw results if OCR stage wasn't run
                raw_results = self.load_existing_results(image_paths, STAGE_OCR)

            if raw_results:
                clean_results = self.process_clean_batch(raw_results)
                self.save_results(clean_results, image_paths, STAGE_MD_TEXT)

        # Stage 4: md_merged processing
        if STAGE_MD_MERGED in stages or STAGE_ALL in stages:
            if raw_results is None:
                # Load existing raw results if OCR stage wasn't run
                raw_results = self.load_existing_results(image_paths, STAGE_OCR)

            if raw_results:
                image_clean_results = self.process_md_merged_batch(raw_results, image_paths)
                self.save_results(image_clean_results, image_paths, STAGE_MD_MERGED)

        # Copy processed images to md_image directories for proper display
        # This needs to be done after md_merged stage to ensure processed_images directory exists
        if STAGE_MD_IMAGE in stages or STAGE_ALL in stages:
            self._copy_processed_images_for_md_image(image_paths)

        print("Batch processing completed!")

    def load_existing_results(self, image_paths, stage):
        """Load existing results for a specific stage to skip reprocessing"""
        stage_dir = os.path.join(self.batch_config['output_dir'], stage)
        results = []

        if not os.path.exists(stage_dir):
            print(f"No existing {stage} results found (directory doesn't exist)")
            return [""] * len(image_paths)

        for image_path in image_paths:
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)

            # For md_image stage, check the subdirectory structure
            if stage == STAGE_MD_IMAGE:
                # Try new format first (subdirectory structure)
                result_dir = os.path.join(stage_dir, f"{name}_{stage}")
                result_path = os.path.join(result_dir, "result_md_image.md")
                # Fallback to old format
                if not os.path.exists(result_path):
                    old_format_path = os.path.join(stage_dir, f"{name}_{stage}.md")
                    if os.path.exists(old_format_path):
                        result_path = old_format_path
            else:
                # For other stages, check direct files in stage directory
                if stage == STAGE_OCR:
                    result_filename = "result_ori.md"
                elif stage == STAGE_MD_TEXT:
                    result_filename = "result_md_text.md"
                elif stage == STAGE_MD_MERGED:
                    result_filename = "result_vl.md"
                else:
                    result_filename = f"result_{stage}.md"

                # Try new format first (direct file with .md extension)
                result_path = os.path.join(stage_dir, f"{name}_{stage}.md")
                # Fallback to old format with .mmd extension
                if not os.path.exists(result_path):
                    old_format_path = os.path.join(stage_dir, f"{name}_{stage}.mmd")
                    if os.path.exists(old_format_path):
                        result_path = old_format_path

            if os.path.exists(result_path):
                with open(result_path, 'r', encoding='utf-8') as f:
                    results.append(f.read())
            else:
                print(f"Warning: No {stage} result found for {filename}")
                results.append("")

        return results