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
from common.text_processing import clean_ref_tags
from common.image_processing import rematch_image, draw_bounding_boxes, process_bounding_boxes
from common.image_analysis import analyze_extracted_images_sync
from common.config_utils import get_ocr_config, get_processing_config

# Import batch-specific configuration
from batch.config import (
    get_batch_config, STAGE_OCR, STAGE_CLEAN, STAGE_IMAGE_CLEAN, STAGE_ALL,
    VL_MODEL_BASE_URL, VL_MODEL_API_KEY, VL_MODEL_NAME, VL_MODEL_ANALYSIS_PROMPT,
    ENHANCEMENT_LLM_BASE_URL, ENHANCEMENT_LLM_MODEL_NAME, ENHANCEMENT_LLM_API_KEY,
    VL_MODEL_ENHANCEMENT_PROMPT, DEFAULT_OCR_PROMPT
)

# Import required modules for image analysis
import base64
import datetime
import uuid
import asyncio
from openai import AsyncOpenAI


class BatchProcessor:
    def __init__(self):
        self.batch_config = get_batch_config()
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
        """Second stage: md_text batch processing"""
        print(f"Processing md_text for {len(raw_results)} results...")

        clean_results = []
        for result in tqdm(raw_results, desc="Cleaning results"):
            clean_result = clean_ref_tags(result)
            clean_results.append(clean_result)

        return clean_results

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
        """Save results to output directory"""
        output_dir = os.path.join(self.batch_config['output_dir'], stage)
        os.makedirs(output_dir, exist_ok=True)

        for result, image_path in zip(results, image_paths):
            filename = os.path.basename(image_path)
            name, _ = os.path.splitext(filename)
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

        # Get image paths
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

        if not image_paths:
            print(f"No images found in {input_dir}")
            return

        print(f"Found {len(image_paths)} images to process")

        # Process in stages
        raw_results = None
        clean_results = None
        image_clean_results = None

        # Stage 1: OCR processing
        if STAGE_OCR in stages or STAGE_ALL in stages:
            raw_results = self.process_ocr_batch(image_paths)
            self.save_results(raw_results, image_paths, STAGE_OCR)

        # Stage 2: md_text processing
        if STAGE_CLEAN in stages or STAGE_ALL in stages:
            if raw_results is None:
                # Load existing raw results if OCR stage wasn't run
                raw_results = self.load_existing_results(image_paths, STAGE_OCR)

            if raw_results:
                clean_results = self.process_clean_batch(raw_results)
                self.save_results(clean_results, image_paths, STAGE_CLEAN)

        # Stage 3: md_merged processing
        if STAGE_IMAGE_CLEAN in stages or STAGE_ALL in stages:
            if raw_results is None:
                # Load existing raw results if OCR stage wasn't run
                raw_results = self.load_existing_results(image_paths, STAGE_OCR)

            if raw_results:
                image_clean_results = self.process_md_merged_batch(raw_results, image_paths)
                self.save_results(image_clean_results, image_paths, STAGE_IMAGE_CLEAN)

        print("Batch processing completed!")

    def load_existing_results(self, image_paths, stage):
        """Load existing results from a previous stage"""
        results = []
        stage_dir = os.path.join(self.batch_config['output_dir'], stage)

        if not os.path.exists(stage_dir):
            print(f"No existing {stage} results found")
            return None

        for image_path in image_paths:
            filename = os.path.basename(image_path)
            name, _ = os.path.splitext(filename)
            result_path = os.path.join(stage_dir, f"{name}_{stage}.md")

            if os.path.exists(result_path):
                with open(result_path, 'r', encoding='utf-8') as f:
                    results.append(f.read())
            else:
                print(f"Warning: No {stage} result found for {filename}")
                results.append("")

        return results