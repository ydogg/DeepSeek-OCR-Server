import asyncio
import base64
import io
import os
import sys
import threading
import time
import uuid
from typing import Dict, Any
from queue import Queue, Empty
from threading import Thread, Event
from PIL import Image

# Add parent directory and project root to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
sys.path.extend([parent_dir, project_root])

from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import Device
from vllm.engine.async_llm_engine import AsyncEngineDeadError

from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor
from config import MODEL_PATH, PROMPT, CROP_MODE, MAX_CONCURRENCY, CLEAN_REF_TAGS
from server.config import MAX_WORKER_THREADS
from server.schemas.models import OCRRequest
from server.core.utils import clean_ref_tags


class ModelWorker:
    def __init__(self, request_queue: Queue, result_dict: Dict[str, dict], shutdown_event: Event):
        self.request_queue = request_queue
        self.result_dict = result_dict
        self.shutdown_event = shutdown_event
        self.engine = None
        self.sampling_params = None

    def initialize_model(self):
        """Initialize a single model instance with error handling"""
        print(f"Initializing model with path: {MODEL_PATH}")
        
        engine_args = EngineArgs(
            model=MODEL_PATH,
            hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
            block_size=256,
            max_model_len=4096,
            enforce_eager=False,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.6,
        )
        try:
            # If we have an existing engine, try to clean it up
            if self.engine is not None:
                try:
                    # Attempt to shutdown the existing engine gracefully
                    if hasattr(self.engine, 'shutdown_background_loop'):
                        self.engine.shutdown_background_loop()
                except Exception as e:
                    print(f"Warning: Failed to shutdown existing engine: {e}")
                finally:
                    self.engine = None
            
            self.engine = LLMEngine.from_engine_args(engine_args)
        except Exception as e:
            print(f"Failed to initialize vLLM engine: {e}")
            self.engine = None
            raise
            
        logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=30, window_size=90, whitelist_token_ids={128821, 128822})]
        
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            logits_processors=logits_processors,
            skip_special_tokens=False,
        )

    def process_image_with_model(self, image: Image.Image, prompt: str = None) -> str:
        """Process image using the model"""
        # Check if engine is initialized
        if self.engine is None:
            raise RuntimeError("Engine is not initialized")
            
        # Use default prompt if not provided
        if prompt is None:
            prompt = PROMPT
            
        # Process image
        if '<image>' in prompt:
            image_features = DeepseekOCRProcessor().tokenize_with_images(images=[image], bos=True, eos=True, cropping=CROP_MODE)
        else:
            image_features = ''
            
        request_id = f"request-{int(time.time() * 1000000)}-{uuid.uuid4().hex[:8]}"
        
        if image_features and '<image>' in prompt:
            request = {
                "prompt": prompt,
                "multi_modal_data": {"image": image_features}
            }
        elif prompt:
            request = {
                "prompt": prompt
            }
        else:
            raise ValueError('Prompt is empty!')
            
        # Add request to engine
        self.engine.add_request(request_id, request, self.sampling_params)
        
        final_output = ""
        try:
            # Process the request using the engine's step method
            while self.engine.has_unfinished_requests():
                request_outputs = self.engine.step()
                for request_output in request_outputs:
                    if request_output.request_id == request_id and request_output.outputs:
                        final_output = request_output.outputs[0].text
        except Exception as e:
            # Re-raise the exception to be handled by the caller
            raise RuntimeError(f"Error during model generation: {str(e)}") from e
                
        return final_output

    def run(self):
        """Worker thread function with improved error handling"""
        # Initialize model
        try:
            self.initialize_model()
        except Exception as e:
            print(f"Failed to initialize model: {e}")
            # Set shutdown event to prevent other workers from starting
            self.shutdown_event.set()
            return
            
        print(f"Model initialized in worker thread {threading.current_thread().ident}")
        
        # Process requests from queue
        while not self.shutdown_event.is_set():
            try:
                # Get request from queue with timeout
                ocr_request: OCRRequest = self.request_queue.get(timeout=1)
                
                try:
                    # Process the request
                    result = self.process_image_with_model(ocr_request.image, ocr_request.prompt)
                    
                    # Clean ref and det tags if enabled
                    final_result = result
                    if CLEAN_REF_TAGS:
                        final_result = clean_ref_tags(result)
                    
                    # Store result
                    self.result_dict[ocr_request.request_id] = {
                        "status": "completed",
                        "result": final_result
                    }
                except Exception as e:
                    # Handle all exceptions
                    print(f"Error processing request: {e}")
                    
                    # Store error for this request
                    self.result_dict[ocr_request.request_id] = {
                        "status": "error",
                        "error": str(e)
                    }
                finally:
                    # Mark task as done
                    self.request_queue.task_done()
                    
            except Empty:
                # Timeout, continue loop
                continue
            except Exception as e:
                print(f"Worker thread error: {e}")
                continue
        
        print(f"Worker thread {threading.current_thread().ident} shutting down")


class OCRProcessor:
    def __init__(self):
        self.request_queue: Queue = Queue()
        self.result_dict: Dict[str, dict] = {}
        self.shutdown_event = Event()
        self.workers = []
        self.max_models = min(2, MAX_WORKER_THREADS)  # Default to 1, max 2

    def start_workers(self):
        """Initialize and start worker threads"""
        print(f"Starting up with {self.max_models} worker threads")
        
        # Create worker threads
        for i in range(self.max_models):
            worker = ModelWorker(self.request_queue, self.result_dict, self.shutdown_event)
            thread = Thread(target=worker.run, daemon=True)
            thread.start()
            self.workers.append(thread)
            
        print(f"All worker threads started")

    def stop_workers(self):
        """Stop worker threads"""
        print("Shutting down worker threads")
        self.shutdown_event.set()
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)

    async def wait_for_result(self, request_id: str, timeout: int = 300) -> dict:
        """Wait for result from worker thread"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if request_id in self.result_dict:
                result = self.result_dict.pop(request_id)
                return result
            await asyncio.sleep(0.1)
        
        raise Exception("Request timeout")

    def submit_request(self, ocr_request: OCRRequest):
        """Submit a request to the queue"""
        self.request_queue.put(ocr_request)

    def health_check(self):
        """Check if the processor is healthy"""
        # Check if workers are still alive
        alive_workers = [worker.is_alive() for worker in self.workers]
        all_workers_alive = all(alive_workers) if alive_workers else False
        
        # Check if shutdown event is not set
        shutdown_not_set = not self.shutdown_event.is_set()
        
        return all_workers_alive and shutdown_not_set


def load_image_from_base64(image_str: str) -> Image.Image:
    """Load PIL Image from base64 string"""
    image_data = base64.b64decode(image_str)
    image = Image.open(io.BytesIO(image_data))
    return image.convert('RGB')
