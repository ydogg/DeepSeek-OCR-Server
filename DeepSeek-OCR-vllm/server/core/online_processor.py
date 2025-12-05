import asyncio
import base64
import io
from PIL import Image

from server.schemas.models import OCRRequest
from server.config import ONLINE_OCR_BASE_URL, ONLINE_OCR_MODEL_NAME, ONLINE_OCR_API_KEY, OCR_PROMPT

# Use OpenAI client for online OCR
from openai import OpenAI


class OnlineOCRProcessor:
    def __init__(self):
        self.client = OpenAI(
            base_url=ONLINE_OCR_BASE_URL,
            api_key=ONLINE_OCR_API_KEY or "sk-test"  # Use test key if none provided
        )
        self.model_name = ONLINE_OCR_MODEL_NAME

    def start_workers(self):
        """Initialize online OCR processor (no workers needed)"""
        pass

    def stop_workers(self):
        """Stop online OCR processor (no workers to stop)"""
        pass

    async def wait_for_result(self, request_id: str, timeout: int = 300) -> dict:
        """Wait for result - not needed for online mode, but kept for API compatibility"""
        # For online mode, processing is synchronous, so this is just a placeholder
        pass

    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    def submit_request(self, ocr_request: OCRRequest):
        """Submit a request and process it directly using OpenAI client"""
        # For online mode, processing happens directly

        # Use default prompt from config if none provided
        from server.config import OCR_PROMPT
        prompt = ocr_request.prompt if ocr_request.prompt is not None else OCR_PROMPT
        #print(f"[Online OCR] Using prompt: {repr(prompt)}")

        try:
            result = self.process_image(ocr_request.image, prompt)
            # Store result for compatibility with wait_for_result
            self._last_result = result
            self._last_request_id = ocr_request.request_id
        except Exception as e:
            # Store error for this request
            self._last_result = {
                "status": "error",
                "error": str(e)
            }
            self._last_request_id = ocr_request.request_id

    async def wait_for_result(self, request_id: str, timeout: int = 300) -> dict:
        """Wait for result - for online mode, return the result directly"""
        # For online mode, processing is synchronous, so we return the result directly
        # But we need to make sure it's the right request
        if hasattr(self, '_last_request_id') and self._last_request_id == request_id:
            return self._last_result
        else:
            # This shouldn't happen in normal operation, but just in case
            return {
                "status": "error",
                "error": "No result available for this request"
            }

    def process_image(self, image: Image.Image, prompt: str = '') -> dict:
        """
        Process image using online OCR API directly with OpenAI client
        
        Args:
            image: PIL Image to process
            prompt: OCR prompt (optional)
            
        Returns:
            dict: OCR result with status and result/error
        """
        try:
            # Convert image to base64
            image_base64 = self.image_to_base64(image)
            
            # Prepare the request
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        },
                        {
                             "type": "text",
                             "text": prompt
                        }
                    ]
                }
            ]
            
            # Make the API call using OpenAI client
            print(f"[Online OCR] Making API call to {self.client.base_url} with model {self.model_name}")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=8100,
                temperature=0.0,
                extra_body={
                    "skip_special_tokens": False,
                    # args used to control custom logits processor
                    "vllm_xargs": {
                        "ngram_size": 30,
                        "window_size": 90,
                        # whitelist: <td>, </td>
                        "whitelist_token_ids": [128821, 128822],
                    },
                },
            )

            # Process the response
            if response and response.choices:
                result_text = response.choices[0].message.content
                return {
                    "status": "completed",
                    "result": result_text
                }
            else:
                return {
                    "status": "error",
                    "error": "No response from OCR API"
                }
                
        except Exception as e:
            # Handle all exceptions
            return {
                "status": "error",
                "error": str(e)
            }

    def health_check(self):
        """Check if the online processor is healthy"""
        # For now, we'll assume it's healthy
        return True
