# Server-specific configuration
# These are configurations specific to the server implementation

# Server address and port
ADDRESS = "0.0.0.0"
PORT = 8001

# Maximum number of worker threads (default: 1, max: 2)
MAX_WORKER_THREADS = 1

# Whether to enable streaming mode for responses
STREAMING_MODE = False

# OCR mode configuration
# ONLINE_OCR_MODE: True for online API mode, False for offline local mode
ONLINE_OCR_MODE = False

# Online OCR API configuration
ONLINE_OCR_BASE_URL = "http://localhost:8000/v1"  # Base URL for the online OCR API
ONLINE_OCR_MODEL_NAME = "deepseek-ai/DeepSeek-OCR"  # Model name for the online OCR API
ONLINE_OCR_API_KEY = "test"  # API key for the online OCR API if needed

# VL model configuration for image analysis
VL_MODEL_BASE_URL = "http://172.28.71.194:8000/v1"  # Default to local server
VL_MODEL_API_KEY = "test"  # API key for the VL model if needed
VL_MODEL_NAME = "gpt-5-chat"  # Model name for the VL model
VL_MODEL_ANALYSIS_PROMPT = """Analyze this image and provide a detailed description.
If it's a chart or graph, describe the data, trends, and key insights.
If it's a table, extract and present the data in a structured format.
If it's a diagram or flowchart, explain the process or relationships shown.
If it's a photograph, describe the content and context."""  # Prompt for VL model analysis

# LLM model configuration for output enhancement
ENHANCEMENT_LLM_BASE_URL = "http://llm.necsoft.jn.com.cn:8000/v1"
ENHANCEMENT_LLM_MODEL_NAME = "qwen3-coder"
ENHANCEMENT_LLM_API_KEY = "test"
VL_MODEL_ENHANCEMENT_PROMPT = "如果文字内容描述了流程图等UML风格的内容，将这些文字转为mermaid格式，其他无关文字直接保留即可。"

# Default OCR prompt
DEFAULT_OCR_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."

# Inherited configuration from main config.py
# These configurations are imported from the main config.py file

# Model configuration
# OCR configuration
BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True
MIN_CROPS= 2
MAX_CROPS= 6 # max:9; If your GPU memory is small, it is recommended to set it to 6.
MAX_CONCURRENCY = 100 # If you have limited GPU memory, lower the concurrency count.
NUM_WORKERS = 64 # image pre-process (resize/padding) workers
PRINT_NUM_VIS_TOKENS = False
SKIP_REPEAT = True

OCR_MODEL_PATH = '/home/ai/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-OCR'  # change to your model path

OCR_PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'
#OCR_PROMPT = '<image>\nFree OCR.'

# Tokenizer
from transformers import AutoTokenizer
TOKENIZER = AutoTokenizer.from_pretrained(OCR_MODEL_PATH, trust_remote_code=True)
