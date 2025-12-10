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
VL_MODEL_BASE_URL = "http://localhost:8000/v1"
VL_MODEL_API_KEY = "test"
VL_MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
VL_MODEL_ANALYSIS_PROMPT = """<image>/nAnalyze this image and provide a detailed description.
If it's a chart or graph, describe the data, trends, and key insights.
If it's a table, extract and present the data in a structured format.
If it's a diagram or flowchart, explain the process or relationships shown.
If it's a photograph, describe the content and context.""" 

# LLM model configuration for output enhancement
#ENHANCEMENT_LLM_BASE_URL = "http://llm.necsoft.jn.com.cn:4000/v1"
ENHANCEMENT_LLM_BASE_URL = "http://localhost:4000/v1"
ENHANCEMENT_LLM_MODEL_NAME = "qwen3-coder"
ENHANCEMENT_LLM_API_KEY = "sk-1234"
VL_MODEL_ENHANCEMENT_PROMPT = "如果文字内容描述了流程图等UML风格的内容，将这些文字转为mermaid格式，其他内容保留。"

# Default OCR prompt
OCR_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
