# Server-specific configuration
# These are configurations specific to the server implementation

# Server address and port
ADDRESS = "0.0.0.0"
PORT = 8888

# Maximum number of worker threads (default: 1, max: 2)
MAX_WORKER_THREADS = 1

# Whether to clean ref and det tags from the output
CLEAN_REF_TAGS = True  # Set to True to remove ref and det tags from the output


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

MODEL_PATH = '/home/ai/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-OCR'  # change to your model path

PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'

# Tokenizer
from transformers import AutoTokenizer
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
