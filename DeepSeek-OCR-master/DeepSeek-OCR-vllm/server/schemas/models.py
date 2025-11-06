from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.0
    max_tokens: int = 8192
    stream: bool = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Optional[Dict[str, Any]] = None


class ImageRequest(BaseModel):
    image: str  # base64 encoded image
    prompt: Optional[str] = None


class OCRRequest:
    def __init__(self, request_id: str, image, prompt: str = None):
        from config import PROMPT
        self.request_id = request_id
        self.image = image
        self.prompt = prompt or PROMPT