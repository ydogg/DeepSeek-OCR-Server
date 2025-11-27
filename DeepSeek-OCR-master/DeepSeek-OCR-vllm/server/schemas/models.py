from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union




class ImageUrl(BaseModel):
    url: str


class ContentImage(BaseModel):
    type: str
    image_url: ImageUrl


class ContentText(BaseModel):
    type: str
    text: str


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Union[ContentImage, ContentText]]]


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
    level: str = "clean"  # Processing level: raw, clean, image_clean


class OCRRequest:
    def __init__(self, request_id: str, image, prompt: str = None):
        from config import OCR_PROMPT
        self.request_id = request_id
        self.image = image
        self.prompt = prompt or OCR_PROMPT