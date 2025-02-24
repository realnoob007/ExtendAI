from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Union, Literal, Dict, Any
from enum import Enum

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class ContentType(str, Enum):
    TEXT = "text"
    IMAGE_URL = "image_url"
    DOCUMENT = "document"

class ImageUrl(BaseModel):
    url: str

class Document(BaseModel):
    """Document model for storing document content and metadata"""
    page_content: str
    metadata: Dict[str, Any]
    id: Optional[str] = None

class MessageContent(BaseModel):
    type: ContentType
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None
    document: Optional[Document] = None

class Message(BaseModel):
    role: Role
    content: Union[str, List[MessageContent]]

class ChatRequest(BaseModel):
    messages: List[Message]
    model: str
    stream: bool = False

class SearchAnalysis(BaseModel):
    """Schema for analyzing whether search context is needed"""
    needs_search: bool
    search_keywords: Optional[List[str]] = None

class SearchResult(BaseModel):
    """Schema for search results"""
    url: str
    title: str
    content: str 