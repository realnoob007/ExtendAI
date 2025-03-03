import os
from dotenv import load_dotenv
from enum import Enum

# Load environment variables from .env
load_dotenv()

class SearchEngine(Enum):
    GOOGLE = "google"
    SEARXNG = "searxng"

# Server Settings
PORT = int(os.getenv("PORT", "8096"))
HOST = os.getenv("HOST", "0.0.0.0")

# Document processing settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
MAX_CHUNKS_PER_DOC = int(os.getenv("MAX_CHUNKS_PER_DOC", "5"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "50"))
MIN_CHUNK_LENGTH = int(os.getenv("MIN_CHUNK_LENGTH", "10"))

# Vector store settings
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "faiss")  # "faiss", "postgres", or "pinecone"
POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING", "")
POSTGRES_COLLECTION_NAME = os.getenv("POSTGRES_COLLECTION_NAME", "document_vectors")

# Pinecone settings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_xxx")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "extendai")
# Cache settings
VECTOR_CACHE_DIR = os.getenv("VECTOR_CACHE_DIR", "cache/vectors")
VECTOR_CACHE_TTL = int(os.getenv("VECTOR_CACHE_TTL", str(2 * 60 * 60)))  # 2 hours in seconds

# File size limits (in bytes)
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(10 * 1024 * 1024)))  # 10MB default
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", str(10 * 1024 * 1024)))  # 10MB default

# Supported file formats
SUPPORTED_IMAGE_FORMATS = {
    # MIME types
    "image/jpeg", "image/png", "image/gif", "image/webp", "image/tiff", "image/bmp",
    # Extensions
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".tiff", ".bmp"
}

SUPPORTED_DOCUMENT_FORMATS = {
    # MIME types
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "text/plain",
    "text/csv",
    "text/html",
    "text/xml",
    "text/markdown",
    "application/epub+zip",
    # Extensions
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".txt", ".csv", ".html", ".htm", ".xml", ".md", ".rst", ".epub"
}

# API URLs and Keys
TARGET_MODEL_BASE_URL = os.getenv("TARGET_MODEL_BASE_URL", "https://api.openai.com")
TARGET_MODEL_API_URL = f"{TARGET_MODEL_BASE_URL}/v1/chat/completions"
TARGET_MODEL_API_KEY = os.getenv("TARGET_MODEL_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
OPENAI_API_URL = f"{OPENAI_BASE_URL}/v1/chat/completions"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENHANCE_MODEL = os.getenv("OPENAI_ENHANCE_MODEL")  # Model for image/document enhancement

# Embedding Settings
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", OPENAI_API_KEY)  # Default to OpenAI key if not set
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

MY_API_KEY = os.getenv("MY_API_KEY")

# Search Settings
SEARXNG_URL = os.getenv("SEARXNG_URL", "https://searxng.example.com")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "deepseek-r1")
SEARCH_ENGINE = os.getenv("SEARCH_ENGINE", "google").lower()

# Search Result Settings
SEARCH_RESULT_LIMIT = int(os.getenv("SEARCH_RESULT_LIMIT", "5"))  # Number of results to return
SEARCH_RESULT_MULTIPLIER = int(os.getenv("SEARCH_RESULT_MULTIPLIER", "2"))  # Multiplier for raw results to fetch
WEB_CONTENT_CHUNK_SIZE = int(os.getenv("WEB_CONTENT_CHUNK_SIZE", "512"))  # Size of each content chunk
WEB_CONTENT_CHUNK_OVERLAP = int(os.getenv("WEB_CONTENT_CHUNK_OVERLAP", "50"))  # Overlap between chunks
WEB_CONTENT_MAX_CHUNKS = int(os.getenv("WEB_CONTENT_MAX_CHUNKS", "5"))  # Number of most relevant chunks to return

# Feature Switches
ENABLE_PROGRESS_MESSAGES = os.getenv("ENABLE_PROGRESS_MESSAGES", "true").lower() == "true"
ENABLE_IMAGE_ANALYSIS = os.getenv("ENABLE_IMAGE_ANALYSIS", "true").lower() == "true"
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true"
ENABLE_DOCUMENT_ANALYSIS = os.getenv("ENABLE_DOCUMENT_ANALYSIS", "true").lower() == "true"

# Progress Messages
PROGRESS_MESSAGES = {
    "image_analysis": os.getenv("PROGRESS_MSG_IMAGE", "Analyzing image content..."),
    "document_analysis": os.getenv("PROGRESS_MSG_DOC", "Analyzing document content..."),
    "document_search": os.getenv("PROGRESS_MSG_DOC_SEARCH", "Searching document for relevant content..."),
    "web_search": os.getenv("PROGRESS_MSG_WEB_SEARCH", "Doing web search for relevant information...")
}

# Proxy Settings
PROXY_ENABLED = os.getenv("PROXY_ENABLED", "false").lower() == "true"
PROXY_HOST = os.getenv("PROXY_HOST", "")
PROXY_PORT = int(os.getenv("PROXY_PORT", ""))
PROXY_USERNAME = os.getenv("PROXY_USERNAME", "")  # Using -res-any for any region
PROXY_PASSWORD = os.getenv("PROXY_PASSWORD", "")

# Timeouts
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "600"))
SEARCH_TIMEOUT = 10
REQUEST_TIMEOUT = {
    "connect": 3,
    "read": 5,
    "write": 3,
    "pool": 3
}

# Headers
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1"
}

# Search Engine Weights
SEARCH_ENGINE_WEIGHTS = {
    "google": 3,
    "bing": 2,
    "duckduckgo": 2,
    "brave": 1,
    "qwant": 1
} 