import os
import logging
import base64
from typing import Tuple
from urllib.parse import urlparse, parse_qs, unquote
from app.config.settings import (
    SUPPORTED_IMAGE_FORMATS, 
    SUPPORTED_DOCUMENT_FORMATS
)

logger = logging.getLogger(__name__)

# Known source file extensions
KNOWN_SOURCE_EXT = [
    "go", "py", "java", "sh", "bat", "ps1", "cmd", "js", "ts", "css",
    "cpp", "hpp", "h", "c", "cs", "sql", "log", "ini", "pl", "pm",
    "r", "dart", "dockerfile", "env", "php", "hs", "hsc", "lua",
    "nginxconf", "conf", "m", "mm", "plsql", "perl", "rb", "rs",
    "db2", "scala", "bash", "swift", "vue", "svelte", "msg", "ex",
    "exs", "erl", "tsx", "jsx", "hs", "lhs"
]

def extract_filename_from_url(url: str) -> str:
    """Extract filename from URL using various methods"""
    # Try to get filename from URL parameters
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    
    # Check for filename in query parameters
    for param in ['filename', 'rscd']:
        if param in query_params:
            value = query_params[param][0]
            if 'filename=' in value:
                # Extract filename from rscd parameter
                filename = value.split('filename=')[-1].strip()
                # Remove any quotes or additional parameters
                filename = filename.split(';')[0].strip('"\'')
                return unquote(filename)
    
    # Fallback to path
    path = parsed_url.path
    if path:
        return os.path.basename(path)
    
    return ""

def is_base64(s: str) -> bool:
    """Check if a string is base64 encoded"""
    try:
        # Check if string starts with data URI scheme
        if s.startswith('data:'):
            return True
            
        # Skip URLs that look like normal web URLs
        if s.startswith(('http://', 'https://')):
            return False
            
        # Try to decode the string if it looks like base64
        if len(s) % 4 == 0 and not s.startswith(('/', '\\')):
            try:
                # Check if string contains valid base64 characters
                if not all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in s):
                    return False
                    
                decoded = base64.b64decode(s)
                # Check if decoded data looks like binary data
                return len(decoded) > 0 and not all(32 <= byte <= 126 for byte in decoded)
            except:
                pass
        return False
    except:
        return False

def detect_file_type(url: str, content_type: str, content_disposition: str = "") -> Tuple[str, bool]:
    """
    Detect if a file is an image or document based on URL, content type and content disposition
    Returns: (file_type, is_supported)
    where file_type is either 'image' or 'document'
    """
    # Handle base64 image data
    if is_base64(url):
        return 'image', True
    
    # Try to get filename from Content-Disposition
    filename = ""
    if content_disposition and "filename=" in content_disposition:
        filename = content_disposition.split("filename=")[-1].strip('"\'')
        filename = unquote(filename)
    
    # If no filename in Content-Disposition, try URL
    if not filename:
        filename = extract_filename_from_url(url)
    
    # Get file extension
    ext = os.path.splitext(filename.lower())[1] if filename else ""
    
    # Check content type and extension
    if (content_type in SUPPORTED_IMAGE_FORMATS or 
        ext in SUPPORTED_IMAGE_FORMATS):
        return 'image', True
    elif (content_type in SUPPORTED_DOCUMENT_FORMATS or 
          ext in SUPPORTED_DOCUMENT_FORMATS):
        return 'document', True
    elif ext.lstrip('.') in KNOWN_SOURCE_EXT:
        return 'document', True
    
    # If content type starts with image/, treat as image
    if content_type.startswith('image/'):
        return 'image', False
    
    # Default to document for all other types
    return 'document', False 