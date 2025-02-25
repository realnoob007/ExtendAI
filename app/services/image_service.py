import httpx
import logging
import base64
from fastapi import HTTPException
from app.utils.file_utils import is_base64

from app.config.settings import (
    OPENAI_API_URL, OPENAI_API_KEY,
    API_TIMEOUT, MAX_IMAGE_SIZE,
    OPENAI_ENHANCE_MODEL
)

logger = logging.getLogger(__name__)

def get_headers(api_key: str) -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

async def process_image(image_url: str) -> str:
    """Process image using OpenAI and return description"""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    
    try:
        # Handle base64 image data
        if is_base64(image_url):
            # If it's a data URI, get the actual base64 data
            if image_url.startswith('data:'):
                base64_data = image_url.split(',')[1]
            else:
                base64_data = image_url
            
            # Check size of decoded data
            try:
                decoded_data = base64.b64decode(base64_data)
                if len(decoded_data) > MAX_IMAGE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Image size ({len(decoded_data)} bytes) exceeds maximum allowed size ({MAX_IMAGE_SIZE} bytes)"
                    )
            except Exception as e:
                logger.error(f"Error decoding base64 data: {str(e)}")
                raise HTTPException(status_code=400, detail="Invalid base64 image data")
            
            # Process base64 image directly
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    OPENAI_API_URL,
                    json={
                        "model": OPENAI_ENHANCE_MODEL,
                        "messages": [{
                            "role": "user",
                            "content": [{
                                "type": "text",
                                "text": "Only output the ocr result of user's uploaded image. If the image contains data structures like a graph or table, convert the info into text. If this is a image of an object, describe it as detailed as possible."
                            }, {
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            }]
                        }],
                        "stream": False
                    },
                    headers=get_headers(OPENAI_API_KEY),
                    timeout=API_TIMEOUT
                )
                response.raise_for_status()
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Handle regular URL
        async with httpx.AsyncClient() as client:
            # First check image size
            head_response = await client.head(image_url, follow_redirects=True)
            head_response.raise_for_status()
            
            # Check content length if available
            content_length = head_response.headers.get("content-length")
            if content_length:
                file_size = int(content_length)
                if file_size > MAX_IMAGE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Image size ({file_size} bytes) exceeds maximum allowed size ({MAX_IMAGE_SIZE} bytes)"
                    )
            
            # Proceed with image processing
            response = await client.post(
                OPENAI_API_URL,
                json={
                    "model": OPENAI_ENHANCE_MODEL,
                    "messages": [{
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": "Only output the ocr result of user's uploaded image. If the image contains data structures like a graph or table, convert the info into text. If this is a image of an object, describe it as detailed as possible."
                        }, {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        }]
                    }],
                    "stream": False
                },
                headers=get_headers(OPENAI_API_KEY),
                timeout=API_TIMEOUT
            )
            response.raise_for_status()
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")