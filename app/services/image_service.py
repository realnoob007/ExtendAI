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
            
            image_data = image_url
        else:
            # Handle regular URL
            # Check if this is an S3 presigned URL
            is_s3_url = "X-Amz-Algorithm=AWS4-HMAC-SHA256" in image_url and "X-Amz-Credential" in image_url
            
            # Set headers based on URL type
            headers = {
                "Accept": "*/*",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            async with httpx.AsyncClient() as client:
                if is_s3_url:
                    # For S3 presigned URLs, skip HEAD request and directly download
                    response = await client.get(
                        image_url,
                        headers=headers,
                        follow_redirects=True
                    )
                    response.raise_for_status()
                    
                    # Check content length after download
                    image_bytes = response.content
                    if len(image_bytes) > MAX_IMAGE_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Image size ({len(image_bytes)} bytes) exceeds maximum allowed size ({MAX_IMAGE_SIZE} bytes)"
                        )
                    
                    # Detect content type
                    content_type = response.headers.get("content-type", "image/jpeg")
                    if not content_type.startswith("image/"):
                        content_type = "image/jpeg"  # Default to jpeg if no valid content type
                    
                    # Create data URI
                    image_data = f"data:{content_type};base64,{base64.b64encode(image_bytes).decode()}"
                else:
                    # For regular URLs, do HEAD request first
                    head_response = await client.head(
                        image_url,
                        headers=headers,
                        follow_redirects=True
                    )
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
                    
                    # For regular URLs, pass the URL directly
                    image_data = image_url
        
        # Process image with OpenAI
        async with httpx.AsyncClient() as client:
            openai_request = {
                "model": OPENAI_ENHANCE_MODEL,
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": "Only output the ocr result of user's uploaded image. If the image contains data structures like a graph or table, convert the info into text. If this is a image of an object, describe it as detailed as possible."
                    }, {
                        "type": "image_url",
                        "image_url": {"url": image_data}
                    }]
                }],
                "stream": False
            }
            
            # Add headers for non-base64 and non-S3 URLs
            if not is_base64(image_data) and not is_s3_url:
                openai_request["messages"][0]["content"][1]["image_url"]["headers"] = headers
            
            response = await client.post(
                OPENAI_API_URL,
                json=openai_request,
                headers=get_headers(OPENAI_API_KEY),
                timeout=API_TIMEOUT
            )
            response.raise_for_status()
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")