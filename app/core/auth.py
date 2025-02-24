from fastapi import HTTPException, Request
from app.config.settings import MY_API_KEY

async def verify_api_key(request: Request) -> str:
    """Verify the API key from the request header"""
    if not MY_API_KEY:
        raise HTTPException(status_code=500, detail="Server API key not configured")
    
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    try:
        scheme, token = auth_header.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authentication scheme")
        if token != MY_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return token
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization header format") 