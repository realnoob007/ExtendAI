from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from app.api.chat import router as chat_router
from app.core.logging import setup_logging
from app.config.settings import PORT, HOST

# Setup logging
logger = setup_logging()

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", PORT))
    host = os.getenv("HOST", HOST)
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port) 