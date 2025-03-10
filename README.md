# ExtendAI

ExtendAI is a universal framework that can extend any AI model to achieve multi-modal ability, network search ability, and document analysis ability.

## Features

- **Multi-format Document Support**: Process various document formats including:
  - PDF files
  - Microsoft Office documents (Word, Excel, PowerPoint)
  - Markdown files
  - Text files
  - Source code files
  - HTML/XML files
  - CSV files
  - RST files
  - Outlook MSG files
  - EPub files

- **Advanced Document Processing**:
  - Automatic text encoding detection and fixing
  - Mutiple Documents Batch Processing
  - Smart document chunking with configurable size and overlap
  - Ultra fast processing of large documents
  - File type detection based on content and extensions

- **Intelligent Web Search**:
  - Integration with multiple search engines (Google, Bing)
  - SearXNG support for privacy-focused searches
  - Smart result filtering and ranking
  - Configurable search depth and limits
  - Automatic content extraction and processing

- **Advanced Image Comprehension**:
  - Deep image analysis and understanding
  - Mutiple Images Batch Processing
  - Text extraction from images (OCR)
  - Support for multiple image formats
  - Context-aware image interpretation

- **Flexible Vector Store Integration**:
  - PostgreSQL vector store support
  - Pinecone vector store support
  - Local FAISS vector store with caching
  - Automatic fallback mechanisms

- **Performance Optimizations**:
  - Asynchronous document processing
  - Efficient caching system for FAISS vectors
  - Batched embedding generation
  - Configurable chunk sizes and overlap
  - File size limits and safety checks

- **Robust Error Handling**:
  - Graceful fallbacks for failed operations
  - Comprehensive logging system
  - Automatic cleanup of temporary files
  - Cache invalidation and management

## Screenshots
<div align="center">
  <img src="https://github.com/user-attachments/assets/5f107c04-89ee-46b0-9f22-906fe420ab41" width="800" alt="ExtendAI Screenshot 1" />
  <br/><br/>
  <img src="https://github.com/user-attachments/assets/b5c558b9-9f23-479e-ae42-1c7e5ebb82d8" width="800" alt="ExtendAI Screenshot 2" />
  <br/><br/>
  <img src="https://github.com/user-attachments/assets/988419ce-9824-44cd-abf8-fbb27162ce83" width="800" alt="ExtendAI Screenshot 3" />

</div>

## Configuration

The system is highly configurable through environment variables:

```env
# Document Processing Settings
CHUNK_SIZE=1000                # Size of text chunks for processing (in characters)
CHUNK_OVERLAP=200              # Overlap between chunks to maintain context
MAX_CHUNKS_PER_DOC=5          # Maximum number of chunks to return per document
EMBEDDING_BATCH_SIZE=50        # Number of chunks to process in parallel for embeddings
MIN_CHUNK_LENGTH=10           # Minimum length of a chunk to be processed
MAX_FILE_SIZE=10485760        # Maximum file size in bytes (10MB)
MAX_IMAGE_SIZE=5242880        # Maximum image size in bytes (5MB)

# Vector Store Settings
VECTOR_STORE_TYPE=faiss       # Vector store backend: 'faiss', 'postgres', or 'pinecone'
POSTGRES_CONNECTION_STRING=    # PostgreSQL connection URL for vector storage
POSTGRES_COLLECTION_NAME=      # Collection name for storing vectors in PostgreSQL

# Pinecone Settings
PINECONE_API_KEY=             # API key for Pinecone vector database
PINECONE_INDEX_NAME=          # Name of the Pinecone index to use

# Cache Settings
VECTOR_CACHE_DIR=cache/vectors # Directory for storing FAISS vector cache
VECTOR_CACHE_TTL=7200         # Cache time-to-live in seconds (2 hours)

API_TIMEOUT=600                    # API Timeout (in seconds)
MY_API_KEY=sk-planetzero-api-key   # Authorization key

# Model API Settings
TARGET_MODEL_BASE_URL=        # Base URL for the target AI model API
TARGET_MODEL_API_KEY=         # API key for the target model
OPENAI_BASE_URL=             # Base URL for OpenAI API (or compatible endpoint)
OPENAI_API_KEY=              # OpenAI API key (or compatible key)
OPENAI_ENHANCE_MODEL=        # The model used to facilitate image processing and search analysis

# Embedding API Settings
EMBEDDING_BASE_URL=          # Base URL for embedding API
EMBEDDING_API_KEY=           # API key for embedding service
EMBEDDING_MODEL=text-embedding-3-small  # Model to use for text embeddings
EMBEDDING_DIMENSIONS=1536    # Dimension of the embedding vectors

# Search Settings
SEARXNG_URL=                # URL for SearXNG instance (for web search)
DEFAULT_MODEL=deepseek-r1   # Default AI model to use
SEARCH_ENGINE=google        # Search engine to use (google, bing, etc.)
SEARCH_RESULT_LIMIT=5       # Number of search results to return
SEARCH_RESULT_MULTIPLIER=2  # Multiplier for raw results to fetch
WEB_CONTENT_CHUNK_SIZE=512  # Size of chunks for web content
WEB_CONTENT_CHUNK_OVERLAP=50 # Overlap for web content chunks
WEB_CONTENT_MAX_CHUNKS=5    # Maximum chunks to process from web content

# Proxy Settings (Optional)
PROXY_ENABLED=false         # Whether to use proxy for requests
PROXY_HOST=                # Proxy server hostname
PROXY_PORT=                # Proxy server port
PROXY_USERNAME=            # Proxy authentication username
PROXY_PASSWORD=            # Proxy authentication password
PROXY_COUNTRY=             # Preferred proxy server country
PROXY_SESSION_ID=          # Session ID for proxy (if required)

# Feature Switches
ENABLE_PROGRESS_MESSAGES=false  # Enable/disable progress message updates
ENABLE_IMAGE_ANALYSIS=true     # Enable/disable image analysis capability
ENABLE_WEB_SEARCH=true        # Enable/disable web search capability
ENABLE_DOCUMENT_ANALYSIS=true  # Enable/disable document analysis capability

# Progress Messages (Customizable)
PROGRESS_MSG_IMAGE="Analyzing image content..."       # Message shown during image analysis
PROGRESS_MSG_DOC="Analyzing document..."             # Message shown during document processing
PROGRESS_MSG_DOC_SEARCH="Searching document..."      # Message shown during document search
PROGRESS_MSG_WEB_SEARCH="Searching web content..."   # Message shown during web search
```

## Advanced Configuration Details

### Document Processing
- OpenAI format, just need to pass the document url through image_url parameter
- The chunking system breaks down documents into manageable pieces while maintaining context through overlap
- Batch processing helps optimize embedding generation and API usage
- File size limits protect against resource exhaustion and API limitations

### Vector Store Options
1. **FAISS** (Local):
   - Fast, efficient local vector storage
   - Good for development and smaller deployments
   - Includes local caching system for performance

2. **PostgreSQL**:
   - Persistent vector storage in PostgreSQL database
   - Suitable for production deployments
   - Supports concurrent access and backups

3. **Pinecone**:
   - Cloud-based vector database
   - Excellent for large-scale deployments
   - Provides automatic scaling and management

### API Integration
- Supports multiple model endpoints (OpenAI-compatible)
- Configurable embedding services
- Automatic fallbacks and error handling

### Search Capabilities
- Integrated web search through SearXNG and Google
- Configurable search engines and result limits
- Smart content chunking for web results

### Progress Tracking
- Customizable progress messages
- Feature toggles for different capabilities
- Localization support for messages

## Installation & Deployment

### Option 1: Docker Compose (Recommended for Production)

1. Clone the repository:
```bash
git clone https://github.com/realnoob007/ExtendAI.git
cd ExtendAI
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration (API keys, model endpoints, etc.)
```

3. Start the services using Docker Compose:
```bash
docker compose up -d
```

This will start:
- ExtendAI application on port 8096
- PostgreSQL database on port 6023
- PostgreSQL with pgvector extension on port 6024

To view logs:
```bash
docker compose logs -f
```

To stop the services:
```bash
docker compose down
```

### Option 2: Docker (Single Container)

If you want to run only the ExtendAI application container and use external databases:

1. Clone and configure:
```bash
git clone https://github.com/realnoob007/ExtendAI.git
cd ExtendAI
cp .env.example .env
# Edit .env with your configuration
```

2. Run the container:
```bash
docker run -d \
  --name extendai \
  -p 8096:8096 \
  --env-file .env \
  -v $(pwd)/cache:/app/cache \
  -v $(pwd)/.env:/app/.env:ro \
  chasney/extendai:latest
```

Useful Docker commands:
```bash
# View logs
docker logs -f extendai

# Stop container
docker stop extendai

# Remove container
docker rm extendai

# Rebuild image (if you made changes)
docker build --no-cache -t extendai .
```

### Option 3: Local Development

For development purposes, you can run the application directly:

1. Create a virtual environment (Python 3.11+ recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run the application:
```bash
python main.py
```

## Architecture

(not fully refactored yet):

- `main.py`: Application entry point and FastAPI setup
- `app/api/`: API route definitions
- `app/services/`: Core business logic services
- `app/models/`: Data models and schemas
- `app/utils/`: Utility functions and helpers
- `app/core/`: Core system components
- `app/config/`: Configuration management

#### Example Request Payload (Post Request to http://0.0.0.0:8096/v1/chat/completions)
```json
{
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "what is this document about?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://xxx.com/document.pdf"
            }
          }
        ]
      }
    ],
    "model": "deepseek-r1-all",
    "stream": false
}
```

#### Content Types Support
The API supports multiple content types in messages:

1. **Text Content**
```json
{
    "type": "text",
    "text": "your question or prompt here"
}
```

2. **Image URL**
```json
{
    "type": "image_url",
    "image_url": {
        "url": "https://example.com/image.jpg"
    }
}
```

3. **Document URL**
```json
{
    "type": "image_url",
    "image_url": {
        "url": "https://example.com/document.pdf"
    }
}
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| messages | array | Yes | Array of message objects |
| model | string | No | Model to use (default: deepseek-r1) |
| stream | boolean | No | Whether to stream the response (default: false) |

#### Non-Streaming Response
```json
{
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677858242,
    "model": "deepseek-r1-all",
    "usage": {
        "prompt_tokens": 56,
        "completion_tokens": 31,
        "total_tokens": 87
    },
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "The document appears to be about..."
            },
            "finish_reason": "stop",
            "index": 0
        }
    ]
}
```

#### Streaming Response
When `stream` is set to `true`, the response will be sent as Server-Sent Events (SSE):
```http
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"deepseek-r1-all","choices":[{"delta":{"role":"assistant","content":"The"},"index":0,"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"deepseek-r1-all","choices":[{"delta":{"content":" document"},"index":0,"finish_reason":null}]}

data: [DONE]
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Sponsorship (You can run this project with this api platform to save 20%)
[PlanetZero API](https://api.planetzeroapi.com/)

## License

MIT License
