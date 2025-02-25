import logging
import datetime
import httpx
from typing import List, Tuple, Dict
from fastapi import HTTPException
import asyncio
import time
from cachetools import TTLCache

from app.models.chat import Message, Role, SearchAnalysis, ContentType
from app.config.settings import (
    OPENAI_API_URL, OPENAI_API_KEY,
    API_TIMEOUT, PROGRESS_MESSAGES,
    ENABLE_PROGRESS_MESSAGES, ENABLE_IMAGE_ANALYSIS,
    ENABLE_WEB_SEARCH, ENABLE_DOCUMENT_ANALYSIS,
    VECTOR_CACHE_TTL, OPENAI_ENHANCE_MODEL
)
from app.services.search_service import SearchService
from app.services.image_service import process_image
from app.services.document_service import DocumentService
from app.utils.file_utils import detect_file_type

logger = logging.getLogger(__name__)

# Initialize services
document_service = DocumentService()
search_service = SearchService()

# Global cache for processed files
# Cache structure: {file_url: {'type': 'image'|'document', 'content': str, 'timestamp': float}}
file_cache = TTLCache(maxsize=1000, ttl=VECTOR_CACHE_TTL)

def get_headers(api_key: str) -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

async def analyze_search_need(query: str) -> SearchAnalysis:
    """Analyze if the query needs search context"""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                OPENAI_API_URL,
                json={
                    "model": OPENAI_ENHANCE_MODEL,
                    "messages": [{
                        "role": "system",
                        "content": "You are an expert at analyzing whether queries need real-time information. Return true only for queries about current events, real-time info, or those needing factual verification."
                    }, {
                        "role": "user",
                        "content": query
                    }],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "search_analysis",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "needs_search": {"type": "boolean"},
                                    "search_keywords": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                },
                                "required": ["needs_search", "search_keywords"],
                                "additionalProperties": False
                            },
                            "strict": True
                        }
                    }
                },
                headers=get_headers(OPENAI_API_KEY),
                timeout=API_TIMEOUT
            )
            response.raise_for_status()
            result = response.json()
            
            content = result.get("choices", [{}])[0].get("message", {}).get("content", {})
            if isinstance(content, str):
                import json
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse response content as JSON: {content}")
                    content = {"needs_search": False, "search_keywords": []}
            
            analysis = SearchAnalysis.model_validate(content)
            
            logger.info(f"Search analysis for query '{query}':")
            logger.info(f"Needs search: {analysis.needs_search}")
            if analysis.search_keywords:
                logger.info(f"Search keywords: {analysis.search_keywords}")
            
            return analysis
        except Exception as e:
            logger.error(f"Search analysis error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Search analysis failed: {str(e)}")

async def process_messages(messages: List[Message], stream: bool = False, progress_queue: asyncio.Queue = None) -> Tuple[List[Message], List[str]]:
    """Process messages and handle any images or documents in them"""
    
    async def send_progress(message_type: str):
        if ENABLE_PROGRESS_MESSAGES and stream and progress_queue:
            await progress_queue.put(PROGRESS_MESSAGES[message_type])
    
    processed_messages = []
    system_content_parts = []
    current_search_parts = []
    current_search_urls = []
    
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    system_content_parts.append(f"Current date: {current_date}")
    
    # Track processed files in current request to avoid duplicates
    request_processed_files = set()
    
    for message in messages:
        if isinstance(message.content, list):
            text_parts = []
            query_text = ""
            
            for content in message.content:
                if content.type == ContentType.TEXT:
                    query_text = content.text
                    text_parts.append(query_text)
                    # Truncate long text in logs
                    log_text = query_text[:50] + "..." if len(query_text) > 50 else query_text
                    logger.info(f"Added text content: {log_text}")
                elif content.type == ContentType.IMAGE_URL and content.image_url:
                    file_url = content.image_url.url
                    
                    # Skip if already processed in this request
                    if file_url in request_processed_files:
                        continue
                    request_processed_files.add(file_url)
                    
                    # For base64 URLs, show truncated version in logs
                    log_url = file_url
                    if file_url.startswith('data:'):
                        log_url = file_url[:30] + "..." + file_url[-10:]
                    elif len(file_url) > 100:
                        log_url = file_url[:50] + "..." + file_url[-50:]
                    
                    # Check cache first
                    cache_hit = file_cache.get(file_url)
                    if cache_hit:
                        logger.info(f"Cache hit for file: {log_url}")
                        if cache_hit['type'] == 'image':
                            text_parts.append(f"[Image Content: {cache_hit['content']}]")
                            logger.info("Added cached image content")
                        elif cache_hit['type'] == 'document' and query_text:
                            # For documents, we still need to search with the current query
                            logger.info("Using cached document for search")
                            await send_progress("document_search")
                            relevant_chunks = await document_service.search_similar(query_text, source_url=file_url)
                            if relevant_chunks:
                                chunks_text = "\n\n".join(
                                    f"Relevant section {i+1} from {file_url}:\n{chunk.page_content}"
                                    for i, chunk in enumerate(relevant_chunks)
                                )
                                text_parts.append(f"[Document Content:\n{chunks_text}]")
                                logger.info(f"Added {len(relevant_chunks)} relevant document sections from cache")
                            else:
                                text_parts.append(f"[No relevant content found in cached document: {log_url}]")
                        continue
                    
                    logger.info(f"\nProcessing file URL: {log_url}")
                    
                    # Check if it's base64 data
                    file_type, is_supported = detect_file_type(file_url, "")
                    
                    # Skip processing based on feature flags
                    if file_type == 'image':
                        if not ENABLE_IMAGE_ANALYSIS:
                            continue
                        await send_progress("image_analysis")
                    else:  # document type
                        if not ENABLE_DOCUMENT_ANALYSIS:
                            continue
                        await send_progress("document_analysis")
                    
                    try:
                        if file_type == 'image' and is_supported and ENABLE_IMAGE_ANALYSIS:
                            logger.info("Processing as IMAGE")
                            image_content = await process_image(file_url)
                            if image_content:
                                # Cache the result
                                file_cache[file_url] = {
                                    'type': 'image',
                                    'content': image_content,
                                    'timestamp': time.time()
                                }
                                text_parts.append(f"[Image Content: {image_content}]")
                                logger.info("Image processing completed and cached")
                        else:
                            if ENABLE_DOCUMENT_ANALYSIS:
                                logger.info("Processing as DOCUMENT")
                                documents = await document_service.process_document(file_url)
                                if documents:
                                    # Cache the document processing status
                                    file_cache[file_url] = {
                                        'type': 'document',
                                        'content': 'processed',
                                        'timestamp': time.time()
                                    }
                                    if query_text:
                                        logger.info("Searching document for relevant content...")
                                        await send_progress("document_search")
                                        relevant_chunks = await document_service.search_similar(query_text, source_url=file_url)
                                        if relevant_chunks:
                                            chunks_text = "\n\n".join(
                                                f"Relevant section {i+1} from {file_url}:\n{chunk.page_content}"
                                                for i, chunk in enumerate(relevant_chunks)
                                            )
                                            text_parts.append(f"[Document Content:\n{chunks_text}]")
                                            logger.info(f"Added {len(relevant_chunks)} relevant document sections")
                                        else:
                                            text_parts.append(f"[No relevant content found in document: {file_url}]")
                    except Exception as e:
                        logger.error(f"Error processing file {file_url}: {str(e)}")
                        text_parts.append(f"[Error processing file {file_url}: {str(e)}]")
            
            # Only perform web search if this is the last user message
            if query_text and message == messages[-1] and message.role == Role.USER:
                await send_progress("web_search")
                
                if ENABLE_WEB_SEARCH:
                    search_analysis = await analyze_search_need(query_text)
                    
                    if search_analysis.needs_search and search_analysis.search_keywords:
                        logger.info("Performing web search...")
                        search_results = await search_service.search(query_text)
                        
                        search_context_parts = []
                        current_search_urls = []
                        
                        for idx, result in enumerate(search_results, 1):
                            url = result["url"]
                            title = result["title"]
                            content = result["content"]
                            similarity = result["similarity"]
                            
                            current_search_urls.append(f"> [{idx}] {url}")
                            search_context_parts.append(
                                f"Source [{idx}] (Similarity: {similarity:.2f}): {title}\n{content}"
                            )
                        
                        if search_context_parts:
                            search_context = "\n\n".join(search_context_parts)
                            current_search_parts = [
                                "When using information from the search results, cite the sources using [n] format where n is the source number.",
                                f"Search Results:\n{search_context}",
                                f"Include these references at the end of your response:\n{chr(10).join(current_search_urls)}"
                            ]
                            logger.info(f"Added {len(search_context_parts)} search results to context")
                else:
                    logger.info("Web search is disabled")
            
            if text_parts:
                processed_messages.append(Message(
                    role=message.role,
                    content=" ".join(text_parts)
                ))
        else:
            # Handle string content (direct text input)
            query_text = message.content
            processed_messages.append(message)
            
            # Only perform web search if this is the last user message and it's a text query
            if query_text and message == messages[-1] and message.role == Role.USER:
                await send_progress("web_search")
                
                if ENABLE_WEB_SEARCH:
                    search_analysis = await analyze_search_need(query_text)
                    
                    if search_analysis.needs_search and search_analysis.search_keywords:
                        logger.info("Performing web search...")
                        search_results = await search_service.search(query_text)
                        
                        search_context_parts = []
                        current_search_urls = []
                        
                        for idx, result in enumerate(search_results, 1):
                            url = result["url"]
                            title = result["title"]
                            content = result["content"]
                            similarity = result["similarity"]
                            
                            current_search_urls.append(f"> [{idx}] {url}")
                            search_context_parts.append(
                                f"Source [{idx}] (Similarity: {similarity:.2f}): {title}\n{content}"
                            )
                        
                        if search_context_parts:
                            search_context = "\n\n".join(search_context_parts)
                            current_search_parts = [
                                "When using information from the search results, cite the sources using [n] format where n is the source number.",
                                f"Search Results:\n{search_context}",
                                f"Include these references at the end of your response:\n{chr(10).join(current_search_urls)}"
                            ]
                            logger.info(f"Added {len(search_context_parts)} search results to context")
                else:
                    logger.info("Web search is disabled")
    
    final_system_parts = system_content_parts + current_search_parts
    
    if final_system_parts:
        final_system_content = "\n\n".join(final_system_parts)
        processed_messages.insert(0, Message(
            role=Role.SYSTEM,
            content=final_system_content
        ))
    
    return processed_messages, current_search_urls