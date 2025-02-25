import logging
import datetime
import httpx
from typing import List, Tuple
from fastapi import HTTPException
import asyncio

from app.models.chat import Message, Role, SearchAnalysis, ContentType
from app.config.settings import (
    OPENAI_API_URL, OPENAI_API_KEY,
    API_TIMEOUT, PROGRESS_MESSAGES,
    ENABLE_PROGRESS_MESSAGES, ENABLE_IMAGE_ANALYSIS,
    ENABLE_WEB_SEARCH, ENABLE_DOCUMENT_ANALYSIS
)
from app.services.search_service import SearchService
from app.services.image_service import process_image
from app.services.document_service import DocumentService
from app.utils.file_utils import detect_file_type

logger = logging.getLogger(__name__)

# Initialize services
document_service = DocumentService()
search_service = SearchService()

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
                    "model": "gpt-4o-mini",
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
    last_file_content = None
    
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    system_content_parts.append(f"Current date: {current_date}")
    
    last_file_url = None
    last_file_type = None
    last_file_message = None
    
    for message in reversed(messages):
        if isinstance(message.content, list):
            for content in reversed(message.content):
                if content.type == ContentType.IMAGE_URL and content.image_url:
                    file_url = content.image_url.url
                    logger.info(f"\nFound file URL: {file_url}")
                    
                    # Check if it's base64 data
                    file_type, is_supported = detect_file_type(file_url, "")
                    
                    # Skip processing based on feature flags
                    if file_type == 'image':
                        if not ENABLE_IMAGE_ANALYSIS:
                            continue
                        # Send progress message for image
                        await send_progress("image_analysis")
                    else:  # document type
                        if not ENABLE_DOCUMENT_ANALYSIS:
                            continue
                        # For documents, send document progress message
                        await send_progress("document_analysis")
                    
                    if file_type == 'image' and is_supported:
                        last_file_url = file_url
                        last_file_type = file_type
                        last_file_message = message
                        logger.info(f"✓ Selected file for processing: {file_type} (base64 data)")
                        break
                    
                    # If not base64, check URL as before
                    async with httpx.AsyncClient() as client:
                        try:
                            response = await client.head(file_url)
                            content_type = response.headers.get("content-type", "")
                            content_disposition = response.headers.get("content-disposition", "")
                            
                            # Update progress message based on content type
                            if 'image' in content_type.lower():
                                await send_progress("image_analysis")
                            
                            file_type, is_supported = detect_file_type(
                                file_url, 
                                content_type,
                                content_disposition
                            )
                            
                            if is_supported:
                                last_file_url = file_url
                                last_file_type = file_type
                                last_file_message = message
                                logger.info(f"✓ Selected file for processing: {file_type} at {file_url}")
                                break
                            else:
                                logger.warning(f"⚠ Unsupported file format: {file_type} at {file_url}")
                        except Exception as e:
                            logger.error(f"Error checking file type: {str(e)}")
                            continue
            if last_file_url:
                break
    
    # Process the last file if found
    if last_file_url:
        try:
            if last_file_type == 'image':
                if ENABLE_IMAGE_ANALYSIS:
                    logger.info("Processing as IMAGE")
                    last_file_content = await process_image(last_file_url)
                    logger.info("Image processing completed")
                else:
                    logger.info("Image analysis is disabled, skipping processing")
                    last_file_content = None
            else:
                if ENABLE_DOCUMENT_ANALYSIS:
                    logger.info("Processing as DOCUMENT")
                    documents = await document_service.process_document(last_file_url)
                    if documents:
                        last_file_content = "Document processed."
                    else:
                        logger.warning("Document processing yielded no content")
                        last_file_content = None
                else:
                    last_file_content = "Document analysis is disabled."
        except Exception as e:
            logger.error(f"Error processing file {last_file_url}: {str(e)}")
            last_file_content = None
            if stream and progress_queue:
                await progress_queue.put("\nFailed to process file. Please try again with a different file format.\n")
    
    # Find the last user message for web search
    last_user_message = None
    for message in reversed(messages):
        if message.role == Role.USER:
            if isinstance(message.content, list):
                for content in message.content:
                    if content.type == ContentType.TEXT:
                        last_user_message = content.text
                        break
            else:
                last_user_message = message.content
            if last_user_message:
                break
    
    for message in messages:
        if message.role == Role.SYSTEM:
            if not any(keyword in message.content.lower() for keyword in ["search results:", "references:", "[1]:", "[2]:", "[3]:"]):
                system_content_parts.append(message.content)
                logger.info(f"Added system message: {message.content[:100]}...")
            continue
            
        if isinstance(message.content, list):
            text_parts = []
            query_text = ""
            
            for content in message.content:
                if content.type == ContentType.TEXT:
                    query_text = content.text
                    text_parts.append(query_text)
                    logger.info(f"Added text content: {query_text[:100]}...")
                elif content.type == ContentType.IMAGE_URL and content.image_url:
                    if content.image_url.url == last_file_url and message == last_file_message:
                        if last_file_type == 'document':
                            if not ENABLE_DOCUMENT_ANALYSIS:
                                logger.info("Document analysis is disabled, skipping content search")
                                continue
                            if query_text:
                                logger.info("Searching document for relevant content...")
                                await send_progress("document_search")
                                relevant_chunks = await document_service.search_similar(query_text, source_url=last_file_url)
                                if relevant_chunks:
                                    chunks_text = "\n\n".join(
                                        f"Relevant section {i+1}:\n{chunk.page_content}"
                                        for i, chunk in enumerate(relevant_chunks)
                                    )
                                    text_parts.append(f"[Document Content:\n{chunks_text}]")
                                    logger.info(f"Added {len(relevant_chunks)} relevant document sections")
                                else:
                                    text_parts.append("[No relevant content found in document]")
                        elif last_file_type == 'image' and ENABLE_IMAGE_ANALYSIS and last_file_content:
                            text_parts.append(f"[File Content: {last_file_content}]")
                            logger.info("Added file content to message")
            
            # Only perform web search if this is the last user message
            if query_text and query_text == last_user_message:
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
            # Only perform web search if this is the last user message
            if message.role == Role.USER and message.content == last_user_message:
                await send_progress("web_search")
                
                if ENABLE_WEB_SEARCH:
                    search_analysis = await analyze_search_need(message.content)
                    if search_analysis.needs_search and search_analysis.search_keywords:
                        
                        search_results = await search_service.search(message.content)
                        
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
            processed_messages.append(message)
    
    final_system_parts = system_content_parts + current_search_parts
    
    if final_system_parts:
        final_system_content = "\n\n".join(final_system_parts)
        processed_messages.insert(0, Message(
            role=Role.SYSTEM,
            content=final_system_content
        ))
  
    
    return processed_messages, current_search_urls