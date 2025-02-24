import json
import logging
import asyncio
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from app.models.chat import ChatRequest
from app.services.message_service import process_messages
from app.config.settings import (
    TARGET_MODEL_BASE_URL, TARGET_MODEL_API_KEY,
    DEFAULT_MODEL
)
from app.core.auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize OpenAI client
client = AsyncOpenAI(
    base_url=f"{TARGET_MODEL_BASE_URL}/v1",
    api_key=TARGET_MODEL_API_KEY
)

async def stream_progress_message(message: str):
    """Stream a progress message"""
    chunk_data = {
        "id": "progress_msg",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": DEFAULT_MODEL,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": message},
            "finish_reason": None
        }]
    }
    yield f"data: {json.dumps(chunk_data)}\n\n"

async def stream_response(response):
    """Stream the response from the upstream API"""
    try:
        buffer = ""
        async for line in response.aiter_lines():
            if not line.strip():
                continue
                
            buffer += line
            if buffer.strip() == 'data: [DONE]':
                yield 'data: [DONE]\n\n'
                break
                
            try:
                if buffer.startswith('data: '):
                    json.loads(buffer[6:]) 
                    yield f"{buffer}\n\n"
                    buffer = ""
            except json.JSONDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error parsing stream chunk: {str(e)}")
                continue
    except Exception as e:
        logger.error(f"Error in stream_response: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@router.post("/v1/chat/completions")
async def chat(
    request: ChatRequest,
    raw_request: Request,
    api_key: str = Depends(verify_api_key)
):
    """Handle chat requests with support for streaming and image processing"""
    if not TARGET_MODEL_API_KEY:
        raise HTTPException(status_code=500, detail="TARGET_MODEL_API_KEY not configured")
    
    try:
        logger.info("Processing chat request...")
        
        if request.stream:
            # Create a queue for progress messages
            progress_queue = asyncio.Queue()
            
            # Process messages in background task
            async def process_messages_task():
                try:
                    processed_messages, _ = await process_messages(
                        request.messages,
                        stream=True,
                        progress_queue=progress_queue
                    )
                    # Convert messages for API request
                    api_messages = [{"role": msg.role, "content": msg.content} for msg in processed_messages]
                    
                    # Create streaming completion
                    stream = await client.chat.completions.create(
                        messages=api_messages,
                        model=DEFAULT_MODEL,
                        stream=True
                    )
                    return stream
                except Exception as e:
                    logger.error(f"Error in process_messages_task: {str(e)}")
                    raise

            # Start processing task
            processing_task = asyncio.create_task(process_messages_task())
            
            async def generate():
                progress_task = None
                current_progress_message = None
                progress_generator = None
                had_progress_message = False
                
                try:
                    while True:
                        # Check for new progress message
                        try:
                            progress_message = progress_queue.get_nowait()
                            if had_progress_message:
                                chunk_data = {
                                    "id": "progress_msg",
                                    "object": "chat.completion.chunk",
                                    "created": 0,
                                    "model": DEFAULT_MODEL,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"role": "assistant", "content": "\n"},
                                        "finish_reason": None
                                    }]
                                }
                                yield f"data: {json.dumps(chunk_data)}\n\n"
                            progress_generator = stream_progress_message(progress_message)
                            current_progress_message = progress_message
                            yield await anext(progress_generator)
                            had_progress_message = True
                        except asyncio.QueueEmpty:
                            pass
                        except StopAsyncIteration:
                            pass
                        
                        if processing_task.done():
                            break
                            
                        await asyncio.sleep(0.1)
                    
                    stream = await processing_task
                    
                    if had_progress_message:
                        chunk_data = {
                            "id": "progress_msg",
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": DEFAULT_MODEL,
                            "choices": [{
                                "index": 0,
                                "delta": {"role": "assistant", "content": "\n"},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                    
                    last_content_chunk = None
                    async for chunk in stream:
                        chunk_dict = chunk.model_dump()
                        
                        if chunk.choices[0].delta.content is not None:
                            if last_content_chunk:
                                yield f"data: {json.dumps(last_content_chunk)}\n\n"
                            last_content_chunk = chunk_dict
                        else:
                            if last_content_chunk:
                                last_content_chunk["choices"][0]["finish_reason"] = "stop"
                                yield f"data: {json.dumps(last_content_chunk)}\n\n"
                                last_content_chunk = None
                            if not (chunk.choices[0].finish_reason == "stop" and not chunk.choices[0].delta.content):
                                yield f"data: {json.dumps(chunk_dict)}\n\n"
                    
                    if last_content_chunk:
                        last_content_chunk["choices"][0]["finish_reason"] = "stop"
                        yield f"data: {json.dumps(last_content_chunk)}\n\n"
                    
                    yield "data: [DONE]\n\n"
                    
                except Exception as e:
                    logger.error(f"Error in generate: {str(e)}")
                    if progress_task and not progress_task.done():
                        progress_task.cancel()
                    raise

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream",
                    "Transfer-Encoding": "chunked",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            processed_messages, _ = await process_messages(request.messages)
            
            response = await client.chat.completions.create(
                messages=[{"role": msg.role, "content": msg.content} for msg in processed_messages],
                model=DEFAULT_MODEL,
                stream=False
            )
            return response.model_dump()
            
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 