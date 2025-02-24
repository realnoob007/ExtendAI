import httpx
import asyncio
import logging
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from googlesearch import search
from typing import List, Optional, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_postgres import PGVector
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from app.config.settings import (
    PROXY_ENABLED, PROXY_HOST, PROXY_PORT, PROXY_USERNAME,
    PROXY_PASSWORD, SEARCH_RESULT_LIMIT, SEARCH_RESULT_MULTIPLIER,
    WEB_CONTENT_CHUNK_SIZE, WEB_CONTENT_CHUNK_OVERLAP,
    EMBEDDING_MODEL, EMBEDDING_BASE_URL, EMBEDDING_API_KEY, EMBEDDING_DIMENSIONS,
    VECTOR_STORE_TYPE, POSTGRES_CONNECTION_STRING, POSTGRES_COLLECTION_NAME,
    PINECONE_API_KEY, PINECONE_INDEX_NAME
)
from app.models.chat import SearchResult

logger = logging.getLogger(__name__)

def get_proxy_url() -> str:
    """Generate basic proxy URL"""
    return f"http://{PROXY_USERNAME}:{PROXY_PASSWORD}@{PROXY_HOST}:{PROXY_PORT}"

class SearchService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=WEB_CONTENT_CHUNK_SIZE,
            chunk_overlap=WEB_CONTENT_CHUNK_OVERLAP,
            add_start_index=True,
            separators=["\n\n", "\n", "。", ". ", " ", ""],
            length_function=len,
            is_separator_regex=False
        )
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=EMBEDDING_API_KEY,
            openai_api_base=EMBEDDING_BASE_URL,
            dimensions=EMBEDDING_DIMENSIONS,
            request_timeout=60.0,
            show_progress_bar=True,
            retry_min_seconds=1,
            retry_max_seconds=60,
            max_retries=3,
            skip_empty=True,
        )
        # Initialize vector store
        self.vector_store = None
        self._init_vector_store()
    
    def _init_vector_store(self):
        """Initialize vector store based on configuration"""
        try:
            if VECTOR_STORE_TYPE == "postgres" and POSTGRES_CONNECTION_STRING:
                logger.info("Initializing PostgreSQL vector store")
                self.vector_store = PGVector(
                    connection=POSTGRES_CONNECTION_STRING,
                    collection_name=f"{POSTGRES_COLLECTION_NAME}_web_search",
                    embeddings=self.embeddings
                )
            elif VECTOR_STORE_TYPE == "pinecone":
                logger.info("Initializing Pinecone vector store")
                pc = Pinecone(api_key=PINECONE_API_KEY)
                index = pc.Index(PINECONE_INDEX_NAME)
                self.vector_store = PineconeVectorStore(
                    embedding=self.embeddings,
                    index=index,
                    namespace="web_search"
                )
            else:
                logger.info("Initializing FAISS vector store")
                self.vector_store = FAISS.from_texts(
                    ["placeholder"],  # Need at least one document to initialize
                    self.embeddings,
                    metadatas=[{"placeholder": True}]
                )
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            logger.warning("Falling back to FAISS vector store")
            self.vector_store = FAISS.from_texts(
                ["placeholder"],
                self.embeddings,
                metadatas=[{"placeholder": True}]
            )

    async def _get_proxy_config(self) -> Optional[str]:
        """Get proxy configuration if enabled"""
        if not PROXY_ENABLED or not all([PROXY_HOST, PROXY_PORT, PROXY_USERNAME, PROXY_PASSWORD]):
            return None
        
        return f"http://{PROXY_USERNAME}:{PROXY_PASSWORD}@{PROXY_HOST}:{PROXY_PORT}"

    async def _fetch_page_content(self, url: str) -> str:
        """Fetch and extract text content from a webpage"""
        try:
            proxy = await self._get_proxy_config()
            async with httpx.AsyncClient(proxy=proxy, timeout=30.0) as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove unwanted elements
                for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'iframe', 'aside', 'form']):
                    element.decompose()
                
                # First try to find main content area
                main_content = ""
                content_tags = soup.find_all(['article', 'main', 'div'], 
                    class_=lambda x: x and any(c in str(x).lower() for c in ['content', 'article', 'post', 'entry', 'main', 'text']))
                
                if content_tags:
                    # Get the tag with most text content
                    main_tag = max(content_tags, key=lambda x: len(x.get_text().strip()))
                    
                    # Extract paragraphs from main content area
                    paragraphs = []
                    for p in main_tag.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                        text = p.get_text().strip()
                        if len(text) > 20:  # Skip very short paragraphs
                            paragraphs.append(text)
                    
                    main_content = '\n'.join(paragraphs)
                
                # Fallback to all paragraphs if no main content found
                if not main_content:
                    paragraphs = []
                    for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                        text = p.get_text().strip()
                        if len(text) > 20:
                            paragraphs.append(text)
                    main_content = '\n'.join(paragraphs)
                
                return main_content.strip()
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {str(e)}")
            return ""

    async def search(self, query: str) -> List[Dict[str, Any]]:
        """Perform web search and return relevant content chunks"""
        try:
            # Get more initial results to account for failures
            raw_limit = SEARCH_RESULT_LIMIT * SEARCH_RESULT_MULTIPLIER
            
            # Configure proxy for Google search
            proxy_url = get_proxy_url() if PROXY_ENABLED else None
            
            # Get search results using googlesearch
            search_urls = []
            try:
                results = search(
                    query,
                    num_results=raw_limit,
                    proxy=proxy_url,
                    ssl_verify=False  # Skip SSL verification for proxy
                )
                search_urls = list(results)  # Convert generator to list
                logger.info(f"Found {len(search_urls)} results from Google search")
            except Exception as e:
                logger.error(f"Google search error: {str(e)}")
                return []
            
            if not search_urls:
                return []
            
            # Deduplicate URLs while keeping order
            seen_urls = set()
            unique_results = []
            for url in search_urls:
                if not url or url in seen_urls:
                    continue
                try:
                    parsed = urlparse(url)
                    if not all([parsed.scheme, parsed.netloc]):
                        continue
                    if any(parsed.path.lower().endswith(ext) for ext in [
                        '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx',
                        '.zip', '.rar', '.7z', '.tar', '.gz', '.mp3', '.mp4', '.avi'
                    ]):
                        continue
                    seen_urls.add(url)
                    unique_results.append({
                        "url": url,
                        "title": url.split("/")[-1] or parsed.netloc,  # Simple title extraction
                    })
                except:
                    continue
            
            logger.info(f"Found {len(unique_results)} unique URLs to process")
            
            # Process URLs in batches until we have enough valid results
            valid_chunks = []
            batch_size = SEARCH_RESULT_LIMIT
            current_index = 0
            
            while current_index < len(unique_results) and len(valid_chunks) < SEARCH_RESULT_LIMIT:
                batch = unique_results[current_index:current_index + batch_size]
                current_index += batch_size
                
                # Fetch content from URLs in parallel
                proxies = await self._get_proxy_config()
                async with httpx.AsyncClient(proxy=proxies, timeout=30.0, verify=False) as client:
                    tasks = []
                    for result in batch:
                        if "url" in result:
                            tasks.append(self._fetch_page_content(result["url"]))
                    
                    # Wait for all tasks with timeout
                    try:
                        contents = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        # Process successful results
                        for result, content in zip(batch, contents):
                            if isinstance(content, Exception):
                                logger.error(f"Failed to fetch {result.get('url')}: {str(content)}")
                                continue
                                
                            if not content:
                                continue
                            
                            # Split content into chunks
                            chunks = self.text_splitter.split_text(content)
                            logger.info(f"Split content from {result.get('url')} into {len(chunks)} chunks")
                            
                            # Create documents with metadata
                            for chunk in chunks:
                                chunk = chunk.strip()
                                if not chunk or len(chunk) < 50:  # Skip very short chunks
                                    continue
                                doc = Document(
                                    page_content=chunk,
                                    metadata={
                                        "url": result.get("url", ""),
                                        "title": result.get("title", ""),
                                        "source": "web_search"
                                    }
                                )
                                valid_chunks.append(doc)
                                
                                # Break if we have enough chunks
                                if len(valid_chunks) >= SEARCH_RESULT_LIMIT * 2:  # Get extra for diversity
                                    break
                            
                            if len(valid_chunks) >= SEARCH_RESULT_LIMIT * 2:
                                break
                            
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout processing batch starting at index {current_index}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing batch: {str(e)}")
                        continue
            
            if not valid_chunks:
                return []
            
            logger.info(f"Total valid chunks collected: {len(valid_chunks)}")
            
            # Create new FAISS store for similarity search
            vector_store = FAISS.from_documents(valid_chunks, self.embeddings)
            
            # Search for similar chunks using MMR for diversity
            similar_chunks = vector_store.max_marginal_relevance_search(
                query,
                k=min(SEARCH_RESULT_LIMIT, len(valid_chunks)),
                fetch_k=min(SEARCH_RESULT_LIMIT * 2, len(valid_chunks)),
                lambda_mult=0.7
            )
            
            # Return relevant chunks with metadata
            relevant_chunks = []
            logger.info("\nMatched chunks content:")
            logger.info("="*80)
            
            # Get embeddings for scoring
            query_embedding = self.embeddings.embed_query(query)
            chunk_embeddings = self.embeddings.embed_documents([doc.page_content for doc in similar_chunks])
            
            from numpy import dot
            from numpy.linalg import norm
            
            for i, (doc, chunk_embedding) in enumerate(zip(similar_chunks, chunk_embeddings), 1):
                # Calculate cosine similarity
                similarity = dot(query_embedding, chunk_embedding)/(norm(query_embedding)*norm(chunk_embedding))
                
                logger.info(f"\nChunk [{i}] (similarity: {similarity:.3f})")
                logger.info(f"Source: {doc.metadata['title']} ({doc.metadata['url']})")
                logger.info(f"Content:\n{doc.page_content}")
                logger.info("-"*80)
                
                relevant_chunks.append({
                    "content": doc.page_content,
                    "url": doc.metadata["url"],
                    "title": doc.metadata["title"],
                    "similarity": float(similarity)
                })
            
            logger.info(f"\nSelected {len(relevant_chunks)} most relevant chunks")
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return [] 