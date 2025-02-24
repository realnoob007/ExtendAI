import os
import logging
import tempfile
import httpx
import ftfy
import json
import time
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple
import faiss
from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    Docx2txtLoader,
    OutlookMessageLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredRSTLoader,
    UnstructuredXMLLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS
from langchain_postgres import PGVector
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_core.documents import Document as LangchainDocument
from fastapi import HTTPException
import asyncio

from app.models.chat import Document
from app.config.settings import (
    EMBEDDING_BASE_URL, EMBEDDING_API_KEY,
    EMBEDDING_MODEL, EMBEDDING_DIMENSIONS,
    CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNKS_PER_DOC,
    EMBEDDING_BATCH_SIZE, MIN_CHUNK_LENGTH,
    MAX_FILE_SIZE, VECTOR_CACHE_DIR, VECTOR_CACHE_TTL,
    VECTOR_STORE_TYPE, POSTGRES_CONNECTION_STRING,
    POSTGRES_COLLECTION_NAME, PINECONE_API_KEY,
    PINECONE_INDEX_NAME
)

logger = logging.getLogger(__name__)

# Suppress FAISS GPU warning since we're explicitly using CPU
faiss.get_num_gpus = lambda: 0

# Known source file extensions
KNOWN_SOURCE_EXT = [
    "go", "py", "java", "sh", "bat", "ps1", "cmd", "js", "ts", "css",
    "cpp", "hpp", "h", "c", "cs", "sql", "log", "ini", "pl", "pm",
    "r", "dart", "dockerfile", "env", "php", "hs", "hsc", "lua",
    "nginxconf", "conf", "m", "mm", "plsql", "perl", "rb", "rs",
    "db2", "scala", "bash", "swift", "vue", "svelte", "msg", "ex",
    "exs", "erl", "tsx", "jsx", "hs", "lhs"
]

class DocumentService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True,
            separators=["\n\n", "\n", ". ", " ", ""],
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
        self.vector_store: Optional[VectorStore] = None
        
        # Initialize vector store based on configuration
        logger.info(f"Initializing vector store with type: {VECTOR_STORE_TYPE}")
        
        if VECTOR_STORE_TYPE == "postgres":
            if not POSTGRES_CONNECTION_STRING:
                logger.error("PostgreSQL connection string is not configured")
                logger.warning("Falling back to FAISS vector store")
                self._init_faiss_store()
            else:
                try:
                    logger.info(f"Attempting to connect to PostgreSQL with collection: {POSTGRES_COLLECTION_NAME}")
                    self.vector_store = PGVector(
                        connection=POSTGRES_CONNECTION_STRING,
                        collection_name=POSTGRES_COLLECTION_NAME,
                        embeddings=self.embeddings,
                    )
                    logger.info(f"Successfully initialized PostgreSQL vector store with collection: {POSTGRES_COLLECTION_NAME}")
                except Exception as e:
                    logger.error(f"Failed to initialize PostgreSQL vector store: {str(e)}")
                    logger.warning("Falling back to FAISS vector store")
                    self._init_faiss_store()
        elif VECTOR_STORE_TYPE == "pinecone":
            try:
                pc = Pinecone(api_key=PINECONE_API_KEY)
                index = pc.Index(PINECONE_INDEX_NAME)
                self.vector_store = PineconeVectorStore(
                    embedding=self.embeddings,
                    index=index,
                    namespace="default"
                )
                logger.info(f"Initialized Pinecone vector store with index: {PINECONE_INDEX_NAME}")
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone vector store: {str(e)}")
                logger.warning("Falling back to FAISS vector store")
                self._init_faiss_store()
        else:
            logger.info("Using FAISS vector store as configured")
            self._init_faiss_store()
    
    def _init_faiss_store(self):
        """Initialize FAISS vector store and ensure cache directory exists"""
        if VECTOR_STORE_TYPE == "faiss":
            os.makedirs(VECTOR_CACHE_DIR, exist_ok=True)
            logger.info("Initialized FAISS vector store (CPU mode)")
    
    def _create_vector_store(self, documents: List[LangchainDocument]) -> VectorStore:
        """Create a new vector store instance"""
        try:
            if VECTOR_STORE_TYPE == "postgres" and POSTGRES_CONNECTION_STRING:
                store = PGVector.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    collection_name=POSTGRES_COLLECTION_NAME,
                    connection=POSTGRES_CONNECTION_STRING,
                )
                logger.info("Created new PostgreSQL vector store")
                return store
            elif VECTOR_STORE_TYPE == "pinecone":
                pc = Pinecone(api_key=PINECONE_API_KEY)
                index = pc.Index(PINECONE_INDEX_NAME)
                store = PineconeVectorStore.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    index=index,
                    namespace="default"
                )
                logger.info("Created new Pinecone vector store")
                return store
        except Exception as e:
            logger.error(f"Failed to create vector store: {str(e)}")
            logger.warning("Falling back to FAISS vector store")
        
        # Default or fallback to FAISS
        store = FAISS.from_documents(
            documents,
            self.embeddings,
            distance_strategy="COSINE"
        )
        logger.info("Created new FAISS vector store")
        return store

    def _add_to_vector_store(self, store: VectorStore, documents: List[LangchainDocument]):
        """Add documents to existing vector store"""
        try:
            if isinstance(store, PGVector):
                store.add_documents(documents)
                logger.info(f"Added {len(documents)} documents to PostgreSQL vector store")
            elif isinstance(store, PineconeVectorStore):
                store.add_documents(documents)
                logger.info(f"Added {len(documents)} documents to Pinecone vector store")
            elif isinstance(store, FAISS):
                store.add_documents(documents)
                logger.info(f"Added {len(documents)} documents to FAISS vector store")
            else:
                raise ValueError(f"Unsupported vector store type: {type(store)}")
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {str(e)}")
            raise

    def _get_cache_key(self, filename: str, file_size: int) -> str:
        """Generate cache key from filename and size"""
        return hashlib.md5(f"{filename}_{file_size}".encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Tuple[Path, Path]:
        """Get cache file paths for documents and vectors (FAISS only)"""
        docs_path = Path(VECTOR_CACHE_DIR) / f"{cache_key}_docs.json"
        vectors_path = Path(VECTOR_CACHE_DIR) / f"{cache_key}_vectors.faiss"
        return docs_path, vectors_path
    
    def _save_to_cache(self, cache_key: str, documents: List[Document], vector_store: VectorStore):
        """Save documents and vector store to cache (FAISS only)"""
        if not isinstance(vector_store, FAISS):
            return
            
        try:
            docs_path, vectors_path = self._get_cache_path(cache_key)
            
            # Save documents and timestamp
            docs_data = {
                "timestamp": time.time(),
                "documents": [doc.model_dump() for doc in documents]
            }
            with open(docs_path, 'w', encoding='utf-8') as f:
                json.dump(docs_data, f)
            logger.info(f"Saved document cache to {docs_path}")
            
            # Save FAISS index
            vector_store.save_local(str(vectors_path))
            logger.info(f"Saved FAISS vectors to {vectors_path}")
            
        except Exception as e:
            logger.error(f"Failed to save cache: {str(e)}")
    
    def _load_from_cache(self, cache_key: str) -> Tuple[Optional[List[Document]], Optional[VectorStore]]:
        """Load documents and vector store from cache if valid (FAISS only)"""
        try:
            docs_path, vectors_path = self._get_cache_path(cache_key)
            
            # Check if both cache files exist
            if not docs_path.exists() or not vectors_path.exists():
                return None, None
            
            # Load and validate documents cache
            try:
                with open(docs_path, 'r', encoding='utf-8') as f:
                    docs_data = json.load(f)
            except (PermissionError, json.JSONDecodeError) as e:
                logger.error(f"Failed to read document cache: {str(e)}")
                return None, None
            
            # Check if cache is expired
            if time.time() - docs_data["timestamp"] > VECTOR_CACHE_TTL:
                logger.info("Cache expired, will reprocess document")
                # Clean up expired cache files
                try:
                    if docs_path.exists():
                        os.unlink(docs_path)
                    if vectors_path.exists():
                        os.unlink(vectors_path)
                except PermissionError as e:
                    logger.warning(f"Failed to clean up expired cache: {str(e)}")
                return None, None
            
            # Restore documents
            try:
                documents = [Document.model_validate(doc) for doc in docs_data["documents"]]
            except Exception as e:
                logger.error(f"Failed to validate documents: {str(e)}")
                return None, None
            
            # Restore FAISS vector store
            try:
                vector_store = FAISS.load_local(
                    str(vectors_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True  # Allow since we control the cache
                )
                logger.info(f"Loaded FAISS vectors from {vectors_path}")
                return documents, vector_store
            except Exception as e:
                logger.error(f"Failed to load vector store: {str(e)}")
                # Clean up invalid cache
                try:
                    if docs_path.exists():
                        os.unlink(docs_path)
                    if vectors_path.exists():
                        os.unlink(vectors_path)
                except PermissionError as e:
                    logger.warning(f"Failed to clean up invalid cache: {str(e)}")
                return None, None
            
        except Exception as e:
            logger.error(f"Failed to load cache: {str(e)}")
            return None, None

    async def download_file(self, url: str) -> tuple[str, str, str]:
        """Download file from URL and return file path, name and content type"""
        async with httpx.AsyncClient() as client:
            try:
                # First do a HEAD request to check content length
                head_response = await client.head(url, follow_redirects=True)
                head_response.raise_for_status()
                
                # Check content length if available
                content_length = head_response.headers.get("content-length")
                if content_length:
                    file_size = int(content_length)
                    if file_size > MAX_FILE_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail=f"File size ({file_size} bytes) exceeds maximum allowed size ({MAX_FILE_SIZE} bytes)"
                        )
                
                # Proceed with download using streaming to enforce size limit
                async with client.stream("GET", url, follow_redirects=True) as response:
                    response.raise_for_status()
                    
                    # Get filename from URL or Content-Disposition
                    content_disposition = response.headers.get("content-disposition")
                    if content_disposition and "filename=" in content_disposition:
                        filename = content_disposition.split("filename=")[-1].strip('"')
                    else:
                        filename = url.split("/")[-1]
                    
                    content_type = response.headers.get("content-type", "")
                    
                    # Create temporary file with proper extension
                    ext = os.path.splitext(filename)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                        total_size = 0
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            total_size += len(chunk)
                            if total_size > MAX_FILE_SIZE:
                                # Clean up and raise error
                                temp_file.close()
                                os.unlink(temp_file.name)
                                raise HTTPException(
                                    status_code=413,
                                    detail=f"File size exceeds maximum allowed size ({MAX_FILE_SIZE} bytes)"
                                )
                            temp_file.write(chunk)
                        
                        return temp_file.name, filename, content_type
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error downloading file from {url}: {str(e)}")
                raise

    def _get_loader(self, filename: str, content_type: str, file_path: str):
        """Get appropriate document loader based on file type"""
        file_ext = filename.split(".")[-1].lower() if "." in filename else ""
        
        if file_ext == "pdf":
            return PyPDFLoader(file_path)
        elif file_ext == "csv":
            return CSVLoader(file_path)
        elif file_ext == "rst":
            return UnstructuredRSTLoader(file_path, mode="elements")
        elif file_ext == "xml":
            return UnstructuredXMLLoader(file_path)
        elif file_ext in ["htm", "html"]:
            return BSHTMLLoader(file_path, open_encoding="unicode_escape")
        elif file_ext == "md":
            return UnstructuredMarkdownLoader(file_path)
        elif content_type == "application/epub+zip":
            return UnstructuredEPubLoader(file_path)
        elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or file_ext == "docx":
            return Docx2txtLoader(file_path)
        elif content_type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"] or file_ext in ["xls", "xlsx"]:
            return UnstructuredExcelLoader(file_path)
        elif content_type in ["application/vnd.ms-powerpoint", "application/vnd.openxmlformats-officedocument.presentationml.presentation"] or file_ext in ["ppt", "pptx"]:
            return UnstructuredPowerPointLoader(file_path)
        elif file_ext == "msg":
            return OutlookMessageLoader(file_path)
        elif file_ext in KNOWN_SOURCE_EXT or (content_type and content_type.find("text/") >= 0):
            return TextLoader(file_path, autodetect_encoding=True)
        else:
            return TextLoader(file_path, autodetect_encoding=True)

    async def _process_chunk_batch(self, batch: List[LangchainDocument]) -> None:
        """Process a batch of document chunks asynchronously"""
        try:
            if not self.vector_store:
                logger.info("Initializing new vector store")
                self.vector_store = self._create_vector_store(batch)
            else:
                logger.info("Adding documents to existing vector store")
                self._add_to_vector_store(self.vector_store, batch)
        except Exception as e:
            logger.error(f"Failed to process batch: {str(e)}")
            raise

    async def process_document(self, url: str) -> List[Document]:
        """Process a document from URL and store it in vector store"""
        temp_file_path = None
        try:
            # Download file
            temp_file_path, filename, content_type = await self.download_file(url)
            file_size = os.path.getsize(temp_file_path)
            
            # Log file information
            logger.info("="*50)
            logger.info("File Type Detection:")
            logger.info(f"URL: {url}")
            logger.info(f"Content-Type: {content_type}")
            logger.info(f"Filename: {filename}")
            logger.info(f"File Size: {file_size} bytes")
            logger.info(f"File Extension: {os.path.splitext(filename)[1]}")
            logger.info("="*50)
            
            # For PostgreSQL, check if document already exists
            if VECTOR_STORE_TYPE == "postgres" and isinstance(self.vector_store, PGVector):
                try:
                    # Query by source URL and file size
                    existing_docs = self.vector_store.similarity_search(
                        "",
                        k=1000,
                        filter={
                            "source": url,
                            "file_size": file_size
                        }
                    )
                    if existing_docs:
                        logger.info(f"Found existing document in PostgreSQL: {filename}")
                        return [
                            Document(
                                page_content=doc.page_content,
                                metadata=doc.metadata,
                                id=f"{url}_{i}"
                            ) for i, doc in enumerate(existing_docs)
                        ]
                except Exception as e:
                    logger.error(f"Failed to query PostgreSQL: {str(e)}")
            # For FAISS, check file cache
            elif VECTOR_STORE_TYPE == "faiss":
                cache_key = self._get_cache_key(filename, file_size)
                cached_docs, cached_store = self._load_from_cache(cache_key)
                if cached_docs and cached_store:
                    logger.info("Using cached vectors from FAISS")
                    self.vector_store = cached_store
                    return cached_docs
            
            logger.info(f"Processing document: {filename} ({file_size} bytes)")
            
            # Get appropriate loader
            loader = self._get_loader(filename, content_type, temp_file_path)
            logger.info(f"Using loader: {loader.__class__.__name__}")
            
            # Load document
            docs = loader.load()
            logger.info(f"Loaded {len(docs)} document sections")
            
            # Fix text encoding and create metadata
            fixed_docs = []
            for doc in docs:
                if isinstance(doc.page_content, (str, bytes)):
                    content = ftfy.fix_text(str(doc.page_content))
                else:
                    content = str(doc.page_content)
                
                fixed_doc = LangchainDocument(
                    page_content=content,
                    metadata={
                        **doc.metadata,
                        "source": url,
                        "filename": filename,
                        "content_type": content_type,
                        "file_size": file_size  # Add file size to metadata for future lookups
                    }
                )
                fixed_docs.append(fixed_doc)
            
            # Split into chunks
            splits = self.text_splitter.split_documents(fixed_docs)
            logger.info(f"Split into {len(splits)} chunks")
            
            # Convert to our Document model first
            documents = []
            for i, split in enumerate(splits):
                doc = Document(
                    page_content=split.page_content,
                    metadata=split.metadata,
                    id=f"{url}_{i}"
                )
                documents.append(doc)
            
            # Try to update vector store if needed
            try:
                # Verify splits have content
                valid_splits = []
                for split in splits:
                    if not isinstance(split.page_content, str):
                        continue
                    content = split.page_content.strip()
                    if not content:
                        continue
                    if len(content) < MIN_CHUNK_LENGTH:
                        continue
                    valid_splits.append(split)
                
                logger.info(f"Found {len(valid_splits)} valid chunks for embedding")
                logger.info(f"Average chunk size: {sum(len(split.page_content) for split in valid_splits) / len(valid_splits) if valid_splits else 0:.0f} characters")
                
                # Process valid splits in parallel batches
                if valid_splits:
                    tasks = []
                    for i in range(0, len(valid_splits), EMBEDDING_BATCH_SIZE):
                        batch = valid_splits[i:i + EMBEDDING_BATCH_SIZE]
                        logger.info(f"Processing batch {i//EMBEDDING_BATCH_SIZE + 1} of {(len(valid_splits) + EMBEDDING_BATCH_SIZE - 1)//EMBEDDING_BATCH_SIZE} ({len(batch)} chunks)")
                        tasks.append(self._process_chunk_batch(batch))
                    
                    # Process all batches concurrently
                    await asyncio.gather(*tasks)
                    logger.info(f"Processed all {len(valid_splits)} chunks")
                else:
                    logger.warning("No valid content found for vector store update")
            except Exception as ve:
                logger.error(f"Vector store operation failed: {str(ve)}")
                # Continue without vector store update
                pass
            
            if documents:
                logger.info(f"Document processed into {len(documents)} sections")
                # Don't add unnecessary description
                last_file_description = None
            else:
                logger.warning("No valid content found for vector store update")
            
            # Only save to cache if using FAISS
            if VECTOR_STORE_TYPE == "faiss":
                cache_key = self._get_cache_key(filename, file_size)
                self._save_to_cache(cache_key, documents, self.vector_store)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing document from {url}: {str(e)}")
            raise
        finally:
            # Clean up temporary file
            if temp_file_path:
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {temp_file_path}: {str(e)}")
                    pass
    
    async def search_similar(self, query: str, source_url: str = None, k: int = None) -> List[Document]:
        """Search for similar documents"""
        try:
            if not self.vector_store:
                logger.info("Vector store not initialized")
                return []
            
            # Use configured max chunks if k is not specified
            if k is None:
                k = MAX_CHUNKS_PER_DOC
            
            try:    
                # Add source URL filter if provided
                filter_dict = {"source": source_url} if source_url else None
                
                results = self.vector_store.similarity_search(
                    query, 
                    k=k,
                    filter=filter_dict
                )
            except Exception as e:
                logger.error(f"Vector search failed: {str(e)}")
                return []
            
            return [
                Document(
                    page_content=doc.page_content,
                    metadata=doc.metadata,
                    id=f"{doc.metadata.get('source', '')}_{i}"
                ) for i, doc in enumerate(results)
            ]
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return [] 