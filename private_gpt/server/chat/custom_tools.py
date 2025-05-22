from typing import List, Dict, Any, Optional
from llama_index.core.tools import BaseTool
from llama_index.core.schema import NodeWithScore, Document
from llama_index.core.storage import StorageContext
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from private_gpt.open_ai.extensions.context_filter import ContextFilter
import requests
from bs4 import BeautifulSoup
import logging
from dataclasses import dataclass
import re
from urllib.parse import urlparse
import time
from functools import lru_cache, wraps
import asyncio
from aiohttp import ClientSession, ClientTimeout
from aiohttp.client_exceptions import ClientError

logger = logging.getLogger(__name__)

# Rate limiting configuration
RATE_LIMIT_DELAY = 1.0  # seconds between requests
MAX_CONCURRENT_REQUESTS = 3
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3

# URL validation patterns
ALLOWED_DOMAINS = {
    'example.com',
    'wikipedia.org',
    'github.com',
    # Add more allowed domains as needed
}

class RateLimiter:
    """Rate limiter for web requests."""
    def __init__(self, delay: float = RATE_LIMIT_DELAY):
        self.delay = delay
        self.last_request_time = 0
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire rate limit lock."""
        async with self._lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.delay:
                await asyncio.sleep(self.delay - time_since_last)
            self.last_request_time = time.time()

class WebRequestManager:
    """Manages web requests with rate limiting and retries."""
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self.session = None
    
    async def get_session(self) -> ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            timeout = ClientTimeout(total=REQUEST_TIMEOUT)
            self.session = ClientSession(timeout=timeout)
        return self.session
    
    async def close(self):
        """Close the session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL for security."""
        try:
            parsed = urlparse(url)
            return (
                parsed.scheme in {'http', 'https'} and
                parsed.netloc in ALLOWED_DOMAINS
            )
        except Exception:
            return False
    
    async def fetch_url(self, url: str) -> str:
        """Fetch URL content with rate limiting and retries."""
        if not self.validate_url(url):
            raise ValueError(f"URL not allowed: {url}")
        
        async with self.semaphore:
            for attempt in range(MAX_RETRIES):
                try:
                    await self.rate_limiter.acquire()
                    session = await self.get_session()
                    async with session.get(url) as response:
                        if response.status == 200:
                            return await response.text()
                        else:
                            raise ClientError(f"HTTP {response.status}")
                except Exception as e:
                    if attempt == MAX_RETRIES - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

# Global web request manager
web_manager = WebRequestManager()

# Cleanup function for web manager
async def cleanup_web_manager():
    """Cleanup web request manager resources."""
    await web_manager.close()

# Register cleanup
import atexit

def cleanup():
    """Synchronous cleanup wrapper."""
    loop = asyncio.get_event_loop()
    if loop.is_running():
        loop.create_task(cleanup_web_manager())
    else:
        loop.run_until_complete(cleanup_web_manager())

atexit.register(cleanup)

@dataclass
class DocumentStats:
    doc_id: str
    filename: str
    page_count: int
    word_count: int
    last_modified: str
    metadata: Dict[str, Any]

def with_retry(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying operations with exponential backoff."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s...")
                        await asyncio.sleep(wait_time)
            raise last_exception

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
            raise last_exception

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

class DocumentStatsTool(BaseTool):
    """Tool for retrieving document statistics."""
    
    def __init__(
        self,
        storage_context: StorageContext,
        context_filter: Optional[ContextFilter] = None
    ):
        self.storage_context = storage_context
        self.context_filter = context_filter
        self.name = "DocumentStatsTool"
        self.description = "Retrieves statistics about documents in the knowledge base, including metadata, page counts, and word counts."
        
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get tool metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": {
                "doc_ref": {
                    "type": "string",
                    "description": "Optional document reference (ID or filename or file_name) to get stats for a specific document"
                }
            }
        }
        
    def _find_document(self, doc_ref: Optional[str]) -> Optional[Document]:
        """Find a document by ID, filename, or other reference."""
        if not doc_ref:
            return None
            
        doc = self.storage_context.docstore.get_document(doc_ref)
        if doc:
            return doc
            
        docs = self.storage_context.docstore.docs
        if self.context_filter and self.context_filter.docs_ids:
            docs = {k: v for k, v in docs.items() if k in self.context_filter.docs_ids}
            
        # Try exact filename match
        for doc in docs.values():
            if doc.metadata.get("file_name", "").lower() == doc_ref.lower():
                return doc
                
        # Try partial filename match
        for doc in docs.values():
            filename = doc.metadata.get("file_name", "").lower()
            if doc_ref.lower() in filename:
                return doc
                
        return None
        
    def __call__(self, doc_ref: Optional[str] = None) -> str:
        """Get statistics about documents in the knowledge base."""
        try:
            if doc_ref:
                doc = self._find_document(doc_ref)
                if not doc:
                    return f"Document '{doc_ref}' not found. Please check the document reference and try again."
                    
                stats = DocumentStats(
                    doc_id=doc.doc_id,
                    filename=doc.metadata.get("file_name", "Unknown"),
                    page_count=doc.metadata.get("page_count", 0),
                    word_count=len(doc.text.split()),
                    last_modified=doc.metadata.get("last_modified", "Unknown"),
                    metadata=doc.metadata
                )
                return f"Document Statistics:\n{stats}"
            else:
                # Get stats for all documents or filtered documents
                docs = self.storage_context.docstore.docs
                if self.context_filter and self.context_filter.docs_ids:
                    docs = {k: v for k, v in docs.items() if k in self.context_filter.docs_ids}
                
                total_docs = len(docs)
                total_pages = sum(doc.metadata.get("page_count", 0) for doc in docs.values())
                total_words = sum(len(doc.text.split()) for doc in docs.values())
                
                return f"Knowledge Base Statistics:\nTotal Documents: {total_docs}\nTotal Pages: {total_pages}\nTotal Words: {total_words}"
                
        except Exception as e:
            logger.error(f"Error getting document stats: {e}")
            return f"Error retrieving document statistics: {str(e)}"
    
    @with_retry(max_retries=3)
    async def acall(self, doc_ref: Optional[str] = None) -> str:
        """Async version of document statistics retrieval."""
        try:
            if doc_ref:
                doc = self._find_document(doc_ref)
                if not doc:
                    return f"Document '{doc_ref}' not found. Please check the document reference and try again."
                    
                stats = DocumentStats(
                    doc_id=doc.doc_id,
                    filename=doc.metadata.get("file_name", "Unknown"),
                    page_count=doc.metadata.get("page_count", 0),
                    word_count=len(doc.text.split()),
                    last_modified=doc.metadata.get("last_modified", "Unknown"),
                    metadata=doc.metadata
                )
                return f"Document Statistics:\n{stats}"
            else:
                docs = self.storage_context.docstore.docs
                if self.context_filter and self.context_filter.docs_ids:
                    docs = {k: v for k, v in docs.items() if k in self.context_filter.docs_ids}
                
                total_docs = len(docs)
                total_pages = sum(doc.metadata.get("page_count", 0) for doc in docs.values())
                total_words = sum(len(doc.text.split()) for doc in docs.values())
                
                return f"Knowledge Base Statistics:\nTotal Documents: {total_docs}\nTotal Pages: {total_pages}\nTotal Words: {total_words}"
                
        except Exception as e:
            logger.error(f"Error getting document stats: {e}")
            return f"Error retrieving document statistics: {str(e)}"

class DocumentSummaryTool(BaseTool):
    """Tool for generating document summaries."""
    
    def __init__(
        self,
        storage_context: StorageContext,
        llm: Any,
        context_filter: Optional[ContextFilter] = None
    ):
        self.storage_context = storage_context
        self.llm = llm
        self.context_filter = context_filter
        self.name = "DocumentSummaryTool"
        self.description = "Generates concise summaries of documents in the knowledge base."
        
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get tool metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": {
                "doc_ref": {
                    "type": "string",
                    "description": "Document reference (ID or filename) to generate a summary for"
                }
            }
        }
        
    def _find_document(self, doc_ref: Optional[str]) -> Optional[Document]:
        """Find a document by ID, filename, or other reference."""
        if not doc_ref:
            return None
            
        # First try direct document ID lookup
        doc = self.storage_context.docstore.get_document(doc_ref)
        if doc:
            return doc
            
        # If not found, try to find by filename
        docs = self.storage_context.docstore.docs
        if self.context_filter and self.context_filter.docs_ids:
            docs = {k: v for k, v in docs.items() if k in self.context_filter.docs_ids}
            
        # Try exact filename match
        for doc in docs.values():
            if doc.metadata.get("file_name", "").lower() == doc_ref.lower():
                return doc
                
        # Try partial filename match
        for doc in docs.values():
            filename = doc.metadata.get("file_name", "").lower()
            if doc_ref.lower() in filename:
                return doc
                
        return None
        
    def __call__(self, doc_ref: str) -> str:
        """Generate a summary of the specified document."""
        try:
            # Find document by reference
            doc = self._find_document(doc_ref)
            if not doc:
                return f"Document '{doc_ref}' not found. Please check the document reference and try again."
            
            response_synthesizer = get_response_synthesizer(
                response_mode="tree_summarize",
                llm=self.llm
            )
            
            summary = response_synthesizer.synthesize(
                query="Summarize this document concisely",
                nodes=[doc]
            )
            
            return f"Document Summary for {doc.metadata.get('file_name', doc_ref)}:\n{summary.response}"
            
        except Exception as e:
            logger.error(f"Error generating document summary: {e}")
            return f"Error generating summary: {str(e)}"
    
    @with_retry(max_retries=3)
    async def acall(self, doc_ref: str) -> str:
        """Async version of document summary generation."""
        try:
            # Find document by reference
            doc = self._find_document(doc_ref)
            if not doc:
                return f"Document '{doc_ref}' not found. Please check the document reference and try again."
            
            response_synthesizer = get_response_synthesizer(
                response_mode="tree_summarize",
                llm=self.llm
            )
            
            summary = await response_synthesizer.asynthesize(
                query="Summarize this document concisely",
                nodes=[doc]
            )
            
            return f"Document Summary for {doc.metadata.get('file_name', doc_ref)}:\n{summary.response}"
            
        except Exception as e:
            logger.error(f"Error generating document summary: {e}")
            return f"Error generating summary: {str(e)}"

class RetrievalTool(BaseTool):
    """Tool for retrieving relevant documents from the vector store."""
    
    def __init__(
        self,
        retriever: BaseRetriever,
        node_postprocessors: Optional[List[Any]] = None
    ):
        self.retriever = retriever
        self.node_postprocessors = node_postprocessors or []
        self.name = "RetrievalTool"
        self.description = "Retrieves relevant documents from the vector store based on semantic similarity to the query."
        
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get tool metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": {
                "query": {
                    "type": "string",
                    "description": "The query to search for relevant documents"
                }
            }
        }
        
    def __call__(self, query: str) -> str:
        """Retrieve relevant documents for a query."""
        try:
            nodes = self.retriever.retrieve(query)
            
            # Apply postprocessors
            for processor in self.node_postprocessors:
                nodes = processor.postprocess_nodes(nodes)
            
            # Format results
            results = []
            for node in nodes:
                doc_id = node.node.ref_doc_id
                filename = node.node.metadata.get("file_name", "Unknown")
                score = node.score
                text = node.node.get_content()
                
                results.append(
                    f"Document: {filename} (ID: {doc_id})\n"
                    f"Relevance Score: {score:.3f}\n"
                    f"Content: {text[:200]}...\n"
                )
            
            return "\n".join(results) if results else "No relevant documents found"
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return f"Error retrieving documents: {str(e)}"

    @with_retry(max_retries=3)
    async def acall(self, query: str) -> str:
        """Async version of document retrieval."""
        try:
            # Retrieve nodes
            nodes = await self.retriever.aretrieve(query)
            
            # Apply postprocessors
            for processor in self.node_postprocessors:
                nodes = processor.postprocess_nodes(nodes)
            
            # Format results
            results = []
            for node in nodes:
                doc_id = node.node.ref_doc_id
                filename = node.node.metadata.get("file_name", "Unknown")
                score = node.score
                text = node.node.get_content()
                
                results.append(
                    f"Document: {filename} (ID: {doc_id})\n"
                    f"Relevance Score: {score:.3f}\n"
                    f"Content: {text[:200]}...\n"
                )
            
            return "\n".join(results) if results else "No relevant documents found"
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return f"Error retrieving documents: {str(e)}"

class WebScrapingTool(BaseTool):
    """Tool for scraping content from web pages with security measures."""
    
    def __init__(self):
        self.name = "WebScrapingTool"
        self.description = "Extracts content from a specific webpage URL with security measures and rate limiting."
        
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get tool metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": {
                "url": {
                    "type": "string",
                    "description": "The URL of the webpage to scrape"
                }
            }
        }
    
    def __call__(self, url: str) -> str:
        """Extract content from a webpage."""
        try:
            if not WebRequestManager.validate_url(url):
                return f"URL not allowed: {url}"
            
            # Use synchronous requests for backward compatibility
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract main content
                title = soup.title.string if soup.title else "No title found"
                
                # Get main content with security measures
                main_content = []
                for elem in soup.find_all(['p', 'h1', 'h2', 'h3']):
                    text = elem.get_text().strip()
                    # Sanitize text
                    text = re.sub(r'<[^>]+>', '', text)  # Remove any HTML tags
                    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
                    if text:
                        main_content.append(text)
                
                content = "\n".join(main_content)
                
                return f"Title: {title}\n\nContent:\n{content[:2000]}..."
            else:
                return f"Failed to retrieve webpage. Status code: {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error scraping webpage: {e}", exc_info=True)
            return f"Error scraping webpage: {str(e)}"
            
    async def acall(self, url: str) -> str:
        """Async version of webpage content extraction."""
        try:
            if not WebRequestManager.validate_url(url):
                return f"URL not allowed: {url}"
            
            async with web_manager.semaphore:
                await web_manager.rate_limiter.acquire()
                session = await web_manager.get_session()
                
                async with session.get(url) as response:
                    if response.status == 200:
                        text = await response.text()
                        soup = BeautifulSoup(text, 'html.parser')
                        
                        title = soup.title.string if soup.title else "No title found"
                        main_content = []
                        
                        for elem in soup.find_all(['p', 'h1', 'h2', 'h3']):
                            text = elem.get_text().strip()
                            text = re.sub(r'<[^>]+>', '', text)
                            text = re.sub(r'\s+', ' ', text)
                            if text:
                                main_content.append(text)
                        
                        content = "\n".join(main_content)
                        return f"Title: {title}\n\nContent:\n{content[:2000]}..."
                    else:
                        return f"Failed to retrieve webpage. Status code: {response.status}"
                        
        except Exception as e:
            logger.error(f"Error scraping webpage: {e}", exc_info=True)
            return f"Error scraping webpage: {str(e)}"

class WebCrawlingTool(BaseTool):
    """Tool for crawling websites with security measures."""
    
    def __init__(self):
        self.name = "WebCrawlingTool"
        self.description = "Crawls a website and extracts content from multiple pages with security measures and rate limiting."
        
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get tool metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": {
                "url": {
                    "type": "string",
                    "description": "The starting URL of the website to crawl"
                },
                "max_pages": {
                    "type": "integer",
                    "description": "Maximum number of pages to crawl (default: 3)"
                }
            }
        }
    
    def __call__(self, url: str, max_pages: int = 3) -> str:
        """Crawl a website and extract content from multiple pages."""
        try:
            if not WebRequestManager.validate_url(url):
                return f"URL not allowed: {url}"
            
            visited_urls = set()
            urls_to_visit = {url}
            results = []
            
            while urls_to_visit and len(visited_urls) < max_pages:
                current_url = urls_to_visit.pop()
                if current_url in visited_urls:
                    continue
                
                if not WebRequestManager.validate_url(current_url):
                    continue
                    
                response = requests.get(current_url, timeout=REQUEST_TIMEOUT)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract content with security measures
                    title = soup.title.string if soup.title else "No title found"
                    content = []
                    for elem in soup.find_all(['p', 'h1', 'h2', 'h3']):
                        text = elem.get_text().strip()
                        # Sanitize text
                        text = re.sub(r'<[^>]+>', '', text)
                        text = re.sub(r'\s+', ' ', text)
                        if text:
                            content.append(text)
                    
                    results.append(
                        f"Page: {title}\n"
                        f"URL: {current_url}\n"
                        f"Content: {' '.join(content[:3])}...\n"
                    )
                    
                    # Find new links with security checks
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        if href.startswith('/'):
                            href = f"{url.rstrip('/')}{href}"
                        if href.startswith('http') and href not in visited_urls:
                            if WebRequestManager.validate_url(href):
                                urls_to_visit.add(href)
                    
                    visited_urls.add(current_url)
                    time.sleep(RATE_LIMIT_DELAY)  # Rate limiting
            
            return "\n".join(results) if results else "No content found"
            
        except Exception as e:
            logger.error(f"Error crawling website: {e}", exc_info=True)
            return f"Error crawling website: {str(e)}"
            
    async def acall(self, url: str, max_pages: int = 3) -> str:
        """Async version of website crawling."""
        try:
            if not WebRequestManager.validate_url(url):
                return f"URL not allowed: {url}"
            
            visited_urls = set()
            urls_to_visit = {url}
            results = []
            
            while urls_to_visit and len(visited_urls) < max_pages:
                current_url = urls_to_visit.pop()
                if current_url in visited_urls:
                    continue
                
                if not WebRequestManager.validate_url(current_url):
                    continue
                
                async with web_manager.semaphore:
                    await web_manager.rate_limiter.acquire()
                    session = await web_manager.get_session()
                    
                    async with session.get(current_url) as response:
                        if response.status == 200:
                            text = await response.text()
                            soup = BeautifulSoup(text, 'html.parser')
                            
                            title = soup.title.string if soup.title else "No title found"
                            content = []
                            
                            for elem in soup.find_all(['p', 'h1', 'h2', 'h3']):
                                text = elem.get_text().strip()
                                text = re.sub(r'<[^>]+>', '', text)
                                text = re.sub(r'\s+', ' ', text)
                                if text:
                                    content.append(text)
                            
                            results.append(
                                f"Page: {title}\n"
                                f"URL: {current_url}\n"
                                f"Content: {' '.join(content[:3])}...\n"
                            )
                            
                            # Find new links with security checks
                            for link in soup.find_all('a', href=True):
                                href = link['href']
                                if href.startswith('/'):
                                    href = f"{url.rstrip('/')}{href}"
                                if href.startswith('http') and href not in visited_urls:
                                    if WebRequestManager.validate_url(href):
                                        urls_to_visit.add(href)
                            
                            visited_urls.add(current_url)
            
            return "\n".join(results) if results else "No content found"
            
        except Exception as e:
            logger.error(f"Error crawling website: {e}", exc_info=True)
            return f"Error crawling website: {str(e)}" 