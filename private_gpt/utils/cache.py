"""Cache utilities for PrivateGPT.

This module provides caching functionality for embeddings and document chunks
to improve performance and reduce redundant computations.
"""

import asyncio
import logging
import time
import weakref
from collections import OrderedDict
from typing import Any, Optional, Dict, Union
from functools import wraps

logger = logging.getLogger(__name__)


class LRUCache:
    """Simple LRU Cache implementation."""
    
    def __init__(self, maxsize: int = 128):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.maxsize:
            # Remove least recently used item
            self.cache.popitem(last=False)
        self.cache[key] = value

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()


class AsyncCache:
    """Async cache with TTL support."""
    
    def __init__(self, maxsize: int = 1000, ttl: int = 3600):
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache: Dict[str, tuple] = {}  # key -> (value, expiry_time)
        self.access_order = OrderedDict()  # for LRU tracking
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            current_time = time.time()
            if key in self.cache:
                value, expiry_time = self.cache[key]
                if current_time < expiry_time:
                    # Move to end (most recently used)
                    self.access_order.move_to_end(key)
                    return value
                else:
                    # Expired, remove it
                    del self.cache[key]
                    if key in self.access_order:
                        del self.access_order[key]
            return None

    async def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        async with self._lock:
            current_time = time.time()
            expiry_time = current_time + self.ttl
            
            if key in self.cache:
                # Update existing entry
                self.cache[key] = (value, expiry_time)
                self.access_order.move_to_end(key)
            else:
                # Add new entry
                if len(self.cache) >= self.maxsize:
                    # Remove least recently used item
                    lru_key, _ = self.access_order.popitem(last=False)
                    del self.cache[lru_key]
                
                self.cache[key] = (value, expiry_time)
                self.access_order[key] = None

    async def clear(self) -> None:
        """Clear the cache."""
        async with self._lock:
            self.cache.clear()
            self.access_order.clear()


# Global cache instances
_embedding_cache = AsyncCache(maxsize=1000, ttl=3600)  # 1 hour TTL
_chunk_cache = AsyncCache(maxsize=5000, ttl=1800)      # 30 min TTL
_document_cache = LRUCache(maxsize=100)                 # Simple LRU for documents

# Cache for frequently accessed data
_frequent_cache = weakref.WeakValueDictionary()


def cache_embedding(func):
    """Decorator to cache embedding results."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Create cache key from arguments
        cache_key = f"embedding:{hash(str(args) + str(kwargs))}"
        
        # Try to get from cache first
        cached_result = await _embedding_cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for embedding: {cache_key}")
            return cached_result
            
        # If not in cache, compute and store
        logger.debug(f"Cache miss for embedding: {cache_key}")
        result = await func(*args, **kwargs)
        await _embedding_cache.put(cache_key, result)
        return result
    return wrapper


def cache_chunk(func):
    """Decorator to cache chunk results."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Create cache key from arguments
        cache_key = f"chunk:{hash(str(args) + str(kwargs))}"
        
        # Try to get from cache first
        cached_result = await _chunk_cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for chunk: {cache_key}")
            return cached_result
            
        # If not in cache, compute and store
        logger.debug(f"Cache miss for chunk: {cache_key}")
        result = await func(*args, **kwargs)
        await _chunk_cache.put(cache_key, result)
        return result
    return wrapper


def get_embedding_cache() -> AsyncCache:
    """Get the global embedding cache instance."""
    return _embedding_cache


def get_chunk_cache() -> AsyncCache:
    """Get the global chunk cache instance."""
    return _chunk_cache


def get_document_cache() -> LRUCache:
    """Get the global document cache instance."""
    return _document_cache