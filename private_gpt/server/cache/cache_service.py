from typing import List, Optional, Dict, Any, TYPE_CHECKING
import logging
import json
import time
import numpy as np
from pydantic import BaseModel, Field
from injector import inject, singleton

from sqlalchemy import func

from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.cache.cache_component import CacheComponent, CacheStatus
from private_gpt.users.schemas import FAQ 

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CacheStats(BaseModel):
    """Model for cache statistics."""
    cache_status: str = Field(description="Current cache status")
    redis_connected: bool = Field(description="Whether Redis is connected")
    embedding_model_available: bool = Field(description="Whether embedding model is available")
    total_faqs: int = Field(default=0, description="Total number of FAQs in cache")
    cache_keys: int = Field(default=0, description="Total number of cache keys")
    embedding_dimensions: Optional[int] = Field(default=None, description="Embedding vector dimensions")
    error: Optional[str] = Field(default=None, description="Error message if any")


class FAQSearchResult(BaseModel):
    """Model for FAQ search results with similarity scores."""
    faq: FAQ = Field(description="The FAQ item")
    similarity: Optional[float] = Field(default=None, description="Similarity score")


@singleton
class CacheService:
    """Service for managing FAQ cache operations with graceful degradation."""
    
    @inject
    def __init__(
        self,
        cache_component: CacheComponent,
        embedding_component: EmbeddingComponent,
    ) -> None:
        """Initialize the cache service with required components."""
        self.cache_component = cache_component
        self.embedding_component = embedding_component
        self._embedding_dimensions = None  # Cache the embedding dimensions
        
        # Initialize the cache component with embedding model
        self._initialize_cache()
        
        logger.info("CacheService initialized")
    
    def _initialize_cache(self) -> None:
        """Initialize the cache component with embedding model."""
        try:
            embedding_model = self.embedding_component.embedding_model
            success = self.cache_component.initialize(embedding_model=embedding_model)
            
            if success:
                # Test embedding generation to get dimensions
                self._determine_embedding_dimensions()
                logger.info("Cache component initialized successfully")
            else:
                logger.warning("Cache component initialization failed, operating in degraded mode")
        except Exception as e:
            logger.error(f"Failed to initialize cache component: {str(e)}")

    def initialize_with_warming(self, db_session=None, warm_cache: bool = True, eviction_limit: int = 1000) -> bool:
        """Initialize the cache service and optionally warm it with frequently accessed FAQs.
        
        Args:
            db_session: Database session for fetching FAQs (required if warm_cache=True)
            warm_cache: Whether to warm the cache with frequently accessed FAQs
            eviction_limit: Maximum number of FAQs to keep in cache after warming
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        # Initialize the cache component first
        self._initialize_cache()
        
        if not self.is_connected:
            logger.warning("Cache not connected, skipping initialization with warming")
            return False
            
        # Warm cache with frequently accessed FAQs if requested
        if warm_cache and db_session:
            try:
                self.warm_cache_with_frequently_accessed_faqs(db_session, limit=eviction_limit)
            except Exception as e:
                logger.error(f"Failed to warm cache during initialization: {str(e)}")
        
        # Perform initial eviction to ensure we're within limits
        try:
            self.evict_least_recently_used_faqs(max_faqs=eviction_limit)
        except Exception as e:
            logger.error(f"Failed to perform initial cache eviction: {str(e)}")
            
        logger.info("CacheService initialized with warming and eviction policies")
        return True
    
    def _determine_embedding_dimensions(self) -> None:
        """Determine the embedding dimensions by generating a test embedding."""
        try:
            if self.embedding_component.embedding_model:
                test_embedding = self.embedding_component.embedding_model.get_text_embedding("test")
                self._embedding_dimensions = len(test_embedding)
                logger.info(f"Embedding dimensions determined: {self._embedding_dimensions}")
        except Exception as e:
            logger.warning(f"Failed to determine embedding dimensions: {str(e)}")
            self._embedding_dimensions = None
    
    def _get_embedding_safely(self, text: str) -> Optional[np.ndarray]:
        """Safely generate embedding and convert to consistent numpy array format."""
        try:
            if not self.embedding_component.embedding_model:
                return None
            
            # Get embedding from LlamaIndex
            embedding_result = self.embedding_component.embedding_model.get_text_embedding(text)
            
            # Convert to numpy array with consistent dtype
            embedding_array = np.array(embedding_result, dtype=np.float32)
            
            # Validate dimensions
            if self._embedding_dimensions is None:
                self._embedding_dimensions = len(embedding_array)
                logger.info(f"Setting embedding dimensions to: {self._embedding_dimensions}")
            elif len(embedding_array) != self._embedding_dimensions:
                logger.error(f"Embedding dimension mismatch: expected {self._embedding_dimensions}, got {len(embedding_array)}")
                return None
            
            return embedding_array
            
        except Exception as e:
            logger.error(f"Failed to generate embedding safely: {str(e)}")
            return None
    
    def _store_embedding_safely(self, redis_client, embedding_key: str, embedding: np.ndarray) -> bool:
        """Safely store embedding in Redis with metadata."""
        try:
            # Store the embedding as bytes
            redis_client.set(embedding_key, embedding.tobytes())
            
            # Store metadata about the embedding
            metadata_key = f"{embedding_key}:meta"
            metadata = {
                "dimensions": len(embedding),
                "dtype": str(embedding.dtype)
            }
            redis_client.set(metadata_key, json.dumps(metadata))
            
            return True
        except Exception as e:
            logger.error(f"Failed to store embedding safely: {str(e)}")
            return False
    
    def _load_embedding_safely(self, redis_client, embedding_key: str) -> Optional[np.ndarray]:
        """Safely load embedding from Redis with validation."""
        try:
            # Load embedding data
            embedding_data = redis_client.get(embedding_key)
            if not embedding_data:
                return None
            
            # Load metadata
            metadata_key = f"{embedding_key}:meta"
            metadata_data = redis_client.get(metadata_key)
            
            if metadata_data:
                metadata = json.loads(metadata_data.decode('utf-8'))
                expected_dimensions = metadata.get("dimensions")
                dtype = metadata.get("dtype", "float32")
            else:
                # Fallback: assume float32 and current embedding dimensions
                expected_dimensions = self._embedding_dimensions
                dtype = "float32"
            
            # Convert bytes back to numpy array
            embedding = np.frombuffer(embedding_data, dtype=np.float32)
            
            # Validate dimensions
            if expected_dimensions and len(embedding) != expected_dimensions:
                logger.warning(f"Stored embedding dimension mismatch: expected {expected_dimensions}, got {len(embedding)}")
                # Try to determine if we can still use it
                if self._embedding_dimensions and len(embedding) != self._embedding_dimensions:
                    logger.error(f"Cannot use stored embedding: dimension {len(embedding)} != current model dimension {self._embedding_dimensions}")
                    return None
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to load embedding safely: {str(e)}")
            return None

    @property
    def is_connected(self) -> bool:
        """Check if the cache is properly connected."""
        return self.cache_component.status == CacheStatus.CONNECTED
    
    @property
    def cache_status(self) -> CacheStatus:
        """Get current cache status."""
        return self.cache_component.status
    
    def add_faq(self, faq: FAQ) -> bool:
        """Add a FAQ to the cache.
        
        Args:
            faq: The FAQ item to add
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.debug(f"Adding FAQ {faq.id} to cache")
        
        if not self.is_connected:
            logger.warning("Cache not connected, FAQ will not be cached")
            return False
        
        try:
            return self._store_faq_in_cache(faq)
        except Exception as e:
            logger.error(f"Failed to add FAQ {faq.id} to cache: {str(e)}")
            return False
    
    def get_faq(self, faq_id: int) -> Optional[FAQ]:
        """Get a FAQ from the cache by ID.
        
        Args:
            faq_id: The ID of the FAQ to retrieve
            
        Returns:
            FAQ or None if not found or cache unavailable
        """
        if not self.is_connected:
            logger.debug("Cache not connected, cannot retrieve FAQ")
            return None
        
        try:
            return self._retrieve_faq_from_cache(faq_id)
        except Exception as e:
            logger.error(f"Failed to get FAQ {faq_id} from cache: {str(e)}")
            return None
    
    def update_faq(self, faq: FAQ) -> bool:
        """Update a FAQ in the cache.
        
        Args:
            faq: The updated FAQ item
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.debug(f"Updating FAQ {faq.id} in cache")
        
        if not self.is_connected:
            logger.warning("Cache not connected, FAQ will not be updated")
            return False
        
        try:
            return self._update_faq_in_cache(faq)
        except Exception as e:
            logger.error(f"Failed to update FAQ {faq.id} in cache: {str(e)}")
            return False
    
    def delete_faq(self, faq_id: int) -> bool:
        """Delete a FAQ from the cache.
        
        Args:
            faq_id: The ID of the FAQ to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.debug(f"Deleting FAQ {faq_id} from cache")
        
        if not self.is_connected:
            logger.warning("Cache not connected, FAQ cannot be deleted")
            return False
        
        try:
            return self._delete_faq_from_cache(faq_id)
        except Exception as e:
            logger.error(f"Failed to delete FAQ {faq_id} from cache: {str(e)}")
            return False
    
    def search_faqs(
        self, 
        query: str, 
        limit: int = 10, 
        similarity_threshold: float = 0.9
    ) -> List[FAQSearchResult]:
        """Search FAQs by vector similarity.
        
        Args:
            query: The search query
            limit: Maximum number of FAQs to return
            similarity_threshold: Minimum similarity score (0-1) for results
            
        Returns:
            List[FAQSearchResult]: List of matching FAQs with similarity scores
        """
        logger.debug(f"Searching FAQs for query: '{query}' (limit={limit}, threshold={similarity_threshold})")
        
        if not self.is_connected:
            logger.warning("Cache not connected, falling back to simple text search")
            return self._simple_text_search(query, limit)
        
        try:
            return self._vector_search_faqs(query, limit, similarity_threshold)
        except Exception as e:
            logger.error(f"Failed to search FAQs: {str(e)}")
            # Fallback to simple text search
            return self._simple_text_search(query, limit)
    
    def get_cache_stats(self) -> CacheStats:
        """Get comprehensive cache statistics.
        
        Returns:
            CacheStats: Statistics about the cache
        """
        try:
            stats = CacheStats(
                cache_status=self.cache_status.value,
                redis_connected=self.is_connected,
                embedding_model_available=self.embedding_component.embedding_model is not None,
                embedding_dimensions=self._embedding_dimensions
            )
            
            if self.is_connected:
                cache_stats = self._get_detailed_cache_stats()
                stats.total_faqs = cache_stats.get("total_faqs", 0)
                stats.cache_keys = cache_stats.get("cache_keys", 0)
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get cache stats: {str(e)}")
            return CacheStats(
                cache_status=self.cache_status.value,
                redis_connected=False,
                embedding_model_available=False,
                error=str(e)
            )
    
    # Private helper methods
    def _as_schema_faq(self, faq_obj: FAQ) -> FAQ:
        """Ensure the given object is a Pydantic FAQ schema instance.

        In many code paths, a SQLAlchemy ORM model is passed in. Since we
        serialize with Pydantic's model_dump_json, convert ORM to schema first.
        """
        if isinstance(faq_obj, FAQ):
            return faq_obj
        # Convert from ORM using from_attributes=True in schema Config
        return FAQ.model_validate(faq_obj)
    
    def _store_faq_in_cache(self, faq: FAQ) -> bool:
        """Store FAQ in Redis cache with embedding."""
        redis_client = self.cache_component.redis_client
        
        # Store the FAQ item as a JSON string
        schema_faq = self._as_schema_faq(faq)
        key = f"faq:{schema_faq.id}"
        value = schema_faq.model_dump_json()
        redis_client.set(key, value)
        
        # Generate and store embedding for the question
        question_embedding = self._get_embedding_safely(schema_faq.question)
        if question_embedding is not None:
            embedding_key = f"faq_embedding:{schema_faq.id}"
            if not self._store_embedding_safely(redis_client, embedding_key, question_embedding):
                logger.warning(f"Failed to store embedding for FAQ {schema_faq.id}")
        else:
            logger.warning(f"Failed to generate embedding for FAQ {schema_faq.id}")
        
        # Add to sorted set for frequency tracking
        redis_client.zadd("faq:frequency", {key: schema_faq.frequency})
        
        # Add to category set if category exists
        if schema_faq.category:
            redis_client.sadd(f"faq:category:{schema_faq.category}", key)
        
        # Add to recent FAQs list (keep last 100)
        # Remove first if it already exists to avoid duplicates
        redis_client.lrem("faq:recent", 0, key)
        redis_client.lpush("faq:recent", key)
        redis_client.ltrim("faq:recent", 0, 99)
        
        return True
    
    def _retrieve_faq_from_cache(self, faq_id: int) -> Optional[FAQ]:
        """Retrieve FAQ from Redis cache."""
        redis_client = self.cache_component.redis_client
        
        key = f"faq:{faq_id}"
        value = redis_client.get(key)
        
        if value:
            # Increment frequency when accessed
            try:
                redis_client.zincrby("faq:frequency", 1, key)
            except Exception as e:
                logger.warning(f"Failed to increment frequency for FAQ {faq_id}: {str(e)}")
            
            faq_dict = json.loads(value.decode('utf-8'))
            return FAQ(**faq_dict)
        
        return None
    
    def _update_faq_in_cache(self, faq: FAQ) -> bool:
        """Update FAQ in Redis cache."""
        redis_client = self.cache_component.redis_client
        
        schema_faq = self._as_schema_faq(faq)
        key = f"faq:{schema_faq.id}"
        
        # First, get the existing FAQ to check its current category
        old_value = redis_client.get(key)
        old_category = None
        if old_value:
            try:
                old_faq_dict = json.loads(old_value.decode('utf-8'))
                old_category = old_faq_dict.get("category")
            except Exception as e:
                logger.warning(f"Failed to parse existing FAQ {schema_faq.id} for category update: {str(e)}")
        
        # Store the updated FAQ item as a JSON string
        value = schema_faq.model_dump_json()
        redis_client.set(key, value)
        
        # Handle category changes - remove from old category set if needed
        if old_category and old_category != schema_faq.category:
            redis_client.srem(f"faq:category:{old_category}", key)
            logger.debug(f"Removed FAQ {schema_faq.id} from old category '{old_category}'")
        
        # Add to category set if category exists (new or updated)
        if schema_faq.category:
            redis_client.sadd(f"faq:category:{schema_faq.category}", key)
            logger.debug(f"Added FAQ {schema_faq.id} to category '{schema_faq.category}'")
        
        # Update embedding for the question
        question_embedding = self._get_embedding_safely(schema_faq.question)
        if question_embedding is not None:
            embedding_key = f"faq_embedding:{schema_faq.id}"
            if not self._store_embedding_safely(redis_client, embedding_key, question_embedding):
                logger.warning(f"Failed to update embedding for FAQ {schema_faq.id}")
        else:
            logger.warning(f"Failed to generate updated embedding for FAQ {schema_faq.id}")
        
        # Update frequency in sorted set
        redis_client.zadd("faq:frequency", {key: schema_faq.frequency})
        logger.debug(f"Updated frequency for FAQ {schema_faq.id} to {schema_faq.frequency}")
        
        return True
    
    def _delete_faq_from_cache(self, faq_id: int) -> bool:
        """Delete FAQ from Redis cache."""
        redis_client = self.cache_component.redis_client
        
        key = f"faq:{faq_id}"
        embedding_key = f"faq_embedding:{faq_id}"
        embedding_meta_key = f"{embedding_key}:meta"
        
        # Get the FAQ to check its category
        value = redis_client.get(key)
        if value:
            try:
                faq_dict = json.loads(value.decode('utf-8'))
                category = faq_dict.get("category")
                
                # Remove from frequency set
                redis_client.zrem("faq:frequency", key)
                
                # Remove from category set if category exists
                if category:
                    redis_client.srem(f"faq:category:{category}", key)
                
                # Remove from recent list
                redis_client.lrem("faq:recent", 0, key)
                
                # Delete the FAQ, its embedding, and metadata
                redis_client.delete(key)
                redis_client.delete(embedding_key)
                redis_client.delete(embedding_meta_key)
                
                logger.debug(f"Successfully deleted FAQ {faq_id} from cache")
                return True
            except Exception as e:
                logger.error(f"Failed to properly delete FAQ {faq_id} from cache: {str(e)}")
                # Try to delete what we can
                redis_client.delete(key, embedding_key, embedding_meta_key)
                return False
        else:
            logger.debug(f"FAQ {faq_id} not found in cache for deletion")
            return False
    
    def _vector_search_faqs(self, query: str, limit: int, similarity_threshold: float) -> List[FAQSearchResult]:
        """Perform vector similarity search on FAQs."""
        redis_client = self.cache_component.redis_client
        
        # Generate embedding for the query
        query_embedding = self._get_embedding_safely(query)
        if query_embedding is None:
            logger.warning("Failed to generate query embedding, falling back to text search")
            return self._simple_text_search(query, limit)
        
        # Get all FAQ keys
        all_keys = redis_client.keys("faq:*")
        
        similarities = []
        
        for key in all_keys:
            try:
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                
                # Skip non-FAQ keys and metadata keys
                if (key_str in ["faq:frequency", "faq:recent"] or 
                    key_str.startswith("faq_embedding:") or 
                    key_str.startswith("faq:category:") or
                    key_str.endswith(":meta")):
                    continue
                
                # Verify this is a string key
                key_type = redis_client.type(key)
                if key_type != b'string':
                    continue
                
                # Get FAQ embedding
                faq_id = key_str.split(":")[1]
                embedding_key = f"faq_embedding:{faq_id}"
                faq_embedding = self._load_embedding_safely(redis_client, embedding_key)
                
                if faq_embedding is not None:
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(query_embedding, faq_embedding)
                    
                    # Only include results above the similarity threshold
                    if similarity >= similarity_threshold:
                        similarities.append((key, similarity))
                else:
                    logger.debug(f"No embedding found for FAQ {faq_id}")
                        
            except Exception as e:
                logger.error(f"Failed to process key {key_str} in vector search: {str(e)}")
                continue
        
        # Sort by similarity (descending) and limit results
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similarities = similarities[:limit]
        
        # Retrieve FAQ objects for the most similar entries
        matching_faqs = []
        for key, similarity in top_similarities:
            try:
                value = redis_client.get(key)
                if value:
                    faq_dict = json.loads(value.decode('utf-8'))
                    faq = FAQ(**faq_dict)
                    matching_faqs.append(FAQSearchResult(faq=faq, similarity=similarity))
            except Exception as e:
                logger.error(f"Failed to retrieve FAQ data for key {key}: {str(e)}")
                continue
        
        return matching_faqs
    
    def _simple_text_search(self, query: str, limit: int) -> List[FAQSearchResult]:
        """Fallback simple text search implementation."""
        if not self.is_connected:
            return []
        
        redis_client = self.cache_component.redis_client
        all_keys = redis_client.keys("faq:*")
        
        matching_faqs = []
        
        for key in all_keys:
            try:
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                
                # Skip non-FAQ keys and metadata keys
                if (key_str in ["faq:frequency", "faq:recent"] or 
                    key_str.startswith("faq_embedding:") or 
                    key_str.startswith("faq:category:") or
                    key_str.endswith(":meta")):
                    continue
                
                # Verify this is a string key
                key_type = redis_client.type(key)
                if key_type != b'string':
                    continue
                
                value = redis_client.get(key)
                if value:
                    faq_dict = json.loads(value.decode('utf-8'))
                    # Simple text search in question and answer
                    question = faq_dict.get("question", "").lower()
                    answer = faq_dict.get("answer", "").lower()
                    query_lower = query.lower()
                    
                    if query_lower in question or query_lower in answer:
                        faq = FAQ(**faq_dict)
                        matching_faqs.append(FAQSearchResult(faq=faq))
                        
            except Exception as e:
                logger.error(f"Failed to process key {key_str} in simple search: {str(e)}")
                continue
        
        # Sort by frequency and limit results
        matching_faqs.sort(key=lambda x: x.faq.frequency, reverse=True)
        return matching_faqs[:limit]
    
    def _get_frequently_asked_faqs(self, limit: int) -> List[FAQ]:
        """Get FAQs sorted by frequency (descending)."""
        redis_client = self.cache_component.redis_client
        
        # Get FAQs sorted by frequency (descending)
        faq_keys = redis_client.zrevrange("faq:frequency", 0, limit - 1)
        faqs = []
        
        for key in faq_keys:
            value = redis_client.get(key)
            if value:
                faq_dict = json.loads(value.decode('utf-8'))
                faqs.append(FAQ(**faq_dict))
        
        return faqs
    
    def _get_recently_asked_faqs(self, limit: int) -> List[FAQ]:
        """Get recently added FAQs."""
        redis_client = self.cache_component.redis_client
        
        # Get recently added FAQs
        faq_keys = redis_client.lrange("faq:recent", 0, limit - 1)
        faqs = []
        
        for key in faq_keys:
            value = redis_client.get(key)
            if value:
                faq_dict = json.loads(value.decode('utf-8'))
                faqs.append(FAQ(**faq_dict))
        
        return faqs
    
    def _get_faqs_by_category(self, category: str, limit: int) -> List[FAQ]:
        """Get FAQs by category."""
        redis_client = self.cache_component.redis_client
        
        # Get FAQs in the specified category
        faq_keys = redis_client.smembers(f"faq:category:{category}")
        faqs = []
        
        count = 0
        for key in faq_keys:
            if count >= limit:
                break
            value = redis_client.get(key)
            if value:
                faq_dict = json.loads(value.decode('utf-8'))
                faqs.append(FAQ(**faq_dict))
                count += 1
        
        return faqs
    
    def _get_detailed_cache_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics."""
        redis_client = self.cache_component.redis_client
        
        # Get number of FAQs
        faq_keys = redis_client.keys("faq:*")
        # Filter out special keys and metadata keys
        actual_faq_keys = [
            key for key in faq_keys 
            if not key.decode('utf-8') in ["faq:frequency", "faq:recent"] 
            and not key.decode('utf-8').startswith("faq_embedding:")
            and not key.decode('utf-8').startswith("faq:category:")
            and not key.decode('utf-8').endswith(":meta")
        ]
        
        return {
            "total_faqs": len(actual_faq_keys),
            "cache_keys": len(faq_keys)
        }
    
    def _clear_all_faqs(self) -> bool:
        """Clear all FAQ-related data from the cache."""
        redis_client = self.cache_component.redis_client
        
        # Get all FAQ keys including metadata
        all_keys = redis_client.keys("faq*")
        
        # Delete all FAQ-related keys
        if all_keys:
            deleted_count = redis_client.delete(*all_keys)
            logger.info(f"Deleted {deleted_count} keys from FAQ cache")
        
        return True
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            # Ensure both vectors are numpy arrays with the same dtype
            vec1 = np.array(vec1, dtype=np.float32)
            vec2 = np.array(vec2, dtype=np.float32)
            
            # Check dimensions match
            if vec1.shape != vec2.shape:
                logger.warning(f"Vector dimension mismatch: {vec1.shape} vs {vec2.shape}")
                return 0.0
            
            # Calculate dot product
            dot_product = np.dot(vec1, vec2)
            
            # Calculate magnitudes
            magnitude1 = np.linalg.norm(vec1)
            magnitude2 = np.linalg.norm(vec2)
            
            # Avoid division by zero
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = float(dot_product / (magnitude1 * magnitude2))
            
            # Ensure the result is within [-1, 1] range
            return max(-1.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}")
            return 0.0

    def generate_faq_id(self) -> Optional[int]:
        """Generate a new FAQ ID using Redis atomic increment.
        
        Returns:
            int: A new unique FAQ ID, or None if cache is not available
        """
        if not self.is_connected:
            logger.warning("Cache not connected, cannot generate FAQ ID")
            return None
        
        try:
            redis_client = self.cache_component.redis_client
            # Use Redis atomic increment to generate a new ID
            # This ensures unique IDs even across multiple instances
            new_id = redis_client.incr("faq:id_counter")
            logger.debug(f"Generated new FAQ ID from cache: {new_id}")
            return new_id
        except Exception as e:
            logger.error(f"Failed to generate FAQ ID from cache: {str(e)}")
            return None

    def initialize_faq_id_counter(self, db_session) -> bool:
        """Initialize the FAQ ID counter in cache based on the maximum ID in database.
        
        This ensures that cache-generated IDs don't conflict with existing database IDs.
        
        Args:
            db_session: Database session to query for maximum FAQ ID
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if not self.is_connected:
            logger.warning("Cache not connected, cannot initialize FAQ ID counter")
            return False
        
        try:
            # Get the maximum ID from the database
            from private_gpt.users.models.faq import FAQ
            max_id = db_session.query(func.max(FAQ.id)).scalar()
            
            # If no FAQs exist, start from 1
            if max_id is None:
                max_id = 0
                
            # Set the Redis counter to the maximum ID
            # This ensures new IDs generated by Redis will be unique
            redis_client = self.cache_component.redis_client
            redis_client.set("faq:id_counter", max_id)
            
            logger.info(f"FAQ ID counter initialized to {max_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize FAQ ID counter: {str(e)}")
            return False

    def warm_cache_with_frequently_accessed_faqs(self, db_session, limit: int = 50) -> int:
        """Pre-load frequently accessed FAQs into cache.
        
        Args:
            db_session: Database session to fetch FAQs
            limit: Maximum number of FAQs to preload
            
        Returns:
            int: Number of FAQs successfully loaded into cache
        """
        if not self.is_connected:
            logger.warning("Cache not connected, skipping cache warming")
            return 0
            
        try:
            from private_gpt.users.crud import faq as faq_crud
            # Get frequently asked FAQs from database
            frequently_asked_faqs = faq_crud.get_frequently_asked(db_session, limit=limit)
            
            loaded_count = 0
            for faq in frequently_asked_faqs:
                try:
                    if self.add_faq(faq):
                        loaded_count += 1
                except Exception as e:
                    logger.warning(f"Failed to preload FAQ {faq.id} into cache: {str(e)}")
                    
            logger.info(f"Cache warming completed: {loaded_count}/{len(frequently_asked_faqs)} FAQs loaded")
            return loaded_count
            
        except Exception as e:
            logger.error(f"Failed to warm cache with frequently accessed FAQs: {str(e)}")
            return 0

    def evict_least_recently_used_faqs(self, max_faqs: int = 1000) -> int:
        """Evict least recently used FAQs from cache to manage memory.
        
        Args:
            max_faqs: Maximum number of FAQs to keep in cache
            
        Returns:
            int: Number of FAQs evicted from cache
        """
        if not self.is_connected:
            logger.warning("Cache not connected, skipping cache eviction")
            return 0
            
        try:
            redis_client = self.cache_component.redis_client
            
            # Get all FAQ keys
            all_faq_keys = redis_client.keys("faq:*")
            
            # Filter out special keys (frequency, recent, etc.)
            actual_faq_keys = [
                key for key in all_faq_keys 
                if not key.decode('utf-8') in ["faq:frequency", "faq:recent"] 
                and not key.decode('utf-8').startswith("faq_embedding:")
                and not key.decode('utf-8').startswith("faq:category:")
                and not key.decode('utf-8').endswith(":meta")
            ]
            
            # If we're under the limit, no eviction needed
            if len(actual_faq_keys) <= max_faqs:
                logger.debug(f"Cache size {len(actual_faq_keys)} is under limit {max_faqs}, no eviction needed")
                return 0
            
            # Get FAQ access frequencies to determine LRU candidates
            # We'll use the frequency sorted set to identify least frequently used
            faq_frequencies = redis_client.zrange("faq:frequency", 0, -1, withscores=True)
            
            # Sort by frequency (ascending) to get least frequently used
            least_frequently_used = sorted(faq_frequencies, key=lambda x: x[1])
            
            # Calculate how many to evict
            evict_count = len(actual_faq_keys) - max_faqs
            evicted_count = 0
            
            # Evict the least frequently used FAQs
            for i in range(min(evict_count, len(least_frequently_used))):
                try:
                    faq_key_bytes, frequency = least_frequently_used[i]
                    faq_key = faq_key_bytes.decode('utf-8')
                    faq_id = int(faq_key.split(":")[1])
                    
                    # Delete the FAQ from cache
                    if self.delete_faq(faq_id):
                        evicted_count += 1
                        logger.debug(f"Evicted FAQ {faq_id} from cache (frequency: {frequency})")
                        
                except Exception as e:
                    logger.warning(f"Failed to evict FAQ: {str(e)}")
                    continue
                    
            logger.info(f"Cache eviction completed: {evicted_count} FAQs evicted")
            return evicted_count
            
        except Exception as e:
            logger.error(f"Failed to evict least recently used FAQs: {str(e)}")
            return 0

    def health_check(self) -> dict:
        """Perform a health check on the cache service.
        
        Returns:
            dict: Health status information
        """
        try:
            if not self.is_connected:
                return {
                    "status": "disconnected",
                    "redis_connected": False,
                    "error": "Redis not connected"
                }
            
            redis_client = self.cache_component.redis_client
            
            # Test Redis connectivity
            redis_client.ping()
            
            # Get basic stats
            stats = self._get_detailed_cache_stats()
            
            return {
                "status": "healthy",
                "redis_connected": True,
                "total_faqs": stats.get("total_faqs", 0),
                "total_cache_keys": stats.get("cache_keys", 0),
                "embedding_dimensions": self._embedding_dimensions,
                "last_checked": time.time()
            }
            
        except Exception as e:
            logger.error(f"Cache health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "redis_connected": self.is_connected,
                "error": str(e),
                "last_checked": time.time()
            }

    def clear_cache(self) -> bool:
        """Clear all FAQ-related data from the cache.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.is_connected:
                logger.warning("Cache not connected, cannot clear cache")
                return False
                
            return self._clear_all_faqs()
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
            return False

    def refresh_faq_cache(self, db_session, faq_id: int) -> bool:
        """Refresh a specific FAQ in cache with the latest data from database.
        
        Args:
            db_session: Database session
            faq_id: ID of the FAQ to refresh
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected:
            logger.warning("Cache not connected, cannot refresh FAQ cache")
            return False
            
        try:
            # Import here to avoid circular imports
            from private_gpt.users.crud import faq as faq_crud
            
            # Get the latest FAQ from database
            db_faq = faq_crud.get(db_session, id=faq_id)
            if not db_faq:
                logger.warning(f"FAQ {faq_id} not found in database, removing from cache")
                # If not in database, remove from cache
                return self.delete_faq(faq_id)
            
            # Update in cache
            return self.update_faq(db_faq)
            
        except Exception as e:
            logger.error(f"Failed to refresh FAQ {faq_id} cache: {str(e)}")
            return False

    def refresh_all_faq_cache(self, db_session, limit: int = 1000) -> int:
        """Refresh all FAQ cache with the latest data from database.
        
        Args:
            db_session: Database session
            limit: Maximum number of FAQs to refresh
            
        Returns:
            int: Number of FAQs successfully refreshed
        """
        if not self.is_connected:
            logger.warning("Cache not connected, cannot refresh all FAQ cache")
            return 0
            
        try:
            # Import here to avoid circular imports
            from private_gpt.users.crud import faq as faq_crud
            
            # Get all active FAQs from database
            db_faqs = faq_crud.get_active_faqs(db_session, skip=0, limit=limit)
            
            refreshed_count = 0
            for faq in db_faqs:
                try:
                    if self.update_faq(faq):
                        refreshed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to refresh FAQ {faq.id} cache: {str(e)}")
                    continue
                    
            logger.info(f"Refreshed {refreshed_count}/{len(db_faqs)} FAQs in cache")
            return refreshed_count
            
        except Exception as e:
            logger.error(f"Failed to refresh all FAQ cache: {str(e)}")
            return 0

    def increment_faq_frequency_cache_only(self, faq_id: int) -> bool:
        """Increment FAQ frequency in cache only (for high-frequency operations).
        
        This method is optimized for high-frequency operations where immediate
        database synchronization is not required. Frequencies can be synced
        to database periodically in batch operations.
        
        Args:
            faq_id: ID of the FAQ to increment frequency for
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected:
            logger.warning("Cache not connected, cannot increment FAQ frequency")
            return False
            
        try:
            redis_client = self.cache_component.redis_client
            key = f"faq:{faq_id}"
            
            # Increment frequency in sorted set
            redis_client.zincrby("faq:frequency", 1, key)
            
            # Also increment in the FAQ object itself if it exists in cache
            value = redis_client.get(key)
            if value:
                try:
                    faq_dict = json.loads(value.decode('utf-8'))
                    faq_dict['frequency'] = faq_dict.get('frequency', 0) + 1
                    updated_value = json.dumps(faq_dict)
                    redis_client.set(key, updated_value)
                except Exception as e:
                    logger.warning(f"Failed to increment frequency in FAQ object {faq_id}: {e}")
            
            logger.debug(f"Incremented frequency for FAQ {faq_id} in cache only")
            return True
        except Exception as e:
            logger.error(f"Failed to increment FAQ {faq_id} frequency in cache: {e}")
            return False
