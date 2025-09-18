import logging
from typing import List, Optional
from sqlalchemy.orm import Session
from injector import inject, singleton

from private_gpt.server.cache.cache_service import CacheService
from private_gpt.users import crud
from private_gpt.users.schemas.faq import FAQ, FAQCreate, FAQUpdate

logger = logging.getLogger(__name__)


@singleton
class FAQService:
    """Service layer for FAQ operations that integrates database and cache."""
    
    @inject
    def __init__(self, cache_service: CacheService):
        # Use injected cache service instead of getting it from global injector
        self.cache_service = cache_service
        logger.debug(f"FAQService initialized with cache service: {self.cache_service is not None}")
        logger.debug(f"FAQService cache status: {self.cache_service.cache_status}")
    
    def initialize_cache_with_warming(self, db: Session, warm_cache: bool = True, eviction_limit: int = 1000) -> bool:
        """Initialize cache with warming and eviction policies.
        
        Args:
            db: Database session
            warm_cache: Whether to warm the cache with frequently accessed FAQs
            eviction_limit: Maximum number of FAQs to keep in cache
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Initialize the ID counter to ensure unique IDs
            self.cache_service.initialize_faq_id_counter(db)
            
            return self.cache_service.initialize_with_warming(
                db_session=db, 
                warm_cache=warm_cache, 
                eviction_limit=eviction_limit
            )
        except Exception as e:
            logger.error(f"Failed to initialize cache with warming: {str(e)}")
            return False
    
    def create_faq(self, db: Session, faq_in: FAQCreate, user_id: int) -> FAQ:
        """Create a new FAQ and add it to cache.
        
        Args:
            db: Database session
            faq_in: FAQ creation data
            user_id: ID of the user creating the FAQ
            
        Returns:
            FAQ: Created FAQ
        """
        # First, try to get an ID from the cache
        # This follows the user's request to get the ID from cache first
        faq_id = None
        try:
            faq_id = self.cache_service.generate_faq_id()
            logger.debug(f"Got FAQ ID {faq_id} from cache")
        except Exception as e:
            logger.warning(f"Failed to get FAQ ID from cache: {e}")
        
        # Create in database using the ID from cache if available
        # If cache ID generation failed, let database auto-generate the ID
        db_faq = crud.faq.create_with_user(db, obj_in=faq_in, user_id=user_id, faq_id=faq_id)
        
        # Add to cache
        try:
            self.cache_service.add_faq(db_faq)
        except Exception as e:
            logger.warning(f"Failed to add FAQ {db_faq.id} to cache: {e}")
        
        return db_faq
    
    def get_faq(self, db: Session, faq_id: int) -> Optional[FAQ]:
        """Get a FAQ by ID, first checking cache then database.
        
        Args:
            db: Database session
            faq_id: FAQ ID
            
        Returns:
            FAQ or None if not found
        """
        # Try to get from cache first
        try:
            cached_faq = self.cache_service.get_faq(faq_id)
            if cached_faq:
                return cached_faq
        except Exception as e:
            logger.warning(f"Failed to get FAQ {faq_id} from cache: {e}")
        
        # If not in cache, get from database
        db_faq = crud.faq.get(db, id=faq_id)
        if not db_faq:
            return None
        
        # Add to cache for future requests
        try:
            self.cache_service.add_faq(db_faq)
        except Exception as e:
            logger.warning(f"Failed to add FAQ {faq_id} to cache: {e}")
        
        return db_faq
    
    def update_faq(self, db: Session, faq_id: int, faq_in: FAQUpdate, user_id: int) -> Optional[FAQ]:
        """Update a FAQ and update it in cache.
        
        Args:
            db: Database session
            faq_id: FAQ ID to update
            faq_in: FAQ update data
            user_id: ID of the user updating the FAQ
            
        Returns:
            FAQ or None if not found
        """
        # Update in database
        db_faq = crud.faq.get(db, id=faq_id)
        if not db_faq:
            return None
            
        updated_faq = crud.faq.update_with_user(db, db_obj=db_faq, obj_in=faq_in, user_id=user_id)
        
        # Update in cache using the improved method
        try:
            self.cache_service.update_faq(updated_faq)
        except Exception as e:
            logger.warning(f"Failed to update FAQ {faq_id} in cache: {e}")
        
        return updated_faq
    
    def delete_faq(self, db: Session, faq_id: int) -> bool:
        """Delete a FAQ and remove it from cache.
        
        Args:
            db: Database session
            faq_id: FAQ ID to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Delete from database
        db_faq = crud.faq.get(db, id=faq_id)
        if not db_faq:
            # If not in database, try to remove from cache anyway
            try:
                self.cache_service.delete_faq(faq_id)
            except Exception as e:
                logger.warning(f"Failed to remove FAQ {faq_id} from cache: {e}")
            return False
            
        success = crud.faq.remove(db, id=faq_id)
        if not success:
            return False
        
        # Remove from cache
        try:
            self.cache_service.delete_faq(faq_id)
        except Exception as e:
            logger.warning(f"Failed to remove FAQ {faq_id} from cache: {e}")
        
        return True
    
    def get_frequently_asked(self, db: Session, limit: int = 10) -> List[FAQ]:
        """Get the most frequently asked FAQs.
        
        Args:
            db: Database session
            limit: Maximum number of FAQs to return
            
        Returns:
            List[FAQ]: List of frequently asked FAQs
        """
        # Try to get from cache first
        try:
            cached_faqs = self.cache_service.get_frequently_asked(limit)
            if cached_faqs and len(cached_faqs) == limit:
                return cached_faqs
        except Exception as e:
            logger.warning(f"Failed to get frequently asked FAQs from cache: {e}")
        
        # If not in cache or not enough items, get from database
        db_faqs = crud.faq.get_frequently_asked(db, limit=limit)
        
        # Update cache with database results using the improved method
        for faq in db_faqs:
            try:
                self.cache_service.add_faq(faq)
            except Exception as e:
                logger.warning(f"Failed to add FAQ {faq.id} to cache: {e}")
        
        return db_faqs
    
    def get_recently_added(self, db: Session, limit: int = 10) -> List[FAQ]:
        """Get the most recently added FAQs.
        
        Args:
            db: Database session
            limit: Maximum number of FAQs to return
            
        Returns:
            List[FAQ]: List of recently added FAQs
        """
        # Try to get from cache first
        try:
            cached_faqs = self.cache_service.get_recently_asked(limit)
            if cached_faqs and len(cached_faqs) == limit:
                return cached_faqs
        except Exception as e:
            logger.warning(f"Failed to get recently added FAQs from cache: {e}")
        
        # If not in cache or not enough items, get from database
        db_faqs = crud.faq.get_recently_added(db, limit=limit)
        
        # Update cache with database results using the improved method
        for faq in db_faqs:
            try:
                self.cache_service.add_faq(faq)
            except Exception as e:
                logger.warning(f"Failed to add FAQ {faq.id} to cache: {e}")
        
        return db_faqs
    
    def search_faqs(self, db: Session, query: str, limit: int = 3, similarity_threshold: float = 0.9) -> List[FAQ]:
        """Search FAQs by query string using vector similarity.
        
        Args:
            db: Database session
            query: Search query
            limit: Maximum number of FAQs to return
            similarity_threshold: Minimum similarity score (0-1) for results
            
        Returns:
            List[FAQ]: List of matching FAQs
        """
        logger.debug(f"FAQ Service searching for: '{query}' with threshold: {similarity_threshold}")
        
        # Search in cache first using vector similarity
        try:
            cached_results = self.cache_service.search_faqs(query, limit=limit, similarity_threshold=similarity_threshold)
            logger.debug(f"Cache search returned {len(cached_results)} results")
            
            if cached_results:
                # Extract FAQ objects from search results
                return [result.faq for result in cached_results]
        except Exception as e:
            logger.warning(f"Failed to search FAQs in cache: {e}")
        
        # If no results in cache, search in database
        logger.debug("No cache results, searching database")
        db_faqs = crud.faq.search(db, query=query, limit=limit)
        logger.debug(f"Database search returned {len(db_faqs)} results")
        
        # Update cache with search results using the improved method
        for faq in db_faqs:
            try:
                self.cache_service.add_faq(faq)
            except Exception as e:
                logger.warning(f"Failed to add FAQ {faq.id} to cache: {e}")
        
        return db_faqs
    
    def refresh_faq_cache(self, db: Session, faq_id: int) -> bool:
        """Refresh a specific FAQ in cache with the latest data from database.
        
        Args:
            db: Database session
            faq_id: ID of the FAQ to refresh
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            return self.cache_service.refresh_faq_cache(db, faq_id)
        except Exception as e:
            logger.error(f"Failed to refresh FAQ {faq_id} cache through FAQService: {str(e)}")
            return False
    
    def refresh_all_faq_cache(self, db: Session, limit: int = 1000) -> int:
        """Refresh all FAQ cache with the latest data from database.
        
        Args:
            db: Database session
            limit: Maximum number of FAQs to refresh
            
        Returns:
            int: Number of FAQs successfully refreshed
        """
        try:
            return self.cache_service.refresh_all_faq_cache(db, limit)
        except Exception as e:
            logger.error(f"Failed to refresh all FAQ cache through FAQService: {str(e)}")
            return 0

    def increment_cache_hit_frequency(self, db: Session, faq_id: int) -> bool:
        """Increment the frequency count for a FAQ when there's a cache hit.
        
        This method increments the frequency counter in both the database and cache
        to track how often FAQs are accessed. This should be called whenever a FAQ
        is served from cache to maintain accurate usage statistics.
        
        Args:
            db: Database session
            faq_id: ID of the FAQ to increment frequency for
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Processing cache hit frequency increment for FAQ {faq_id}")
        
        try:
            # Increment frequency in database using CRUD operation
            success = crud.faq.increment_frequency(db, faq_id=faq_id)
            if not success:
                logger.warning(f"Failed to increment frequency in database for FAQ {faq_id}")
                return False
            
            # Update the FAQ in cache to reflect the new frequency
            try:
                # Get the updated FAQ from database with new frequency
                updated_faq = crud.faq.get(db, id=faq_id)
                if updated_faq:
                    # Update in cache with new frequency count
                    self.cache_service.update_faq(updated_faq)
                    logger.info(f"Successfully updated frequency count for FAQ {faq_id} in both database and cache")
                else:
                    logger.warning(f"FAQ {faq_id} not found in database after frequency increment")
                    return False
            except Exception as cache_e:
                logger.warning(f"Failed to update FAQ {faq_id} frequency in cache: {cache_e}")
                # Don't return False here as database update was successful
                # Cache will be eventually consistent
            
            return True
            
        except Exception as e:
            logger.error(f"Error incrementing frequency count for FAQ {faq_id}: {e}")
            return False

    def batch_increment_cache_hit_frequency(self, db: Session, faq_ids: List[int]) -> Dict[int, bool]:
        """Batch increment frequency counts for multiple FAQs.
        
        This method provides better performance for multiple FAQ frequency updates
        by reducing database round trips.
        
        Args:
            db: Database session
            faq_ids: List of FAQ IDs to increment frequency for
            
        Returns:
            Dict[int, bool]: Mapping of FAQ IDs to success status
        """
        results = {}
        
        try:
            # Batch increment frequencies in database
            for faq_id in faq_ids:
                try:
                    success = crud.faq.increment_frequency(db, faq_id=faq_id)
                    results[faq_id] = success
                except Exception as e:
                    logger.error(f"Failed to increment frequency for FAQ {faq_id}: {e}")
                    results[faq_id] = False
            
            # Batch update cache with new frequencies
            try:
                updated_faqs = []
                for faq_id in faq_ids:
                    if results[faq_id]:  # Only update cache if database update succeeded
                        updated_faq = crud.faq.get(db, id=faq_id)
                        if updated_faq:
                            updated_faqs.append(updated_faq)
                
                # Batch update all FAQs in cache
                for updated_faq in updated_faqs:
                    try:
                        self.cache_service.update_faq(updated_faq)
                        logger.info(f"Successfully updated FAQ {updated_faq.id} in cache")
                    except Exception as cache_e:
                        logger.warning(f"Failed to update FAQ {updated_faq.id} in cache: {cache_e}")
                        results[updated_faq.id] = False  # Mark as failed if cache update fails
                        
            except Exception as batch_cache_e:
                logger.error(f"Failed to batch update FAQs in cache: {batch_cache_e}")
                
        except Exception as e:
            logger.error(f"Error in batch increment frequency operation: {e}")
            
        return results
