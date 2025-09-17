import logging
import threading
import time
from typing import Optional
from sqlalchemy.orm import Session
from injector import inject, singleton

from private_gpt.server.cache.cache_service import CacheService
from private_gpt.server.cache.faq_service import FAQService
from private_gpt.users import crud

logger = logging.getLogger(__name__)


@singleton
class CacheManager:
    """Manager for cache warming and eviction policies."""
    
    @inject
    def __init__(self, cache_service: CacheService, faq_service: FAQService):
        self.cache_service = cache_service
        self.faq_service = faq_service
        self._background_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_running = False
        
    def start_background_cache_management(self, db_session_factory, warm_interval: int = 3600, eviction_interval: int = 1800) -> bool:
        """Start background thread for cache warming and eviction.
        
        Args:
            db_session_factory: Factory function to create database sessions
            warm_interval: Interval in seconds between cache warming operations
            eviction_interval: Interval in seconds between cache eviction operations
            
        Returns:
            bool: True if background thread started successfully, False otherwise
        """
        if self._is_running:
            logger.warning("Cache management background thread is already running")
            return False
            
        try:
            self._stop_event.clear()
            self._background_thread = threading.Thread(
                target=self._cache_management_loop,
                args=(db_session_factory, warm_interval, eviction_interval),
                daemon=True,
                name="CacheManagementThread"
            )
            self._background_thread.start()
            self._is_running = True
            logger.info("Cache management background thread started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start cache management background thread: {str(e)}")
            return False
    
    def stop_background_cache_management(self) -> bool:
        """Stop background cache management thread.
        
        Returns:
            bool: True if thread stopped successfully, False otherwise
        """
        if not self._is_running:
            logger.warning("Cache management background thread is not running")
            return False
            
        try:
            self._stop_event.set()
            if self._background_thread and self._background_thread.is_alive():
                self._background_thread.join(timeout=5)
            self._is_running = False
            logger.info("Cache management background thread stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop cache management background thread: {str(e)}")
            return False
    
    def _cache_management_loop(self, db_session_factory, warm_interval: int, eviction_interval: int):
        """Background loop for cache management operations."""
        last_warm_time = 0
        last_eviction_time = 0
        last_refresh_time = 0
        refresh_interval = 7200  # Refresh all cache every 2 hours
        
        while not self._stop_event.is_set():
            try:
                current_time = time.time()
                
                # Perform cache warming
                if current_time - last_warm_time >= warm_interval:
                    try:
                        db = db_session_factory()
                        self.faq_service.initialize_cache_with_warming(
                            db=db, 
                            warm_cache=True, 
                            eviction_limit=500
                        )
                        db.close()
                        last_warm_time = current_time
                        logger.debug("Cache warming completed")
                    except Exception as e:
                        logger.error(f"Error during cache warming: {str(e)}")
                
                # Perform cache eviction
                if current_time - last_eviction_time >= eviction_interval:
                    try:
                        evicted_count = self.cache_service.evict_least_recently_used_faqs(max_faqs=1000)
                        last_eviction_time = current_time
                        logger.debug(f"Cache eviction completed, {evicted_count} FAQs evicted")
                    except Exception as e:
                        logger.error(f"Error during cache eviction: {str(e)}")
                
                # Perform cache refresh (less frequently)
                if current_time - last_refresh_time >= refresh_interval:
                    try:
                        db = db_session_factory()
                        refreshed_count = self.faq_service.refresh_all_faq_cache(db, limit=1000)
                        db.close()
                        last_refresh_time = current_time
                        logger.debug(f"Cache refresh completed, {refreshed_count} FAQs refreshed")
                    except Exception as e:
                        logger.error(f"Error during cache refresh: {str(e)}")
                
                # Sleep for a short interval before checking again
                self._stop_event.wait(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in cache management loop: {str(e)}")
                self._stop_event.wait(60)  # Wait before retrying
    
    def warm_cache_on_startup(self, db: Session, limit: int = 100) -> int:
        """Warm cache with frequently accessed FAQs on application startup.
        
        Args:
            db: Database session
            limit: Maximum number of FAQs to preload
            
        Returns:
            int: Number of FAQs successfully loaded into cache
        """
        try:
            logger.info("Warming cache with frequently accessed FAQs on startup")
            loaded_count = self.cache_service.warm_cache_with_frequently_accessed_faqs(db, limit=limit)
            logger.info(f"Cache warming on startup completed: {loaded_count} FAQs loaded")
            return loaded_count
        except Exception as e:
            logger.error(f"Failed to warm cache on startup: {str(e)}")
            return 0
    
    def perform_eviction_if_needed(self, max_faqs: int = 1000) -> int:
        """Perform cache eviction if needed to maintain size limits.
        
        Args:
            max_faqs: Maximum number of FAQs to keep in cache
            
        Returns:
            int: Number of FAQs evicted from cache
        """
        try:
            evicted_count = self.cache_service.evict_least_recently_used_faqs(max_faqs=max_faqs)
            if evicted_count > 0:
                logger.info(f"Cache eviction performed: {evicted_count} FAQs evicted")
            return evicted_count
        except Exception as e:
            logger.error(f"Failed to perform cache eviction: {str(e)}")
            return 0
    
    def refresh_cache_for_faq(self, db: Session, faq_id: int) -> bool:
        """Refresh cache for a specific FAQ with latest data from database.
        
        Args:
            db: Database session
            faq_id: ID of the FAQ to refresh
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            return self.faq_service.refresh_faq_cache(db, faq_id)
        except Exception as e:
            logger.error(f"Failed to refresh cache for FAQ {faq_id}: {str(e)}")
            return False
    
    def refresh_all_cache(self, db: Session, limit: int = 1000) -> int:
        """Refresh all FAQ cache with latest data from database.
        
        Args:
            db: Database session
            limit: Maximum number of FAQs to refresh
            
        Returns:
            int: Number of FAQs successfully refreshed
        """
        try:
            refreshed_count = self.faq_service.refresh_all_faq_cache(db, limit=limit)
            logger.info(f"Cache refresh completed: {refreshed_count} FAQs refreshed")
            return refreshed_count
        except Exception as e:
            logger.error(f"Failed to refresh all cache: {str(e)}")
            return 0