from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from sqlalchemy.orm import Session
import logging

from private_gpt.users import crud, schemas, models
from private_gpt.users.api import deps
from private_gpt.server.cache.faq_service import FAQService
from private_gpt.server.cache.cache_service import CacheService, FAQSearchResult, CacheStats

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/faq", tags=["FAQ Management"])

@router.post("/", response_model=schemas.FAQ, status_code=status.HTTP_201_CREATED)
def create_faq(
    *,
    request: Request,
    db: Session = Depends(deps.get_db),
    faq_in: schemas.FAQCreate,
    current_user: models.User = Depends(deps.get_current_user),
) -> schemas.FAQ:
    """
    Create a new FAQ and add it to the cache.
    """
    # Get services from injector
    faq_service = request.state.injector.get(FAQService)
    cache_service = request.state.injector.get(CacheService)
    
    # Create FAQ in database and cache
    faq = faq_service.create_faq(db, faq_in=faq_in, user_id=current_user.id)
    
    # Add to cache (graceful degradation if cache unavailable)
    try:
        cache_service.add_faq(faq)
        logger.info(f"FAQ {faq.id} added to cache successfully")
    except Exception as e:
        logger.warning(f"Failed to add FAQ {faq.id} to cache", extra={"error": str(e)})
        # Continue without cache - not a critical failure
    
    return faq


@router.get("/{faq_id}", response_model=schemas.FAQ)
def get_faq(
    *,
    request: Request,
    db: Session = Depends(deps.get_db),
    faq_id: int,
) -> schemas.FAQ:
    """
    Get a FAQ by ID, first checking cache then database.
    """
    # Get services from injector
    faq_service = request.state.injector.get(FAQService)
    cache_service = request.state.injector.get(CacheService)
    
    # Try to get from cache first
    try:
        cached_faq = cache_service.get_faq(faq_id)
        if cached_faq:
            logger.debug(f"FAQ {faq_id} retrieved from cache")
            return cached_faq
    except Exception as e:
        logger.warning(f"Failed to get FAQ {faq_id} from cache", extra={"error": str(e)})
    
    # Fallback to database
    faq = faq_service.get_faq(db, faq_id=faq_id)
    if not faq:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="FAQ not found",
        )
    
    # Add to cache for future requests (if not already there)
    try:
        cache_service.add_faq(faq)
        logger.debug(f"FAQ {faq_id} added to cache from database")
    except Exception as e:
        logger.warning(f"Failed to cache FAQ {faq_id}", extra={"error": str(e)})
    
    return faq


@router.get("/", response_model=schemas.FAQList)
def list_faqs(
    *,
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
) -> schemas.FAQList:
    """
    List all active FAQs with pagination.
    """
    faqs = crud.faq.get_active_faqs(db, skip=skip, limit=limit)
    total = crud.faq.get_total_count(db)
    return schemas.FAQList(faqs=faqs, total=total)


@router.put("/{faq_id}", response_model=schemas.FAQ)
def update_faq(
    *,
    request: Request,
    db: Session = Depends(deps.get_db),
    faq_id: int,
    faq_in: schemas.FAQUpdate,
    current_user: models.User = Depends(deps.get_current_user),
) -> schemas.FAQ:
    """
    Update a FAQ in database and cache.
    """
    # Get services from injector
    faq_service = request.state.injector.get(FAQService)
    cache_service = request.state.injector.get(CacheService)
    
    faq = faq_service.update_faq(db, faq_id=faq_id, faq_in=faq_in, user_id=current_user.id)
    if not faq:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="FAQ not found",
        )
    
    # Update in cache
    try:
        cache_service.update_faq(faq)
        logger.info(f"FAQ {faq_id} updated in cache successfully")
    except Exception as e:
        logger.warning(f"Failed to update FAQ {faq_id} in cache", extra={"error": str(e)})
    
    return faq


@router.delete("/{faq_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_faq(
    *,
    request: Request,
    db: Session = Depends(deps.get_db),
    faq_id: int,
    current_user: models.User = Depends(deps.get_current_user),
) -> None:
    """
    Delete a FAQ from database and cache.
    """
    # Get services from injector
    faq_service = request.state.injector.get(FAQService)
    cache_service = request.state.injector.get(CacheService)
    
    success = faq_service.delete_faq(db, faq_id=faq_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="FAQ not found",
        )
    
    # Remove from cache
    try:
        cache_service.delete_faq(faq_id)
        logger.info(f"FAQ {faq_id} removed from cache successfully")
    except Exception as e:
        logger.warning(f"Failed to remove FAQ {faq_id} from cache", extra={"error": str(e)})


@router.get("/frequently-asked/", response_model=List[schemas.FAQ])
def get_frequently_asked_faqs(
    *,
    request: Request,
    db: Session = Depends(deps.get_db),
    limit: int = Query(10, ge=1, le=100),
) -> List[schemas.FAQ]:
    """
    Get the most frequently asked FAQs, preferring cache when available.
    """
    # Get services from injector
    faq_service = request.state.injector.get(FAQService)
    cache_service = request.state.injector.get(CacheService)
    
    try:
        cached_faqs = cache_service.get_frequently_asked(limit=limit)
        if cached_faqs:
            logger.debug(f"Retrieved {len(cached_faqs)} frequently asked FAQs from cache")
            return cached_faqs
    except Exception as e:
        logger.warning("Failed to get frequently asked FAQs from cache", extra={"error": str(e)})
    
    # Fallback to database
    faqs = faq_service.get_frequently_asked(db, limit=limit)
    logger.debug(f"Retrieved {len(faqs)} frequently asked FAQs from database")
    return faqs


@router.get("/recent/", response_model=List[schemas.FAQ])
def get_recent_faqs(
    *,
    request: Request,
    db: Session = Depends(deps.get_db),
    limit: int = Query(10, ge=1, le=100),
) -> List[schemas.FAQ]:
    """
    Get the most recently added FAQs, preferring cache when available.
    """
    # Get services from injector
    faq_service = request.state.injector.get(FAQService)
    cache_service = request.state.injector.get(CacheService)
    
    try:
        cached_faqs = cache_service.get_recently_asked(limit=limit)
        if cached_faqs:
            logger.debug(f"Retrieved {len(cached_faqs)} recent FAQs from cache")
            return cached_faqs
    except Exception as e:
        logger.warning("Failed to get recent FAQs from cache", extra={"error": str(e)})
    
    # Fallback to database
    faqs = faq_service.get_recently_added(db, limit=limit)
    logger.debug(f"Retrieved {len(faqs)} recent FAQs from database")
    return faqs


@router.get("/categories/", response_model=List[schemas.FAQCategory])
def get_faq_categories(
    *,
    db: Session = Depends(deps.get_db),
) -> List[schemas.FAQCategory]:
    """
    Get all FAQ categories with count.
    """
    categories = crud.faq.get_categories(db)
    return [schemas.FAQCategory(category=cat[0], count=cat[1]) for cat in categories]


@router.get("/category/{category}", response_model=schemas.FAQList)
def get_faqs_by_category(
    *,
    request: Request,
    db: Session = Depends(deps.get_db),
    category: str,
    skip: int = 0,
    limit: int = 100,
) -> schemas.FAQList:
    """
    Get FAQs by category, preferring cache when available.
    """
    # Get services from injector
    faq_service = request.state.injector.get(FAQService)
    cache_service = request.state.injector.get(CacheService)
    
    # Try cache first (only for first page without skip)
    if skip == 0:
        try:
            cached_faqs = cache_service.get_by_category(category, limit=limit)
            if cached_faqs:
                logger.debug(f"Retrieved {len(cached_faqs)} FAQs for category '{category}' from cache")
                return schemas.FAQList(faqs=cached_faqs, total=len(cached_faqs))
        except Exception as e:
            logger.warning(f"Failed to get FAQs for category '{category}' from cache", extra={"error": str(e)})
    
    # Fallback to database
    faqs = crud.faq.get_by_category(db, category=category, skip=skip, limit=limit)
    total = len(faqs)
    logger.debug(f"Retrieved {len(faqs)} FAQs for category '{category}' from database")
    return schemas.FAQList(faqs=faqs, total=total)


@router.post("/search/", response_model=List[FAQSearchResult])
def search_faqs(
    *,
    request: Request,
    db: Session = Depends(deps.get_db),
    search_in: schemas.FAQSearch,
) -> List[FAQSearchResult]:
    """
    Search FAQs by query string using vector similarity when available.
    """
    # Get services from injector
    faq_service = request.state.injector.get(FAQService)
    cache_service = request.state.injector.get(CacheService)
    
    try:
        search_results = cache_service.search_faqs(
            query=search_in.query,
            limit=search_in.limit,
            similarity_threshold=search_in.similarity_threshold
        )
        
        if search_results:
            logger.info(f"Vector search returned {len(search_results)} results for query: '{search_in.query}'")
            # Convert to response format
            return [
                FAQSearchResult(
                    faq=result.faq,
                    similarity_score=result.similarity
                ) for result in search_results
            ]
    except Exception as e:
        logger.warning(f"Cache-based search failed for query '{search_in.query}'", extra={"error": str(e)})
    
    # Fallback to database search
    faqs = faq_service.search_faqs(
        db,
        query=search_in.query,
        limit=search_in.limit,
        similarity_threshold=search_in.similarity_threshold
    )
    
    logger.info(f"Database search returned {len(faqs)} results for query: '{search_in.query}'")
    
    # Return without similarity scores for database results
    return [
        FAQSearchResult(faq=faq, similarity_score=None)
        for faq in faqs
    ]


# Cache management endpoints
@router.get("/cache/stats", response_model=CacheStats)
def get_cache_stats(
    *,
    request: Request,
    current_user: models.User = Depends(deps.get_current_user),
) -> CacheStats:
    """
    Get cache statistics (admin only).
    """
    # Add permission check if needed
    # if not current_user.is_admin:
    #     raise HTTPException(status_code=403, detail="Admin access required")
    cache_service = request.state.injector.get(CacheService)
    return cache_service.get_cache_stats()


@router.get("/cache/health", response_model=Dict[str, Any])
def get_cache_health(
    *,
    request: Request,
    current_user: models.User = Depends(deps.get_current_user),
) -> Dict[str, Any]:
    """
    Get cache health status (admin only).
    """
    # Add permission check if needed
    # if not current_user.is_admin:
    #     raise HTTPException(status_code=403, detail="Admin access required")
    cache_service = request.state.injector.get(CacheService)
    return cache_service.health_check()


@router.delete("/cache/clear", status_code=status.HTTP_204_NO_CONTENT)
def clear_cache(
    *,
    request: Request,
    current_user: models.User = Depends(deps.get_current_user),
) -> None:
    """
    Clear all FAQ cache data (admin only).
    """
    # Add permission check if needed
    # if not current_user.is_admin:
    #     raise HTTPException(status_code=403, detail="Admin access required")
    cache_service = request.state.injector.get(CacheService)
    success = cache_service.clear_cache()
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear cache"
        )
    
    logger.info(f"Cache cleared by user {current_user.id}")


@router.post("/cache/warm-up", status_code=status.HTTP_202_ACCEPTED)
def warm_up_cache(
    *,
    request: Request,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_user),
    limit: int = Query(1000, ge=1, le=10000),
) -> Dict[str, Any]:
    """
    Warm up the cache by loading FAQs from database (admin only).
    """
    # Add permission check if needed
    # if not current_user.is_admin:
    #     raise HTTPException(status_code=403, detail="Admin access required")
    cache_service = request.state.injector.get(CacheService)
    faq_service = request.state.injector.get(FAQService)
    
    try:
        # Get FAQs from database
        faqs = crud.faq.get_active_faqs(db, skip=0, limit=limit)
        
        # Add them to cache
        cached_count = 0
        failed_count = 0
        
        for faq in faqs:
            try:
                success = cache_service.add_faq(faq)
                if success:
                    cached_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                logger.error(f"Failed to cache FAQ {faq.id} during warm-up", extra={"error": str(e)})
                failed_count += 1
        
        result = {
            "message": "Cache warm-up completed",
            "cached_count": cached_count,
            "failed_count": failed_count,
            "total_processed": len(faqs)
        }
        
        logger.info(f"Cache warm-up completed by user {current_user.id}", extra=result)
        return result
        
    except Exception as e:
        logger.error("Cache warm-up failed", extra={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cache warm-up failed: {str(e)}"
        )


@router.post("/cache/warm-frequently-accessed", status_code=status.HTTP_200_OK)
def warm_frequently_accessed_faqs(
    *,
    request: Request,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_user),
    limit: int = Query(100, ge=1, le=1000),
) -> Dict[str, Any]:
    """
    Warm up the cache with frequently accessed FAQs.
    """
    cache_service = request.state.injector.get(CacheService)
    
    try:
        cached_count = cache_service.warm_cache_with_frequently_accessed_faqs(db, limit=limit)
        
        result = {
            "message": "Cache warmed with frequently accessed FAQs",
            "cached_count": cached_count,
            "limit": limit
        }
        
        logger.info(f"Cache warmed with frequently accessed FAQs by user {current_user.id}", extra=result)
        return result
        
    except Exception as e:
        logger.error("Failed to warm cache with frequently accessed FAQs", extra={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cache warming failed: {str(e)}"
        )


@router.post("/cache/evict-lru", status_code=status.HTTP_200_OK)
def evict_least_recently_used(
    *,
    request: Request,
    current_user: models.User = Depends(deps.get_current_user),
    max_faqs: int = Query(1000, ge=1, le=10000),
) -> Dict[str, Any]:
    """
    Evict least recently used FAQs from cache to manage memory.
    """
    cache_service = request.state.injector.get(CacheService)
    
    try:
        evicted_count = cache_service.evict_least_recently_used_faqs(max_faqs=max_faqs)
        
        result = {
            "message": "Least recently used FAQs evicted from cache",
            "evicted_count": evicted_count,
            "max_faqs": max_faqs
        }
        
        logger.info(f"LRU FAQs evicted from cache by user {current_user.id}", extra=result)
        return result
        
    except Exception as e:
        logger.error("Failed to evict LRU FAQs from cache", extra={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cache eviction failed: {str(e)}"
        )


@router.get("/cache/auto-manage/status", status_code=status.HTTP_200_OK)
def get_cache_management_status(
    *,
    request: Request,
    current_user: models.User = Depends(deps.get_current_user),
) -> Dict[str, Any]:
    """
    Get the status of automatic cache management.
    """
    try:
        from private_gpt.server.cache.cache_manager import CacheManager
        cache_manager = request.state.injector.get(CacheManager)
        
        result = {
            "message": "Cache management status",
            "is_background_thread_running": cache_manager._is_running,
            "background_thread_name": cache_manager._background_thread.name if cache_manager._background_thread else None,
            "is_stop_event_set": cache_manager._stop_event.is_set() if cache_manager._stop_event else False
        }
        
        return result
        
    except Exception as e:
        logger.error("Failed to get cache management status", extra={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cache management status: {str(e)}"
        )


@router.post("/cache/auto-manage/start", status_code=status.HTTP_200_OK)
def start_cache_management(
    *,
    request: Request,
    current_user: models.User = Depends(deps.get_current_user),
) -> Dict[str, Any]:
    """
    Start automatic cache management background thread.
    """
    try:
        from private_gpt.server.cache.cache_manager import CacheManager
        from private_gpt.users.db.session import SessionLocal
        cache_manager = request.state.injector.get(CacheManager)
        
        success = cache_manager.start_background_cache_management(
            db_session_factory=SessionLocal,
            warm_interval=3600,  # Every hour
            eviction_interval=1800  # Every 30 minutes
        )
        
        if success:
            result = {
                "message": "Automatic cache management started successfully",
                "is_background_thread_running": cache_manager._is_running
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to start automatic cache management"
            )
            
        return result
        
    except Exception as e:
        logger.error("Failed to start automatic cache management", extra={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start automatic cache management: {str(e)}"
        )


@router.post("/cache/auto-manage/stop", status_code=status.HTTP_200_OK)
def stop_cache_management(
    *,
    request: Request,
    current_user: models.User = Depends(deps.get_current_user),
) -> Dict[str, Any]:
    """
    Stop automatic cache management background thread.
    """
    try:
        from private_gpt.server.cache.cache_manager import CacheManager
        cache_manager = request.state.injector.get(CacheManager)
        
        success = cache_manager.stop_background_cache_management()
        
        if success:
            result = {
                "message": "Automatic cache management stopped successfully",
                "is_background_thread_running": cache_manager._is_running
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to stop automatic cache management"
            )
            
        return result
        
    except Exception as e:
        logger.error("Failed to stop automatic cache management", extra={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop automatic cache management: {str(e)}"
        )


@router.post("/cache/refresh/{faq_id}", status_code=status.HTTP_200_OK)
def refresh_faq_cache(
    *,
    request: Request,
    db: Session = Depends(deps.get_db),
    faq_id: int,
    current_user: models.User = Depends(deps.get_current_user),
) -> Dict[str, Any]:
    """
    Refresh a specific FAQ in cache with the latest data from database.
    """
    faq_service = request.state.injector.get(FAQService)
    
    try:
        success = faq_service.refresh_faq_cache(db, faq_id)
        
        if success:
            result = {
                "message": f"FAQ {faq_id} cache refreshed successfully",
                "faq_id": faq_id
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"FAQ {faq_id} not found or failed to refresh"
            )
            
        return result
        
    except Exception as e:
        logger.error(f"Failed to refresh FAQ {faq_id} cache", extra={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to refresh FAQ {faq_id} cache: {str(e)}"
        )


@router.post("/cache/refresh-all", status_code=status.HTTP_200_OK)
def refresh_all_faq_cache(
    *,
    request: Request,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_user),
    limit: int = Query(1000, ge=1, le=10000),
) -> Dict[str, Any]:
    """
    Refresh all FAQ cache with the latest data from database.
    """
    faq_service = request.state.injector.get(FAQService)
    
    try:
        refreshed_count = faq_service.refresh_all_faq_cache(db, limit)
        
        result = {
            "message": "All FAQ cache refreshed successfully",
            "refreshed_count": refreshed_count,
            "limit": limit
        }
        
        logger.info(f"All FAQ cache refreshed by user {current_user.id}", extra=result)
        return result
        
    except Exception as e:
        logger.error("Failed to refresh all FAQ cache", extra={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to refresh all FAQ cache: {str(e)}"
        )
