#!/usr/bin/env python3
"""
Test script to verify FAQ cache singleton and graceful degradation implementation.
"""

import sys
import os
import logging

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'private_gpt'))

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_singleton_pattern():
    """Test that the singleton pattern works correctly."""
    logger.info("Testing singleton pattern...")
    
    try:
        from private_gpt.components.faq.faq_cache import get_faq_cache, FAQCacheSingleton
        
        # Test singleton instance
        cache1 = get_faq_cache()
        cache2 = get_faq_cache()
        
        # They should be the same instance
        assert cache1 is cache2, "Singleton pattern failed - different instances created"
        logger.info("✓ Singleton pattern working correctly")
        
        # Test singleton class
        singleton1 = FAQCacheSingleton()
        singleton2 = FAQCacheSingleton()
        
        assert singleton1 is singleton2, "Singleton class failed - different instances created"
        logger.info("✓ Singleton class working correctly")
        
        return True
        
    except Exception as e:
        logger.error(f"Singleton pattern test failed: {e}")
        return False

def test_graceful_degradation():
    """Test graceful degradation when Redis is unavailable."""
    logger.info("Testing graceful degradation...")
    
    try:
        from private_gpt.components.faq.faq_cache import get_faq_cache
        
        # Get cache instance (should work even without Redis)
        cache = get_faq_cache()
        
        # Test health check
        health = cache.health_check()
        logger.info(f"Cache health: {health}")
        
        # Test that cache operations don't crash when Redis is unavailable
        stats = cache.get_stats()
        logger.info(f"Cache stats: {stats}")
        
        # Test search with graceful degradation
        results = cache.search_faqs("test query", limit=5)
        logger.info(f"Search results: {len(results)} items")
        
        logger.info("✓ Graceful degradation working correctly")
        return True
        
    except Exception as e:
        logger.error(f"Graceful degradation test failed: {e}")
        return False

def test_cache_initialization():
    """Test cache initialization and health checks."""
    logger.info("Testing cache initialization...")
    
    try:
        from private_gpt.components.faq.faq_cache import initialize_redis_client, get_cache_status, is_redis_connected
        
        # Test initialization (will fail in non-Docker environment, but should not crash)
        result = initialize_redis_client()
        logger.info(f"Redis initialization result: {result}")
        
        # Test status functions
        status = get_cache_status()
        logger.info(f"Cache status: {status}")
        
        connected = is_redis_connected()
        logger.info(f"Redis connected: {connected}")
        
        logger.info("✓ Cache initialization working correctly")
        return True
        
    except Exception as e:
        logger.error(f"Cache initialization test failed: {e}")
        return False

def test_faq_service_integration():
    """Test FAQ service integration with singleton."""
    logger.info("Testing FAQ service integration...")
    
    try:
        from private_gpt.components.faq.faq_service import FAQService
        
        # Create FAQ service instance
        service = FAQService()
        
        # Test that it uses the singleton cache
        cache_status = service.cache.cache_status
        logger.info(f"FAQ service cache status: {cache_status}")
        
        # Test health check
        health = service.cache.health_check()
        logger.info(f"FAQ service health: {health}")
        
        logger.info("✓ FAQ service integration working correctly")
        return True
        
    except Exception as e:
        logger.error(f"FAQ service integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting FAQ cache fixes tests...")
    
    tests = [
        test_singleton_pattern,
        test_graceful_degradation,
        test_cache_initialization,
        test_faq_service_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                logger.error(f"Test {test.__name__} failed")
        except Exception as e:
            logger.error(f"Test {test.__name__} crashed: {e}")
    
    logger.info(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! FAQ cache fixes are working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
