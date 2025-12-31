import logging
from enum import Enum
from typing import Optional, Any
import threading
import redis
from redis.exceptions import RedisError, ConnectionError, TimeoutError
from injector import inject, singleton

from private_gpt.settings.settings import settings

logger = logger = logging.getLogger(__name__)


class CacheStatus(Enum):
    """Enum for cache status."""
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    DEGRADED = "degraded"  # Redis unavailable, using fallback


@singleton
class CacheComponent:
    """Component responsible for managing FAQ cache backed by Redis."""

    _lock = threading.Lock()

    @inject
    def __init__(self) -> None:
        """Initialize the FAQ cache component."""
        self._redis_client: Optional[redis.Redis] = None
        self._embedding_model: Optional[Any] = None
        self._status: CacheStatus = CacheStatus.DISCONNECTED
        logger.debug("FAQComponent created (status=DISCONNECTED)")

    @property
    def status(self) -> CacheStatus:
        """Current cache status."""
        return self._status

    @property
    def redis_client(self) -> Optional[redis.Redis]:
        """Expose Redis client if connected."""
        return self._redis_client

    def initialize(self, embedding_model: Optional[Any] = None, force_reconnect: bool = False) -> bool:
        """
        Initialize Redis connection and embedding model.

        Args:
            embedding_model: Optional embedding model for similarity search.
            force_reconnect: If True, force reinitialization even if already connected.

        Returns:
            bool: True if Redis is connected, False otherwise.
        """
        with self._lock:
            if not settings().faq.enabled:
                logger.info("FAQ feature is disabled in settings, skipping Redis initialization.")
                self._status = CacheStatus.DISCONNECTED
                return False

            if self._redis_client is not None and not force_reconnect:
                logger.debug("Redis client already initialized, skipping reconnect")
                return self._status == CacheStatus.CONNECTED

            try:
                logger.info("Initializing Redis client for FAQ cache...")
                redis_config = settings().redis

                self._redis_client = redis.Redis(
                    host=redis_config.host,
                    port=redis_config.port,
                    password=redis_config.password or None,
                    db=redis_config.db,
                    decode_responses=False,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30,
                )

                # Test connection
                logger.debug("Pinging Redis...")
                self._redis_client.ping()
                self._status = CacheStatus.CONNECTED
                logger.info("Successfully connected to Redis for FAQ cache")

                # Set embedding model if provided
                self._embedding_model = embedding_model
                if embedding_model:
                    logger.info("Embedding model set for FAQ cache")
                else:
                    logger.warning("No embedding model provided for FAQ cache")

                return True

            except (ConnectionError, TimeoutError, RedisError) as e:
                logger.error("Failed to connect to Redis for FAQ cache", error=str(e))
                self._redis_client = None
                self._status = CacheStatus.DEGRADED
                logger.warning("FAQ cache will operate in degraded mode (fallback only)")
                return False

            except Exception as e:
                logger.exception("Unexpected error during FAQ cache initialization")
                self._redis_client = None
                self._status = CacheStatus.DEGRADED
                return False
