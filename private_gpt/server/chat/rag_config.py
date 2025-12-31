"""
Centralized RAG configuration module.

This module contains all configurable parameters for the RAG system,
making it easy to tune performance and behavior.
"""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class FAQCacheConfig:
    """Configuration for FAQ caching"""
    enabled: bool = True
    similarity_threshold: float = 0.9
    result_limit: int = 3
    ttl_seconds: int = 3600  # 1 hour


@dataclass
class RetrievalConfig:
    """Configuration for document retrieval"""
    similarity_top_k: int = 5
    similarity_cutoff: float = 0.7
    enable_reranking: bool = True
    rerank_top_n: int = 10
    enable_hybrid_search: bool = False  # Vector + keyword
    keyword_weight: float = 0.3  # For hybrid search


@dataclass
class QueryProcessingConfig:
    """Configuration for query processing"""
    max_sub_queries: int = 3
    enable_query_decomposition: bool = True
    decomposition_threshold_words: int = 15  # Min words to consider decomposition
    condense_timeout_seconds: float = 30.0
    decompose_timeout_seconds: float = 30.0
    retrieval_timeout_seconds: float = 60.0


@dataclass
class MemoryConfig:
    """Configuration for chat memory"""
    token_limit_ratio: float = 0.7  # Ratio of context window
    max_messages: int = 100
    enable_compression: bool = False


@dataclass
class ResponseConfig:
    """Configuration for response generation"""
    max_response_tokens: Optional[int] = None
    temperature: float = 0.1  # Low for factual responses
    enable_streaming: bool = True
    citation_format: str = "[{filename}]"  # or "[{filename}, p. {page}]"


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization"""
    enable_caching: bool = True
    cache_ttl_seconds: int = 1800  # 30 minutes
    parallel_retrieval: bool = True
    max_concurrent_retrievals: int = 5
    enable_early_stopping: bool = False
    early_stop_confidence: float = 0.95


@dataclass
class ObservabilityConfig:
    """Configuration for logging and monitoring"""
    enable_structured_logging: bool = True
    log_level: str = "INFO"
    enable_metrics: bool = True
    enable_tracing: bool = False
    verbose_mode: bool = False


@dataclass
class RAGConfig:
    """Main RAG configuration combining all sub-configurations"""
    faq_cache: FAQCacheConfig = field(default_factory=FAQCacheConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    query_processing: QueryProcessingConfig = field(default_factory=QueryProcessingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    response: ResponseConfig = field(default_factory=ResponseConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)

    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Create configuration from environment variables"""
        config = cls()
        
        # FAQ Cache
        if os.getenv("FAQ_CACHE_ENABLED"):
            config.faq_cache.enabled = os.getenv("FAQ_CACHE_ENABLED").lower() == "true"
        if os.getenv("FAQ_SIMILARITY_THRESHOLD"):
            config.faq_cache.similarity_threshold = float(os.getenv("FAQ_SIMILARITY_THRESHOLD"))
        
        # Retrieval
        if os.getenv("SIMILARITY_TOP_K"):
            config.retrieval.similarity_top_k = int(os.getenv("SIMILARITY_TOP_K"))
        if os.getenv("SIMILARITY_CUTOFF"):
            config.retrieval.similarity_cutoff = float(os.getenv("SIMILARITY_CUTOFF"))
        if os.getenv("ENABLE_RERANKING"):
            config.retrieval.enable_reranking = os.getenv("ENABLE_RERANKING").lower() == "true"
        
        # Query Processing
        if os.getenv("MAX_SUB_QUERIES"):
            config.query_processing.max_sub_queries = int(os.getenv("MAX_SUB_QUERIES"))
        if os.getenv("CONDENSE_TIMEOUT"):
            config.query_processing.condense_timeout_seconds = float(os.getenv("CONDENSE_TIMEOUT"))
        
        # Performance
        if os.getenv("ENABLE_CACHING"):
            config.performance.enable_caching = os.getenv("ENABLE_CACHING").lower() == "true"
        if os.getenv("PARALLEL_RETRIEVAL"):
            config.performance.parallel_retrieval = os.getenv("PARALLEL_RETRIEVAL").lower() == "true"
        
        # Observability
        if os.getenv("LOG_LEVEL"):
            config.observability.log_level = os.getenv("LOG_LEVEL")
        if os.getenv("VERBOSE_MODE"):
            config.observability.verbose_mode = os.getenv("VERBOSE_MODE").lower() == "true"
        
        return config

    def validate(self) -> None:
        """Validate configuration values"""
        assert 0.0 <= self.faq_cache.similarity_threshold <= 1.0, "FAQ similarity threshold must be between 0 and 1"
        assert 0.0 <= self.retrieval.similarity_cutoff <= 1.0, "Similarity cutoff must be between 0 and 1"
        assert self.retrieval.similarity_top_k > 0, "Top-k must be positive"
        assert self.query_processing.max_sub_queries > 0, "Max sub-queries must be positive"
        assert 0.0 <= self.memory.token_limit_ratio <= 1.0, "Token limit ratio must be between 0 and 1"
        assert 0.0 <= self.response.temperature <= 2.0, "Temperature must be between 0 and 2"


# Global default configuration
DEFAULT_RAG_CONFIG = RAGConfig()

# Load from environment if available
try:
    RAG_CONFIG = RAGConfig.from_env()
    RAG_CONFIG.validate()
except Exception as e:
    print(f"Warning: Failed to load RAG config from environment: {e}")
    print("Using default configuration")
    RAG_CONFIG = DEFAULT_RAG_CONFIG
