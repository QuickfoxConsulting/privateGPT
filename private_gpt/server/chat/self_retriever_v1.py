import logging
from typing import List, Tuple, Optional, Dict, Union, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import re
import hashlib
import time
import json
from functools import lru_cache
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from llama_index.core import PromptTemplate, ServiceContext
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import BaseNode, NodeWithScore, TextNode, Document
from llama_index.core import QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from private_gpt.components.llm.llm_component import LLMComponent

logger = logging.getLogger(__name__)

class RetrievalDecision(Enum):
    RETRIEVE = "RETRIEVE"  # Perform new retrieval
    SKIP = "SKIP"          # Skip retrieval entirely
    FOLLOWUP = "FOLLOWUP"  # Use existing context
    HYBRID = "HYBRID"      # Combine existing context with new retrieval

@dataclass
class RetrievalMetrics:
    total_nodes: int = 0
    filtered_nodes: int = 0
    retrieval_time: float = 0.0
    filtering_time: float = 0.0
    decision: RetrievalDecision = RetrievalDecision.RETRIEVE
    decision_reason: str = ""
    cache_hits: int = 0
    cache_misses: int = 0
    llm_call_count: int = 0
    llm_call_time: float = 0.0
    error_count: int = 0
    parallel_tasks: int = 0
    node_sources: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging/analysis"""
        return {
            "total_nodes": self.total_nodes,
            "filtered_nodes": self.filtered_nodes,
            "retrieval_time": round(self.retrieval_time, 3),
            "filtering_time": round(self.filtering_time, 3),
            "decision": self.decision.value,
            "decision_reason": self.decision_reason,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_ratio": round(self.cache_hits / (self.cache_hits + self.cache_misses or 1), 2),
            "llm_call_count": self.llm_call_count,
            "llm_call_time": round(self.llm_call_time, 3),
            "error_count": self.error_count,
            "parallel_tasks": self.parallel_tasks,
            "node_sources": dict(self.node_sources)
        }

class ThreadSafeLRUCache:
    """Thread-safe LRU cache implementation with expiration"""
    def __init__(self, capacity: int, ttl: int = 3600):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.ttl = ttl  # Time-to-live in seconds
        self.timestamps = {}  # Track when entries were added
        from threading import RLock
        self._lock = RLock()  # RLock allows reentrant locking
    
    def get(self, key: str) -> Any:
        with self._lock:
            if key not in self.cache:
                return None
                
            # Check expiration
            if self.ttl > 0:
                timestamp = self.timestamps.get(key, 0)
                if time.time() - timestamp > self.ttl:
                    # Entry expired
                    self.cache.pop(key, None)
                    self.timestamps.pop(key, None)
                    return None
            
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
    
    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.capacity:
                # Remove oldest item
                self.cache.popitem(last=False)
                
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def clear(self) -> None:
        with self._lock:
            self.cache.clear()
            self.timestamps.clear()
    
    def get_all_keys(self) -> Set[str]:
        with self._lock:
            return set(self.cache.keys())
    
    def __len__(self) -> int:
        with self._lock:
            return len(self.cache)

@dataclass
class SelfRAGConfig:
    """Configuration class for SelfRAGRetriever"""
    # Retrieval settings
    max_nodes: int = 5
    relevance_threshold: float = 0.6
    dynamic_relevance: bool = True  # Adjust threshold based on context
    
    # Caching
    cache_size: int = 512
    content_cache_size: int = 1000
    relevance_cache_size: int = 1000
    context_history_size: int = 10
    cache_ttl: int = 3600  # Cache time-to-live in seconds
    enable_caching: bool = True
    
    # Scoring
    high_relevance_bonus: float = 0.1
    recency_bonus: float = 0.05  # Bonus for recently retrieved docs
    cross_validation_threshold: int = 3  # Validate with multiple instances
    
    # Performance
    parallel_critique: bool = True
    max_critique_workers: int = 5
    timeout_per_critique: float = 5.0
    
    # Misc 
    debug_mode: bool = False
    max_retries: int = 3
    fallback_to_base: bool = True  # Fallback to base retriever on failure
    
    # Query rewriting
    enable_query_rewriting: bool = False
    query_rewriting_threshold: float = 0.3  # Rewrite if score < threshold
    
    def verify(self) -> None:
        """Verify configuration is valid"""
        if self.max_nodes <= 0:
            raise ValueError("max_nodes must be positive")
        if not 0 <= self.relevance_threshold <= 1:
            raise ValueError("relevance_threshold must be between 0 and 1")
        if self.enable_caching and self.cache_size <= 0:
            raise ValueError("cache_size must be positive when caching is enabled")

class SelfRAGRetriever(BaseRetriever):
    """Enhanced Self-RAG Retriever with advanced filtering capabilities"""
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm: LLMComponent,
        config: Optional[SelfRAGConfig] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        critique_prompt: Optional[str] = None,
        query_rewrite_prompt: Optional[str] = None,
        service_context: Optional[ServiceContext] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.base_retriever = base_retriever
        self.llm = llm
        self.config = config or SelfRAGConfig()
        self.config.verify()
        self._node_postprocessors = node_postprocessors or []
        self._service_context = service_context
        
        # Initialize improved caches with TTL
        if self.config.enable_caching:
            self._content_cache = ThreadSafeLRUCache(self.config.content_cache_size, self.config.cache_ttl)
            self._relevance_cache = ThreadSafeLRUCache(self.config.relevance_cache_size, self.config.cache_ttl)
            self._context_history = ThreadSafeLRUCache(self.config.context_history_size, self.config.cache_ttl)
            self._query_rewrite_cache = ThreadSafeLRUCache(self.config.cache_size, self.config.cache_ttl)
        
        # Store critique results per session for cross-validation
        self._critique_history = defaultdict(list)
        
        # Enhanced prompts
        self.critique_prompt_template = PromptTemplate(
            template=critique_prompt or self._get_default_critique_prompt()
        )
        
        self.query_rewrite_prompt_template = PromptTemplate(
            template=query_rewrite_prompt or self._get_default_query_rewrite_prompt()
        )
        
        # Thread pool for parallel critiques
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_critique_workers)
        
        # Current metrics
        self.metrics: Optional[RetrievalMetrics] = None

    def _get_default_critique_prompt(self) -> str:
        """Returns an enhanced critique prompt with more specific evaluation criteria"""
        return """
        You are an expert evaluator analyzing whether a passage is relevant to a specific query.
        
        Passage content:
        -------------------
        {context}

        User query:
        ---------------
        {query}

        Evaluate on multiple dimensions (score 0-10):
        1. Direct relevance: How directly does it answer the query?
        2. Information quality: How specific, detailed, and accurate is the information?
        3. Context completeness: Does it provide sufficient context to be useful?
        4. Authority/credibility: How authoritative is this information?
        
        Analyze how the passage would contribute to answering the query. Consider:
        - Factual precision and specificity to the query
        - Whether the passage contains key information needed in a response
        - If the passage provides unique context not found in general knowledge
        
        Required format:
        Direct relevance: [score 0-10]
        Information quality: [score 0-10] 
        Context completeness: [score 0-10]
        Authority/credibility: [score 0-10]
        
        Overall confidence: [0.0-1.0]
        RELEVANT: [YES/NO/PARTIAL]
        
        Reasoning: [2-3 sentence explanation]
        """

    def _get_default_query_rewrite_prompt(self) -> str:
        """Returns prompt for query rewriting when initial results are poor"""
        return """
        Based on the initial search results and user query, determine if the query needs to be rewritten 
        to improve search effectiveness.
        
        Original Query: {query}
        
        Initial Results Quality: {results_quality}
        
        Your task:
        1. Analyze why the original query might not be retrieving optimal results
        2. Identify any ambiguities, missing terms, or domain-specific language needed
        3. If needed, rewrite the query to be more precise and effective
        
        If rewriting:
        - Preserve the original intent completely
        - Be more specific and include key technical terms
        - Break down complex questions into clearer components
        - Do not introduce new requirements not in the original query
        
        Output:
        REWRITE NEEDED: [YES/NO]
        REWRITTEN QUERY: [New query if needed, otherwise "N/A"]
        REASONING: [Brief explanation]
        """

    def _safe_llm_call(self, prompt: str, max_retries: int = None, timeout: Optional[float] = None) -> str:
        """Make LLM calls with retries, error handling and metrics tracking"""
        if max_retries is None:
            max_retries = self.config.max_retries
            
        metrics = self.metrics or RetrievalMetrics()
        metrics.llm_call_count += 1
        
        start_time = time.time()
        for attempt in range(max_retries):
            try:
                # Use a timeout if specified
                result = self.llm.complete(prompt, timeout=timeout).text.strip()
                metrics.llm_call_time += (time.time() - start_time)
                return result
            except Exception as e:
                if self.config.debug_mode:
                    logger.error(f"LLM call failed (attempt {attempt+1}/{max_retries}): {e}")
                metrics.error_count += 1
                if attempt == max_retries - 1:
                    metrics.llm_call_time += (time.time() - start_time)
                    logger.warning(f"LLM call failed after {max_retries} attempts")
                    return ""
                # Exponential backoff with jitter
                delay = (2 ** attempt) + (0.1 * (time.time() % 1.0))
                time.sleep(min(delay, 10))  # Cap at 10 seconds
                
        metrics.llm_call_time += (time.time() - start_time)
        return ""

    def _critique_node(self, node: NodeWithScore, query: str) -> Tuple[bool, float, str]:
        """Evaluate node relevance with enhanced criteria"""
        content = node.node.get_content()
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Store content in cache for future reference
        if self.config.enable_caching:
            self._content_cache.put(content_hash, content)
            
        cache_key = f"{query}:{content_hash}"
        cached_result = self._relevance_cache.get(cache_key) if self.config.enable_caching else None
        
        if cached_result:
            self.metrics.cache_hits += 1
            return cached_result
            
        self.metrics.cache_misses += 1
        result = self._get_critique_result(content, query)
        
        if self.config.enable_caching:
            self._relevance_cache.put(cache_key, result)
            
        # Store in critique history for cross-validation
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        self._critique_history[query_hash].append((content_hash, result))
            
        return result

    def _get_critique_result(self, context: str, query: str) -> Tuple[bool, float, str]:
        """Get critique result from LLM with enhanced error handling"""
        # Truncate context if too long
        max_context_len = 4000
        if len(context) > max_context_len:
            context = context[:max_context_len] + "..."
            
        prompt = self.critique_prompt_template.format(
            context=context,
            query=query
        )
        
        response = self._safe_llm_call(
            prompt, 
            timeout=self.config.timeout_per_critique if self.config.parallel_critique else None
        )
        
        return self._parse_critique_response(response)

    def _parse_critique_response(self, response: str) -> Tuple[bool, float, str]:
        """Parse LLM critique response with improved robustness"""
        scores = {
            'direct': 0.0,
            'quality': 0.0,
            'completeness': 0.0,
            'authority': 0.0,
            'confidence': 0.0,
            'relevant': False,
            'reasoning': ""
        }
        
        # Default fallback reasoning
        fallback_reasoning = "Unable to properly evaluate this content."
        
        try:
            # Normalize response for case-insensitive matching
            normalized = response.lower()
            
            # Extract all scores with improved regex pattern
            direct_match = re.search(r"direct relevance:?\s*(\d+\.?\d*)", normalized)
            if direct_match:
                scores['direct'] = min(10.0, float(direct_match.group(1)))
            
            quality_match = re.search(r"information quality:?\s*(\d+\.?\d*)", normalized)
            if quality_match:
                scores['quality'] = min(10.0, float(quality_match.group(1)))
                
            completeness_match = re.search(r"context completeness:?\s*(\d+\.?\d*)", normalized)
            if completeness_match:
                scores['completeness'] = min(10.0, float(completeness_match.group(1)))
                
            authority_match = re.search(r"authority/credibility:?\s*(\d+\.?\d*)", normalized)
            if authority_match:
                scores['authority'] = min(10.0, float(authority_match.group(1)))
            
            # Extract Confidence
            confidence_match = re.search(r"overall confidence:?\s*([01]\.?\d*)", normalized)
            if confidence_match:
                scores['confidence'] = float(confidence_match.group(1))
            
            # Determine Relevance with support for PARTIAL
            relevant_match = re.search(r"relevant:?\s*(yes|no|partial)", normalized)
            if relevant_match:
                relevance = relevant_match.group(1).lower()
                if relevance == 'yes':
                    scores['relevant'] = True
                elif relevance == 'partial':
                    # Partial relevance - check if it meets threshold
                    composite_score = self._calculate_composite_score(scores)
                    scores['relevant'] = composite_score >= self.config.relevance_threshold
                else:
                    scores['relevant'] = False
            else:
                # Fallback: Determine relevance based on composite score
                composite_score = self._calculate_composite_score(scores)
                scores['relevant'] = composite_score >= self.config.relevance_threshold
            
            # Extract reasoning
            reasoning_match = re.search(r"reasoning:?\s*(.+?)(?:\n|$)", normalized, re.DOTALL)
            if reasoning_match:
                scores['reasoning'] = reasoning_match.group(1).strip()
            
            # Calculate final composite score
            composite_score = self._calculate_composite_score(scores)
            
            return scores['relevant'], composite_score, scores['reasoning'] or fallback_reasoning
            
        except Exception as e:
            if self.config.debug_mode:
                logger.error(f"Error parsing critique response: {e}")
            self.metrics.error_count += 1
            return False, 0.0, fallback_reasoning
    
    def _calculate_composite_score(self, scores: Dict[str, Any]) -> float:
        """Calculate weighted composite score from different dimensions"""
        # Weighted average of all dimensions
        weighted_score = (
            0.4 * (scores['direct'] / 10) +
            0.3 * (scores['quality'] / 10) +
            0.2 * (scores['completeness'] / 10) +
            0.1 * (scores['authority'] / 10)
        ) 
        
        # Apply confidence as a multiplier
        composite_score = weighted_score * (scores['confidence'] or 0.5)
        
        return min(1.0, max(0.0, composite_score))  # Clamp between 0 and 1

    def _should_retrieve(self, query: str, context: Optional[List[NodeWithScore]] = None) -> Tuple[RetrievalDecision, str]:
        """Enhanced decision logic for retrieval necessity"""
        # If no context at all, we definitely need to retrieve
        if not context:
            return RetrievalDecision.RETRIEVE, "No existing context available"
        
        # Extract recent chat history (last 3 exchanges)
        chat_history = "\n".join([
            f"{i+1}. {n.node.get_content()[:150]}..." 
            for i, n in enumerate(context[-3:])
        ])
        
        # Calculate average relevance score of existing context
        avg_relevance = sum(n.score for n in context) / len(context) if context else 0
        
        prompt = f"""Analyze if new information retrieval is needed for this query.
        
        [EXISTING CONTEXT]
        {chat_history}
        
        [NEW QUERY]
        {query}
        
        [CONTEXT DETAILS]
        Average relevance: {avg_relevance:.2f} (0-1 scale)
        Number of context items: {len(context)}
        
        [DECISION OPTIONS]
        - RETRIEVE: Get fresh information (when query needs new data)
        - FOLLOWUP: Use existing context (when context contains needed information)
        - HYBRID: Combine existing context with new retrieval (when partial information exists)
        - SKIP: No retrieval needed (when query is clarification or doesn't need data)
        
        First analyze if the query:
        1. Introduces a new topic not in context
        2. Requests information already covered in context
        3. Seeks deeper details on context topics
        4. Is a clarification or follow-up on existing information
        
        Provide your reasoning, then state ONE decision word: RETRIEVE, FOLLOWUP, HYBRID, or SKIP.
        """
        
        response = self._safe_llm_call(prompt)
        return self._parse_retrieval_response(response, context, avg_relevance)

    def _parse_retrieval_response(self, response: str, context: Optional[List[NodeWithScore]], avg_relevance: float) -> Tuple[RetrievalDecision, str]:
        """Parse retrieval decision with fallbacks based on context quality"""
        decision_match = re.search(r"\b(RETRIEVE|SKIP|FOLLOWUP|HYBRID)\b", response, re.IGNORECASE)
        
        reasoning = re.sub(r"\b(RETRIEVE|SKIP|FOLLOWUP|HYBRID)\b", "", response, flags=re.IGNORECASE).strip()
        
        if decision_match:
            decision = RetrievalDecision(decision_match.group(1).upper())
        else:
            # Fallback logic based on context quality
            if not context:
                decision = RetrievalDecision.RETRIEVE
                reasoning = "No context available (fallback decision)"
            elif avg_relevance >= 0.8:
                decision = RetrievalDecision.FOLLOWUP
                reasoning = "High quality context available (fallback decision)"
            elif avg_relevance >= 0.4:
                decision = RetrievalDecision.HYBRID
                reasoning = "Medium quality context available (fallback decision)"
            else:
                decision = RetrievalDecision.RETRIEVE
                reasoning = "Low quality context (fallback decision)"
                
        return decision, reasoning

    def _maybe_rewrite_query(self, query: str, results_quality: str) -> str:
        """Potentially rewrite query to improve retrieval results"""
        if not self.config.enable_query_rewriting:
            return query
            
        cache_key = f"rewrite_{hashlib.md5(query.encode()).hexdigest()}"
        cached_result = self._query_rewrite_cache.get(cache_key) if self.config.enable_caching else None
        
        if cached_result:
            return cached_result
            
        prompt = self.query_rewrite_prompt_template.format(
            query=query,
            results_quality=results_quality
        )
        
        response = self._safe_llm_call(prompt)
        
        # Extract rewritten query if available
        rewrite_match = re.search(r"REWRITE NEEDED:\s*(YES|NO)", response, re.IGNORECASE)
        if rewrite_match and rewrite_match.group(1).upper() == "YES":
            query_match = re.search(r"REWRITTEN QUERY:\s*(.+?)(?:\n|$)", response, re.DOTALL)
            if query_match:
                rewritten_query = query_match.group(1).strip()
                if rewritten_query and rewritten_query.lower() != "n/a":
                    if self.config.enable_caching:
                        self._query_rewrite_cache.put(cache_key, rewritten_query)
                    logger.info(f"Rewrote query: '{query}' → '{rewritten_query}'")
                    return rewritten_query
        
        return query

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Main retrieval method with enhanced error handling"""
        try:
            return self._retrieve(query_bundle)
        except Exception as e:
            logger.error(f"Error during retrieval: {e}", exc_info=True)
            if self.config.fallback_to_base and self.base_retriever:
                logger.info("Falling back to base retriever")
                return self.base_retriever.retrieve(query_bundle)
            return []

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Enhanced retrieval implementation with parallel processing"""
        start_time = time.time()
        
        # Initialize metrics
        metrics = RetrievalMetrics()
        self.metrics = metrics
        
        if isinstance(query_bundle, str):
            query_bundle = QueryBundle(query_bundle)
        
        query_str = query_bundle.query_str
        
        # Check context history
        context_key = hashlib.sha256(query_str.encode()).hexdigest()
        previous_context = self._context_history.get(context_key) if self.config.enable_caching else None
        
        if previous_context:
            # Adjust relevance threshold if dynamic relevance is enabled
            threshold = self.config.relevance_threshold
            if self.config.dynamic_relevance:
                # Calculate dynamic threshold based on context quality
                avg_score = sum(node.score for node in previous_context) / len(previous_context)
                threshold = max(0.4, min(0.8, self.config.relevance_threshold - (0.7 - avg_score)))
                
            high_relevance_threshold = threshold + self.config.high_relevance_bonus
            
            highly_relevant = [
                node for node in previous_context 
                if node.score >= high_relevance_threshold
            ]
            
            if highly_relevant:
                metrics.decision = RetrievalDecision.FOLLOWUP
                metrics.decision_reason = "Found highly relevant previous context"
                metrics.filtered_nodes = len(highly_relevant)
                return highly_relevant[:self.config.max_nodes]
        
        # Determine if retrieval is needed
        decision, reasoning = self._should_retrieve(query_str, previous_context)
        metrics.decision = decision
        metrics.decision_reason = reasoning
        
        if decision == RetrievalDecision.SKIP:
            metrics.filtered_nodes = 0
            return []
            
        if decision == RetrievalDecision.FOLLOWUP and previous_context:
            # Use existing context but add recency bonus to newer items
            for i, node in enumerate(previous_context):
                recency_bonus = self.config.recency_bonus * (len(previous_context) - i) / len(previous_context)
                node.score = min(1.0, node.score + recency_bonus)
                
            filtered = sorted(
                previous_context,
                key=lambda x: x.score,
                reverse=True
            )[:self.config.max_nodes]
            
            metrics.filtered_nodes = len(filtered)
            return filtered
        
        # Perform retrieval with base retriever
        try:
            nodes = self.base_retriever.retrieve(query_bundle)
        except Exception as e:
            logger.error(f"Base retriever error: {e}")
            metrics.error_count += 1
            return []
        
        metrics.total_nodes = len(nodes)
        metrics.retrieval_time = time.time() - start_time
        
        # If no nodes returned, try query rewriting
        if len(nodes) == 0 and self.config.enable_query_rewriting:
            rewritten_query = self._maybe_rewrite_query(
                query_str, 
                "Poor - no results returned"
            )
            
            if rewritten_query != query_str:
                rewritten_bundle = QueryBundle(rewritten_query)
                try:
                    nodes = self.base_retriever.retrieve(rewritten_bundle)
                    metrics.total_nodes = len(nodes)
                except Exception as e:
                    logger.error(f"Rewrite retriever error: {e}")
                    metrics.error_count += 1
        
        # Track node sources (if available)
        for node in nodes:
            if hasattr(node.node, 'metadata') and 'source' in node.node.metadata:
                source = str(node.node.metadata['source'])
                metrics.node_sources[source] += 1
        
        # Filter and score nodes with parallel processing
        filter_start = time.time()
        filtered_nodes = []
        
        if self.config.parallel_critique and len(nodes) > 1:
            # Parallel processing for multiple nodes
            future_to_node = {}
            metrics.parallel_tasks = len(nodes)
            
            with ThreadPoolExecutor(max_workers=min(self.config.max_critique_workers, len(nodes))) as executor:
                for i, node in enumerate(nodes):
                    future = executor.submit(self._critique_node, node, query_str)
                    future_to_node[future] = node
                
                for future in as_completed(future_to_node):
                    node = future_to_node[future]
                    try:
                        is_relevant, score, reasoning = future.result()
                        if is_relevant and score >= self.config.relevance_threshold:
                            filtered_nodes.append(NodeWithScore(node=node.node, score=score))
                    except Exception as e:
                        logger.error(f"Error in parallel node critique: {e}")
                        metrics.error_count += 1
        else:
            # Sequential processing
            for node in nodes:
                is_relevant, score, reasoning = self._critique_node(node, query_str)
                if is_relevant and score >= self.config.relevance_threshold:
                    filtered_nodes.append(NodeWithScore(node=node.node, score=score))
        
        if len(self._critique_history.get(context_key, [])) >= self.config.cross_validation_threshold:
            self._apply_cross_validation(filtered_nodes, context_key)
        
        # Cache results
        if filtered_nodes and self.config.enable_caching:
            self._context_history.put(
                context_key,
                filtered_nodes[:self.config.max_nodes * 2] 
            )
        
        if not filtered_nodes and self.config.enable_query_rewriting:
            results_quality = f"Poor - {len(nodes)} total nodes but none passed relevance filter"
            rewritten_query = self._maybe_rewrite_query(query_str, results_quality)
            
            if rewritten_query != query_str:
                rewritten_bundle = QueryBundle(rewritten_query)
                if not hasattr(rewritten_bundle, '_rewritten'):
                    setattr(rewritten_bundle, '_rewritten', True)
                    return self.retrieve(rewritten_bundle)
        
        if decision == RetrievalDecision.HYBRID and previous_context:
            combined = filtered_nodes + previous_context
            seen_ids = set()
            unique_nodes = []
            for node in combined:
                node_id = node.node.node_id
                if node_id not in seen_ids:
                    seen_ids.add(node_id)
                    unique_nodes.append(node)
            
            # Sort by score
            filtered_nodes = sorted(unique_nodes, key=lambda x: x.score, reverse=True)

        if self._node_postprocessors and nodes:
            for postprocessor in self._node_postprocessors:
                nodes = postprocessor.postprocess_nodes(nodes)
        
        # Sort by relevance score
        filtered_nodes.sort(key=lambda x: x.score, reverse=True)
        final_nodes = filtered_nodes[:self.config.max_nodes]
        
        metrics.filtering_time = time.time() - filter_start
        metrics.filtered_nodes = len(final_nodes)
        
        if self.config.debug_mode:
            logger.info(f"Retrieval metrics: {json.dumps(metrics.to_dict(), indent=2)}")
        
        return final_nodes


    def _apply_cross_validation(self, nodes: List[NodeWithScore], query_hash: str) -> None:
        """Apply cross-validation to refine node scores based on historical critique results"""
        critique_data = self._critique_history.get(query_hash, [])
        if not critique_data or not nodes:
            return
            
        # Build validation map from historical critiques
        validation_map = {}
        for content_hash, (is_relevant, score, _) in critique_data:
            if content_hash not in validation_map:
                validation_map[content_hash] = []
            validation_map[content_hash].append((is_relevant, score))
        
        # Apply validation to adjust scores
        for node in nodes:
            content = node.node.get_content()
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            if content_hash in validation_map and len(validation_map[content_hash]) >= 2:
                # Get average score from past critiques
                past_scores = [score for _, score in validation_map[content_hash]]
                avg_past_score = sum(past_scores) / len(past_scores)
                
                # Blend current score with historical average (75% current, 25% historical)
                node.score = 0.75 * node.score + 0.25 * avg_past_score


    def get_metrics(self) -> Optional[RetrievalMetrics]:
        """Get retrieval metrics"""
        return self.metrics
    
    @property
    def retriever(self) -> BaseRetriever:
        """Get the retriever object."""
        return self._retriever