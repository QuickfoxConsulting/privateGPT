import logging
from typing import List, Tuple, Optional, Dict, Union, Any
from dataclasses import dataclass
from enum import Enum
import re
import hashlib
import time
from functools import lru_cache
from collections import OrderedDict
from llama_index.core import PromptTemplate
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import BaseNode
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from private_gpt.components.llm.llm_component import LLMComponent


logger = logging.getLogger(__name__)

class RetrievalDecision(Enum):
    RETRIEVE = "RETRIEVE"
    SKIP = "SKIP"
    FOLLOWUP = "FOLLOWUP"

@dataclass
class RetrievalMetrics:
    total_nodes: int
    filtered_nodes: int
    retrieval_time: float
    filtering_time: float
    decision: RetrievalDecision
    decision_reason: str
    cache_hits: int = 0
    cache_misses: int = 0

class LRUCache:
    """Thread-safe LRU cache implementation"""
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        from threading import Lock
        self._lock = Lock()
    
    def get(self, key: str) -> Any:
        with self._lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]
    
    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

@dataclass
class SelfRAGConfig:
    """Configuration class for SelfRAGRetriever"""
    max_nodes: int = 3
    relevance_threshold: float = 0.6
    cache_size: int = 512
    content_cache_size: int = 1000
    relevance_cache_size: int = 1000
    context_history_size: int = 10
    high_relevance_bonus: float = 0.1
    enable_caching: bool = True
    debug_mode: bool = False

class SelfRAGRetriever(BaseRetriever):
    """Improved Self-RAG Retriever with synchronous retrieval"""
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm: LLMComponent,
        config: Optional[SelfRAGConfig] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        critique_prompt: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.base_retriever = base_retriever
        self.llm = llm
        self.config = config or SelfRAGConfig()
        self._node_postprocessors = node_postprocessors or []
        
        if self.config.enable_caching:
            self._content_cache = LRUCache(self.config.content_cache_size)
            self._relevance_cache = LRUCache(self.config.relevance_cache_size)
            self._context_history = LRUCache(self.config.context_history_size)
        
        self.critique_prompt_template = PromptTemplate(
            template=critique_prompt or self._get_default_critique_prompt()
        )
        
        self.metrics: Optional[RetrievalMetrics] = None

    def _get_default_critique_prompt(self) -> str:
        """Returns an improved critique prompt"""
        return """
        Task: Evaluate passage relevance for the query with specific scoring criteria.

        Passage to evaluate:
        -------------------
        {context}

        Query to answer:
        ---------------
        {query}

        Scoring criteria:
        1. Direct relevance (0-10):
           - Does it directly answer the query?
           - Contains key information needed?
           
        2. Context quality (0-10):
           - Is information specific and detailed?
           - Is it from a reliable/authoritative section?

        Required format:
        Direct relevance: [score]
        Context quality: [score]
        Overall confidence: [0.0-1.0]
        RELEVANT: [YES/NO]
        Reasoning: [Brief explanation]
        """

    def _safe_llm_call(self, prompt: str, max_retries: int = 3) -> str:
        """Safely make LLM calls with retries and error handling"""
        for attempt in range(max_retries):
            try:
                return self.llm.complete(prompt).text.strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    if self.config.debug_mode:
                        print(f"LLM call failed after {max_retries} attempts: {e}")
                    return ""
                time.sleep(2 ** attempt)  # Exponential backoff
        return ""

    def _critique_node(self, content_hash: str, query: str) -> Tuple[bool, float]:
        """Synchronous node evaluation"""
        cache_key = f"{query}:{content_hash}"
        cached_result = self._relevance_cache.get(cache_key) if self.config.enable_caching else None
        
        if cached_result:
            self.metrics.cache_hits += 1
            return cached_result
            
        self.metrics.cache_misses += 1
        content = self._content_cache.get(content_hash)
        if not content:
            return False, 0.0
            
        result = self._get_critique_result(content, query)
        
        if self.config.enable_caching:
            self._relevance_cache.put(cache_key, result)
            
        return result

    def _get_critique_result(self, context: str, query: str) -> Tuple[bool, float]:
        """Get critique result from LLM"""
        prompt = self.critique_prompt_template.format(
            context=context,
            query=query
        )
        response = self._safe_llm_call(prompt)
        return self._parse_critique_response(response)


    def _parse_critique_response(self, response: str) -> Tuple[bool, float]:
        """Parse LLM critique response with improved robustness"""
        scores = {
            'direct': 0.0,
            'context': 0.0,
            'confidence': 0.0,
            'relevant': False
        }
        
        normalized = response.lower()
        
        direct_match = re.search(r"direct relevance:\s*(\d+\.?\d*)", normalized)
        if direct_match:
            scores['direct'] = min(10.0, float(direct_match.group(1)))
        
        context_match = re.search(r"context quality:\s*(\d+\.?\d*)", normalized)
        if context_match:
            scores['context'] = min(10.0, float(context_match.group(1)))
        
        confidence_match = re.search(r"overall confidence:\s*([0-1]\.\d+)", normalized)
        if confidence_match:
            scores['confidence'] = float(confidence_match.group(1))
        
        relevant_match = re.search(r"relevant:\s*(yes|no)", normalized, re.IGNORECASE)
        if relevant_match:
            scores['relevant'] = relevant_match.group(1).lower() == 'yes'
        else:
            composite_score = (
                0.6 * (scores['direct'] / 10) +
                0.4 * (scores['context'] / 10)
            ) * scores['confidence']
            scores['relevant'] = composite_score >= self.config.relevance_threshold
        
        composite_score = (
            0.6 * (scores['direct'] / 10) +
            0.4 * (scores['context'] / 10)
        ) * scores['confidence']
        
        return scores['relevant'], composite_score
    

    def _should_retrieve(self, query: str, context: Optional[List[NodeWithScore]] = None) -> Tuple[RetrievalDecision, str]:
        """Determine if retrieval is needed"""
        chat_history = "\n".join([n.node.get_content()[:100] for n in (context or [])][-3:])
        
        prompt = f"""Determine if new retrieval is needed based on query and chat history.
        
        [CHAT HISTORY]
        {chat_history if context else "No prior context"}
        
        [QUERY]
        {query}
        
        [DECISION CRITERIA]
        - RETRIEVE: Requires new information not in context
        - FOLLOWUP: Can be answered with existing context
        - SKIP: No additional information needed
        
        First provide brief reasoning, then state ONLY the decision word.
        """
        
        response = self._safe_llm_call(prompt)
        return self._parse_retrieval_response(response)

    def _parse_retrieval_response(self, response: str) -> Tuple[RetrievalDecision, str]:
        """Parse retrieval decision response"""
        decision_match = re.search(r"(RETRIEVE|SKIP|FOLLOWUP)", response, re.IGNORECASE)
        decision = RetrievalDecision(decision_match.group(1).upper()) if decision_match else RetrievalDecision.RETRIEVE
        reasoning = re.sub(r"\b(RETRIEVE|SKIP|FOLLOWUP)\b", "", response, flags=re.IGNORECASE).strip()
        return decision, reasoning

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Main synchronous retrieval method"""
        start_time = time.time()
        
        metrics = RetrievalMetrics(
            total_nodes=0,
            filtered_nodes=0,
            retrieval_time=0.0,
            filtering_time=0.0,
            decision=RetrievalDecision.RETRIEVE,
            decision_reason="",
            cache_hits=0,
            cache_misses=0
        )
        self.metrics = metrics
        
        context_key = hashlib.sha256(query_bundle.query_str.encode()).hexdigest()
        previous_context = self._context_history.get(context_key) if self.config.enable_caching else None
        logger.info(f"Previous context: {previous_context}")
        if previous_context:
            threshold = self.config.relevance_threshold + self.config.high_relevance_bonus
            highly_relevant = [
                node for node in previous_context 
                if node.score >= threshold
            ]
            if highly_relevant:
                metrics.decision = RetrievalDecision.FOLLOWUP
                metrics.decision_reason = "Found highly relevant previous context"
                return highly_relevant[:self.config.max_nodes]
        
        logger.info(f"Retrieving for query: {query_bundle.query_str}")
        decision, reasoning = self._should_retrieve(
            query_bundle.query_str,
            previous_context
        )
        metrics.decision = decision
        metrics.decision_reason = reasoning
        
        if decision == RetrievalDecision.SKIP:
            return []
            
        if decision == RetrievalDecision.FOLLOWUP and previous_context:
            filtered = sorted(
                previous_context,
                key=lambda x: x.score,
                reverse=True
            )[:self.config.max_nodes]
            metrics.filtered_nodes = len(filtered)
            return filtered
        
        try:
            nodes = self.base_retriever.retrieve(query_bundle)
        except Exception as e:
            if self.config.debug_mode:
                print(f"Retrieval error: {e}")
            return []
        
        metrics.total_nodes = len(nodes)
        metrics.retrieval_time = time.time() - start_time
        logger.info(f"Retrieved nodes: {nodes}")
        filter_start = time.time()
        filtered_nodes = []
        
        for node in nodes:
            logger.info(f"Processing node: {node.node}")
            content = node.node.get_content()
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            if self.config.enable_caching:
                self._content_cache.put(content_hash, content)
            
            is_relevant, score = self._critique_node(
                content_hash,
                query_bundle.query_str
            )
            logger.info(f"Node relevance: {is_relevant}, score: {score}")
            if is_relevant and score >= self.config.relevance_threshold:
                filtered_nodes.append(NodeWithScore(node=node.node, score=score))
        
        if self.config.enable_caching:
            self._context_history.put(
                context_key,
                filtered_nodes[:self.config.max_nodes * 2]
            )
        
        # Finalize results
        filtered_nodes.sort(key=lambda x: x.score, reverse=True)
        final_nodes = filtered_nodes[:self.config.max_nodes]
        
        metrics.filtered_nodes = len(final_nodes)
        metrics.filtering_time = time.time() - filter_start
        logger.info(f"Filtered nodes: {final_nodes}")
        
        return final_nodes

    def get_metrics(self) -> Optional[RetrievalMetrics]:
        """Get retrieval metrics"""
        return self.metrics
    
    @property
    def retriever(self) -> BaseRetriever:
        """Get the retriever object."""
        return self._retriever

