import time
from enum import Enum
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Dict

from llama_index.core.llms.llm import LLM
from llama_index.core.prompts.mixin import PromptMixinType
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate
from llama_index.core.response_synthesizers import (
    BaseSynthesizer,
    ResponseMode,
    get_response_synthesizer,
)
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.service_context import ServiceContext
from llama_index.core.settings import (
    Settings,
    callback_manager_from_settings_or_context,
    llm_from_settings_or_context,
)

from private_gpt.components.llm.llm_component import LLMComponent

DEFAULT_DECISION_PROMPT = """
Given the query and previous responses/context, determine if new information retrieval is needed.

Query: {query}

Previous Context: {context}

Consider:
1. Is the information in the context sufficient to answer the query?
2. Would additional information help provide a better answer?
3. Is this a follow-up question that can be answered with existing context?

Make a decision:
RETRIEVE - New information needed
SKIP - Can answer with existing context
FOLLOWUP - Related to previous context, may need combination

Explain your reasoning, then state your decision.
"""

class RetrievalDecision(Enum):
    RETRIEVE = "RETRIEVE"
    SKIP = "SKIP"
    FOLLOWUP = "FOLLOWUP"

@dataclass
class QueryMetrics:
    """Metrics for tracking query performance"""
    decision: RetrievalDecision
    decision_reason: str
    retrieval_time: float = 0.0
    total_nodes: int = 0
    filtered_nodes: int = 0
    used_cached: bool = False


class SelfRetrieverQueryEngine(BaseQueryEngine):
    """Query engine with self-retrieval decision making capabilities."""
    
    def __init__(
        self,
        retriever: BaseRetriever,
        llm_component: LLMComponent,
        response_synthesizer: Optional[BaseSynthesizer] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        callback_manager: Optional[CallbackManager] = None,
        decision_prompt: Optional[str] = None,
        max_context_history: int = 5,
        cache_size: int = 100,
    ) -> None:
        """Initialize the self-retriever query engine."""
        self._retriever = retriever
        self._llm = llm_component
        self._response_synthesizer = response_synthesizer or get_response_synthesizer(
            llm=llm_from_settings_or_context(Settings, retriever.get_service_context()),
            callback_manager=callback_manager
            or callback_manager_from_settings_or_context(
                Settings, retriever.get_service_context()
            ),
        )
        
        self._node_postprocessors = node_postprocessors or []
        self._decision_prompt = PromptTemplate(decision_prompt or DEFAULT_DECISION_PROMPT)
        self._max_context_history = max_context_history
        
        # Initialize context and response caching
        self._context_history: Dict[str, List[NodeWithScore]] = {}
        self._response_cache: Dict[str, RESPONSE_TYPE] = {}
        self._cache_size = cache_size
        
        # Setup callback manager
        callback_manager = callback_manager or self._response_synthesizer.callback_manager
        for node_postprocessor in self._node_postprocessors:
            node_postprocessor.callback_manager = callback_manager
        super().__init__(callback_manager=callback_manager)
        
        self.metrics: Optional[QueryMetrics] = None

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {
            "decision_prompt": self._decision_prompt,
            "response_synthesizer": self._response_synthesizer,
        }

    @classmethod
    def from_args(
        cls,
        retriever: BaseRetriever,
        llm: Optional[LLM] = None,
        llm_component: Optional[LLMComponent] = None,
        response_synthesizer: Optional[BaseSynthesizer] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        decision_prompt: Optional[str] = None,
        response_mode: ResponseMode = ResponseMode.COMPACT,
        text_qa_template: Optional[BasePromptTemplate] = None,
        refine_template: Optional[BasePromptTemplate] = None,
        service_context: Optional[ServiceContext] = None,
        max_context_history: int = 5,
        cache_size: int = 100,
        **kwargs: Any,
    ) -> "SelfRetrieverQueryEngine":
        """Initialize the engine from arguments."""
        if llm_component is None and llm is None:
            raise ValueError("Either llm or llm_component must be provided")
            
        llm = llm or llm_from_settings_or_context(Settings, service_context)
        
        response_synthesizer = response_synthesizer or get_response_synthesizer(
            llm=llm,
            service_context=service_context,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            response_mode=response_mode,
        )

        callback_manager = callback_manager_from_settings_or_context(
            Settings, service_context
        )

        return cls(
            retriever=retriever,
            llm_component=llm_component or llm,
            response_synthesizer=response_synthesizer,
            callback_manager=callback_manager,
            node_postprocessors=node_postprocessors,
            decision_prompt=decision_prompt,
            max_context_history=max_context_history,
            cache_size=cache_size,
        )

    def _make_retrieval_decision(
        self, 
        query: str, 
        context: Optional[List[NodeWithScore]] = None
    ) -> tuple[RetrievalDecision, str]:
        """Decide whether to perform retrieval based on query and context."""
        context_text = "\n".join([
            f"Previous response {i+1}:\n{node.node.get_content()}"
            for i, node in enumerate(context[-3:])  # Use last 3 contexts
        ]) if context else "No previous context available."
        
        prompt = self._decision_prompt.format(
            query=query,
            context=context_text
        )
        
        response = self._llm.complete(prompt).text.strip()
        
        # Parse decision from response
        if "RETRIEVE" in response:
            decision = RetrievalDecision.RETRIEVE
        elif "SKIP" in response:
            decision = RetrievalDecision.SKIP
        else:
            decision = RetrievalDecision.FOLLOWUP
            
        # Get reasoning (everything before the decision)
        reasoning = response.split(decision.value)[0].strip()
        
        return decision, reasoning

    def _apply_node_postprocessors(
        self, 
        nodes: List[NodeWithScore], 
        query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        """Apply post-processing to retrieved nodes."""
        for node_postprocessor in self._node_postprocessors:
            nodes = node_postprocessor.postprocess_nodes(
                nodes, query_bundle=query_bundle
            )
        return nodes

    def retrieve_with_decision(
        self, 
        query_bundle: QueryBundle
    ) -> tuple[List[NodeWithScore], QueryMetrics]:
        """Retrieve nodes with decision making and metrics tracking."""
        start_time = time.time()
        
        # Get previous context
        prev_context = self._context_history.get(query_bundle.query_str, [])
        
        # Make retrieval decision
        decision, reasoning = self._make_retrieval_decision(
            query_bundle.query_str, 
            prev_context
        )
        
        metrics = QueryMetrics(
            decision=decision,
            decision_reason=reasoning
        )
        
        if decision == RetrievalDecision.SKIP:
            nodes = prev_context
            metrics.used_cached = True
        elif decision == RetrievalDecision.FOLLOWUP:
            new_nodes = self._retriever.retrieve(query_bundle)
            nodes = prev_context + new_nodes[:2]  # Limited new nodes
            metrics.total_nodes = len(new_nodes)
        else:
            nodes = self._retriever.retrieve(query_bundle)
            metrics.total_nodes = len(nodes)
        
        filtered_nodes = self._apply_node_postprocessors(nodes, query_bundle)
        metrics.filtered_nodes = len(filtered_nodes)
        metrics.retrieval_time = time.time() - start_time
        
        self._context_history[query_bundle.query_str] = (
            filtered_nodes[-self._max_context_history:]
        )
        
        # Manage cache size
        if len(self._context_history) > self._cache_size:
            oldest_key = next(iter(self._context_history))
            del self._context_history[oldest_key]
        
        print("*"*50)
        print(f"Filtered nodes: {filtered_nodes}")
        print(f"Metrics: {metrics}")
        print("*"*50)
        return filtered_nodes, metrics

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Process a query with self-retrieval decision making."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            if query_bundle.query_str in self._response_cache:
                response = self._response_cache[query_bundle.query_str]
                query_event.on_end(payload={
                    EventPayload.RESPONSE: response,
                    "cached": True
                })
                return response
            
            nodes, metrics = self.retrieve_with_decision(query_bundle)
            self.metrics = metrics
            
            response = self._response_synthesizer.synthesize(
                query=query_bundle,
                nodes=nodes,
            )
            self._response_cache[query_bundle.query_str] = response
            if len(self._response_cache) > self._cache_size:
                oldest_key = next(iter(self._response_cache))
                del self._response_cache[oldest_key]
            
            query_event.on_end(payload={
                EventPayload.RESPONSE: response,
                "metrics": metrics
            })

        return response

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Async version of query processing."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            if query_bundle.query_str in self._response_cache:
                response = self._response_cache[query_bundle.query_str]
                query_event.on_end(payload={
                    EventPayload.RESPONSE: response,
                    "cached": True
                })
                return response
            
            nodes = await self._retriever.aretrieve(query_bundle)
            nodes = self._apply_node_postprocessors(nodes, query_bundle)
            
            response = await self._response_synthesizer.asynthesize(
                query=query_bundle,
                nodes=nodes,
            )
            
            self._response_cache[query_bundle.query_str] = response
            if len(self._response_cache) > self._cache_size:
                oldest_key = next(iter(self._response_cache))
                del self._response_cache[oldest_key]
            
            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response

    @property
    def retriever(self) -> BaseRetriever:
        """Get the retriever object."""
        return self.retrieve_with_decision

    def get_metrics(self) -> Optional[QueryMetrics]:
        """Get the latest query metrics."""
        return self.metrics