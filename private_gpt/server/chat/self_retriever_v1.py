import re
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
from llama_index.core import PromptTemplate
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import BaseNode
from private_gpt.components.llm.llm_component import LLMComponent

from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore

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

DEFAULT_CRITIQUE_PROMPT = """
Instruction: Evaluate the relevance of this passage for answering the given query.

Consider:
1. Direct relevance: Does the passage directly address the query's main points?
2. Context quality: Is the information specific and detailed enough?

Rate each aspect from 0-10 and provide an overall relevance decision.

Passage: {context}
Query: {query}

Format your response as:
Direct relevance: [score]
Context quality: [score]

Overall confidence: [0-1]
RELEVANT: [YES/NO]
"""


class SelfRAGRetriever(BaseRetriever):
    """Enhanced Retriever with Self-RAG capabilities"""
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm: LLMComponent,
        critique_prompt: Optional[str] = None,
        max_nodes: int = 5,
        relevance_threshold: float = 0.7,
        enable_caching: bool = True,
        **kwargs
    ) -> None:
        self.base_retriever = base_retriever
        self.llm = llm
        self.critique_prompt_template = PromptTemplate(
            critique_prompt if critique_prompt is not None else DEFAULT_CRITIQUE_PROMPT
        )
        self.max_nodes = max_nodes
        self.relevance_threshold = relevance_threshold
        self.enable_caching = enable_caching
        self._cache = {} if enable_caching else None
        self.metrics: Optional[RetrievalMetrics] = None
        super().__init__(**kwargs)

    # def _should_retrieve(self, query: str, context: Optional[str] = None) -> Tuple[RetrievalDecision, str]:
    #     """Enhanced decision making for retrieval necessity"""
    #     prompt = f"""Determine if new retrieval is needed based on query and context history.
        
    #     Previous context: {context or 'None'}
    #     Query: {query}
        
    #     Instructions:
    #     1. If the query requires new factual information, respond with exactly 'RETRIEVE'
    #     2. If this is a follow-up question using previous context, respond with exactly 'FOLLOWUP'
    #     3. If this can be answered without additional context, respond with exactly 'SKIP'
        
    #     First, explain your reasoning in one sentence.
    #     Then on a new line, write ONLY ONE of: RETRIEVE, FOLLOWUP, or SKIP
    #     """
        
    #     response = self.llm.complete(prompt)
    #     reasoning, decision = self._parse_retrieval_response(response.text)
        
    #     try:
    #         return RetrievalDecision(decision), reasoning
    #     except ValueError:
    #         return RetrievalDecision.RETRIEVE, f"Failed to parse decision: {decision}. Defaulting to retrieve."

    def _should_retrieve(self, query: str, context: Optional[List[NodeWithScore]] = None) -> Tuple[RetrievalDecision, str]:
        """RAG-optimized retrieval decision maker"""
        context_summary = "\n".join([n.node.get_content()[:150] for n in context[-3:]]) if context else "No prior context"
        
        prompt = f"""You are a retrieval decision maker for a RAG system. Default to retrieval unless CONTEXT fully answers the query.

        # CONTEXT HISTORY (last 3 nodes):
        {context_summary}

        # CURRENT QUERY:
        {query}

        # DECISION RULES:
        1. RETRIEVE (default) unless:
        - Query is fully answerable using CONTEXT
        - Query is a direct follow-up requiring no new information
        2. FOLLOWUP only if:
        - Query explicitly references CONTEXT
        - Answer requires combining multiple CONTEXT elements
        3. SKIP only if:
        - Query is answered COMPLETELY in CONTEXT
        - Query is a simple reformulation of CONTEXT

        # RESPONSE FORMAT:
        <1-sentence reasoning>
        Decision: RETRIEVE|FOLLOWUP|SKIP
        """
        response = self.llm.complete(prompt).text.strip()
        return self._parse_retrieval_response(response)

    # def _parse_retrieval_response(self, response: str) -> Tuple[str, str]:
    #     lines = [line.strip() for line in response.strip().split('\n') if line.strip()]        
    #     decision = lines[-1].strip().upper()
    #     decision = decision.split(":")[-1].strip()        
    #     reasoning = ' '.join(lines[:-1]) if len(lines) > 1 else ""
        
    #     return reasoning, decision

    def _parse_retrieval_response(self, response: str) -> Tuple[RetrievalDecision, str]:
        """Enhanced parsing for structured response"""
        # Extract reasoning (everything before "Decision:")
        reasoning_match = re.search(r"^(.*?)(?=Decision:)", response, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
        
        # Find decision (case-insensitive)
        decision_match = re.search(r"Decision:\s*(\w+)", response, re.IGNORECASE)
        
        try:
            decision = RetrievalDecision[decision_match.group(1).upper()] if decision_match else RetrievalDecision.RETRIEVE
        except KeyError:
            decision = RetrievalDecision.RETRIEVE
            
        return decision, reasoning

    def _critique_node(self, node: BaseNode, query: str) -> Tuple[bool, float]:
        """Enhanced node evaluation with relevance scoring"""
        if self.enable_caching:
            cache_key = (node.get_content()[:100], query)
            if cache_key in self._cache:
                return self._cache[cache_key]

        prompt = self.critique_prompt_template.format(
            context=node.get_content(),
            query=query
        )
        
        response = self.llm.complete(prompt)
        is_relevant, score = self._parse_critique_response(response.text)
        
        if self.enable_caching:
            self._cache[cache_key] = (is_relevant, score)
            
        return is_relevant, score

    def _parse_critique_response(self, response: str) -> Tuple[bool, float]:
        """Parse LLM response to get relevance and confidence score"""
        response = response.strip().upper()
        is_relevant = "RELEVANT: YES" in response
        
        score = 1.0 if is_relevant else 0.0
        if "CONFIDENCE:" in response:
            try:
                score = float(response.split("CONFIDENCE:")[-1].strip())
            except ValueError:
                pass
            
        return is_relevant, score

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Enhanced retrieve method with metrics and filtering"""
        import time
        start_time = time.time()
        
        decision, reasoning = self._should_retrieve(
            query_bundle.query_str,
            getattr(query_bundle, 'context', None)
        )
        
        if decision != RetrievalDecision.RETRIEVE:
            self.metrics = RetrievalMetrics(0, 0, 0.0, 0.0)
            return []

        nodes = self.base_retriever._retrieve(query_bundle)
        retrieval_time = time.time() - start_time
        
        filtered_nodes = []
        filter_start = time.time()
        
        for node in nodes:
            is_relevant, score = self._critique_node(node.node, query_bundle.query_str)
            if is_relevant and score >= self.relevance_threshold:
                filtered_nodes.append(
                    NodeWithScore(node=node.node, score=score)
                )
        
        filtered_nodes.sort(key=lambda x: x.score, reverse=True)
        filtered_nodes = filtered_nodes[:self.max_nodes]
        
        filtering_time = time.time() - filter_start
        
        self.metrics = RetrievalMetrics(
            total_nodes=len(nodes),
            filtered_nodes=len(filtered_nodes),
            retrieval_time=retrieval_time,
            filtering_time=filtering_time
        )
        print(f"METRICS:  {self.metrics}\n NODES: {filtered_nodes}")
        
        return filtered_nodes

    def get_metrics(self) -> Optional[RetrievalMetrics]:
        """Return metrics from the last retrieval operation"""
        return self.metrics