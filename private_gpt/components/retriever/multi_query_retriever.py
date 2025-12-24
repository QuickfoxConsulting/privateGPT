import asyncio
import logging
from typing import List, Optional

from llama_index.core import QueryBundle
from llama_index.core.llms import LLM
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore

logger = logging.getLogger(__name__)

QUERY_GEN_PROMPT = """You are an AI assistant tasked with generating three 
different versions of the given user query to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user query, your goal is to help 
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative queries separated by newlines. 
Original query: {query}"""

class MultiQueryRetriever(BaseRetriever):
    """
    Retriever that generates multiple queries from a single query
    and retrieves nodes for all of them.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm: LLM,
        num_queries: int = 3,
    ) -> None:
        self._base_retriever = base_retriever
        self._llm = llm
        self._num_queries = num_queries
        super().__init__()

    async def _agenerate_queries(self, query: str) -> List[str]:
        """Generate multiple queries from the original query."""
        prompt = QUERY_GEN_PROMPT.format(query=query)
        response = await self._llm.acomplete(prompt)
        
        # Split by newline and clean up
        raw_queries = [q.strip() for q in response.text.split("\n") if q.strip()]
        
        cleaned_queries = []
        for q in raw_queries:
            # Remove numbering like "1. ", "1) ", "- "
            import re
            cleaned_q = re.sub(r'^\d+[\.\)]\s*', '', q)
            cleaned_q = re.sub(r'^-\s*', '', cleaned_q)
            cleaned_q = cleaned_q.strip()
            if cleaned_q:
                cleaned_queries.append(cleaned_q)

        # Truncate if necessary and always include the original query
        queries = cleaned_queries[:self._num_queries - 1]
        queries.append(query)
        
        logger.info(f"Generated multi-queries: {queries}")
        return list(set(queries))  # De-duplicate

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Synchronous retrieve (not used in async chat flow but required by interface)."""
        # In a real sync scenario, we'd use loop.run_until_complete or similar,
        # but for privateGPT we mostly use a-methods.
        # Fallback to base retriever for simplicity in sync mode.
        return self._base_retriever.retrieve(query_bundle)

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Asynchronously retrieve nodes for multiple queries."""
        queries = await self._agenerate_queries(query_bundle.query_str)
        
        tasks = []
        for query in queries:
            sub_bundle = QueryBundle(query_str=query)
            tasks.append(self._base_retriever.aretrieve(sub_bundle))
        
        results = await asyncio.gather(*tasks)
        
        # Flatten and de-duplicate nodes
        all_nodes: List[NodeWithScore] = []
        node_ids = set()
        
        for result_list in results:
            for node_with_score in result_list:
                if node_with_score.node.node_id not in node_ids:
                    all_nodes.append(node_with_score)
                    node_ids.add(node_with_score.node.node_id)
        
        # Sort by score (descending)
        all_nodes.sort(key=lambda x: x.score or 0.0, reverse=True)
        
        return all_nodes
