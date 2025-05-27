import logging
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, Optional, List

from llama_index.core.llms import LLM
from llama_index.core.tools import BaseTool
from llama_index.core.schema import NodeWithScore
from llama_index.core.callbacks import CallbackManager
from llama_index.core import (
    Document,
    StorageContext,
    SummaryIndex,
)
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.prompts import PromptTemplate
from llama_index.core.tools.types import ToolMetadata, ToolOutput
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)

logger = logging.getLogger(__name__)

DEFAULT_SUMMARIZE_PROMPT = (
    "You are a senior research analyst tasked with producing precise, well-structured, and evidence-based summaries.\n\n"
    
    "CONTEXT:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    
    "INSTRUCTIONS:\n"
    "1. Only use the provided context. Do not add external information or assumptions.\n"
    "2. Write in clear, technical, and neutral language. Focus strictly on factual accuracy.\n"
    "3. Use professional Markdown with clear section headers, bullets, and subheadings.\n"
    "4. Support every claim with inline citations using the format `[page_number](document_name)`.\n"
    "5. Synthesize related points across the document into coherent themes.\n"
    "6. If context is missing, ambiguous, or contradictory, document it under 'Limitations'.\n"
    "7. Include all available metadata (e.g., version, dates) if referenced in the context.\n\n"
    
    "RESPONSE FORMAT (use exactly this structure):\n"
    "# Executive Summary\n"
    "- Brief overview of the document's purpose and findings\n"
    "- Summary of key themes or insights\n\n"
    "- Document Structure\n"
    "- High-level breakdown of the document's layout and intent\n"
    "- Major sections and what they cover\n\n"
    "# Key Findings\n"
    "## <Theme 1: Descriptive Title>\n"
    "- Core finding with citation [page](document)\n"
    "- Supporting explanation or evidence\n\n"
    "## <Theme 2: Descriptive Title>\n"
    "- Insight or issue described\n"
    "- Supporting facts or examples with citation [page](document)\n\n"
    "# Supporting Evidence\n"
    "- Bullet list of key quotes, data points, or references\n"
    "- Format each: \"Quote or data point\" — [page](document)\n\n"
    "# Limitations & Considerations\n"
    "- Scope constraints or omitted areas\n"
    "- Ambiguities, contradictions, or unclear points in the document\n"
    "- Recommendations for further analysis (if applicable)\n"
    "QUERY:\n"
    "{query_str}\n\n"
    "BEGIN YOUR SUMMARY:"
)

class DocumentSummaryTool(BaseTool):
    """Tool for getting a summary of a document."""

    def __init__(
        self,
        file_name: str,
        llm: LLM,
        index: VectorStoreIndex,
        node_store_component: NodeStoreComponent,
        vector_store_component: VectorStoreComponent,
        callback_manager: Optional[CallbackManager] = None,
        tool_name_prefix: str = "summary",
        citation_format: str = "[Document: {file_name}, Page {page}]",
        similarity_top_k: int = 4,
        verbose: bool = False,
        max_retries: int = 3,
        cache_size: int = 100,
        streaming: bool = False,
        response_mode: str = "tree_summarize",
    ):
        self.file_name = file_name
        self.llm = llm
        self.index = index
        self.callback_manager = callback_manager or CallbackManager([])
        self.similarity_top_k = similarity_top_k
        self.citation_format = citation_format
        self.verbose = verbose
        self.max_retries = max_retries
        self.cache_size = cache_size
        self.streaming = streaming
        self.response_mode = response_mode

        self._name = f"{tool_name_prefix}_{self._generate_safe_name(file_name)}"
        self._description = self._generate_tool_description(file_name)

        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store_component.vector_store,
            docstore=node_store_component.doc_store,
            index_store=node_store_component.index_store,
        )

        if self.verbose:
            logger.info(f"[INIT] Summary tool created for '{self.file_name}' with tool name '{self._name}'")

    @staticmethod
    @lru_cache(maxsize=100)
    def _generate_safe_name(file_name: str) -> str:
        return ''.join(c if c.isalnum() else '_' for c in file_name).lower().strip('_')

    def _generate_tool_description(self, file_name: str) -> str:
        return (
            f"Summary tool for the document: {file_name}. Use this tool to get comprehensive summaries and "
            "structural analysis of the document.\n"
            "Example queries:\n"
            f" - Provide a summary of {file_name}\n"
            f" - What are the main topics in {file_name}?\n"
            f" - Give me an executive summary of {file_name}\n"
            f" - What are the key points in {file_name}?"
        )

    def get_doc_ids_by_filename(self, filename: str) -> List[NodeWithScore]:
        """Get document nodes filtered by filename."""
        try:
            nodes = []
            docstore = self.storage_context.docstore
            for node in docstore.docs.values():
                if node.metadata is not None and node.metadata.get("file_name") == filename:
                    nodes.append(NodeWithScore(node=node, score=1.0))
            
            if self.verbose:
                logger.info(f"Found {len(nodes)} nodes for file '{filename}'")
            return nodes
        except Exception as e:
            logger.error(f"Error getting nodes for file '{filename}': {e}")
            return []

    def _wrap_query(self, query: str) -> str:
        return f"Regarding document '{self.file_name}': {query}"

    def _log_elapsed(self, start_time: datetime, label: str):
        if self.verbose:
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"{label} took {elapsed:.2f}s for '{self.file_name}'")

    async def _run_query(self, query: str, is_async: bool = False) -> str:
        start_time = datetime.now()
        retry_count = 0
        enhanced_query = self._wrap_query(query)

        filtered_nodes = self.get_doc_ids_by_filename(self.file_name)
        if not filtered_nodes:
            return f"No content found for file: {self.file_name}"

        # Create nodes list from NodeWithScore objects
        nodes = [nws.node for nws in filtered_nodes]

        summary_index = SummaryIndex(
            nodes=nodes,
            storage_context=self.storage_context,
            show_progress=True,
        )

        query_engine = summary_index.as_query_engine(
            llm=self.llm,
            response_mode=ResponseMode.TREE_SUMMARIZE,
            streaming=self.streaming,
            use_async=is_async,
            text_qa_template=PromptTemplate(DEFAULT_SUMMARIZE_PROMPT)
        )

        while retry_count < self.max_retries:
            try:
                if is_async:
                    response = await query_engine.aquery(enhanced_query)
                else:
                    response = query_engine.query(enhanced_query)

                self._log_elapsed(start_time, "Summary query")
                # Ensure response is converted to string
                if response is None:
                    return f"No response generated for file: {self.file_name}"
                return str(response)
            except Exception as e:
                retry_count += 1
                logger.warning(f"[RETRY] {retry_count}/{self.max_retries} for '{self.file_name}' - {e}")
                if retry_count == self.max_retries:
                    logger.error(f"[FAIL] Final failure on '{self.file_name}' - {e}", exc_info=self.verbose)
                    return f"Error retrieving summary from '{self.file_name}': {str(e)}"

    def __call__(self, query: str) -> ToolOutput:
        """Run a query with retry logic and fallback."""
        try:
            # Create a synchronous version of the query
            result = self._run_query_sync(query)
            if not isinstance(result, str):
                result = str(result)
            return ToolOutput(
                content=result,
                tool_name=self._name,
                raw_input={"query": query},
                raw_output=result,
                is_error=False
            )
        except Exception as e:
            error_msg = f"Error retrieving summary from '{self.file_name}': {str(e)}"
            logger.error(error_msg, exc_info=self.verbose)
            return ToolOutput(
                content=error_msg,
                tool_name=self._name,
                raw_input={"query": query},
                raw_output=str(e),
                is_error=True
            )

    def _run_query_sync(self, query: str) -> str:
        """Synchronous wrapper for _run_query."""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._run_query(query, is_async=False))
            finally:
                loop.close()
        
        # Run the async code in a separate thread to avoid event loop conflicts
        with ThreadPoolExecutor() as executor:
            return executor.submit(run_in_thread).result()

    async def acall(self, query: str) -> ToolOutput:
        """Run a query asynchronously with retry logic."""
        try:
            result = await self._run_query(query, is_async=True)
            if not isinstance(result, str):
                result = str(result)
            return ToolOutput(
                content=result,
                tool_name=self._name,
                raw_input={"query": query},
                raw_output=result,
                is_error=False
            )
        except Exception as e:
            error_msg = f"Error retrieving summary from '{self.file_name}': {str(e)}"
            logger.error(error_msg, exc_info=self.verbose)
            return ToolOutput(
                content=error_msg,
                tool_name=self._name,
                raw_input={"query": query},
                raw_output=str(e),
                is_error=True
            )

    @property
    def metadata(self) -> ToolMetadata:
        """Get tool metadata."""
        return ToolMetadata(
            name=self._name,
            description=self._description
        )

    @classmethod
    def from_defaults(
        cls,
        retriever: BaseRetriever,
        file_name: str,
        llm: LLM,
        **kwargs
    ) -> "DocumentSummaryTool":
        if not file_name:
            raise ValueError("Missing required parameter: file_name")
        if not llm:
            raise ValueError("Missing required parameter: llm")
        return cls(file_name=file_name, llm=llm, **kwargs)
