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

from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)

DEFAULT_SUMMARIZE_PROMPT = (
    "You are a senior research analyst responsible for delivering accurate, structured, and well-sourced summaries. "
    "Your summaries must reflect deep reasoning, clear structure, and proper citation based only on the provided context.\n\n"
    
    "CONTEXT:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    
    "INSTRUCTIONS:\n"
    "1. **Use only the provided context**. Do not make assumptions or add external knowledge.\n"
    "2. **Organize the response in professional Markdown**, using clear section headers and bullet points.\n"
    "3. **Break down complex ideas** into smaller, easy-to-follow sections with proper headings.\n"
    "4. **Support all claims with specific citations**:\n"
    "   - Use format: `[page_number](document_name)`\n"
    "   - Prefer inline citations immediately after each fact or quote.\n"
    "5. **Synthesize multiple sections** if they cover the same topic.\n"
    "6. If relevant information is **missing or ambiguous**, clearly state it under the 'Limitations' section.\n"
    "7. Always use a clear, neutral, and technical tone. Focus on facts, not opinions.\n"
    "8. Include all known metadata (e.g., document version, dates) when relevant.\n\n"
    
    "RESPONSE FORMAT (use exactly this structure):\n"
    "```markdown\n"
    "# Executive Summary\n"
    "- Concise summary of the document\n"
    "- Key themes and main points\n\n"
    "# Document Structure\n"
    "- Overview of document organization\n"
    "- Major sections and their purposes\n\n"
    "# Key Findings\n"
    "## Topic 1: <Descriptive Header>\n"
    "- Main point with citation [page_number](document_name)\n"
    "- Supporting evidence\n"
    "- Examples and data\n\n"
    "## Topic 2: <Another Key Theme>\n"
    "- Key insight with citation [page_number](document_name)\n"
    "- Supporting evidence\n"
    "- Additional metadata if applicable\n\n"
    "# Supporting Evidence\n"
    "- Bullet list of important quotes or data points with sources\n"
    "- Use this section to provide raw references\n\n"
    "# Limitations & Considerations\n"
    "- Document scope and coverage\n"
    "- Potential gaps or assumptions\n"
    "- Areas needing further investigation\n"
    "```\n\n"
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
        callback_manager: Optional[CallbackManager] = None,
        tool_name_prefix: str = "summary",
        citation_format: str = "[Document: {file_name}, Page {page}]",
        similarity_top_k: int = 4,
        verbose: bool = False,
        max_retries: int = 3,
        cache_size: int = 100,
        streaming: bool = False,
        response_mode: str = "tree_summarize",
        settings: Optional[Settings] = None,
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
        self.settings = settings or Settings()

        self._name = f"{tool_name_prefix}_{self._generate_safe_name(file_name)}"
        self._description = self._generate_tool_description(file_name)

        # Initialize storage context from the index
        self.storage_context = self.index.storage_context

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
            use_async=self.settings.summarize.use_async,
            text_qa_template=PromptTemplate(DEFAULT_SUMMARIZE_PROMPT)
        )

        while retry_count < self.max_retries:
            try:
                if is_async:
                    response = await query_engine.aquery(enhanced_query)
                else:
                    response = query_engine.query(enhanced_query)

                self._log_elapsed(start_time, "Summary query")
                return str(response)
            except Exception as e:
                retry_count += 1
                logger.warning(f"[RETRY] {retry_count}/{self.max_retries} for '{self.file_name}' - {e}")
                if retry_count == self.max_retries:
                    logger.error(f"[FAIL] Final failure on '{self.file_name}' - {e}", exc_info=self.verbose)
                    return f"Error retrieving summary from '{self.file_name}': {str(e)}"

    def __call__(self, query: str) -> str:
        return self._run_query(query, is_async=False)

    async def acall(self, query: str) -> str:
        return await self._run_query(query, is_async=True)

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self._name,
            "description": self._description,
            "args_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": f"Query to run on document '{self.file_name}'."
                    }
                },
                "required": ["query"]
            },
            "document_info": {
                "file_name": self.file_name,
                "tool_type": "document_summary",
                "created_at": datetime.now().isoformat()
            }
        }

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
