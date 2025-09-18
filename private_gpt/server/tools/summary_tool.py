import logging
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, Optional, List

from llama_index.core.llms import LLM
from llama_index.core.tools import BaseTool
from llama_index.core import get_response_synthesizer
from llama_index.core.callbacks import CallbackManager
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.tools.types import ToolMetadata, ToolOutput
from llama_index.core.schema import NodeWithScore

from private_gpt.components.vector_store.vector_store_component import VectorStoreComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent

logger = logging.getLogger(__name__)

class DocumentSummaryTool(BaseTool):
    """Tool for generating document summaries with enhanced cross-page context handling."""

    def __init__(
        self,
        file_name: str,
        llm: LLM,
        vector_store_component: VectorStoreComponent,
        node_store_component: NodeStoreComponent,
        index: VectorStoreIndex,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        callback_manager: Optional[CallbackManager] = None,
        tool_name_prefix: str = "summary",
        citation_format: str = "[Document: {file_name}, Page {page}]",
        similarity_top_k: int = 10,
        verbose: bool = False,
        max_retries: int = 3,
        cache_size: int = 100,
        streaming: bool = False,
        response_mode: str = "tree_summarize",
        # Add new parameters for enhanced summary generation
        preserve_page_flow: bool = True,
        include_page_references: bool = True,
        cross_page_context_window: int = 2,
    ):
        self.file_name = file_name
        self.llm = llm
        self.index = index
        self.vector_store_component = vector_store_component
        self.node_store_component = node_store_component
        self.node_postprocessors = node_postprocessors or []
        self.callback_manager = callback_manager or CallbackManager([])
        self.similarity_top_k = similarity_top_k
        self.citation_format = citation_format
        self.verbose = verbose
        self.max_retries = max_retries
        self.cache_size = cache_size
        self.streaming = streaming
        self.response_mode = response_mode
        # New parameters for enhanced summary generation
        self.preserve_page_flow = preserve_page_flow
        self.include_page_references = include_page_references
        self.cross_page_context_window = cross_page_context_window

        self._name = f"{tool_name_prefix}_{self._generate_safe_name(file_name)}"
        self._description = self._generate_tool_description(file_name)

        self.document_retriever = self._create_document_retriever()

        if self.verbose:
            logger.info(f"[INIT] Summary tool created for '{self.file_name}' with tool name '{self._name}'")

    @staticmethod
    @lru_cache(maxsize=100)
    def _generate_safe_name(file_name: str) -> str:
        """Generate a safe, cacheable name based on file name."""
        return ''.join(c if c.isalnum() else '_' for c in file_name).lower().strip('_')

    def _generate_tool_description(self, file_name: str) -> str:
        """Generate a descriptive summary with query examples."""
        return (
            f"Summary tool for the document: {file_name}. Generates comprehensive summaries "
            "with page-aware context preservation.\n"
            "Example queries:\n"
            f" - Generate a summary of {file_name}\n"
            f" - What are the key points in {file_name}?\n"
            f" - Provide a structured overview of {file_name}\n"
            f" - Extract main themes from {file_name}"
        )

    def _create_document_retriever(self) -> BaseRetriever:
        """Create a document retriever optimized for summary generation."""
        try:
            # Use enhanced file vector retriever with summary-specific parameters
            return self.vector_store_component.file_vector_retriever(
                index=self.index,
                file_name=self.file_name,
                similarity_top_k=self.similarity_top_k,
                # Enable enhanced features for better summaries
                enable_cross_page_retrieval=True,
                preserve_page_context=True,
                context_window_expansion=2.0,  # Expand context for better summaries
            )
        except Exception as e:
            logger.exception(f"[ERROR] Failed to create retriever for '{self.file_name}': {e}")
            raise RuntimeError(f"Unable to create retriever for '{self.file_name}': {str(e)}")

    def _get_summary_template(self) -> PromptTemplate:
        """Define a summary prompt template with enhanced page handling."""
        return PromptTemplate(
            (
                "You are a document summarization expert. Your task is to create a comprehensive, "
                "well-structured summary of the provided document content.\n\n"
                f"Document: '{self.file_name}'\n\n"
                "CONTENT TO SUMMARIZE:\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n\n"
                "INSTRUCTIONS:\n"
                "1. Create a detailed, structured summary organized by main sections or themes.\n"
                "2. Preserve the logical flow and relationships between different parts of the document.\n"
                "3. Highlight key points, findings, and important details.\n"
                "4. Maintain accurate page references throughout the summary.\n"
                "5. For content spanning multiple pages, indicate the full page range.\n"
                "6. Use clear section headings and bullet points where appropriate.\n"
                "7. Ensure technical accuracy and preserve specific terminology.\n"
                "8. Include metadata such as document sections, chapters, or other structural elements.\n\n"
                "SUMMARY FORMAT:\n"
                "- Begin with a brief executive overview\n"
                "- Organize by main sections or themes\n"
                "- Include specific page references for key points\n"
                "- End with key takeaways and conclusions\n\n"
                "Query: {query_str}\n\n"
                "Comprehensive Summary:"
            )
        )
    
    @property
    def metadata(self) -> ToolMetadata:
        """Expose tool metadata and schema for integration."""
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
