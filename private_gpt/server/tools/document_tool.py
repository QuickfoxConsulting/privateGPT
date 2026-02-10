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

from private_gpt.components.vector_store.vector_store_component import VectorStoreComponent

logger = logging.getLogger(__name__)

class DocumentSpecificTool(BaseTool):
    """Tool for querying specific documents in the knowledge base with robust context formatting and retry logic."""

    def __init__(
        self,
        file_name: str,
        llm: LLM,
        vector_store_component: VectorStoreComponent,
        index: VectorStoreIndex,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        callback_manager: Optional[CallbackManager] = None,
        tool_name_prefix: str = "doc",
        citation_format: str = "[Page {page}]({file_name})", 
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
        self.vector_store_component = vector_store_component
        self.node_postprocessors = node_postprocessors or []
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

        self.document_retriever = self._create_document_retriever()

        if self.verbose:
            logger.info(f"[INIT] Document tool created for '{self.file_name}' with tool name '{self._name}'")

    @staticmethod
    @lru_cache(maxsize=100)
    def _generate_safe_name(file_name: str) -> str:
        """Generate a safe, cacheable name based on file name."""
        return ''.join(c if c.isalnum() else '_' for c in file_name).lower().strip('_')

    def _generate_tool_description(self, file_name: str) -> str:
        """Generate a descriptive summary with query examples."""
        return (
            f"Query tool for the document: {file_name}. Use this tool to extract structured information, "
            "summaries, and detailed answers specific to this document.\n"
            "Example queries:\n"
            f" - What does {file_name} say about topic X?\n"
            f" - Summarize sections in {file_name}\n"
            f" - Extract statistics or metadata from {file_name}\n"
            f" - Compare discussions across sections in {file_name}"
        )

    def _create_document_retriever(self) -> BaseRetriever:
        """Create a document retriever filtered by file name."""
        try:
            return self.vector_store_component.file_vector_retriever(
                index=self.index,
                file_name=self.file_name,
                similarity_top_k=self.similarity_top_k
            )
        except Exception as e:
            logger.exception(f"[ERROR] Failed to create retriever for '{self.file_name}': {e}")
            raise RuntimeError(f"Unable to create retriever for '{self.file_name}': {str(e)}")

    def _get_qa_template(self) -> PromptTemplate:
        """Define a document-specific prompt template with detailed citation formatting."""
        return PromptTemplate(
            (
                "You are a helpful assistant for answering questions about a specific document.\n"
                f"Document: '{self.file_name}'\n\n"
                "Use ONLY the context below:\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Instructions:\n"
                "1. Base the response solely on the document content. Do not use outside knowledge.\n"
                "2. Organize complex answers into structured sections.\n"
                f"3. Cite your sources using Markdown format: {self.citation_format.format(file_name=self.file_name, page='X')}\n"
                "4. If data is missing or the answer is not in the context, explicitly say 'I cannot find this information in the document'.\n"
                "5. Be precise and avoid hallucination.\n\n"
                "Query: {query_str}\n\n"
                "Comprehensive Answer:"
            )
        )
    
    @property
    def metadata(self) -> ToolMetadata:
        """Expose tool metadata and schema for integration."""
        return ToolMetadata(
            name=self._name,
            description=self._description
        )

    def _run_query(self, query: str, is_async: bool = False) -> str:
        """Run a query with retry logic."""
        start_time = datetime.now()
        retry_count = 0
        enhanced_query = self._wrap_query(query)

        while retry_count < self.max_retries:
            try:
                response_synthesizer = get_response_synthesizer(
                    llm=self.llm,
                    response_mode=self.response_mode,
                    structured_answer_filtering=True,
                    text_qa_template=self._get_qa_template(),
                    callback_manager=self.callback_manager,
                    streaming=self.streaming,
                )

                self.query_engine = RetrieverQueryEngine(
                    retriever=self.document_retriever,
                    response_synthesizer=response_synthesizer,
                    node_postprocessors=self.node_postprocessors,
                    callback_manager=self.callback_manager,
                )
                if is_async:
                    response = self.query_engine.aquery(enhanced_query)
                else:
                    response = self.query_engine.query(enhanced_query)

                self._log_elapsed(start_time, "Document query")
                return str(response)
            except Exception as e:
                retry_count += 1
                logger.warning(f"[RETRY] {retry_count}/{self.max_retries} for '{self.file_name}' - {e}")
                if retry_count == self.max_retries:
                    logger.error(f"[FAIL] Final failure on '{self.file_name}' - {e}", exc_info=self.verbose)
                    return f"Error retrieving answer from '{self.file_name}': {str(e)}"

    def __call__(self, input: str) -> ToolOutput:
        """Run a query with retry logic and fallback."""
        try:
            result = self._run_query(input, is_async=False)
            if not isinstance(result, str):
                result = str(result)
            return ToolOutput(
                content=result,
                tool_name=self._name,
                raw_input={"input": input},
                raw_output=result,
                is_error=False
            )
        except Exception as e:
            error_msg = f"Error retrieving answer from '{self.file_name}': {str(e)}"
            logger.error(error_msg, exc_info=self.verbose)
            return ToolOutput(
                content=error_msg,
                tool_name=self._name,
                raw_input={"input": input},
                raw_output=str(e),
                is_error=True
            )

    async def acall(self, input: str) -> ToolOutput:
        """Run a query asynchronously with retry logic."""
        try:
            result = await self._run_query(input, is_async=True)
            if not isinstance(result, str):
                result = str(result)
            return ToolOutput(
                content=result,
                tool_name=self._name,
                raw_input={"input": input},
                raw_output=result,
                is_error=False
            )
        except Exception as e:
            error_msg = f"Error retrieving answer from '{self.file_name}': {str(e)}"
            logger.error(error_msg, exc_info=self.verbose)
            return ToolOutput(
                content=error_msg,
                tool_name=self._name,
                raw_input={"input": input},
                raw_output=str(e),
                is_error=True
            )

    def _log_elapsed(self, start_time: datetime, label: str):
        if self.verbose:
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"{label} took {elapsed:.2f}s for '{self.file_name}'")

    def _wrap_query(self, query: str) -> str:
        return f"Regarding document '{self.file_name}': {query}"

    @classmethod
    def from_defaults(
        cls,
        retriever: BaseRetriever,
        file_name: str,
        llm: LLM,
        **kwargs
    ) -> "DocumentSpecificTool":
        """Factory method for instantiation with minimal args."""
        if not file_name:
            raise ValueError("Missing required parameter: file_name")
        if not llm:
            raise ValueError("Missing required parameter: llm")
        return cls(retriever=retriever, file_name=file_name, llm=llm, **kwargs)
