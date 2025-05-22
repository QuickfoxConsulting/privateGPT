import logging
from typing import Any, Dict, Optional, List
from functools import lru_cache
from llama_index.core.tools import BaseTool
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.callbacks import CallbackManager
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.llms import LLM
from datetime import datetime
from private_gpt.components.vector_store.vector_store_component import VectorStoreComponent
from llama_index.core.indices.vector_store import VectorIndexRetriever, VectorStoreIndex

logger = logging.getLogger(__name__)

class DocumentSpecificTool(BaseTool):
    """Tool for querying specific documents in the knowledge base with enhanced filtering and response formatting."""
    
    def __init__(
        self,
        file_name: str,
        llm: LLM,
        vector_store_component: VectorStoreComponent,
        index: VectorStoreIndex,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        callback_manager: Optional[CallbackManager] = None,
        tool_name_prefix: str = "doc",
        citation_format: str = "[Document: {file_name}, Page {page}]",
        similarity_top_k: int = 4,
        verbose: bool = False,
        max_retries: int = 3,
        cache_size: int = 100
    ):
        self.index = index
        self.file_name = file_name
        self.llm = llm
        self.vector_store_component = vector_store_component
        self.node_postprocessors = node_postprocessors or []
        self.callback_manager = callback_manager or CallbackManager([])
        self.citation_format = citation_format
        self.verbose = verbose
        self.max_retries = max_retries
        self.cache_size = cache_size
        
        self.document_retriever = self._create_document_retriever(file_name)
        self.document_retriever.similarity_top_k = similarity_top_k
        
        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
            llm=self.llm,
            structured_answer_filtering=True,
            text_qa_template=self._get_qa_template(),
            callback_manager=self.callback_manager,
            streaming=False 
        )
        
        self.query_engine = RetrieverQueryEngine(
            retriever=self.document_retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=self.node_postprocessors,
            callback_manager=self.callback_manager
        )
        
        safe_name = self._generate_safe_name(file_name)
        self._name = f"{tool_name_prefix}_{safe_name}"
        self._description = self._generate_tool_description(file_name)
        
        if self.verbose:
            logger.info(f"Created document tool for '{file_name}'")
    
    @staticmethod
    @lru_cache(maxsize=100)
    def _generate_safe_name(file_name: str) -> str:
        """Generate a safe name for the tool with caching."""
        safe_name = ''.join(c if c.isalnum() else '_' for c in file_name)
        return safe_name.lower().strip('_')
    
    def _generate_tool_description(self, file_name: str) -> str:
        """Generate a detailed tool description with examples."""
        return (
            f"Query tool for document: {file_name}\n"
            "Use this tool to ask questions specifically about this document.\n"
            f"Example queries:\n"
            f"- What does {file_name} say about X?\n"
            f"- Summarize the key points in {file_name}\n"
            f"- Find information about Y in {file_name}\n"
            f"- Extract specific details or statistics from {file_name}\n"
            f"- Compare different sections of {file_name}"
        )
    
    def _create_document_retriever(self, file_name: str) -> BaseRetriever:
        """Create document retriever with error handling."""
        try:
            return self.vector_store_component.file_vector_retriever(
                index=self.index,
                file_name=file_name,
                similarity_top_k=self.document_retriever.similarity_top_k
            )
        except Exception as e:
            logger.error(f"Error creating document retriever for '{file_name}': {str(e)}")
            raise ValueError(f"Failed to create document retriever: {str(e)}")
    
    def _get_qa_template(self) -> PromptTemplate:
        """Get enhanced QA template with document-specific context."""
        return PromptTemplate(
            "You are a helpful assistant tasked with answering questions about a specific document. "
            f"The document is: '{self.file_name}'\n\n"
            "Use ONLY the context below to answer the query.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Instructions:\n"
            "1. Provide a comprehensive answer based solely on information from this document\n"
            "2. Organize complex information into clear sections or steps\n"
            f"3. When citing information, use this format: {self.citation_format.format(file_name=self.file_name, page='X')}\n"
            "4. If information cannot be found in the context, clearly state what is missing\n"
            "5. Focus on accuracy and completeness\n"
            "6. Include relevant metadata when available (e.g., page numbers, sections)\n"
            "7. If the answer spans multiple sections, synthesize them coherently\n\n"
            "Query: {query_str}\n\n"
            "Comprehensive Answer:"
        )
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get enhanced tool metadata with improved schema."""
        return {
            "name": self._name,
            "description": self._description,
            "args_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": f"The specific question to ask about the document: {self.file_name}"
                    }
                },
                "required": ["query"]
            },
            "document_info": {
                "file_name": self.file_name,
                "tool_type": "document_specific",
                "created_at": datetime.now().isoformat()
            }
        }
    
    def __call__(self, query: str) -> str:
        """Query the specific document with retry logic and performance tracking."""
        start_time = datetime.now()
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                enhanced_query = f"Regarding document '{self.file_name}': {query}"
                response = self.query_engine.query(enhanced_query)
                
                if self.verbose:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    logger.info(f"Document query took {elapsed:.2f}s for '{self.file_name}'")
                
                return str(response)
            except Exception as e:
                retry_count += 1
                if retry_count == self.max_retries:
                    logger.error(f"Error querying document '{self.file_name}' after {self.max_retries} retries: {str(e)}", exc_info=self.verbose)
                    return f"Error retrieving information from '{self.file_name}': {str(e)}"
                logger.warning(f"Retry {retry_count}/{self.max_retries} for document '{self.file_name}'")
                continue
    
    async def acall(self, query: str) -> str:
        """Async version of document querying with retry logic."""
        start_time = datetime.now()
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                enhanced_query = f"Regarding document '{self.file_name}': {query}"
                response = await self.query_engine.aquery(enhanced_query)
                
                if self.verbose:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    logger.info(f"Async document query took {elapsed:.2f}s for '{self.file_name}'")
                
                return str(response)
            except Exception as e:
                retry_count += 1
                if retry_count == self.max_retries:
                    logger.error(f"Error in async query for '{self.file_name}' after {self.max_retries} retries: {str(e)}", exc_info=self.verbose)
                    return f"Error retrieving information from '{self.file_name}': {str(e)}"
                logger.warning(f"Retry {retry_count}/{self.max_retries} for document '{self.file_name}'")
                continue
    
    @classmethod
    def from_defaults(
        cls,
        retriever: BaseRetriever,
        file_name: str,
        llm: LLM,
        **kwargs
    ) -> "DocumentSpecificTool":
        """Factory method with validation."""
        if not file_name:
            raise ValueError("file_name is required")
        if not llm:
            raise ValueError("llm is required")
            
        return cls(
            retriever=retriever,
            file_name=file_name,
            llm=llm,
            **kwargs
        )