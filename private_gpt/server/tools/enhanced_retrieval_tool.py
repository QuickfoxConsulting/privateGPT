"""
Enhanced Document Retrieval Tool with automatic query expansion.

This tool wraps a standard query engine and enhances queries before retrieval
to improve result quality by adding context clues, synonyms, and related terms.
"""
import logging
from typing import Optional, Any

from llama_index.core.tools import BaseTool, ToolMetadata, ToolOutput
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.llms import LLM

logger = logging.getLogger(__name__)


class EnhancedRetrievalTool(BaseTool):
    """
    Enhanced document retrieval tool with automatic query expansion.
    
    This tool improves retrieval quality by:
    1. Expanding queries with context clues and related terms
    2. Adding domain-specific terminology
    3. Including common synonyms and variations
    4. Maintaining natural language readability
    """
    
    def __init__(
        self,
        query_engine: RetrieverQueryEngine,
        llm: LLM,
        enable_query_expansion: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize the enhanced retrieval tool.
        
        Args:
            query_engine: The underlying query engine to use for retrieval
            llm: Language model for query enhancement
            enable_query_expansion: Whether to enhance queries (can disable for testing)
            verbose: Whether to log detailed information
        """
        self._query_engine = query_engine
        self._llm = llm
        self._enable_query_expansion = enable_query_expansion
        self._verbose = verbose
        
        self.name = "document_retriever"
        self.description = (
            "A comprehensive search tool for the entire knowledge base with automatic query enhancement. "
            "This tool intelligently expands your search query with relevant context, synonyms, and related terms "
            "to find the most relevant information in the uploaded documents. "
            "Use this tool for ALL queries to check for relevant information, "
            "regardless of whether the query seems 'internal' or general. "
            "Always verify if the answer exists in the documents before using other tools or external knowledge."
        )
        super().__init__()
    
    @property
    def metadata(self) -> ToolMetadata:
        """Get the tool metadata."""
        return ToolMetadata(
            name=self.name,
            description=self.description,
        )
    
    def _enhance_query(self, query: str) -> str:
        """
        Enhance query with context clues, synonyms, and related terms.
        
        Args:
            query: Original search query
            
        Returns:
            Enhanced query optimized for retrieval
        """
        if not self._enable_query_expansion:
            return query
        
        # Skip enhancement for very short queries that are already specific
        if len(query.split()) <= 2:
            logger.debug(f"Skipping enhancement for short query: '{query}'")
            return query
        
        enhancement_prompt = f"""Given this search query, enhance it to improve document retrieval.

Original query: {query}

Your task: Create an enhanced version that:
1. Adds relevant context clues and related terms that might appear in documents
2. Includes common synonyms and terminology variations
3. Adds domain-specific terms when applicable
4. Maintains natural, readable language (not just keywords)
5. Keeps the core intent of the original query

Guidelines:
- If the query mentions specific items (like "Table 13"), include related terms (data, statistics, information, contains, shows)
- If the query mentions time periods, include variations (2082-083, 2082-83, fiscal year 2082-083)
- If the query is about a concept, include related concepts and applications
- Keep it concise but comprehensive (aim for 2-3x the original length)

Return ONLY the enhanced query, no explanation or commentary.

Enhanced query:"""
        
        try:
            response = self._llm.complete(enhancement_prompt)
            enhanced = str(response).strip()
            
            # Remove quotes if LLM added them
            if enhanced.startswith('"') and enhanced.endswith('"'):
                enhanced = enhanced[1:-1]
            if enhanced.startswith("'") and enhanced.endswith("'"):
                enhanced = enhanced[1:-1]
            
            if self._verbose or logger.isEnabledFor(logging.INFO):
                logger.info(f"Query enhancement:\n  Original: '{query}'\n  Enhanced: '{enhanced}'")
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Query enhancement failed: {e}, using original query")
            return query
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the tool (delegates to call)."""
        return self.call(*args, **kwargs)

    def call(self, input: str) -> ToolOutput:
        """
        Execute enhanced retrieval.
        
        Args:
            input: User's search query
            
        Returns:
            ToolOutput containing the retrieval results
        """
        try:
            # Enhance the query
            enhanced_query = self._enhance_query(input)
            
            # Retrieve with enhanced query
            response = self._query_engine.query(enhanced_query)
            
            return ToolOutput(
                content=str(response),
                tool_name=self.metadata.name,
                raw_input={"input": input, "enhanced_query": enhanced_query},
                raw_output=response,
            )
            
        except Exception as e:
            logger.error(f"Enhanced retrieval failed for query '{input}': {e}")
            return ToolOutput(
                content=f"I encountered an error while searching the documents: {str(e)}",
                tool_name=self.metadata.name,
                raw_input={"input": input},
                raw_output=None,
                is_error=True,
            )
    
    async def acall(self, input: str) -> ToolOutput:
        """
        Async version of enhanced retrieval.
        
        Args:
            input: User's search query
            
        Returns:
            ToolOutput containing the retrieval results
        """
        try:
            # Enhance the query (async)
            if self._enable_query_expansion and len(input.split()) > 2:
                enhancement_prompt = f"""Given this search query, enhance it to improve document retrieval.

Original query: {input}

Your task: Create an enhanced version that:
1. Adds relevant context clues and related terms
2. Includes common synonyms and variations
3. Adds domain-specific terminology
4. Maintains natural language readability

Return ONLY the enhanced query, no explanation.

Enhanced query:"""
                
                try:
                    response = await self._llm.acomplete(enhancement_prompt)
                    enhanced_query = str(response).strip()
                    
                    # Remove quotes
                    if enhanced_query.startswith('"') and enhanced_query.endswith('"'):
                        enhanced_query = enhanced_query[1:-1]
                    if enhanced_query.startswith("'") and enhanced_query.endswith("'"):
                        enhanced_query = enhanced_query[1:-1]
                    
                    if self._verbose:
                        logger.info(f"Async query enhancement:\n  Original: '{input}'\n  Enhanced: '{enhanced_query}'")
                        
                except Exception as e:
                    logger.warning(f"Async query enhancement failed: {e}, using original")
                    enhanced_query = input
            else:
                enhanced_query = input
            
            # Retrieve with enhanced query
            response = await self._query_engine.aquery(enhanced_query)
            
            return ToolOutput(
                content=str(response),
                tool_name=self.metadata.name,
                raw_input={"input": input, "enhanced_query": enhanced_query},
                raw_output=response,
            )
            
        except Exception as e:
            logger.error(f"Async enhanced retrieval failed for query '{input}': {e}")
            return ToolOutput(
                content=f"I encountered an error while searching the documents: {str(e)}",
                tool_name=self.metadata.name,
                raw_input={"input": input},
                raw_output=None,
                is_error=True,
            )
