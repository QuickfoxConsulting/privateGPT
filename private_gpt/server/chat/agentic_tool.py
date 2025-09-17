"""
RAG ReAct Agent with improved context handling and source tracking.

This module provides an AgenticRAGEngine that combines document retrieval, web search,
and reasoning capabilities through a ReAct agent architecture.
"""
import logging
import random
import asyncio
import time
from datetime import datetime
from threading import Thread
from typing import Any, List, Optional, Dict, Union

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.callbacks import CallbackManager, trace_method
from llama_index.core.chat_engine.types import (
    AgentChatResponse, 
    StreamingAgentChatResponse, 
    BaseChatEngine
)
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import BaseTool, QueryEngineTool
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import LLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
from llama_index.core.indices.vector_store import VectorStoreIndex

from private_gpt.server.tools.time_tool import TimeTool
from private_gpt.server.tools.web_tool import Crawl4AITool
from private_gpt.server.tools.document_tool import DocumentSpecificTool
from private_gpt.server.tools.summary_tool import DocumentSummaryTool
from private_gpt.server.tools.websearch import SerperSearchToolSpec

from private_gpt.components.vector_store.vector_store_component import VectorStoreComponent
from llama_index.core.agent.react.formatter import ReActChatFormatter
from private_gpt.components.node_store.node_store_component import NodeStoreComponent

logger = logging.getLogger(__name__)

class AgenticRAGEngine(BaseChatEngine):
    """
    Advanced RAG-enhanced ReAct Agent with improved context handling and source tracking.
    
    This chat engine combines document retrieval, web search, and reasoning capabilities
    through a ReAct agent architecture. It provides comprehensive document analysis,
    real-time web search, and intelligent source attribution.
    """

    def __init__(
        self,
        llm: LLM,
        index: VectorStoreIndex,
        vector_store_component: VectorStoreComponent,
        node_store_component: NodeStoreComponent,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        system_prompt: Optional[str] = None,
        memory: Optional[ChatMemoryBuffer] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        max_iterations: int = 10,
        document_files: Optional[List[str]] = None,
        tool_name_prefix: str = "doc",
        citation_format: str = "[Document: {file_name}, Page {page}]",
        similarity_top_k: int = 5,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: tuple[float, float] = (0.1, 0.3)
    ) -> None:
        """Initialize the AgenticRAGEngine with all necessary components."""
        # Core components
        self._llm = llm
        self._index = index
        self._vector_store_component = vector_store_component
        self._node_postprocessors = node_postprocessors or []
        self._node_store = node_store_component
        
        # Configuration
        self._verbose = verbose
        self._max_iterations = max_iterations
        self._document_files = document_files or []
        self._tool_name_prefix = tool_name_prefix
        self._citation_format = citation_format
        self._similarity_top_k = similarity_top_k
        
        # Rate limit handling
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._jitter = jitter
        
        # Callback management
        self.callback_manager = callback_manager or CallbackManager([])
        
        # Memory initialization
        self._memory = memory or self._create_default_memory()
        
        # Agent initialization
        self._agent = self._create_agent(system_prompt)
        
        logger.info(f"AgenticRAGEngine initialized with {len(self._document_files)} documents")

    def _create_default_memory(self) -> ChatMemoryBuffer:
        """Create default memory buffer with appropriate token limits."""
        token_limit = int(self._llm.metadata.context_window * 0.5)
        return ChatMemoryBuffer.from_defaults(token_limit=token_limit)

    def _create_agent(self, system_prompt: Optional[str]) -> ReActAgent:
        """Create the ReAct agent with all tools and configuration."""
        tools = self._build_tools()
        
        system_prompt_str = system_prompt if system_prompt else self._get_default_system_prompt()
        chat_formatter = ReActChatFormatter.from_defaults(system_header=system_prompt_str)
        return ReActAgent.from_tools(
            tools=tools,
            llm=self._llm,
            memory=self._memory,
            react_chat_formatter=chat_formatter,
            verbose=self._verbose,
            callback_manager=self.callback_manager,
            max_iterations=self._max_iterations,
            tool_retrieval_mode="structured"
        )

    @property
    def chat_history(self) -> List[ChatMessage]:
        return self._memory.get_all()

    def reset(self) -> None:
        """Reset the chat memory and agent state."""
        self._memory.reset()
        logger.debug("Chat memory reset")

    def _build_tools(self) -> List[BaseTool]:
        tools = []
        
        # Primary document retrieval tool
        rag_tool = self._create_primary_rag_tool()
        tools.append(rag_tool)
        
        # Document-specific tools
        doc_tools = self._create_document_specific_tools()
        tools.extend(doc_tools)
        
        # # Summary tools
        summary_tools = self._create_summary_tools()
        tools.extend(summary_tools)
        
        # Web search tools
        web_tools = self._create_web_tools()
        tools.extend(web_tools)
        
        # Add time tool
        time_tool = TimeTool()
        tools.append(time_tool)
        
        # Validate and return tools
        validated_tools = self._validate_tools(tools)
        
        if self._verbose:
            logger.info(f"Built {len(validated_tools)} total tools")
            for tool in validated_tools:
                logger.debug(f"  - {tool.metadata.name}: {tool.metadata.description[:50]}...")
        
        return validated_tools

    def _create_primary_rag_tool(self) -> QueryEngineTool:
        """Create the primary document retrieval tool."""
        query_engine = self._create_query_engine()
        
        return QueryEngineTool.from_defaults(
            query_engine=query_engine,
            name="document_retriever",
            description=(
                "Search and retrieve information from the organizational knowledge base. "
                "This tool provides access to all indexed documents and should be your "
                "primary source for factual information about:\n"
                "• Company policies and procedures\n"
                "• Technical specifications and documentation\n"
                "• Historical records and data\n"
                "• Process guidelines and workflows\n"
                "• Regulatory and compliance information\n\n"
                "Always consult this tool first for document-based queries. "
                "It returns comprehensive, contextual information with proper citations."
            )
        )

    def _create_query_engine(self) -> RetrieverQueryEngine:
        """Create the core query engine with optimized configuration."""
        retriever = self._vector_store_component.get_retriever(
            index=self._index,
            similarity_top_k=self._similarity_top_k
        )
        
        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
            llm=self._llm,
            structured_answer_filtering=True,
            text_qa_template=self._get_qa_template(),
            streaming=False,
            callback_manager=self.callback_manager
        )
        
        return RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=self._node_postprocessors,
            callback_manager=self.callback_manager
        )

    def _create_document_specific_tools(self) -> List[BaseTool]:
        """Create tools for specific document analysis."""
        tools = []
        
        if not self._document_files:
            logger.warning("No document files provided for document-specific tools")
            return tools
            
        for file_name in self._document_files:
            try:
                tool = DocumentSpecificTool(
                    index=self._index,
                    file_name=file_name,
                    llm=self._llm,
                    vector_store_component=self._vector_store_component,
                    node_postprocessors=self._node_postprocessors,
                    callback_manager=self.callback_manager,
                    tool_name_prefix=self._tool_name_prefix,
                    citation_format=self._citation_format,
                    similarity_top_k=self._similarity_top_k,
                    verbose=self._verbose
                )
                tools.append(tool)
                logger.info(f"Created document tool for {file_name}")
                
            except Exception as e:
                logger.error(f"Failed to create document tool for {file_name}: {e}")
                
        return tools

    def _create_summary_tools(self) -> List[BaseTool]:
        """Create document summary tools."""
        tools = []
        
        if not self._document_files:
            logger.warning("No document files provided for summary tools")
            return tools
            
        for file_name in self._document_files:
            try:
                tool = DocumentSummaryTool(
                    index=self._index,
                    llm=self._llm,
                    node_store_component=self._node_store,
                    vector_store_component=self._vector_store_component,
                    file_name=file_name,
                    callback_manager=self.callback_manager,
                    tool_name_prefix="summary",
                    citation_format=self._citation_format,
                    similarity_top_k=self._similarity_top_k,
                    verbose=self._verbose,
                )
                tools.append(tool)
                logger.info(f"Created summary tool for {file_name}")
                
            except Exception as e:
                logger.error(f"Failed to create summary tool for {file_name}: {e}")
                
        return tools

    def _create_web_tools(self) -> List[BaseTool]:
        """Create web search and crawling tools."""
        tools = []
        
        try:
            search_tool_spec = SerperSearchToolSpec()
            search_tools = search_tool_spec.to_tool_list()
            tools.extend(search_tools)
            
            crawl_tool = Crawl4AITool()
            tools.append(crawl_tool)
            
            if self._verbose:
                logger.info(f"Created {len(tools)} web tools")
                
        except Exception as e:
            logger.error(f"Failed to create web tools: {e}")
        return tools

    def _validate_tools(self, tools: List[BaseTool]) -> List[BaseTool]:
        """Validate tools for name uniqueness and proper configuration."""
        seen_names = set()
        valid_tools = []
        
        for tool in tools:
            tool_name = tool.metadata.name
            
            if tool_name in seen_names:
                logger.error(f"Duplicate tool name detected: {tool_name}")
                raise ValueError(f"Duplicate tool name: {tool_name}")
                
            if not tool.metadata.description:
                logger.warning(f"Tool {tool_name} has no description")
                
            seen_names.add(tool_name)
            valid_tools.append(tool)
        return valid_tools

    def _get_qa_template(self) -> PromptTemplate:
        """Get the enhanced summarization template for high-quality synthesis with citations and structured reasoning."""
        template_str = (
            "You are a document grounded assistant. Your task is to generate a clear, structured, and well-sourced answer "
            "strictly based on the provided context, which may come from research papers, books, technical manuals, reports, or other documents.\n\n"

            "CONTEXT:\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n\n"

            "INSTRUCTIONS:\n"
            "1. **Use only the provided context**. Do not use outside knowledge unless the context includes explicit web references.\n"
            "2. **Write in professional Markdown**, with informative section headers and bullet points when appropriate.\n"
            "3. **Decompose complex content** into manageable, logically ordered sections.\n"
            "4. **Cite sources precisely and consistently**:\n"
            "   - Do NOT use numeric citations like [1], [2], etc. For every web source, always use a markdown link in the format [Article Title](https://example.com) directly after the relevant statement.\n"
            "   - From documents: `[Page 5](document.pdf)`\n"
            "   - From web sources: `[Article Title](https://example.com)`\n"
            "   - Place citations **inline immediately after each referenced statement**.\n"
            "5. **Synthesize overlapping information** across sources and highlight agreement or conflict between them.\n"
            "6. **If context is incomplete or unclear**, explicitly state these gaps in the Limitations section.\n"
            "7. Use a formal, factual, and neutral tone. Avoid speculative or subjective language.\n"
            "8. Include key metadata (e.g., document titles, authors, publication years, tools, version numbers) when mentioned in the context.\n"
            "9. Follow a step-by-step reasoning approach where appropriate, especially when summarizing technical or multi-part content.\n"
            "10. **At the end of your answer, always include a 'Sources' section listing all documents or URLs referenced in your answer.**\n\n"
            "QUERY:\n"
            "{query_str}\n\n"
            "BEGIN YOUR RESPONSE:"
        )
        return PromptTemplate(template_str)

    def _get_default_system_prompt(self) -> str:
        """Generate an optimized system prompt that follows ReAct format with RAG capabilities."""
        return """
            You are QuickREF, an intelligent reasoning agent that systematically solves complex tasks through structured thinking and strategic tool usage.

            ## Core Capabilities
            - **Analytical Reasoning**: Break down complex problems into manageable steps
            - **Tool Orchestration**: Use multiple tools efficiently to gather comprehensive information
            - **Quality Assurance**: Validate accuracy and source credibility
            - **Adaptive Communication**: Match response style to user needs and expertise level
            
            ## Available Tools
            {tool_desc}

            ## Tool Selection Strategy
            **Document Queries**:
            - Specific document mentioned → Use `doc_[document_name]` (e.g., `doc_compliance_manual_pdf`)
            - General document search → Use `document_retriever`
            - Document summary needed → Use `summary_[document_name]`
            
            **Web Information**:
            - Real-time search → `serper_web_search`
            - Instant answers → `serper_instant_search` 
            - Recent news → `serper_news_search`
            - Web content analysis → `crawl4ai_scraper`
            
            **Efficiency Rules**:
            - Maximum 5 tool calls per query
            - Each tool call must add new value
            - Answer when sufficient information is obtained
            - Avoid redundant queries

            ## Response Standards
            Write accurate, well-cited responses using markdown formatting. Ensure:
            - **Factual accuracy** based on reliable sources
            - **Proper citations** with source attribution
            - **Language consistency** - respond in user's query language
            - **Appropriate depth** matched to user expertise

            ## Citation Format
            - **Documents**: `[page 42](manual.pdf)` or `[Section 3.2](spec.docx)`
            - **Web sources**: `[Article Title](https://example.com)`
            - **Multiple sources**: `[Source 1](ref1), [Source 2](ref2)`

            ## Error Handling
            When tools fail:
            1. Acknowledge limitation clearly
            2. Try alternative approaches when possible
            3. Provide partial answers based on available information
            4. Explain what went wrong and suggest alternatives

            ## Language Handling
            If user query is not in English:
            1. Translate query to English for tool usage
            2. Use tools with English queries
            3. Translate final response back to user's language

            ## Output Format
            Always use this ReAct format:

            ```
            Thought: [Your reasoning about what you need to accomplish]
            Action: [tool_name] (if using a tool)
            Action Input: {"query": "your question in English"}
            ```

            **Format Rules**:
            - ALWAYS start with Thought
            - Use valid JSON for Action Input: {"key": "value"}
            - NEVER surround response with markdown code markers
            - Continue until sufficient information is gathered

            End with either:
            ```
            Thought: I can answer without using any more tools.
            Answer: [comprehensive answer in user's language]
            ```
            OR
            ```
            Thought: I cannot answer with available tools.
            Answer: [explanation in user's language]
            ```

            ## Query-Specific Instructions

            **Academic Research**: Provide detailed, scientific write-ups with methodology and limitations
            **News**: Summarize concisely, use lists, highlight titles, prioritize recent and diverse sources
            **Weather**: Short forecast only, include warnings if present
            **People**: Brief biography, separate multiple people, focus on achievements
            **Coding**: Use code blocks with syntax highlighting, code first then explanation
            **Recipes**: Step-by-step with ingredients, amounts, timing, and tips
            **Translation**: Direct translation only, no citations needed
            **Math/Science**: Show work for complex problems, use LaTeX format \(x^2\) and \[formula\]
            **URL Lookup**: Summarize only that URL's content
            **Shopping**: Organize by categories, limit to 5 results, include prices and features

            ## User Profile (use when relevant)
            - **Platform**: Manjaro Linux, Android-focused (no iPhone)
            - **Technical Level**: Linux-savvy, open-source familiar
            - **Language**: English

            ## Success Criteria
            Deliver accurate, comprehensive, well-sourced answers efficiently while maintaining transparency in your reasoning process.
        """

    def _sync_memory(self, chat_history: Optional[List[ChatMessage]]) -> None:
        """Synchronize memory with provided chat history."""
        if chat_history:
            self._memory.reset()
            for message in chat_history:
                self._memory.put(message)

    def _format_response(self, response: Any) -> AgentChatResponse:
        """Format agent response into standardized chat response, appending a Sources section if not present."""
        answer = getattr(response, 'response', str(response))
        sources = getattr(response, 'sources', [])
        source_nodes = getattr(response, 'source_nodes', [])
        return AgentChatResponse(
            response=answer,
            sources=sources,
            source_nodes=source_nodes
        )

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if the error is a rate limit error."""
        error_str = str(error).lower()
        return any(msg in error_str for msg in [
            "rate limit",
            "too many requests",
            "429",
            "resource exhausted",
            "quota exceeded"
        ])

    def _get_retry_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        delay = min(self._base_delay * (2 ** attempt), self._max_delay)
        jitter = random.uniform(*self._jitter)
        return delay * (1 + jitter)

    async def _handle_rate_limit(self, attempt: int, error: Exception) -> None:
        """Handle rate limit error with exponential backoff."""
        delay = self._get_retry_delay(attempt)
        logger.warning(
            f"Rate limit hit, retrying in {delay:.1f} seconds "
            f"(attempt {attempt + 1}/{self._max_retries})"
        )
        await asyncio.sleep(delay)

    @trace_method("chat")
    def chat(
        self, 
        message: str, 
        chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        """
        Synchronous chat method following BaseChatEngine interface.
        
        Args:
            message: User's message/query
            chat_history: Optional chat history to restore context
            
        Returns:
            AgentChatResponse with answer, sources, and source nodes
        """
        self._sync_memory(chat_history)
        
        for attempt in range(self._max_retries):
            try:
                response = self._agent.chat(message)
                
                # Ensure message is stored in memory
                if hasattr(response, 'message') and response.message:
                    self._memory.put(response.message)
                else:
                    # Fallback: create message from response
                    message = ChatMessage(
                        content=getattr(response, 'response', str(response)),
                        role=MessageRole.ASSISTANT
                    )
                    self._memory.put(message)
                
                return self._format_response(response)
                
            except Exception as e:
                if self._is_rate_limit_error(e) and attempt < self._max_retries - 1:
                    delay = self._get_retry_delay(attempt)
                    logger.warning(
                        f"Rate limit hit, retrying in {delay:.1f} seconds "
                        f"(attempt {attempt + 1}/{self._max_retries})"
                    )
                    time.sleep(delay)
                    continue
                
                logger.error(f"Chat error: {e}", exc_info=self._verbose)
                return AgentChatResponse(
                    response="I encountered an error while processing your request. Please try again.",
                    sources=[],
                    source_nodes=[]
                )

    @trace_method("stream_chat")
    def stream_chat(
        self, 
        message: str, 
        chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        """
        Streaming chat method following BaseChatEngine interface.
        
        Args:
            message: User's message/query
            chat_history: Optional chat history to restore context
            
        Returns:
            StreamingAgentChatResponse with streaming generator
        """
        self._sync_memory(chat_history)
        
        for attempt in range(self._max_retries):
            try:
                response = self._agent.stream_chat(message)
                
                result = StreamingAgentChatResponse(
                    chat_stream=response.response_gen,
                    sources=getattr(response, 'sources', []),
                    source_nodes=getattr(response, 'source_nodes', [])
                )
                
                # Handle memory update in background thread
                Thread(
                    target=result.write_response_to_history, 
                    args=(self._memory,),
                    daemon=True
                ).start()
                
                return result
                
            except Exception as e:
                if self._is_rate_limit_error(e) and attempt < self._max_retries - 1:
                    delay = self._get_retry_delay(attempt)
                    logger.warning(
                        f"Rate limit hit, retrying in {delay:.1f} seconds "
                        f"(attempt {attempt + 1}/{self._max_retries})"
                    )
                    time.sleep(delay)
                    continue
                
                logger.error(f"Stream chat error: {e}", exc_info=self._verbose)
                return StreamingAgentChatResponse(
                    chat_stream=iter(["I encountered an error while processing your request."]),
                    sources=[],
                    source_nodes=[]
                )

    @trace_method("achat")
    async def achat(
        self, 
        message: str, 
        chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        """
        Asynchronous chat method following BaseChatEngine interface.
        
        Args:
            message: User's message/query
            chat_history: Optional chat history to restore context
            
        Returns:
            AgentChatResponse with answer, sources, and source nodes
        """
        self._sync_memory(chat_history)
        
        for attempt in range(self._max_retries):
            try:
                response = await self._agent.achat(message)
                
                # Ensure message is stored in memory
                if hasattr(response, 'message') and response.message:
                    self._memory.put(response.message)
                else:
                    message = ChatMessage(
                        content=getattr(response, 'response', str(response)),
                        role=MessageRole.ASSISTANT
                    )
                    self._memory.put(message)
                
                return self._format_response(response)
                
            except Exception as e:
                if self._is_rate_limit_error(e) and attempt < self._max_retries - 1:
                    await self._handle_rate_limit(attempt, e)
                    continue
                
                logger.error(f"Async chat error: {e}", exc_info=self._verbose)
                return AgentChatResponse(
                    response="I encountered an error while processing your request. Please try again.",
                    sources=[],
                    source_nodes=[]
                )

    @trace_method("astream_chat")
    async def astream_chat(
        self, 
        message: str, 
        chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        """
        Asynchronous streaming chat method following BaseChatEngine interface.
        
        Args:
            message: User's message/query
            chat_history: Optional chat history to restore context
            
        Returns:
            StreamingAgentChatResponse with async streaming generator
        """
        self._sync_memory(chat_history)
        
        for attempt in range(self._max_retries):
            try:
                response = await self._agent.astream_chat(message)
                
                result = StreamingAgentChatResponse(
                    chat_stream=response.response_gen,
                    sources=getattr(response, 'sources', []),
                    source_nodes=getattr(response, 'source_nodes', [])
                )
                
                # Handle memory update in background thread
                Thread(
                    target=result.write_response_to_history, 
                    args=(self._memory,),
                    daemon=True
                ).start()
                
                return result
                
            except Exception as e:
                if self._is_rate_limit_error(e) and attempt < self._max_retries - 1:
                    await self._handle_rate_limit(attempt, e)
                    continue
                
                logger.error(f"Async stream chat error: {e}", exc_info=self._verbose)
                return StreamingAgentChatResponse(
                    chat_stream=iter(["I encountered an error while processing your request."]),
                    sources=[],
                    source_nodes=[]
                )