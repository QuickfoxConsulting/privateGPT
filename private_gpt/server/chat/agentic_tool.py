"""
Advanced RAG-enhanced ReAct Agent with improved context handling and source tracking.

This module provides an AgenticRAGEngine that combines document retrieval, web search,
and reasoning capabilities through a ReAct agent architecture.
"""

import logging
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
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec

from private_gpt.server.tools.document_tool import DocumentSpecificTool
from private_gpt.server.tools.summary_tool import DocumentSummaryTool
from private_gpt.server.tools.web_tool import Crawl4AITool
from private_gpt.components.vector_store.vector_store_component import VectorStoreComponent
from llama_index.core.agent.react.formatter import ReActChatFormatter


logger = logging.getLogger(__name__)


class AgenticRAGEngine(BaseChatEngine):
    """
    Advanced RAG-enhanced ReAct Agent with improved context handling and source tracking.
    
    This chat engine combines document retrieval, web search, and reasoning capabilities
    through a ReAct agent architecture. It provides comprehensive document analysis,
    real-time web search, and intelligent source attribution.
    
    Features:
    - Multi-modal tool integration (documents, web search, summaries)
    - Enhanced context management and memory handling
    - Structured reasoning with source tracking
    - Configurable similarity search and post-processing
    - Async and streaming support
    
    Args:
        llm: The language model to use for reasoning and generation
        index: Vector store index for document retrieval
        vector_store_component: Component for vector store operations
        node_postprocessors: Optional list of node post-processors
        system_prompt: Custom system prompt (uses default if None)
        memory: Chat memory buffer (creates default if None)
        callback_manager: Callback manager for tracing
        verbose: Enable verbose logging
        max_iterations: Maximum ReAct iterations
        document_files: List of document files for specific tools
        tool_name_prefix: Prefix for document-specific tool names
        citation_format: Format string for citations
        similarity_top_k: Number of similar documents to retrieve
    """

    def __init__(
        self,
        llm: LLM,
        index: VectorStoreIndex,
        vector_store_component: VectorStoreComponent,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        system_prompt: Optional[str] = None,
        memory: Optional[ChatMemoryBuffer] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        max_iterations: int = 10,
        document_files: Optional[List[str]] = None,
        tool_name_prefix: str = "doc",
        citation_format: str = "[Document: {file_name}, Page {page}]",
        similarity_top_k: int = 5
    ) -> None:
        """Initialize the AgenticRAGEngine with all necessary components."""
        # Core components
        self._llm = llm
        self._index = index
        self._vector_store_component = vector_store_component
        self._node_postprocessors = node_postprocessors or []
        
        # Configuration
        self._verbose = verbose
        self._max_iterations = max_iterations
        self._document_files = document_files or []
        self._tool_name_prefix = tool_name_prefix
        self._citation_format = citation_format
        self._similarity_top_k = similarity_top_k
        
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
        """
        Get the chat history from memory.
        
        Returns:
            List of chat messages from the current session
        """
        return self._memory.get_all()

    def reset(self) -> None:
        """Reset the chat memory and agent state."""
        self._memory.reset()
        logger.debug("Chat memory reset")

    def _build_tools(self) -> List[BaseTool]:
        """
        Construct comprehensive toolset with enhanced RAG tool configuration.
        
        Returns:
            List of configured tools for the agent
        """
        tools = []
        
        # Primary document retrieval tool
        rag_tool = self._create_primary_rag_tool()
        tools.append(rag_tool)
        
        # Document-specific tools
        doc_tools = self._create_document_specific_tools()
        tools.extend(doc_tools)
        
        # Summary tools
        summary_tools = self._create_summary_tools()
        tools.extend(summary_tools)
        
        # Web search tools
        web_tools = self._create_web_tools()
        tools.extend(web_tools)
        
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
                
            except Exception as e:
                logger.error(f"Failed to create document tool for {file_name}: {e}")
                
        return tools

    def _create_summary_tools(self) -> List[BaseTool]:
        """Create document summary tools."""
        tools = []
        
        for file_name in self._document_files:
            try:
                tool = DocumentSummaryTool(
                    index=self._index,
                    llm=self._llm,
                    file_name=file_name,
                    callback_manager=self.callback_manager,
                    tool_name_prefix="summary",
                    citation_format=self._citation_format,
                    similarity_top_k=self._similarity_top_k,
                    verbose=self._verbose,
                    settings=None  # Optional settings parameter
                )
                tools.append(tool)
                
                if self._verbose:
                    logger.info(f"Created summary tool for {file_name}")
                
            except Exception as e:
                logger.error(f"Failed to create summary tool for {file_name}: {e}")
                
        return tools

    def _create_web_tools(self) -> List[BaseTool]:
        """Create web search and crawling tools."""
        tools = []
        
        try:
            # DuckDuckGo search tool
            search_tool_spec = DuckDuckGoSearchToolSpec()
            search_tools = search_tool_spec.to_tool_list()
            tools.extend(search_tools)
            
            # Web crawling tool
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
        print(f"VALID TOOLS: {valid_tools}")
        return valid_tools

    def _get_qa_template(self) -> PromptTemplate:
        """Get the enhanced Q&A template for high-quality synthesis with citations."""
        template_str = (
            "You are a senior research analyst responsible for delivering accurate, structured, and well-sourced responses. "
            "Your answers must reflect deep reasoning, clear structure, and proper citation based only on the provided context.\n\n"
            
            "CONTEXT:\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n\n"
            
            "INSTRUCTIONS:\n"
            "1. **Use only the provided context**. Do not make assumptions or add external knowledge unless specifically cited from a web source.\n"
            "2. **Organize the response in professional Markdown**, using clear section headers and bullet points where appropriate.\n"
            "3. **Break down complex ideas** into smaller, easy-to-follow sections with proper headings.\n"
            "4. **Support all claims with specific citations**:\n"
            "   - From documents: `[Page 3](technical_manual.pdf)`\n"
            "   - From web: `[Title](https://example.com)`\n"
            "   - Prefer inline citations immediately after each fact or quote.\n"
            "5. **Synthesize multiple sources** if they cover the same topic, and explicitly mention conflicting or reinforcing perspectives.\n"
            "6. If relevant information is **missing, outdated, or ambiguous**, clearly state it under the 'Limitations' section.\n"
            "7. Always use a clear, neutral, and technical tone. Focus on facts, not opinions.\n"
            "8. Include all known metadata (e.g., document version, dates, tools used, procedures) when relevant.\n"
            "9. Think step-by-step and show your reasoning process where appropriate.\n\n"
            
            "RESPONSE FORMAT (use exactly this structure):\n"
            "```markdown\n"
            "# Executive Summary\n"
            "- Concise summary of findings\n"
            "- Mention scope of available evidence\n\n"
            # Thinking Process section helps show transparency of reasoning
            "# Thinking Process\n"
            "1. Clarified the nature of the query\n"
            "2. Chose appropriate sources from the context\n"
            "3. Extracted and synthesized insights\n\n"
            "# Detailed Analysis\n"
            "## Topic 1: <Descriptive Header>\n"
            "- Explanation with citation [Page 2](policy_guidelines.pdf)\n"
            "- Quote or paraphrased info with supporting data\n"
            "- Synthesis if multiple sources relate\n\n"
            "## Topic 2: <Another Key Insight>\n"
            "- Insight with citation [OpenAI Blog](https://openai.com)\n"
            "- Contrasting or reinforcing evidence\n"
            "- Additional metadata if applicable\n\n"
            "# Supporting Evidence\n"
            "- Bullet list of raw data, quotes, or technical points with sources\n"
            "- Use this section to provide raw references without interpretation\n\n"
            "# Limitations & Considerations\n"
            "- Mention unclear or missing areas in the context\n"
            "- Point out potential gaps or assumptions made\n"
            "- Recommend further reading or follow-up steps if applicable\n"
            "```\n\n"
            "QUERY:\n"
            "{query_str}\n\n"
            "BEGIN YOUR RESPONSE:"
        )
        return PromptTemplate(template_str)

    def _get_default_system_prompt(self) -> str:
        """Generate a comprehensive system prompt that follows ReAct format with RAG capabilities."""
        doc_list = ""
        if self._document_files:
            doc_list = "\nAvailable Documents:\n" + "\n".join(
                f"• {doc}" for doc in self._document_files
            )

        return """
        # ReAct Agent System Prompt
        You are a reasoning agent capable of handling a wide range of tasks—from answering questions and analyzing documents to summarizing content and providing actionable recommendations.
        ## Tools
        You have access to a wide variety of tools. Use them in any sequence necessary to break down and solve the task at hand. You are responsible for decomposing complex tasks and invoking tools for subtasks as needed.
        You have access to the following tools:
        {tool_desc}

        ---
        ## Core ReAct Format (Reasoning + Action)
        Use the following pattern to interact with tools:
        ```
        Thought: The current language of the user is: (user's language). I need to use a tool to help answer the question.
        Action: tool_name
        Action Input: {{"param1": "value1", "param2": "value2"}}
        ```
        Tool output will follow this format:
        ```
        Observation: tool response
        ```
        Repeat the cycle (Thought → Action → Observation) until you can respond confidently. Then conclude with one of:

        **If you can answer:**
        ```
        Thought: I can answer without using any more tools. I'll use the user's language to answer.
        Answer: [your final answer here]
        ```
        **If you cannot answer:**
        ```
        Thought: I cannot answer the question with the provided tools.
        Answer: [reason or limitation in user's language]
        ```
        ---
        ## Response Modes
        Adapt your response style to the type of query:
        ### 1. **Simple Questions / Casual Conversation**
        - Use a direct, friendly tone
        - Answer without formal structure
        - Avoid unnecessary tool usage

        ### 2. **Research / Analytical Tasks**
        Structure your response as:

        ```markdown
        # Executive Summary
        - Key findings and conclusions

        # Analysis
        ## Topic A
        - Insight with citation [page](doc.pdf)
        - Supporting evidence

        ## Topic B
        - Insight with citation
        - Supporting evidence

        # Supporting Evidence
        - Quotes, data, or links

        # Limitations
        - Gaps in data or confidence
        ```

        ### 3. **Technical / Professional Questions**
        - Provide technical accuracy
        - Use domain-appropriate terminology
        - Give specific, actionable insights
        - Add citations or references if applicable

        ## Citation Guidelines

        **Document Citations:** `[page](document_name)`  
        Example: `[42](compliance_manual.pdf)`

        **Web Citations:** `[title](url)`  
        Example: `[OpenAI Docs](https://openai.com/docs)`

        ## Quality Standards

        - **Accuracy:** Validate with available tools and sources
        - **Completeness:** Cover all relevant aspects of the query
        - **Clarity:** Keep responses well-organized and easy to understand
        - **Attribution:** Always cite sources when using external or document data
        - **Relevance:** Focus strictly on what helps answer the user's query

        ## Adaptive Strategy
        1. **Assess the Query Type**
        - Is it conversational, analytical, technical, or something else?

        2. **Match the Response Style**
        - Use structure and formality only if the query calls for it

        3. **Use Tools Thoughtfully**
        - Only when they meaningfully improve the answer

        4. **Stay in the User's Language**
        - Always match the user's preferred language

        5. **Acknowledge Uncertainty**
        - Be transparent about limitations or ambiguous areas
        6. **Put User Needs First**
        - Prioritize helpfulness over rigid format
        ---
        ## Core Principles
        - Begin tool use with a Thought
        - Default to internal documents before external sources
        - Cite and reference carefully following citations guidelines
        - Keep tone professional, helpful, and clear
        - Tailor responses to be either quick or comprehensive as needed
        ---
        ## Your Goal
        Deliver accurate, well-reasoned, and user-appropriate answers—whether it's a fast fact, a structured analysis, or a technical deep dive.
        """

    def _sync_memory(self, chat_history: Optional[List[ChatMessage]]) -> None:
        """Synchronize memory with provided chat history."""
        if chat_history:
            self._memory.reset()
            for message in chat_history:
                self._memory.put(message)

    def _format_response(self, response: Any) -> AgentChatResponse:
        """Format agent response into standardized chat response."""
        return AgentChatResponse(
            response=getattr(response, 'response', str(response)),
            sources=getattr(response, 'sources', []),
            source_nodes=getattr(response, 'source_nodes', [])
        )

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
            logger.error(f"Async stream chat error: {e}", exc_info=self._verbose)
            return StreamingAgentChatResponse(
                chat_stream=iter(["I encountered an error while processing your request."]),
                sources=[],
                source_nodes=[]
            )