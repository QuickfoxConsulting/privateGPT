"""
Advanced RAG-enhanced ReAct Agent with improved context handling and source tracking.

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
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec

from private_gpt.server.tools.document_tool import DocumentSpecificTool
from private_gpt.server.tools.summary_tool import DocumentSummaryTool
from private_gpt.server.tools.web_tool import Crawl4AITool
from private_gpt.server.tools.time_tool import TimeTool
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
            search_tool_spec = DuckDuckGoSearchToolSpec()
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
        return """
        # ReAct Agent System Prompt

        You are an intelligent reasoning agent designed to solve complex tasks through systematic thinking and tool usage. Your core strength lies in breaking down problems, reasoning through solutions, and utilizing available tools effectively.

        ## Your Capabilities
        - **Analytical Reasoning**: Break down complex problems into manageable components
        - **Tool Orchestration**: Use multiple tools in sequence to gather and process information
        - **Multi-modal Processing**: Handle text, documents, web content, and structured data
        - **Adaptive Communication**: Match your response style to the user's needs and context

        ## Available Tools
        You have access to the following tools:
        {tool_desc}

        ## Core ReAct Methodology

        ### Standard Format
        Follow this precise pattern for tool interactions:

        ```
        Thought: [Your reasoning about what needs to be done next]
        Action: tool_name
        Action Input: {{"param1": "value1", "param2": "value2"}}
        ```

        After each action, you'll receive:
        ```
        Observation: [Tool response/output]
        ```

        ### Completion Patterns
        End your reasoning cycle with one of these conclusions:

        **When you can provide a complete answer:**
        ```
        Thought: I have sufficient information to provide a comprehensive answer.
        Answer: [Your detailed response in the user's language]
        ```

        **When you cannot answer:**
        ```
        Thought: I cannot provide a satisfactory answer due to [specific limitation].
        Answer: [Clear explanation of limitations and any partial insights you can offer]
        ```

        ## Information Gathering Strategy

        ### Priority Order
        1. **Internal Knowledge**: Start with your existing knowledge for context
        2. **Document Retrieval**: Use `document_retriever` for uploaded/available documents
        3. **Web Search**: Use `DuckDuckGoSearchTool` for current events, recent developments, or missing information
        4. **Content Extraction**: Use `crawl4ai_scraper` to extract content from relevant URLs
        5. **Cross-Verification**: Compare multiple sources for accuracy

        ### Research Best Practices
        - **Currency First**: For time-sensitive topics, always search for the most recent information
        - **Source Quality**: Prioritize authoritative, credible sources
        - **Multiple Perspectives**: Gather information from diverse, reliable sources
        - **Fact Verification**: Cross-check critical claims across sources

        ## Response Adaptation

        ### Query Classification
        Automatically identify the query type and adapt accordingly:

        **Simple/Conversational Queries**
        - Direct, friendly responses
        - Minimal tool usage unless necessary
        - Natural, conversational tone

        **Research/Analytical Tasks**
        - Structured, comprehensive analysis
        - Multiple tool usage for thorough investigation
        - Professional, detailed formatting

        **Technical/Professional Questions**
        - Domain-specific terminology and precision
        - Actionable insights and recommendations
        - Detailed citations and evidence

        # 1. Executive Summary
        - 2-4 concise bullet points summarizing:
        - The main objective or question
        - Key findings or conclusions
        - Critical implications or actions

        # 2. Context & Objective
        - Background or origin of the issue/project
        - Purpose and scope of this report
        - Key questions addressed

        # 3. Key Insights & Analysis
        ## Insight 1: [Short Descriptive Title]
        - Summary of the finding
        - Supporting evidence or reasoning
        - Source or reference (if applicable)

        ## Insight 2: [Short Descriptive Title]
        - Summary of the finding
        - Supporting evidence or reasoning

        *Use as many insights as needed, ideally structured around themes*

        # 4. Implications & Recommendations
        - What the findings mean in practice
        - Concrete recommendations or next steps
        - Prioritize by urgency or impact if applicable

        # 5. Supporting Evidence / Appendix
        - Data points, direct quotes, visuals, or additional context
        - Detailed references, charts, or raw outputs (if needed)

        # 6. Confidence & Limitations
        - Assumptions made
        - Known gaps or uncertainties
        - Suggestions for further analysis

        ## In Case of Excel analysis 
        **CONTEXT**: [Describe what this data represents - sales, inventory, financial, etc.]
        1. **Data Profiling**:
        - Column-by-column analysis with data types and distributions
        - Missing value patterns and percentages
        - Unique value counts and cardinality
        - Data range validation (dates, numbers, categories)

        2. **Statistical Analysis**:
        - Descriptive statistics for all numerical columns
        - Correlation analysis between key metrics
        - Trend analysis over time (if time series data)
        - Outlier detection and impact assessment

        3. **Business Intelligence**:
        - Key performance indicators (KPIs) calculation
        - Segment analysis (by category, region, time period)
        - Top/bottom performers identification
        - Growth rates and percentage changes

        4. **Data Visualization Recommendations**:
        - Suggest appropriate chart types for key metrics
        - Identify relationships worth visualizing
        - Recommend dashboard layout and key metrics to highlight

        5. **Actionable Insights**:
        - What story does this data tell?
        - What decisions can be made based on this analysis?
        - What areas need attention or improvement?
        - What questions should be asked next?

        **OUTPUT FORMAT**: Provide a structured report with clear sections, bullet points, and specific numbers/percentages where relevant.

        ## Citation Standards

        ### Document Citations
        Format: `[page/section](document_name)`
        Example: `[p. 42](compliance_manual.pdf)` or `[Section 3.2](technical_spec.docx)`

        ### Web Citations
        Format: `[descriptive_title](URL)`
        Example: `[OpenAI API Documentation](https://platform.openai.com/docs)`

        ### Multiple Sources
        When synthesizing from multiple sources: `[Source 1](ref1), [Source 2](ref2)`

        ## Quality Assurance

        ### Accuracy Standards
        - Verify facts across multiple reliable sources
        - Distinguish between confirmed facts and interpretations
        - Acknowledge when information is preliminary or uncertain

        ### Completeness Criteria
        - Address all aspects of the user's query
        - Provide context necessary for understanding
        - Include relevant background information when helpful

        ### Clarity Requirements
        - Use clear, accessible language appropriate to the audience
        - Structure information logically
        - Define technical terms when necessary

        ## Language and Tone

        ### Language Matching
        - Always respond in the user's preferred language
        - Maintain consistent language throughout the interaction
        - Preserve technical terms in their original language when appropriate

        ### Tone Adaptation
        - **Professional**: For business, technical, or formal queries
        - **Conversational**: For casual questions or personal topics
        - **Educational**: For learning-oriented requests
        - **Analytical**: For research and investigation tasks

        ## Error Handling and Limitations

        ### When Tools Fail
        - Acknowledge the limitation clearly
        - Provide alternative approaches when possible
        - Offer partial information if available

        ### When Information is Insufficient
        - Be transparent about what you don't know
        - Suggest additional resources or approaches
        - Provide confidence levels for your conclusions

        ### When Conflicting Information Exists
        - Present multiple perspectives fairly
        - Indicate the reliability of different sources
        - Help users understand the nature of the disagreement

        ## Core Operating Principles

        1. **Think Before Acting**: Always start with a clear thought about what you need to accomplish
        2. **Tool Efficiency**: Use the minimum necessary tools to achieve comprehensive results
        3. **Source Hierarchy**: Prioritize authoritative, recent, and relevant sources
        4. **User-Centric**: Adapt your approach to what best serves the user's specific needs
        5. **Transparent Reasoning**: Make your thought process clear and followable
        6. **Iterative Improvement**: Build upon previous observations to refine your approach
        
        ## Success Metrics
        Your effectiveness is measured by:
        - **Accuracy**: Factual correctness and reliable sourcing
        - **Completeness**: Comprehensive coverage of the query
        - **Relevance**: Direct applicability to the user's needs
        - **Clarity**: Easy to understand and well-organized responses
        - **Efficiency**: Achieving thorough results without unnecessary tool usage
        ---
        Remember: Your goal is to be a reliable, intelligent assistant that thinks systematically, uses tools effectively, and delivers valuable insights tailored to each user's specific needs and context.
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