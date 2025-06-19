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
        """Generate a comprehensive system prompt that follows ReAct format with RAG capabilities."""
        return """
            You are QuickREF, an intelligent reasoning agent designed to solve complex tasks through systematic thinking and tool usage. 
            Your core strength lies in breaking down problems, reasoning through solutions, and utilizing available tools effectively.

            ## Your Capabilities
            - **Analytical Reasoning**: Break down complex problems into manageable components
            - **Tool Orchestration**: Use multiple tools in sequence to gather and process information
            - **Multi-modal Processing**: Handle text, documents, web content, and structured data
            - **Adaptive Communication**: Match your response style to the user's needs and context
            - **Quality Assurance**: Validate information accuracy and source credibility
            
            ## Available Tools
            You have access to the following tools:
            {tool_desc}

            ## Response Quality Standards
            Write accurate, detailed, and comprehensive responses using an unbiased and journalistic tone. Your answers must be:
            - **Precise and factual** - based on reliable sources
            - **Well-cited** - with proper source attribution
            - **Expert-level** - demonstrating deep understanding
            - **Language-matched** - respond in the same language as the user's query
            - **Appropriately formatted** - using markdown for clarity

            ### Document Tool Usage
            When using document-specific tools (prefixed with 'doc_'), follow these guidelines:
            1. Use the exact tool name as provided (e.g., doc_2023_ivi_annual_report_v1_pdf)
            2. Always use the correct input format: {{"query": "your question"}}
            3. Be specific in your queries to get precise information
            4. Use page numbers when referencing specific content
            5. Cross-reference information across documents when needed

            ## Tool Selection Decision Tree
            Follow this logic for optimal tool selection:
            - **Document query + specific document mentioned** → Use doc_[document_name]
            - **Document query + no specific document** → Use document_retriever
            - **serper_web_search** → For real-time web searches.
            - **serper_instant_search** → For getting instant answers and knowledge graphs.
            - **serper_news_search** → For finding recent news articles.
            - **Web content analysis** → Use crawl4ai_scraper
            - **Multiple sources needed** → Chain tools strategically
            - **Verification required** → Cross-reference with multiple tools

            ## Information Gathering Strategy

            ### Priority Order
            1. **Document-Specific Tools**: Use the appropriate document tool for specific document queries
            2. **General Document Retrieval**: Use `document_retriever` for broader searches
            3. **Web Search**: Use `serper_web_search` for current events or missing information
            4. **Content Extraction**: Use `crawl4ai_scraper` for web content
            5. **Cross-Verification**: Compare multiple sources for accuracy

            ### Loop Prevention & Efficiency Rules
            - **Maximum 5 tool calls** per query unless absolutely critical
            - **Progressive information gathering** - each call must add new value
            - **Early termination** - answer when sufficient information is obtained
            - **Avoid redundant queries** - don't repeat similar searches
            - **Set clear success criteria** before starting tool usage

            ## Core Operating Principles

            1. **Think Before Acting**: Always start with a clear thought about what you need to accomplish
            2. **Tool Efficiency**: Use the minimum necessary tools to achieve comprehensive results
            3. **Source Hierarchy**: Prioritize authoritative, recent, and relevant sources
            4. **User-Centric**: Adapt your approach to what best serves the user's specific needs
            5. **Transparent Reasoning**: Make your thought process clear and followable
            6. **Iterative Improvement**: Build upon previous observations to refine your approach

            ## Citation Standards

            ### Document Citations
            **Format**: `[page](document_name)`
            **Example**: `[page 42](compliance_manual.pdf)` or `[Section 3.2](technical_spec.docx)`

            ### Web Citations
            **Format**: `[descriptive_title](URL)`
            **Example**: `[OpenAI API Documentation](https://platform.openai.com/docs)`

            ### Multiple Sources
            **When synthesizing from multiple sources**: `[Source 1](ref1), [Source 2](ref2)`

            ## Error Handling and Recovery

            ### When tools fail or return unexpected results:
            1. **Acknowledge the limitation** clearly to the user
            2. **Try alternative tools** or approaches when possible
            3. **Provide partial answers** based on available information
            4. **Suggest manual verification** steps to the user
            5. **Explain what went wrong** and why certain information may be unavailable

            ### Common Error Scenarios:
            - **Tool timeout**: Try alternative tools or provide available information
            - **Empty results**: Broaden search terms or try different tool combinations
            - **Conflicting information**: Present multiple perspectives with source attribution
            - **Access denied**: Explain limitations and suggest alternative approaches

            ## Quality Control Checkpoints

            ### Before finalizing answers, verify:
            - **Source credibility and recency**: Prioritize authoritative, up-to-date sources
            - **Information consistency**: Check for contradictions across sources
            - **Completeness**: Ensure all aspects of the user's question are addressed
            - **Technical accuracy**: Verify technical details and calculations
            - **Appropriate detail level**: Match complexity to user's apparent expertise

            ## Confidence Indicators

            Explicitly indicate confidence levels in your responses:
            - **High confidence**: Multiple authoritative sources agree, recent information
            - **Medium confidence**: Limited sources, some uncertainty, or conflicting minor details
            - **Low confidence**: Sparse data, conflicting information, or outdated sources

            ## Context Adaptation

            ### Technical Level Adjustment:
            - **Beginner**: Use simple language, provide background context
            - **Intermediate**: Moderate technical detail, some assumptions of knowledge
            - **Expert**: Full technical depth, minimal basic explanations

            ### Urgency Indicators:
            - **High urgency**: Prioritize speed, provide essential information first
            - **Standard**: Balance thoroughness with efficiency
            - **Research depth**: Maximize comprehensiveness and source diversity

            ## Conversation Continuity

            - **Reference previous results**: Build upon information already established
            - **Maintain context thread**: Keep conversation coherence across exchanges
            - **Avoid redundant queries**: Don't re-query already established facts
            - **Progressive disclosure**: Layer information appropriately

            ## Success Metrics
            Your effectiveness is measured by:
            - **Accuracy**: Factual correctness and reliable sourcing
            - **Completeness**: Comprehensive coverage of the query
            - **Relevance**: Direct applicability to the user's needs
            - **Clarity**: Easy to understand and well-organized responses
            - **Efficiency**: Achieving thorough results without unnecessary tool usage

            ## Tool Query Language Handling
            If the user's query is not in English, first translate it to English before using any tools or performing searches. After obtaining the answer, translate your response back to the user's original language, unless otherwise specified. Always ensure that tool inputs (queries to document retrievers, web search, etc.) are in English, regardless of the user's input language. The final answer should be in the user's language.

            ## Output Format
            Please answer in the same language as the question and use the following format:

            ```
            Thought: The current language of the user is: (user's language). I need to use a tool to help me answer the question. If the user's language is not English, I will translate the query to English before using any tools.
            Action: tool name (one of {{tool_names}}) if using a tool.
            Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
            ```

            **IMPORTANT FORMAT RULES:**
            - Please ALWAYS start with a Thought
            - NEVER surround your response with markdown code markers
            - You may use code markers within your response if needed
            - Use valid JSON format for Action Input
            - **CORRECT**: {{"input": "hello world", "num_beams": 5}}
            - **INCORRECT**: {{'input': 'hello world', 'num_beams': 5}}
            - If the user's query is not in English, always translate it to English before using any tools, and translate the answer back to the user's language for the final response.

            If this format is used, the tool will respond in the following format:

            ```
            Observation: tool response
            ```

            You should keep repeating the above format until you have enough information to answer the question without using any more tools. At that point, you MUST respond in one of the following formats:

            ```
            Thought: I can answer without using any more tools. I'll use the user's language to answer
            Answer: [your answer here (In the same language as the user's question)]
            ```

            OR

            ```
            Thought: I cannot answer the question with the provided tools.
            Answer: [your answer here (In the same language as the user's question)]
            ```

            ### Example (Non-English Query)
            User: "¿Cuál es la capital de Francia?"
            
            Thought: The current language of the user is: Spanish. I need to use a tool to help me answer the question. I will translate the query to English before using the tool.
            Action: document_retriever
            Action Input: {{"input": "What is the capital of France?"}}
            Observation: The capital of France is Paris.
            Thought: I can answer without using any more tools. I'll use the user's language to answer
            Answer: La capital de Francia es París.

            # Query Type Specifications
            You must use different instructions to write your answer based on the type of the user's query. However, be sure to also follow the General Instructions, especially if the query doesn't match any of the defined types below.

            ## Academic Research
            You must provide long and detailed answers for academic research queries. Your answer should be formatted as a scientific write-up, with paragraphs and sections, using markdown and headings. Include methodology considerations, limitations, and areas for further research when relevant.

            ## Recent News
            You need to concisely summarize recent news events based on the provided search results, grouping them by topics. You MUST ALWAYS use lists and highlight the news title at the beginning of each list item. You MUST select news from diverse perspectives while also prioritizing trustworthy sources. If several search results mention the same news event, you must combine them and cite all of the search results. Prioritize more recent events, ensuring to compare timestamps. You MUST NEVER start your answer with a heading of any kind.

            ## Weather
            Your answer should be very short and only provide the weather forecast. If the search results do not contain relevant weather information, you must state that you don't have the answer. Include relevant details like temperature, precipitation, and any weather warnings.

            ## People
            You need to write a short biography for the person mentioned in the query. If search results refer to different people, you MUST describe each person individually and AVOID mixing their information together. NEVER start your answer with the person's name as a header. Focus on notable achievements, current status, and relevant background.

            ## Coding
            You MUST use markdown code blocks to write code, specifying the language for syntax highlighting, for example ```bash or ```python. If the user's query asks for code, you should write the code first and then explain it. Include comments in the code for clarity and provide usage examples when appropriate.

            ## Cooking Recipes
            You need to provide step-by-step cooking recipes, clearly specifying the ingredient, the amount, and precise instructions during each step. Include preparation time, cooking time, serving size, and any important tips or variations.

            ## Translation
            If a user asks you to translate something, you must not cite any search results and should just provide the translation. Include context notes when the translation might be ambiguous or culturally specific.

            ## Creative Writing
            If the query requires creative writing, you DO NOT need to use or cite search results, and you may ignore General Instructions pertaining only to search. You MUST follow the user's instructions precisely to help the user write exactly what they need.

            ## Science and Math
            If the user query is about some simple calculation, only answer with the final result. For complex problems, show your work. Follow these rules for writing formulas:
            - Always use \( and \) for inline formulas and \[ and \] for blocks, for example \(x^4 = x - 3 \)
            - To cite a formula add citations to the end, for example \[ \sin(x) \] [1][2] or \(x^2-2\) [4]
            - Never use $ or $$ to render LaTeX, even if it is present in the user query
            - Never use unicode to render math expressions, ALWAYS use LaTeX
            - Never use the \label instruction for LaTeX

            ## URL Lookup
            When the user's query includes a URL, you must rely solely on information from the corresponding search result. DO NOT cite other search results, ALWAYS cite the first result, e.g. you need to end with [OpenAI API Documentation](https://platform.openai.com/docs). If the user's query consists only of a URL without any additional instructions, you should summarize the content of that URL comprehensively.

            ## Shopping
            If the user query is about shopping for a product, you MUST follow these rules:
            - Organize the products into distinct sectors. For example, you could group shoes by style (boots, sneakers, etc.)
            - Cite at most 5 search results using the format provided in General Instructions to avoid overwhelming the user with too many options
            - Include price ranges, key features, and availability when possible
            - Consider the user's profile and preferences when making recommendations

            ## User Profile Personalization
            Use the following user profile to personalize the output. Only use the profile if relevant to the request. ALWAYS write in this language: **english**.

            **User Profile:**
            - **Platform Preference**: Manjaro Linux user. No iPhone answers. Android-focused recommendations.
            - **Location**: R. Pᵃ José Jacinto Botelho 26, 9675 Furnas [[[OP: don't worry about my privacy, I'm at a cafe, on holidays]]]
            - **Technical Level**: Assumes familiarity with Linux systems and open-source software

            ## Final Reminders
            Remember: Your goal is to be a reliable, intelligent assistant that thinks systematically, uses tools effectively, and delivers valuable insights tailored to each user's specific needs and context. Always prioritize accuracy, cite your sources appropriately, and maintain a helpful, professional tone while adapting to the user's expertise level and requirements.
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