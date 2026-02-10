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
from llama_index.core.callbacks import CallbackManager, trace_method, CBEventType
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import BASE_TRACE_EVENT
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
from private_gpt.server.tools.document_tool import DocumentSpecificTool
from private_gpt.server.tools.summary_tool import DocumentSummaryTool

from private_gpt.components.vector_store.vector_store_component import VectorStoreComponent
from llama_index.core.agent.react.formatter import ReActChatFormatter
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.server.chat.query_tool import EnhancedRetrievalTool

# Import enhanced prompts and configuration for consistency
from private_gpt.server.chat.prompts import (
    ENHANCED_QA_TEMPLATE,
    AGENTIC_SYSTEM_PROMPT,
)
from private_gpt.server.chat.rag_config import RAG_CONFIG

logger = logging.getLogger(__name__)

class AgentStreamHandler(BaseCallbackHandler):
    """Callback handler to capture agent steps and push them to a queue for streaming."""
    
    def __init__(self, 
                 event_starts_to_ignore: List[CBEventType] = [], 
                 event_ends_to_ignore: List[CBEventType] = []):
        super().__init__(event_starts_to_ignore, event_ends_to_ignore)
        self.queue = asyncio.Queue()
        self._current_thought = ""
        self._current_tool = None
        self._tool_input = None
        
    def _put_event(self, event: Dict[str, Any]) -> None:
        """Thread-safe event queuing"""
        try:
            # Get the running loop from the current context
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(self.queue.put_nowait, event)
        except RuntimeError:
            # No running loop - try to find one or log warning
            try:
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(self.queue.put_nowait, event)
            except Exception as e:
                logger.warning(f"Failed to queue event: {e}")

    def on_event_start(self, 
                       event_type: CBEventType, 
                       payload: Optional[Dict[str, Any]] = None, 
                       event_id: str = "", 
                       **kwargs: Any) -> str:
        """Capture event starts for context"""
        
        if event_type == CBEventType.LLM and payload:
            # Capture LLM calls - this is where agent reasoning happens
            messages = payload.get("messages", [])
            if messages:
                # The last message is usually the current reasoning
                last_msg = messages[-1] if isinstance(messages, list) else None
                if last_msg:
                    content = getattr(last_msg, 'content', str(last_msg))
                    # Check if this looks like agent reasoning
                    if "Thought:" in content or "Action:" in content:
                        # Optional: signal a new reasoning step
                        pass
        
        elif event_type == CBEventType.RETRIEVE:
            self._put_event({
                "thought": "🔍 Searching knowledge base..."
            })
            
        elif event_type == CBEventType.FUNCTION_CALL or (hasattr(CBEventType, 'TOOL') and event_type == CBEventType.TOOL):
            # Extract tool information
            tool_name = None
            if payload:
                tool_name = payload.get("tool_name") or payload.get("name")
                if not tool_name and "tool" in payload:
                    tool_obj = payload["tool"]
                    tool_name = getattr(tool_obj, "name", None)
            
            self._current_tool = tool_name
            if tool_name:
                self._put_event({
                    "action": tool_name,
                    "action_input": str(payload.get("tool_input")) if payload.get("tool_input") else None
                })
        
        return event_id

    def on_event_end(self, 
                     event_type: CBEventType, 
                     payload: Optional[Dict[str, Any]] = None, 
                     event_id: str = "", 
                     **kwargs: Any) -> None:
        """Capture completed events"""
        
        if event_type == CBEventType.LLM and payload:
            # Capture LLM response - contains agent's reasoning
            response = payload.get("response")
            if response:
                content = getattr(response, 'message', None)
                if content:
                    text = getattr(content, 'content', str(content))
                    
                    # Parse ReAct format: Thought: ... Action: ... Action Input: ...
                    if "Thought:" in text:
                        thought = self._extract_section(text, "Thought:", ["Action:", "Action Input:", "Observation:"])
                        if thought:
                            expected_thought = thought.strip()
                            if expected_thought != self._current_thought:
                                self._current_thought = expected_thought
                                self._put_event({
                                    "thought": expected_thought
                                })
                    
                    if "Action:" in text:
                        action = self._extract_section(text, "Action:", ["Action Input:", "Observation:", "Thought:"])
                        if action:
                            self._put_event({
                                "action": action.strip()
                            })
                    
                    if "Action Input:" in text:
                        action_input = self._extract_section(text, "Action Input:", ["Observation:", "Thought:", "Answer:"])
                        if action_input:
                            self._put_event({
                                "action_input": action_input.strip()
                            })
        
        elif event_type == CBEventType.FUNCTION_CALL or (hasattr(CBEventType, 'TOOL') and event_type == CBEventType.TOOL):
            if payload:
                response = payload.get("response") or payload.get("output")
                tool_name = self._current_tool or payload.get("tool_name") or "tool"
                
                # Truncate long responses for streaming thought
                response_preview = str(response)[:500] if response else "completed"
                
                self._put_event({
                    "observation": response_preview
                })

                # --- Source Extraction for Web/Tool Results ---
                # check if this is a search tool or provides URL-like content
                if any(x in tool_name.lower() for x in ["search", "google", "web", "serp"]):
                    try:
                        import re
                        from private_gpt.server.chunks.chunks_service import Chunk
                        from private_gpt.server.ingest.model import IngestedDoc
                        
                        # Find potential URL patterns and titles
                        # Format in serper_tool: "N. Title\nURL\nSnippet"
                        url_pattern = r"(https?://\S+)"
                        text_response = str(response)
                        urls = re.findall(url_pattern, text_response)
                        
                        if urls:
                            found_sources = []
                            # Try to extract titles (text before the URL)
                            sections = re.split(url_pattern, text_response)
                            for i in range(0, len(sections) - 1, 2):
                                title_text = sections[i].strip().split('\n')[-1].strip()
                                # Clean up number prefix like "1. "
                                title_text = re.sub(r"^\d+\.\s*", "", title_text)
                                url = sections[i+1].strip()
                                
                                # Limit to first 10 sources to prevent SSE overflow
                                if len(found_sources) >= 10:
                                    break
                                    
                                chunk = Chunk(
                                    object="context.chunk",
                                    score=1.0, # Tool results are highly relevant
                                    document=IngestedDoc(
                                        object="ingest.document",
                                        doc_id=f"web-{hash(url)}",
                                        doc_metadata={
                                            "file_name": title_text or url,
                                            "url": url,
                                            "source_type": "web_search",
                                            "domain": url.split('/')[2] if '/' in url else url
                                        }
                                    ),
                                    text=sections[i+2].strip()[:1000] if i+2 < len(sections) else title_text # snippet
                                )
                                found_sources.append(chunk)

                            if found_sources:
                                self._put_event({
                                    "sources": found_sources
                                })
                    except Exception as e:
                        logger.warning(f"Failed to extract web sources from tool output: {e}")
                
                self._current_tool = None
        
        # Handle errors
        if payload and "error" in payload:
            self._put_event({
                "thought": f"❌ Error: {str(payload['error'])}"
            })

    def _extract_section(self, text: str, start_marker: str, end_markers: List[str]) -> Optional[str]:
        """Extract a section between markers"""
        if start_marker not in text:
            return None
        
        try:
            start_idx = text.index(start_marker) + len(start_marker)
            
            # Find the earliest end marker
            end_idx = len(text)
            for end_marker in end_markers:
                if end_marker in text[start_idx:]:
                    marker_idx = text.index(end_marker, start_idx)
                    end_idx = min(end_idx, marker_idx)
            
            return text[start_idx:end_idx].strip()
        except ValueError:
            return None

    def signal_complete(self):
        """Signal that agent execution is complete"""
        self._put_event({"event": "complete"})
        
    def start_trace(self, trace_id: Optional[str] = None) -> None:
        pass

    def end_trace(self, 
                  trace_id: Optional[str] = None, 
                  trace_map: Optional[Dict[str, List[str]]] = None, 
                  **kwargs: Any) -> None:
        """Signal end of trace"""
        self.signal_complete()

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
        jitter: tuple[float, float] = (0.1, 0.3),
        qa_template_str: Optional[str] = None,
        external_tools: Optional[List[BaseTool]] = None,
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
        self._external_tools = external_tools or []
        
        # Rate limit handling
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._jitter = jitter
        self._qa_template_str = qa_template_str
        
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
        
        # Generate tool descriptions for the system prompt
        tool_descriptions = self._format_tool_descriptions(tools)
        logger.info(f"AgenticRAGEngine: Created {len(tools)} tools, descriptions length: {len(tool_descriptions)}")
        logger.debug(f"Tool descriptions preview: {tool_descriptions[:200]}...")
        
        # Resolve system prompt with tool descriptions
        from private_gpt.server.chat.prompts import resolve_system_prompt
        if system_prompt:
            logger.info(f"Using provided system prompt (length: {len(system_prompt)})")
            system_prompt_str = resolve_system_prompt(system_prompt, tool_desc=tool_descriptions)
        else:
            logger.info("Using default AGENTIC_SYSTEM_PROMPT")
            system_prompt_str = resolve_system_prompt(self._get_default_system_prompt(), tool_desc=tool_descriptions)
        
        logger.info(f"Resolved system prompt length: {len(system_prompt_str)}")
        # Check if tool_desc was actually replaced
        if "## Available Tools\n\n\n---" in system_prompt_str:
            logger.error("WARNING: Tool descriptions are EMPTY in resolved prompt!")
        elif tool_descriptions in system_prompt_str:
            logger.info("✓ Tool descriptions successfully injected into prompt")
        
        chat_formatter = ReActChatFormatter.from_defaults(system_header=system_prompt_str)
        return ReActAgent.from_tools(
            tools=tools,
            llm=self._llm,
            memory=self._memory,
            react_chat_formatter=chat_formatter,
            verbose=self._verbose,
            callback_manager=self.callback_manager,
            max_iterations=self._max_iterations,
            tool_retrieval_mode="structured",
            streaming=True
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
        
        # Add time tool
        time_tool = TimeTool()
        tools.append(time_tool)
        
        # Add external tools injected from Registry or Planner
        if self._external_tools:
            tools.extend(self._external_tools)
        
        # Validate and return tools
        validated_tools = self._validate_tools(tools)
        
        if self._verbose:
            logger.info(f"Built {len(validated_tools)} total tools")
            for tool in validated_tools:
                logger.debug(f"  - {tool.metadata.name}: {tool.metadata.description[:50]}...")
        
        return validated_tools

    def _create_primary_rag_tool(self) -> BaseTool:
        """Create the primary document retrieval tool with query enhancement."""        
        query_engine = self._create_query_engine()
        
        return EnhancedRetrievalTool(
            query_engine=query_engine,
            llm=self._llm,
            enable_query_expansion=True,
            verbose=self._verbose,
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
            streaming=True,
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

    def _format_tool_descriptions(self, tools: List[BaseTool]) -> str:
        """Format tool descriptions for inclusion in the system prompt."""
        if not tools:
            return "No tools available."
        
        descriptions = []
        for i, tool in enumerate(tools, 1):
            name = tool.metadata.name
            desc = tool.metadata.description or "No description available"
            descriptions.append(f"{i}. **{name}**: {desc}")
        
        return "\n".join(descriptions)

    def _get_qa_template(self) -> PromptTemplate:
        """Document-grounded QA template with structured synthesis and grouped markdown citations."""
        template_str = (
            "You are a specialized document-grounded assistant. Your task is to produce a clear, structured, and "
            "accurately sourced answer based **exclusively** on the provided context. The context may include research "
            "papers, technical documentation, reports, or internal records.\n\n"

            "INSTRUCTIONS:\n"
            "1. **Strict Grounding**: Use only the information explicitly present in the provided context. "
            "Do NOT rely on external knowledge, assumptions, or prior training data.\n\n"

            "2. **Structured Presentation**: Use professional Markdown formatting. Organize your response with:\n"
            "   - Clear section headers (`##`, `###`)\n"
            "   - Bullet points where appropriate\n"
            "   - Logical flow from overview to details\n\n"

            "3. **Synthesis Over Copying**: When multiple documents discuss the same topic, synthesize them into a "
            "coherent explanation. Explicitly note agreements or contradictions when they exist.\n\n"

            "4. **Citation Rules (MANDATORY)**:\n"
            "   - **Format**: Use markdown link citations in the form `[page X](filename.pdf)` or "
            "`[page X](filename.pdf),[page Y](filename.pdf)`.\n"
            "   - **Grouped Citations**: If multiple items share the same source, place a single citation at the end "
            "of the paragraph or list.\n"
            "   - **Multiple Sources**: When a claim relies on more than one document, include multiple citations, "
            "for example: `[page 4](doc1.pdf),[page 7](doc2.pdf)`.\n"
            "   - **Placement**: Citations must appear at the end of the paragraph, list, or section they support.\n"
            "   - **Density**: Do NOT cite every sentence. Avoid over-citation.\n"
            "   - **Prohibited**: NEVER use superscript citations (e.g., `^[1]`), internal tool names, or fabricated "
            "page numbers.\n\n"

            "5. **Handling Gaps**: If the context is insufficient to fully answer the query, add a final section titled "
            "`## Limitations and Gaps` that clearly explains what information is missing and why the answer is incomplete.\n\n"

            "6. **Tone and Objectivity**: Maintain a professional, neutral, and factual tone. Do not speculate or add "
            "interpretive commentary beyond what the documents support.\n\n"

            "7. **Key Metadata**: When available in the context, incorporate relevant metadata such as document titles, "
            "authors, versions, and publication dates into the narrative.\n\n"

            "8. **No Fabrication Rule**: If a detail (page number, date, author, metric) is not explicitly present in the "
            "context, do not invent it.\n\n"

            "9. **Content Strategy**:\n"
            "   - For follow-up questions: BUILD UPON previous answers, don't repeat facts already stated\n"
            "   - Avoid redundancy: If you've already mentioned a fact, don't repeat it unless directly asked\n"
            "   - Focus on NEW information or different aspects when answering related questions\n\n"

            "QUERY:\n"
            "{query_str}\n\n"

            "CONTEXT:\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n\n"

            "BEGIN YOUR RESPONSE:"
        )

        if self._qa_template_str:
            return PromptTemplate(self._qa_template_str)

        return PromptTemplate(template_str)

    def _get_default_system_prompt(self) -> str:
        """Generate a comprehensive system prompt that follows ReAct format with RAG capabilities."""
        return AGENTIC_SYSTEM_PROMPT
    
    def _detect_detail_level(self, query: str) -> str:
        """Detect if user wants brief, normal, or detailed response."""
        query_lower = query.lower()
        
        brief_indicators = ['brief', 'summary', 'short', 'quick', 'tldr', 'in short', 'concise']
        detailed_indicators = [
            'detailed', 'deep', 'full', 'comprehensive', 'elaborate', 
            'in depth', 'explain more', 'more detail', 'thorough', 'complete'
        ]
        
        if any(ind in query_lower for ind in brief_indicators):
            return "brief"
        elif any(ind in query_lower for ind in detailed_indicators):
            return "detailed"
        return "normal"
    
    def _get_detail_instruction(self, detail_level: str) -> str:
        """Get instruction based on detail level to inject into QA template."""
        if detail_level == "brief":
            return (
                "\n\n**RESPONSE LENGTH**: Provide a BRIEF response (2-3 sentences maximum). "
                "High-level overview only. Do NOT include unnecessary details."
            )
        elif detail_level == "detailed":
            return (
                "\n\n**RESPONSE LENGTH**: Provide a COMPREHENSIVE, DETAILED response. "
                "Include technical specifics, examples, architecture details, and thorough explanations. "
                "Use hierarchical formatting with headings and bullet points."
            )
        return ""  # Normal - no special instruction
    
    def _get_qa_template_with_detail_level(self, detail_level: str) -> PromptTemplate:
        """Get QA template with detail-level instructions injected."""
        base_template = self._get_qa_template()
        detail_instruction = self._get_detail_instruction(detail_level)
        
        if detail_instruction:
            # Inject detail instruction before "BEGIN YOUR RESPONSE:"
            template_str = base_template.template
            template_str = template_str.replace(
                "BEGIN YOUR RESPONSE:",
                f"{detail_instruction}\n\nBEGIN YOUR RESPONSE:"
            )
            return PromptTemplate(template_str)
        
        return base_template

    def _sync_memory(self, chat_history: Optional[List[ChatMessage]]) -> None:
        """Synchronize memory with provided chat history."""
        if chat_history:
            self._memory.reset()
            for message in chat_history:
                self._memory.put(message)

    def _format_response(self, response: Any) -> AgentChatResponse:
        """Format agent response into standardized chat response, appending a Sources section if not present."""
        answer = getattr(response, 'response', str(response))
        
        # Remove trailing code blocks if present and not part of actual content
        answer = self._remove_trailing_empty_code_blocks(answer)
        
        sources = getattr(response, 'sources', [])
        source_nodes = getattr(response, 'source_nodes', [])
        return AgentChatResponse(
            response=answer,
            sources=sources,
            source_nodes=source_nodes
        )

    def _remove_trailing_empty_code_blocks(self, text: str) -> str:
        """Remove trailing empty code blocks that are not part of actual code content."""
        lines = text.split('\n')
        
        # Work backwards from the end to find trailing empty code blocks
        i = len(lines) - 1
        while i >= 0:
            line = lines[i].strip()
            # If we find a non-empty line that's not a code block marker, stop
            if line and line != '```':
                break
            # If we find a code block marker, check if it's at the very end
            if line == '```':
                # Check if this is the last line or if the following lines are just whitespace
                if i == len(lines) - 1 or all(not lines[j].strip() for j in range(i + 1, len(lines))):
                    # Remove this code block marker and any trailing empty lines
                    lines = lines[:i]
                    # Continue checking for more trailing code blocks
                else:
                    break
            i -= 1
            
        return '\n'.join(lines).rstrip()

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
        """
        self._sync_memory(chat_history)
        
        # Strategy: Use a custom callback handler to capture events
        handler = AgentStreamHandler()
        self.callback_manager.add_handler(handler)
        
        # Store source nodes to populate after streaming
        collected_source_nodes = []
        collected_sources = []
        
        async def combined_gen():
            nonlocal collected_source_nodes, collected_sources
            try:
                # Start agent chat in a task
                agent_task = asyncio.create_task(self._agent.astream_chat(message))
                
                # Consume events until reasoning finishes and returns the response object
                # Reasoning is done when agent_task returns the StreamingAgentChatResponse
                while not agent_task.done():
                    while not handler.queue.empty():
                        event = await handler.queue.get()
                        if isinstance(event, dict) and event.get("event") == "complete":
                            continue
                        yield event
                    await asyncio.sleep(0.05)
                
                # Check for remaining events after reasoning task completion
                while not handler.queue.empty():
                    event = await handler.queue.get()
                    if isinstance(event, dict) and event.get("event") == "complete":
                        continue
                    yield event
                
                # Once reasoning loop is done, get the streaming response object
                response = await agent_task
                
                # Extract sources before streaming text
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    collected_source_nodes.extend(response.source_nodes)
                if hasattr(response, 'sources') and response.sources:
                    collected_sources.extend(response.sources)
                
                # Stream the actual text response
                # StreamingAgentChatResponse has multiple possible generator attributes
                # Helper to find the actual async generator among possible attributes
                def get_async_gen(obj):
                    # Try each possible generator attribute
                    # Check if it's callable (method) - if so, call it to get the generator
                    for attr_name in ['async_response_gen', 'chat_stream', 'response_gen']:
                        val = getattr(obj, attr_name, None)
                        if val is None:
                            continue
                        
                        # If callable, it's a method - call it to get the generator
                        if callable(val):
                            try:
                                logger.info(f"AgenticRAG: Calling {attr_name}() to get generator")
                                actual_gen = val()
                                if actual_gen is not None:
                                    return actual_gen, attr_name
                            except Exception as e:
                                logger.warning(f"AgenticRAG: Failed to call {attr_name}(): {e}")
                        else:
                            # It's already a generator - check if it's async
                            import inspect
                            if inspect.isasyncgen(val) or hasattr(val, '__aiter__'):
                                return val, attr_name
                            else:
                                logger.warning(f"AgenticRAG: {attr_name} is a sync generator, need async")
                    return None, None

                active_generator, generator_name = get_async_gen(response)
                
                if active_generator is not None:
                    logger.info(f"AgenticRAG: Using {generator_name} for streaming")
                    logger.info(f"AgenticRAG: Generator type: {type(active_generator)}")
                    
                    # IMPORTANT: For ReAct agents, the callback handler already emitted
                    # structured events (thought, action, observation). The response_gen
                    # contains the RAW ReAct-formatted text which we should NOT stream
                    # because it's redundant and confusing.
                    #
                    # We only need to extract the FINAL ANSWER from the agent's response.
                    # The final answer comes AFTER all the Thought/Action/Observation cycles.
                    
                    full_response = ""
                    async for chunk in active_generator:
                        # Collect the full response
                        if isinstance(chunk, str):
                            full_response += chunk
                    
                    logger.info(f"AgenticRAG: Full response length: {len(full_response)} chars")
                    logger.info(f"AgenticRAG: Response preview: {full_response[:500]}")
                    
                    # Now parse out ONLY the final answer
                    # The agent's response format can be:
                    # 1. Just the answer text (simple queries)
                    # 2. Thought/Action/Observation cycles ending with Answer: <text>
                    # 3. Answer may contain code blocks (```...```)
                    
                    # Strategy: Find "Answer:" that's NOT inside a code block
                    # Then extract everything after it (including any code blocks in the answer)
                    
                    import re
                    
                    # Find all code block positions to avoid matching "Answer:" inside them
                    code_block_ranges = []
                    for match in re.finditer(r'```[\s\S]*?```', full_response):
                        code_block_ranges.append((match.start(), match.end()))
                    
                    # Function to check if a position is inside a code block
                    def is_in_code_block(pos):
                        return any(start <= pos < end for start, end in code_block_ranges)
                    
                    # Find "Answer:" that's not in a code block
                    answer_start = None
                    for match in re.finditer(r'(?:^|\n)\s*Answer:\s*', full_response, re.IGNORECASE):
                        if not is_in_code_block(match.start()):
                            answer_start = match.end()
                            break
                    
                    if answer_start is not None:
                        # Extract from "Answer:" to the end (or until next ReAct marker outside code blocks)
                        # Look for the next Thought:/Action:/Observation: that's not in a code block
                        end_pos = len(full_response)
                        for match in re.finditer(r'\n\s*(?:Thought|Action|Observation):', full_response[answer_start:], re.IGNORECASE):
                            abs_pos = answer_start + match.start()
                            if not is_in_code_block(abs_pos):
                                end_pos = abs_pos
                                break
                        
                        final_answer = full_response[answer_start:end_pos].strip()
                        logger.info(f"AgenticRAG: Extracted answer using pattern match ({len(final_answer)} chars)")
                    else:
                        # Fallback: Check if response is pure ReAct formatting (has Thought/Action but no Answer)
                        if ("Thought:" in full_response or "Action:" in full_response) and "Answer:" not in full_response:
                            logger.warning("AgenticRAG: Response contains only ReAct formatting, no final answer")
                            final_answer = "I apologize, but I wasn't able to formulate a complete answer."
                        else:
                            # The entire response might be the answer (no ReAct formatting)
                            logger.info("AgenticRAG: Using full response as answer (no ReAct markers found)")
                            final_answer = full_response.strip()
                    
                    # Stream the final answer word by word for better UX
                    logger.info(f"AgenticRAG: Streaming answer: {final_answer[:200]}...")
                    words = final_answer.split()
                    for word in words:
                        yield word + " "
                        await asyncio.sleep(0.005)  # Faster streaming
                else:
                    logger.error(f"AgenticRAG: Could not find any valid async generator in {type(response)}")
                    logger.error(f"Available attributes: {[attr for attr in dir(response) if not attr.startswith('_')]}")
                    yield {"thought": "❌ Error: No valid streaming generator found"}
                
                # Final drain of the queue
                while not handler.queue.empty():
                    event = await handler.queue.get()
                    if isinstance(event, dict) and event.get("event") == "complete":
                        continue
                    yield event
                    
            except Exception as e:
                logger.error(f"Error in combined agent stream: {e}", exc_info=True)
                yield {"thought": f"❌ Error: {str(e)}"}
            finally:
                self.callback_manager.remove_handler(handler)

        # Create response object that will be populated with sources during streaming
        streaming_response = StreamingAgentChatResponse(
            chat_stream=combined_gen(),
            sources=[],
            source_nodes=[]
        )
        
        # Attach references to collected data so they can be accessed after streaming
        streaming_response._collected_source_nodes = collected_source_nodes
        streaming_response._collected_sources = collected_sources
        
        return streaming_response