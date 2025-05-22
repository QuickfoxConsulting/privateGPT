from datetime import datetime 
from threading import Thread
from typing import Any, List, Optional, Dict, cast, Union
import logging
from llama_index.core.indices.vector_store import VectorStoreIndex

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.callbacks import CallbackManager, trace_method
from llama_index.core.chat_engine.types import AgentChatResponse, StreamingAgentChatResponse, BaseChatEngine
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import BaseTool, QueryEngineTool
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import LLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
import asyncio
from llama_index.core.prompts.base import PromptTemplate
from .document_tools import DocumentSpecificTool
from private_gpt.components.vector_store.vector_store_component import VectorStoreComponent
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
logger = logging.getLogger(__name__)

class AgenticRAGEngine(BaseChatEngine):
    """Advanced RAG-enhanced ReAct Agent with improved context handling and source tracking."""
    
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
        similarity_top_k: int = 4
    ):
        self.index = index
        self.callback_manager = callback_manager or CallbackManager([])
        self.llm = llm
        self.vector_store_component = vector_store_component
        self.node_postprocessors = node_postprocessors or []
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.document_files = document_files or []
        self.tool_name_prefix = tool_name_prefix
        self.citation_format = citation_format
        self.similarity_top_k = similarity_top_k

        self.memory = memory or ChatMemoryBuffer.from_defaults(
            token_limit=int(llm.metadata.context_window * 0.5) 
        )

        self.agent = ReActAgent.from_tools(
            tools=self._build_tools(),
            llm=llm,
            memory=self.memory,
            system_prompt=system_prompt or self._default_system_prompt(),
            verbose=verbose,
            callback_manager=self.callback_manager,
            max_iterations=max_iterations,
            tool_retrieval_mode="structured"
        )

    def _build_tools(self) -> List[BaseTool]:
        """Construct toolset with enhanced RAG tool configuration.
        
        Args:
            custom_tools: List of custom tools to include
            
        Returns:
            List of configured tools including RAG and document-specific tools
        """
        tools = []
        
        # Add RAG tool for general document queries
        rag_tool = QueryEngineTool.from_defaults(
            query_engine=self._create_query_engine(),
            name="document_retriever",
            description=(
                "Access organizational knowledge base. Use for: "
                "- Factual questions about company documents\n"
                "- Policy and procedure inquiries\n"
                "- Historical data lookup\n"
                "- Technical specifications\n"
                "Always verify information through this tool first."
            )
        )
        tools.append(rag_tool)        
        doc_tools = self._create_document_specific_tools()
        tools.extend(doc_tools)
        
        if self.verbose:
            logger.info(f"Built {len(tools)} tools: {len(doc_tools)} document-specific, custom")
        
        return tools

    def _create_query_engine(self) -> RetrieverQueryEngine:
        """Create a query engine for general document retrieval.
        
        Returns:
            Configured RetrieverQueryEngine instance
        """
        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
            llm=self.llm,
            structured_answer_filtering=True,
            text_qa_template=self._get_qa_template(),
            streaming=False,
            callback_manager=self.callback_manager
        )

        return RetrieverQueryEngine(
            retriever=self.vector_store_component.get_retriever(
                index=self.index,
                similarity_top_k=self.similarity_top_k
            ),
            response_synthesizer=response_synthesizer,
            node_postprocessors=self.node_postprocessors,
            callback_manager=self.callback_manager
        )

    def _create_document_specific_tools(self) -> List[BaseTool]:
        """Create tools for each document in the knowledge base.
        
        Returns:
            List of document-specific tools
        """
        doc_tools = []
        
        for file_name in self.document_files:
            try:
                doc_tool = DocumentSpecificTool(
                    index=self.index,
                    file_name=file_name,
                    llm=self.llm,
                    vector_store_component=self.vector_store_component,
                    node_postprocessors=self.node_postprocessors,
                    callback_manager=self.callback_manager,
                    tool_name_prefix=self.tool_name_prefix,
                    citation_format=self.citation_format,
                    similarity_top_k=self.similarity_top_k,
                    verbose=self.verbose
                )
                doc_tools.append(doc_tool)
                
                if self.verbose:
                    logger.info(f"Created tool for document: {file_name}")
            except Exception as e:
                logger.error(f"Error creating tool for document {file_name}: {str(e)}")
                continue
            
        return doc_tools

    def _validate_tools(self, tools: List[BaseTool]) -> List[BaseTool]:
        """Ensure tool uniqueness and proper configuration.
        
        Args:
            tools: List of tools to validate
            
        Returns:
            List of validated tools
            
        Raises:
            ValueError: If duplicate tool names are found
        """
        seen_names = set()
        validated = []
        for tool in tools:
            if tool.metadata.name in seen_names:
                raise ValueError(f"Duplicate tool name: {tool.metadata.name}")
            seen_names.add(tool.metadata.name)
            validated.append(tool)
        return validated

    def _get_qa_template(self) -> PromptTemplate:
        """Get enhanced QA template with strict citation requirements.
        
        Returns:
            Configured PromptTemplate instance
        """
        return PromptTemplate(
            "You are a helpful assistant. Use ONLY the context below to answer the query as comprehensively as possible.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Instructions:\n"
            "1. Write a detailed, step-by-step answer to the query.\n"
            "2. Synthesize information from multiple context sections if needed.\n"
            "3. If the answer is complex, break it into clear steps or sections.\n"
            "4. Always cite sources using [Document: file_name, Page page].\n"
            "5. If information is missing, state what is missing.\n"
            "Query: {query_str}\n\n"
            "Comprehensive Answer:"
        )

    def _default_system_prompt(self) -> str:
        """Get enhanced system prompt for better reasoning and compliance.
        
        Returns:
            Formatted system prompt string
        """
        doc_tools_description = ""
        if self.document_files:
            doc_tools_description = "\nDocument-specific tools available:\n"
            for file_name in self.document_files:
                doc_tools_description += f"- {file_name}\n"
        
        return (
            "You are a senior research assistant with access to organizational knowledge. "
            "Follow these steps rigorously:\n\n"
            "1. Analyze the query and historical context\n"
            "2. Determine if document access is needed (95% of cases)\n"
            "3. Use tools in this priority:\n"
            "   a) document_retriever for general queries across all documents\n"
            "   b) document-specific tools when the query mentions a specific document\n"
            "   c) other tools for supplementary actions\n"
            "4. Synthesize information from multiple sources when needed\n"
            "5. Always cite sources using exact metadata from documents\n"
            "6. Acknowledge document limitations when applicable\n"
            "7. Maintain professional, technical tone\n"
            f"{doc_tools_description}"
        )

    @trace_method("chat")
    def chat(self, message: str, chat_history: Optional[List[ChatMessage]] = None) -> AgentChatResponse:
        """Enhanced chat with automatic memory synchronization and memory update.
        
        Args:
            message: User message to process
            chat_history: Optional chat history
            
        Returns:
            AgentChatResponse with response and sources
        """
        self._sync_memory(chat_history)
        try:
            response = self.agent.chat(message)
            # Update memory with assistant response
            if hasattr(response, 'message') and response.message:
                self.memory.put(response.message)
            elif hasattr(response, 'response'):
                self.memory.put(ChatMessage(content=response.response, role=MessageRole.ASSISTANT))
            return self._format_response(response)
        except Exception as e:
            logger.error(f"Chat error: {str(e)}", exc_info=self.verbose)
            return AgentChatResponse(
                response="An error occurred while processing your request. Please try again.",
                sources=[],
                source_nodes=[]
            )

    @trace_method("chat")
    def stream_chat(self, message: str, chat_history: Optional[List[ChatMessage]] = None) -> StreamingAgentChatResponse:
        """Stream a response using the ReACT agent and update memory after streaming.
        
        Args:
            message: User message to process
            chat_history: Optional chat history
            
        Returns:
            StreamingAgentChatResponse with streaming response and sources
        """
        self._sync_memory(chat_history)
        try:
            response = self.agent.stream_chat(message)
            streaming_response = StreamingAgentChatResponse(
                chat_stream=response.response_gen,
                sources=getattr(response, 'sources', []),
                source_nodes=getattr(response, 'source_nodes', []),
            )
            thread = Thread(
                target=streaming_response.write_response_to_history, args=(self.memory,)
            )
            thread.start()
            return streaming_response
        except Exception as e:
            logger.error(f"Stream chat error: {str(e)}", exc_info=self.verbose)
            return StreamingAgentChatResponse(chat_stream=iter([]), sources=[], source_nodes=[])

    @trace_method("chat")
    async def achat(self, message: str, chat_history: Optional[List[ChatMessage]] = None) -> AgentChatResponse:
        """Async chat with error handling and memory update.
        
        Args:
            message: User message to process
            chat_history: Optional chat history
            
        Returns:
            AgentChatResponse with response and sources
        """
        self._sync_memory(chat_history)
        try:
            response = await self.agent.achat(message)
            # Update memory with assistant response
            if hasattr(response, 'message') and response.message:
                self.memory.put(response.message)
            elif hasattr(response, 'response'):
                self.memory.put(ChatMessage(content=response.response, role=MessageRole.ASSISTANT))
            return self._format_response(response)
        except Exception as e:
            logger.error(f"Async chat error: {str(e)}", exc_info=self.verbose)
            return AgentChatResponse(
                response="An error occurred during processing. Please try again.",
                sources=[],
                source_nodes=[]
            )

    @trace_method("chat")
    async def astream_chat(self, message: str, chat_history: Optional[List[ChatMessage]] = None) -> StreamingAgentChatResponse:
        """Async streaming chat method using ReACT agent and memory update.
        
        Args:
            message: User message to process
            chat_history: Optional chat history
            
        Returns:
            StreamingAgentChatResponse with streaming response and sources
        """
        self._sync_memory(chat_history)
        try:
            response = await self.agent.astream_chat(message)
            streaming_response = StreamingAgentChatResponse(
                achat_stream=response.response_gen,
                sources=getattr(response, 'sources', []),
                source_nodes=getattr(response, 'source_nodes', []),
            )
            # Start async task to write streamed response to memory after streaming
            asyncio.create_task(streaming_response.awrite_response_to_history(self.memory))
            return streaming_response
        except Exception as e:
            logger.error(f"Async stream chat error: {str(e)}", exc_info=self.verbose)
            class DummyAsyncStream:
                async def __aiter__(self):
                    return self
                async def __anext__(self):
                    raise StopAsyncIteration
            return StreamingAgentChatResponse(achat_stream=DummyAsyncStream(), sources=[], source_nodes=[])

    def reset(self) -> None:
        """Reset chat memory."""
        self.memory.reset()

    @property
    def chat_history(self) -> List[ChatMessage]:
        """Get all chat history from memory.
        
        Returns:
            List of chat messages
        """
        return self.memory.get_all()

    def _sync_memory(self, chat_history: Optional[List[ChatMessage]]) -> None:
        """Synchronize memory with optional external history.
        
        Args:
            chat_history: Optional chat history to sync
        """
        if chat_history:
            self.memory.set(chat_history[-self._max_history_items:])

    @property
    def _max_history_items(self) -> int:
        """Calculate maximum history items based on LLM context window.
        
        Returns:
            Maximum number of history items to keep
        """
        return int(self.llm.metadata.context_window * 0.5 // 100)  # Approx tokens per message

    def _format_response(self, agent_raw_response: Any) -> AgentChatResponse:
        """Improved response formatting with detailed source tracking.
        
        Args:
            agent_raw_response: Raw response from the agent
            
        Returns:
            Formatted AgentChatResponse
        """
        rag_nodes_from_agent = []
        tool_call_sources = []

        if hasattr(agent_raw_response, 'sources') and agent_raw_response.sources:
            for src in agent_raw_response.sources:
                if isinstance(src, NodeWithScore):
                    rag_nodes_from_agent.append(src)
                elif hasattr(src, 'tool_name') and hasattr(src, 'raw_input'):
                    tool_call_sources.append({
                        "tool_name": src.tool_name,
                        "input": src.raw_input,
                        "output_snippet": str(src.content)[:100] + "..." if hasattr(src, 'content') else "N/A",
                        "timestamp": datetime.now().isoformat(),
                    })

        if hasattr(agent_raw_response, 'source_nodes') and agent_raw_response.source_nodes:
            existing_node_ids = {n.node.node_id for n in rag_nodes_from_agent}
            for sn in agent_raw_response.source_nodes:
                if isinstance(sn, NodeWithScore) and sn.node.node_id not in existing_node_ids:
                    rag_nodes_from_agent.append(sn)
        
        processed_rag_nodes = self._process_source_nodes(rag_nodes_from_agent)

        return AgentChatResponse(
            response=str(agent_raw_response.response),
            sources=tool_call_sources,
            source_nodes=processed_rag_nodes
        )

    def _process_source_nodes(self, rag_nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """Process and validate RAG source nodes.
        
        Args:
            rag_nodes: List of nodes to process
            
        Returns:
            List of processed nodes with sanitized metadata
        """
        processed_nodes = []
        for sn in rag_nodes:
            safe_metadata = {}
            if sn.node and sn.node.metadata:
                safe_metadata["file_name"] = sn.node.metadata.get("file_name", "Unknown Document")
                if "page_label" in sn.node.metadata:
                    safe_metadata["page"] = sn.node.metadata.get("page")
                elif "page" in sn.node.metadata:
                    safe_metadata["page"] = sn.node.metadata.get("page")

                if "document_id" in sn.node.metadata:
                    safe_metadata["document_id"] = sn.node.metadata.get("document_id")

            processed_nodes.append(NodeWithScore(node=sn.node, score=sn.score, metadata=safe_metadata))
        return processed_nodes

    def __repr__(self) -> str:
        """Developer-friendly representation.
        
        Returns:
            String representation of the engine
        """
        return (
            f"AgenticRAGEngine("
            f"llm={self.llm.model_name}, "
            f"tools={len(self.agent.tools)}, "
            f"memory={len(self.memory.get())} messages, "
            f"postprocessors={len(self.node_postprocessors)})"
        )