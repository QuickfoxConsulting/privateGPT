import asyncio
import logging
import json
import re
from typing import Any, List, Optional, Tuple, Dict, Set, Union

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.callbacks import CallbackManager, trace_method
from llama_index.core.schema import TextNode
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
    BaseChatEngine,
    StreamingAgentChatResponse,
    ToolOutput,
)
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core.indices.service_context import ServiceContext
from llama_index.core.base.llms.generic_utils import messages_to_history_str
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.schema import MetadataMode, NodeWithScore
from llama_index.core.settings import (
    Settings,
    callback_manager_from_settings_or_context,
    llm_from_settings_or_context,
)
from llama_index.core.types import Thread
from llama_index.core.utilities.token_counting import TokenCounter

from private_gpt.server.chat.citation_utils import CitationHelper
from private_gpt.server.chat.rag_config import RAG_CONFIG
from private_gpt.server.tools.document_tool import DocumentSpecificTool
from private_gpt.server.tools.summary_tool import DocumentSummaryTool
from private_gpt.components.vector_store.vector_store_component import VectorStoreComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.server.chunks.chunks_service import ChunksService, Chunk
from private_gpt.di import global_injector

logger = logging.getLogger(__name__)

# --- Prompt Templates ---

DEFAULT_CONTEXT_PROMPT_TEMPLATE = """
You are a helpful RAG (Retrieval-Augmented Generation) assistant.
Your goal is to answer the user's question using the provided context documents as your primary source of information.

**Original Question:**
{original_question}

**Context Breakdown:**
To gather information, the original question was broken down into these sub-questions:
{sub_questions_str}

**Context Documents:**
The following sections contain relevant information retrieved based on the sub-questions. Each source is numbered:
---------------------
{context_str}
---------------------

**INSTRUCTIONS:**
1.  **Prioritize Context**: Answer the **Original Question** primarily using the information present in the numbered **Context Documents** section above.
2.  **Fill Gaps safely**: If the context doesn't fully answer the question, you may use your general knowledge to provide a complete answer, but you must clearly distinguish between what is in the documents and what is general knowledge.
3.  **Mandatory Citations**: Whenever you use information from the context, you MUST cite your sources using the format `[N]` (e.g., `[1]`, `[2]`) immediately after the sentence or fact derived from that source.
    - **IMPORTANT**: Use ONLY the `[N]` format. Do not create markdown links or append filenames to the citations.
    - Correct: "The revenue grew by 5% [1]."
    - Incorrect: "The revenue grew by 5% [1](Source.pdf)."
    - Incorrect: "The revenue grew by 5%."
4.  **Tone**: Be professional, helpful, and concise.

**Response Structure:**
- Start with a direct answer to the question.
- Use bullet points for lists.
- End with a citation-based summary if multiple sources are used.
"""

# Standard condense prompt
DEFAULT_CONDENSE_PROMPT_TEMPLATE = """
Given the following conversation between a user and an AI assistant and a follow up question from the user,
rephrase the follow up question to be a standalone question, in its original language.
If the follow-up question is already standalone, repeat it.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:"""

# Prompt for query decomposition
DEFAULT_DECOMPOSE_PROMPT_TEMPLATE = """
Decompose the following potentially complex query into {max_sub_queries} or fewer precise and focused sub-queries.
Each sub-query should target a specific aspect of the original query and help retrieve relevant information efficiently.
If the query is already simple, return it as the only sub-query in the list.

**Query to Decompose:**
{query}

**Output Format:**
Return the sub-queries strictly as a Python list of strings. Example:
["What is X?", "How does Y relate to X?"]
"""

class AgenticCondenseChatEngine(BaseChatEngine):
    """
    Agentic Condense Chat Engine.

    Combines chat history condensation, query decomposition, and dynamic tool routing for Advanced RAG.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: LLM,
        memory: BaseMemory,
        context_prompt: Optional[str] = None,
        condense_prompt: Optional[str] = None,
        decompose_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_sub_queries: int = 3,
        skip_condense: bool = False,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        condense_timeout: float = None,
        decompose_timeout: float = None,
        retrieval_timeout: float = None,
    ):
        self._retriever = retriever
        self._llm = llm
        self._memory = memory
        self._context_prompt_template = (
            context_prompt or DEFAULT_CONTEXT_PROMPT_TEMPLATE
        )
        condense_prompt_str = condense_prompt or DEFAULT_CONDENSE_PROMPT_TEMPLATE
        self._condense_prompt_template = PromptTemplate(condense_prompt_str)

        decompose_prompt_str = decompose_prompt or DEFAULT_DECOMPOSE_PROMPT_TEMPLATE
        self._decompose_prompt_template = PromptTemplate(decompose_prompt_str)

        self._system_prompt = system_prompt
        self._max_sub_queries = max_sub_queries
        self._skip_condense = skip_condense
        self._node_postprocessors = node_postprocessors or []
        self.callback_manager = callback_manager or CallbackManager([])
        for node_postprocessor in self._node_postprocessors:
            node_postprocessor.callback_manager = self.callback_manager

        self._token_counter = TokenCounter()
        self._verbose = verbose
        
        # Timeout configuration
        self._condense_timeout = condense_timeout or RAG_CONFIG.query_processing.condense_timeout_seconds
        self._decompose_timeout = decompose_timeout or RAG_CONFIG.query_processing.decompose_timeout_seconds
        self._retrieval_timeout = retrieval_timeout or RAG_CONFIG.query_processing.retrieval_timeout_seconds
        
        # Chunks Service
        self._chunks_service = global_injector.get(ChunksService)

    def _get_default_system_prompt(self) -> str:
        """Generate a balanced default system prompt for RAG."""
        return (
            "You are a dedicated RAG (Retrieval-Augmented Generation) assistant. "
            "Your PRIMARY role is to retrieve and synthesize information from the user's uploaded documents. "
            "ALWAYS prioritize the provided context. "
            "However, if the context is insufficient, you may use your general knowledge to provide a helpful response, "
            "but please indicate when information comes from outside the documents."
        )

    @classmethod
    def from_defaults(
        cls,
        retriever: BaseRetriever,
        llm: Optional[LLM] = None,
        service_context: Optional[ServiceContext] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        system_prompt: Optional[str] = None,
        context_prompt: Optional[str] = None,
        condense_prompt: Optional[str] = None,
        decompose_prompt: Optional[str] = None,
        max_sub_queries: int = 3,
        skip_condense: bool = False,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "AgenticCondenseChatEngine":
        """Initialize an AgenticCondenseChatEngine from default parameters."""
        llm = llm or llm_from_settings_or_context(Settings, service_context)

        chat_history = chat_history or []
        memory = memory or ChatMemoryBuffer.from_defaults(
            chat_history=chat_history, token_limit=llm.metadata.context_window * 0.7 
        )

        return cls(
            retriever=retriever,
            llm=llm,
            memory=memory,
            context_prompt=context_prompt,
            condense_prompt=condense_prompt,
            decompose_prompt=decompose_prompt,
            max_sub_queries=max_sub_queries,
            skip_condense=skip_condense,
            callback_manager=callback_manager_from_settings_or_context(
                Settings, service_context
            ),
            node_postprocessors=node_postprocessors,
            system_prompt=system_prompt,
            verbose=verbose,
        )
    
    # --- Tool Routing ---
    
    def _detect_specific_document_intent(self, query: str) -> Optional[Dict[str, str]]:
        """
        Heuristic to detect if the user wants to query a specific document.
        Returns {'type': 'summary'|'qa', 'file_name': ...} or None.
        """
        # Pattern for summary: "summarize <file>"
        summary_pattern = r"summarize\s+(?:the\s+)?(.+\.[a-zA-Z0-9]+)"
        summary_match = re.search(summary_pattern, query, re.IGNORECASE)
        if summary_match:
            return {"type": "summary", "file_name": summary_match.group(1).strip()}
            
        # Pattern for QA: "in <file>", "from <file>", "what does <file> say"
        qa_pattern = r"(?:in|from|about)\s+(?:the\s+)?(.+\.[a-zA-Z0-9]+)"
        qa_match = re.search(qa_pattern, query, re.IGNORECASE)
        if qa_match:
             return {"type": "qa", "file_name": qa_match.group(1).strip()}
             
        return None

    def _create_and_run_tool(self, intent: Dict[str, str], query: str) -> ToolOutput:
        """Dynamically creates and runs the appropriate document tool."""
        try:
            vector_store_component = global_injector.get(VectorStoreComponent)
            node_store_component = global_injector.get(NodeStoreComponent)
            
            index = getattr(self._retriever, "_index", None) 
            if not index:
                  index = vector_store_component.index
            
            tool = None
            if intent["type"] == "summary":
                tool = DocumentSummaryTool.from_defaults(
                    retriever=self._retriever,
                    file_name=intent["file_name"],
                    llm=self._llm,
                    index=index,
                    node_store_component=node_store_component,
                    vector_store_component=vector_store_component,
                    verbose=self._verbose
                )
            else:
                tool = DocumentSpecificTool.from_defaults(
                     retriever=self._retriever,
                     file_name=intent["file_name"],
                     llm=self._llm,
                     vector_store_component=vector_store_component,
                     index=index,
                     verbose=self._verbose
                )
                
            return tool(query)
            
        except Exception as e:
            logger.error(f"Failed to create/run tool: {e}")
            return ToolOutput(
                content=f"I couldn't access the specific document '{intent['file_name']}'. Falling back to general search.",
                tool_name="error",
                raw_input={"query": query},
                raw_output=str(e),
                is_error=True
            )

    # --- Core Logic ---

    def _condense_question(
        self, chat_history: List[ChatMessage], latest_message: str
    ) -> str:
        """Condense history and latest message to a standalone question."""
        if self._skip_condense or not chat_history:
            return latest_message

        chat_history_str = messages_to_history_str(chat_history)
        if self._verbose:
            logger.info(f"Condensing query with history: {chat_history_str}")

        response = self._llm.predict(
            self._condense_prompt_template,
            question=latest_message,
            chat_history=chat_history_str,
        )
        return response.strip()

    async def _acondense_question(
        self, chat_history: List[ChatMessage], latest_message: str
    ) -> str:
        """Async condense history and latest message."""
        if self._skip_condense or not chat_history:
            return latest_message

        chat_history_str = messages_to_history_str(chat_history)
        if self._verbose:
            logger.info(f"Condensing query with history: {chat_history_str}")

        response = await self._llm.apredict(
            self._condense_prompt_template,
            question=latest_message,
            chat_history=chat_history_str,
        )
        return response.strip()

    def _should_decompose(self, query: str) -> bool:
        """Determine if query benefits from decomposition."""
        word_count = len(query.split())
        if word_count < RAG_CONFIG.query_processing.decomposition_threshold_words:
            return False
        
        complexity_indicators = [
            " and " in query.lower(),
            " or " in query.lower(),
            query.count("?") > 1,
            "compare" in query.lower(),
            "difference between" in query.lower(),
            "both" in query.lower() and word_count > 10,
        ]
        return any(complexity_indicators)

    def _decompose_query(self, query: str) -> List[str]:
        """Decompose a query into sub-queries using LLM."""
        if not self._should_decompose(query):
            return [query]
        
        prompt = self._decompose_prompt_template.format(
            query=query, max_sub_queries=self._max_sub_queries
        )
        response = self._llm.complete(prompt)
        text = response.text.strip()

        # Robust Parsing
        try:
            if text.startswith("[") and text.endswith("]"):
                 try:
                    return json.loads(text)[:self._max_sub_queries]
                 except: pass # Parsing failed, try regex
            
            # Regex fallback
            matches = re.findall(r'"([^"]+)"', text)
            if matches:
                 return matches[:self._max_sub_queries]
            
            # Line fallback
            lines = [l.strip("- ").strip() for l in text.splitlines() if l.strip()]
            return lines[:self._max_sub_queries] if lines else [query]
            
        except Exception as e:
            logger.warning(f"Decomposition parsing failed: {e}. Using original query.")
            return [query]

    async def _adecompose_query(self, query: str) -> List[str]:
        """Async decomposition."""
        if not self._should_decompose(query):
            return [query]
            
        prompt = self._decompose_prompt_template.format(
            query=query, max_sub_queries=self._max_sub_queries
        )
        response = await self._llm.acomplete(prompt)
        text = response.text.strip()
        
        # Robust Parsing
        try:
            if text.startswith("[") and text.endswith("]"):
                 try:
                    return json.loads(text)[:self._max_sub_queries]
                 except: pass
            
            matches = re.findall(r'"([^"]+)"', text)
            if matches:
                 return matches[:self._max_sub_queries]
            
            lines = [l.strip("- ").strip() for l in text.splitlines() if l.strip()]
            return lines[:self._max_sub_queries] if lines else [query]
            
        except Exception:
            return [query]

    def _convert_chunks_to_nodes(self, chunks: List[Chunk]) -> List[NodeWithScore]:
        """Convert Chunks back to NodeWithScore, adding sibling context."""
        nodes = []
        for chunk in chunks:
             # Construct content with siblings
            content = chunk.text
            if chunk.previous_texts:
                content = "\n".join(chunk.previous_texts) + "\n" + content
            if chunk.next_texts:
                content = content + "\n" + "\n".join(chunk.next_texts)
                
            # Create node
            node = NodeWithScore(
                node=TextNode(
                    text=content,
                    id_=chunk.node_id or "missing_node_id",
                    metadata=chunk.document.doc_metadata or {},
                    embedding=None # Not needed for context
                ),
                score=chunk.score
            )
            nodes.append(node)
        return nodes

    def _retrieve_and_process_nodes(self, sub_queries: List[str]) -> Tuple[str, List[NodeWithScore]]:
        """Retrieve using ChunksService, deduplicate, and post-process nodes."""
        all_nodes_map: Dict[str, NodeWithScore] = {}
        
        for sub_q in sub_queries:
            try:
                # Use ChunksService
                chunks = self._chunks_service.retrieve_relevant_sync(sub_q, limit=5, prev_next_chunks=1)
                nodes = self._convert_chunks_to_nodes(chunks)
                
                for node in nodes:
                    if node.node.node_id not in all_nodes_map:
                         all_nodes_map[node.node.node_id] = node
            except Exception as e:
                logger.error(f"Error retrieving for '{sub_q}': {e}")
                
        combined_nodes = list(all_nodes_map.values())
        
        if self._node_postprocessors:
            primary_query_bundle = QueryBundle(sub_queries[0] if sub_queries else "")
            for postprocessor in self._node_postprocessors:
                combined_nodes = postprocessor.postprocess_nodes(
                    combined_nodes, query_bundle=primary_query_bundle
                )
                
        context_parts = []
        for idx, node in enumerate(combined_nodes):
            content = node.node.get_content(metadata_mode=MetadataMode.LLM).strip()
            context_parts.append(f"Source [{idx + 1}]:\n{content}")
            
        context_str = "\n\n".join(context_parts)
        return context_str, combined_nodes

    async def _aretrieve_and_process_nodes(self, sub_queries: List[str]) -> Tuple[str, List[NodeWithScore]]:
        """Async retrieve using ChunksService, deduplicate, and post-process."""
        all_nodes_map: Dict[str, NodeWithScore] = {}
        
        async def retrieve_safe(sq):
            try:
                # Use ChunksService
                chunks = await self._chunks_service.retrieve_relevant(sq, limit=5, prev_next_chunks=1)
                return self._convert_chunks_to_nodes(chunks)
            except Exception as e: 
                logger.error(f"Async retrieval error: {e}")
                return []
            
        results = await asyncio.gather(*[retrieve_safe(sq) for sq in sub_queries])
        
        for nodes in results:
            for node in nodes:
                if node.node.node_id not in all_nodes_map:
                    all_nodes_map[node.node.node_id] = node
                    
        combined_nodes = list(all_nodes_map.values())
        
        if self._node_postprocessors:
            primary_query_bundle = QueryBundle(sub_queries[0] if sub_queries else "")
            for postprocessor in self._node_postprocessors:
                combined_nodes = postprocessor.postprocess_nodes(
                    combined_nodes, query_bundle=primary_query_bundle
                )
        
        context_parts = []
        for idx, node in enumerate(combined_nodes):
            content = node.node.get_content(metadata_mode=MetadataMode.LLM).strip()
            context_parts.append(f"Source [{idx + 1}]:\n{content}")
            
        context_str = "\n\n".join(context_parts)
        return context_str, combined_nodes

    # --- Execution Flows ---

    def _run_agentic_condense_sync(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> Tuple[List[ChatMessage], ToolOutput, List[NodeWithScore]]:
        
        # 0. Tool Routing Check
        tool_intent = self._detect_specific_document_intent(message)
        if tool_intent:
            logger.info(f"Detected tool intent: {tool_intent}")
            tool_output = self._create_and_run_tool(tool_intent, message)
            if not tool_output.is_error:
                 # In this advanced system, we inject the tool output directly as context
                 pass # Fall through to RAG flow but with Tool Context

        # 1. Memory Setup
        if chat_history is not None:
            self._memory.set(chat_history)
        
        # 2. Condense
        current_history = self._memory.get(input=message)
        standalone_question = self._condense_question(current_history, message)
        
        # 3. Decompose
        sub_queries = self._decompose_query(standalone_question)
        
        # 4. Retrieve
        context_str, context_nodes = self._retrieve_and_process_nodes(sub_queries)
        
        # 5. Construct System Message
        sub_questions_str = "\n".join([f"- {sq}" for sq in sub_queries])
        formatted_prompt = self._context_prompt_template.format(
            original_question=standalone_question,
            sub_questions_str=sub_questions_str,
            context_str=context_str
        )
        
        if self._system_prompt:
             final_system_msg = f"{self._system_prompt}\n\n{formatted_prompt}"
        else:
             # Use default strict prompt if none provided
             final_system_msg = f"{self._get_default_system_prompt()}\n\n{formatted_prompt}"
             
        system_message = ChatMessage(
            content=final_system_msg, role=self._llm.metadata.system_role
        )
        
        # 6. Update Memory with User Message
        self._memory.put(ChatMessage(content=message, role=MessageRole.USER))
        
        # 7. Prepare Final History
        initial_token_count = self._token_counter.estimate_tokens_in_messages([system_message])
        final_history = self._memory.get(initial_token_count=initial_token_count)
        
        chat_messages = [system_message] + final_history
        
        context_source = ToolOutput(
            tool_name="advanced_rag",
            content=context_str,
            raw_input={"standalone_query": standalone_question},
            raw_output=context_nodes
        )
        
        return chat_messages, context_source, context_nodes

    async def _arun_agentic_condense(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> Tuple[List[ChatMessage], ToolOutput, List[NodeWithScore]]:
        
        # Async logic following same flow
        if chat_history is not None:
             self._memory.set(chat_history)
             
        current_history = self._memory.get(input=message)
        standalone_question = await self._acondense_question(current_history, message)
        
        sub_queries = await self._adecompose_query(standalone_question)
        
        context_str, context_nodes = await self._aretrieve_and_process_nodes(sub_queries)
        
        sub_questions_str = "\n".join([f"- {sq}" for sq in sub_queries])
        formatted_prompt = self._context_prompt_template.format(
            original_question=standalone_question,
            sub_questions_str=sub_questions_str,
            context_str=context_str
        )
        
        if self._system_prompt:
             final_system_msg = f"{self._system_prompt}\n\n{formatted_prompt}"
        else:
             # Use default strict prompt if none provided
             final_system_msg = f"{self._get_default_system_prompt()}\n\n{formatted_prompt}"
        system_message = ChatMessage(content=final_system_msg, role=self._llm.metadata.system_role)
        
        self._memory.put(ChatMessage(content=message, role=MessageRole.USER))
        
        initial_token_count = self._token_counter.estimate_tokens_in_messages([system_message])
        final_history = self._memory.get(initial_token_count=initial_token_count)
        
        chat_messages = [system_message] + final_history
        
        context_source = ToolOutput(
            tool_name="advanced_rag",
            content=context_str,
            raw_input={"standalone_query": standalone_question},
            raw_output=context_nodes
        )
        
        return chat_messages, context_source, context_nodes

    # --- Public Methods ---

    @trace_method("chat")
    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        chat_messages, context_source, context_nodes = self._run_agentic_condense_sync(
            message, chat_history
        )
        
        chat_response = self._llm.chat(chat_messages)
        assistant_message = chat_response.message
        self._memory.put(assistant_message)
        
        final_response = CitationHelper.smart_citation_replacement(assistant_message.content, context_nodes)
        cited_nodes = CitationHelper.get_cited_nodes(assistant_message.content, context_nodes)
        
        return AgentChatResponse(
            response=final_response,
            sources=[context_source],
            source_nodes=cited_nodes
        )

    @trace_method("chat")
    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        chat_messages, context_source, context_nodes = self._run_agentic_condense_sync(
            message, chat_history
        )
        
        chat_stream = self._llm.stream_chat(chat_messages)
        
        chat_response = StreamingAgentChatResponse(
            chat_stream=chat_stream,
            sources=[context_source],
            source_nodes=context_nodes,
        )
        
        thread = Thread(
            target=chat_response.write_response_to_history, args=(self._memory,)
        )
        thread.start()
        
        return chat_response

    @trace_method("chat")
    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        chat_messages, context_source, context_nodes = await self._arun_agentic_condense(
            message, chat_history
        )
        
        chat_response = await self._llm.achat(chat_messages)
        assistant_message = chat_response.message
        self._memory.put(assistant_message)
        
        final_response = CitationHelper.smart_citation_replacement(assistant_message.content, context_nodes)
        cited_nodes = CitationHelper.get_cited_nodes(assistant_message.content, context_nodes)
        
        return AgentChatResponse(
            response=final_response,
            sources=[context_source],
            source_nodes=cited_nodes
        )

    @trace_method("chat")
    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        chat_messages, context_source, context_nodes = await self._arun_agentic_condense(
            message, chat_history
        )
        
        chat_stream = await self._llm.astream_chat(chat_messages)
        
        chat_response = StreamingAgentChatResponse(
            achat_stream=chat_stream,
            sources=[context_source],
            source_nodes=context_nodes,
        )
        
        asyncio.create_task(chat_response.awrite_response_to_history(self._memory))
        return chat_response

    def reset(self) -> None:
        self._memory.reset()

    @property
    def chat_history(self) -> List[ChatMessage]:
        return self._memory.get_all()