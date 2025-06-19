import asyncio
import logging
from typing import Any, List, Optional, Tuple, Dict, Set

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.callbacks import CallbackManager, trace_method
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

logger = logging.getLogger(__name__)

# --- Prompt Templates ---

# Combines ideas from both previous engines
DEFAULT_CONTEXT_PROMPT_TEMPLATE = """
You are a helpful and knowledgeable AI assistant.
Your goal is to provide a comprehensive and accurate answer based *only* on the provided context documents.

**Original Question:**
{original_question}

**Context Breakdown:**
To gather information, the original question was broken down into these sub-questions:
{sub_questions_str}

**Context Documents:**
The following sections contain relevant information retrieved based on the sub-questions:
---------------------
{context_str}
---------------------

**Instructions:**
1.  Synthesize a single, comprehensive answer to the **Original Question**.
2.  Use **ONLY** the information present in the **Context Documents** section above.
3.  Do **NOT** use any prior knowledge or information outside the provided context.
4.  If the context does not contain enough information to answer the original question fully, state what information is missing or cannot be found in the documents.
5.  Cite relevant source document node IDs or filenames in square brackets (e.g., `[node_id_123]` or `[file_name.pdf]`) after the information they support, if the node ID or filename is available in the context metadata. When using information from a context document, always cite its node ID or filename inline.
6.  Be detailed, clear, and well-organized in your response.
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

    Combines chat history condensation with query decomposition for RAG.
    Flow:
    1. Condense chat history + user message -> standalone query.
    2. Decompose standalone query -> sub-queries.
    3. Retrieve context for each sub-query.
    4. Post-process retrieved nodes.
    5. Synthesize response using LLM with context, history, and query info.
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
        # Adjust memory token limit calculation if needed, considering prompt complexity
        memory = memory or ChatMemoryBuffer.from_defaults(
            chat_history=chat_history, token_limit=llm.metadata.context_window * 0.7 # Conservative estimate
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

    # --- Core Logic Methods ---

    def _condense_question(
        self, chat_history: List[ChatMessage], latest_message: str
    ) -> str:
        """Condense history and latest message to a standalone question."""
        if self._skip_condense or not chat_history:
            return latest_message

        chat_history_str = messages_to_history_str(chat_history)
        if self._verbose:
            print(f"Condensing query with history: {chat_history_str}")

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
            print(f"Condensing query with history: {chat_history_str}")

        response = await self._llm.apredict(
            self._condense_prompt_template,
            question=latest_message,
            chat_history=chat_history_str,
        )
        return response.strip()

    def _decompose_query(self, query: str) -> List[str]:
        """Decompose a query into sub-queries using LLM."""
        prompt = self._decompose_prompt_template.format(
            query=query, max_sub_queries=self._max_sub_queries
        )
        response = self._llm.complete(prompt)
        text = response.text.strip()

        try:
            # Try parsing as Python list literal
            if text.startswith("[") and text.endswith("]"):
                sub_queries = eval(text)
                if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries):
                    return sub_queries[:self._max_sub_queries]
            # Fallback: treat each line as a sub-query
            logger.warning(f"Decomposition result was not a valid list: {text}. Splitting by lines.")
            sub_queries = [line.strip('- ').strip() for line in text.splitlines() if line.strip()]
            return sub_queries[:self._max_sub_queries] if sub_queries else [query]
        except Exception as e:
            logger.error(f"Failed to parse decomposed queries: {e}. Falling back to original query. Raw response: {text}")
            return [query] # Fallback

    async def _adecompose_query(self, query: str) -> List[str]:
        """Async decompose a query into sub-queries."""
        prompt = self._decompose_prompt_template.format(
            query=query, max_sub_queries=self._max_sub_queries
        )
        response = await self._llm.acomplete(prompt)
        text = response.text.strip()

        try:
             if text.startswith("[") and text.endswith("]"):
                sub_queries = eval(text)
                if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries):
                    return sub_queries[:self._max_sub_queries]
             logger.warning(f"Decomposition result was not a valid list: {text}. Splitting by lines.")
             sub_queries = [line.strip('- ').strip() for line in text.splitlines() if line.strip()]
             return sub_queries[:self._max_sub_queries] if sub_queries else [query]
        except Exception as e:
            logger.error(f"Failed to parse decomposed queries: {e}. Falling back to original query. Raw response: {text}")
            return [query]

    async def _aretrieve_and_process_nodes(
        self, query_bundle: QueryBundle, sub_queries: List[str]
    ) -> Tuple[str, List[NodeWithScore]]:
        """Retrieve nodes for sub-queries, deduplicate, post-process, and format context."""
        all_nodes_map: Dict[str, NodeWithScore] = {} # Use node ID for deduplication

        async def retrieve_for_sub_query(sub_q: str):
            try:
                sub_q_bundle = QueryBundle(sub_q)
                nodes = await self._retriever.aretrieve(sub_q_bundle)
                return sub_q, nodes
            except Exception as e:
                logger.error(f"Error retrieving for sub-query '{sub_q}': {e}")
                return sub_q, []

        tasks = [retrieve_for_sub_query(sq) for sq in sub_queries]
        results = await asyncio.gather(*tasks)

        for sub_q, nodes in results:
            for node in nodes:
                if node.node.node_id not in all_nodes_map:
                    # Add metadata about which sub-query retrieved this node (optional)
                    node.node.metadata["retrieved_by_sub_query"] = sub_q
                    all_nodes_map[node.node.node_id] = node

        combined_nodes = list(all_nodes_map.values())

        if self._node_postprocessors:
             for postprocessor in self._node_postprocessors:
                  combined_nodes = postprocessor.postprocess_nodes(
                       combined_nodes, query_bundle=query_bundle # Use original query bundle for postprocessing context
                  )
        context_str = "\n\n".join(
            [n.node.get_content(metadata_mode=MetadataMode.LLM).strip() for n in combined_nodes]
        )
        return context_str, combined_nodes

    def _retrieve_and_process_nodes_sync(
         self, query_bundle: QueryBundle, sub_queries: List[str]
    ) -> Tuple[str, List[NodeWithScore]]:
        """ Synchronous version of node retrieval and processing. """
        all_nodes_map: Dict[str, NodeWithScore] = {}

        for sub_q in sub_queries:
            try:
                sub_q_bundle = QueryBundle(sub_q)
                nodes = self._retriever.retrieve(sub_q_bundle)
                for node in nodes:
                    if node.node.node_id not in all_nodes_map:
                         node.node.metadata["retrieved_by_sub_query"] = sub_q
                         all_nodes_map[node.node.node_id] = node
            except Exception as e:
                logger.error(f"Error retrieving for sub-query '{sub_q}': {e}")

        combined_nodes = list(all_nodes_map.values())

        if self._node_postprocessors:
            for postprocessor in self._node_postprocessors:
                combined_nodes = postprocessor.postprocess_nodes(
                    combined_nodes, query_bundle=query_bundle
                )

        context_str = "\n\n".join(
            [n.node.get_content(metadata_mode=MetadataMode.LLM).strip() for n in combined_nodes]
        )
        return context_str, combined_nodes


    async def _arun_agentic_condense(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> Tuple[List[ChatMessage], ToolOutput, List[NodeWithScore]]:
        """Core async logic for condense -> decompose -> retrieve -> prepare messages."""
        if chat_history is not None:
            self._memory.set(chat_history)

        current_chat_history = self._memory.get(input=message)

        # 1. Condense
        standalone_question = await self._acondense_question(current_chat_history, message)
        logger.info(f"Standalone question: {standalone_question}")
        if self._verbose: print(f"Standalone question: {standalone_question}")

        # 2. Decompose
        sub_queries = await self._adecompose_query(standalone_question)
        logger.info(f"Decomposed sub-queries: {sub_queries}")
        if self._verbose: print(f"Decomposed sub-queries: {sub_queries}")

        # 3. Retrieve & Process Nodes
        query_bundle = QueryBundle(standalone_question) # Use standalone for retrieval context
        context_str, context_nodes = await self._aretrieve_and_process_nodes(
            query_bundle, sub_queries
        )
        context_source = ToolOutput(
            tool_name="agentic_retriever",
            content=context_str, 
            raw_input={
                "standalone_query": standalone_question,
                "sub_queries": sub_queries
            },
            raw_output=context_nodes,
        )
        logger.debug(f"Retrieved Context: {context_str[:500]}...") # Log snippet
        if self._verbose: print(f"Retrieved Context: {context_str[:500]}...")

        # 4. Prepare messages for LLM
        self._memory.put(ChatMessage(content=message, role=MessageRole.USER))

        sub_questions_str = "\n".join([f"- {sq}" for sq in sub_queries])
        formatted_context_prompt = self._context_prompt_template.format(
            original_question=standalone_question, # Use the condensed question here
            sub_questions_str=sub_questions_str,
            context_str=context_str
        )

        final_system_prompt = formatted_context_prompt
        if self._system_prompt: # Add user-defined overall system prompt if provided
            final_system_prompt = f"{self._system_prompt}\n\n{formatted_context_prompt}"

        system_message = ChatMessage(
            content=final_system_prompt, role=self._llm.metadata.system_role
        )

        initial_token_count = self._token_counter.estimate_tokens_in_messages(
            [system_message]
        )
        final_chat_history = self._memory.get(initial_token_count=initial_token_count)
        chat_messages = [system_message] + final_chat_history
        return chat_messages, context_source, context_nodes

    def _run_agentic_condense_sync(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
     ) -> Tuple[List[ChatMessage], ToolOutput, List[NodeWithScore]]:
        """Core sync logic"""
        if chat_history is not None:
            self._memory.set(chat_history)

        current_chat_history = self._memory.get(input=message)

        standalone_question = self._condense_question(current_chat_history, message)
        logger.info(f"Standalone question: {standalone_question}")
        if self._verbose: print(f"Standalone question: {standalone_question}")

        sub_queries = self._decompose_query(standalone_question)
        logger.info(f"Decomposed sub-queries: {sub_queries}")
        if self._verbose: print(f"Decomposed sub-queries: {sub_queries}")

        query_bundle = QueryBundle(standalone_question)
        context_str, context_nodes = self._retrieve_and_process_nodes_sync(
            query_bundle, sub_queries
        )
        context_source = ToolOutput(
            tool_name="agentic_retriever",
            content=context_str,
            raw_input={
                "standalone_query": standalone_question,
                "sub_queries": sub_queries
            },
            raw_output=context_nodes,
        )
        logger.debug(f"Retrieved Context: {context_str[:500]}...")
        if self._verbose: print(f"Retrieved Context: {context_str[:500]}...")

        self._memory.put(ChatMessage(content=message, role=MessageRole.USER))

        sub_questions_str = "\n".join([f"- {sq}" for sq in sub_queries])
        formatted_context_prompt = self._context_prompt_template.format(
            original_question=standalone_question,
            sub_questions_str=sub_questions_str,
            context_str=context_str
        )

        final_system_prompt = formatted_context_prompt
        if self._system_prompt:
            final_system_prompt = f"{self._system_prompt}\n\n{formatted_context_prompt}"

        system_message = ChatMessage(
            content=final_system_prompt, role=self._llm.metadata.system_role
        )

        initial_token_count = self._token_counter.estimate_tokens_in_messages(
            [system_message]
        )
        final_chat_history = self._memory.get(initial_token_count=initial_token_count)
        chat_messages = [system_message] + final_chat_history

        return chat_messages, context_source, context_nodes

    def _extract_cited_node_ids(self, response: str) -> set:
        """Extract cited node IDs or filenames from the LLM response."""
        import re
        # Matches [node_id_123] or [file.pdf]
        pattern = r'\[([\w\-\.]+)\]'
        return set(re.findall(pattern, response))

    def _filter_nodes_by_citation(self, cited_ids: set, nodes: list) -> list:
        """Filter nodes to only those cited in the response (by node_id or filename)."""
        filtered = []
        for node in nodes:
            node_id = getattr(node.node, 'node_id', None)
            filename = node.node.metadata.get('file_name') or node.node.metadata.get('filename')
            if node_id in cited_ids or (filename and filename in cited_ids):
                filtered.append(node)
        return filtered

    def _llm_post_analysis_attribution(self, response: str, nodes: list) -> list:
        """Ask the LLM to identify which nodes were used in the answer."""
        prompt = (
            "Given the following answer and context nodes, return a JSON list of node IDs that contributed information to the answer.\n"
            "Answer:\n"
            f"{response}\n"
            "Context Nodes (format: Node ID: Content):\n"
        )
        for node in nodes:
            node_id = getattr(node.node, 'node_id', None)
            content = node.node.get_content(metadata_mode=MetadataMode.LLM).strip()
            prompt += f"{node_id}: {content}\n"
        prompt += "\nReturn a JSON array of node IDs, e.g. [\"node_id_1\", \"node_id_2\"]"
        analysis_response = self._llm.complete(prompt)
        import json
        try:
            used_node_ids = json.loads(analysis_response.text)
            return [node for node in nodes if getattr(node.node, 'node_id', None) in used_node_ids]
        except Exception as e:
            logger.error(f"LLM post-analysis parsing failed: {e}")
            return []

    def _react_agent_attribution(self, response: str, nodes: list) -> list:
        """Use a ReACT agent to attribute sources if available. Placeholder for integration."""
        # This is a placeholder. You would integrate your ReACT agent here.
        # For now, just return all nodes (no filtering)
        return nodes

    # --- Public Chat Methods ---

    @trace_method("chat")
    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None,
        attribution_method: str = "inline_citation"  # or "llm_post_analysis" or "react_agent"
    ) -> AgentChatResponse:
        chat_messages, context_source, context_nodes = self._run_agentic_condense_sync(
            message, chat_history
        )
        chat_response = self._llm.chat(chat_messages)
        assistant_message = chat_response.message
        self._memory.put(assistant_message)

        # Attribution logic
        if attribution_method == "inline_citation":
            cited_ids = self._extract_cited_node_ids(assistant_message.content)
            used_nodes = self._filter_nodes_by_citation(cited_ids, context_nodes)
        elif attribution_method == "llm_post_analysis":
            used_nodes = self._llm_post_analysis_attribution(assistant_message.content, context_nodes)
        elif attribution_method == "react_agent":
            used_nodes = self._react_agent_attribution(assistant_message.content, context_nodes)
        else:
            used_nodes = context_nodes

        return AgentChatResponse(
            response=str(assistant_message.content),
            sources=[ToolOutput(
                tool_name="agentic_retriever",
                content=context_source.content,
                raw_input=context_source.raw_input,
                raw_output=used_nodes
            )],
            source_nodes=used_nodes,
        )

    @trace_method("chat")
    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        chat_messages, context_source, context_nodes = self._run_agentic_condense_sync(
            message, chat_history
        )

        chat_response = StreamingAgentChatResponse(
            chat_stream=self._llm.stream_chat(chat_messages),
            sources=[context_source],
            source_nodes=context_nodes,
        )
        # Start thread to write streamed response to memory *after* it's finished
        thread = Thread(
            target=chat_response.write_response_to_history, args=(self._memory,)
        )
        thread.start()

        return chat_response

    @trace_method("chat")
    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None,
        attribution_method: str = "inline_citation"
    ) -> AgentChatResponse:
        chat_messages, context_source, context_nodes = await self._arun_agentic_condense(
            message, chat_history
        )
        chat_response = await self._llm.achat(chat_messages)
        assistant_message = chat_response.message
        self._memory.put(assistant_message)

        # Attribution logic
        if attribution_method == "inline_citation":
            cited_ids = self._extract_cited_node_ids(assistant_message.content)
            used_nodes = self._filter_nodes_by_citation(cited_ids, context_nodes)
        elif attribution_method == "llm_post_analysis":
            used_nodes = self._llm_post_analysis_attribution(assistant_message.content, context_nodes)
        elif attribution_method == "react_agent":
            used_nodes = self._react_agent_attribution(assistant_message.content, context_nodes)
        else:
            used_nodes = context_nodes

        return AgentChatResponse(
            response=str(assistant_message.content),
            sources=[ToolOutput(
                tool_name="agentic_retriever",
                content=context_source.content,
                raw_input=context_source.raw_input,
                raw_output=used_nodes
            )],
            source_nodes=used_nodes,
        )

    @trace_method("chat")
    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        chat_messages, context_source, context_nodes = await self._arun_agentic_condense(
            message, chat_history
        )

        chat_response = StreamingAgentChatResponse(
            achat_stream=await self._llm.astream_chat(chat_messages),
            sources=[context_source],
            source_nodes=context_nodes,
        )
        # Start async task to write streamed response to memory
        asyncio.create_task(chat_response.awrite_response_to_history(self._memory))

        return chat_response

    # --- Memory Management ---

    def reset(self) -> None:
        self._memory.reset()

    @property
    def chat_history(self) -> List[ChatMessage]:
        return self._memory.get_all()