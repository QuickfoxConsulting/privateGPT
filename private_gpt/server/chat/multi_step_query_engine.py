import asyncio
from typing import Any, List, Optional, Tuple, Dict, Set

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.callbacks import CallbackManager, trace_method
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
    BaseChatEngine,
    StreamingAgentChatResponse,
    ToolOutput,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from llama_index.core.service_context import ServiceContext
from llama_index.core.settings import (
    Settings,
    callback_manager_from_settings_or_context,
    llm_from_settings_or_context,
)
from llama_index.core.types import Thread

from private_gpt.components.embedding.embedding_component import EmbeddingComponent

DEFAULT_CONTEXT_TEMPLATE = (
    "Context information is below."
    "\n--------------------\n"
    "{context_str}"
    "\n--------------------\n"
)

DEFAULT_SYSTEM_PROMPT_TEMPLATE = """You are answering a complex question that has been broken down into simpler sub-questions to gather more comprehensive information.

Original question: {original_question}

This question was decomposed into the following sub-questions:
{sub_questions}

Using ONLY the provided context information, create a comprehensive answer to the original question.
If the context doesn't contain all necessary information, acknowledge what's missing.
"""


class MultiStepContextChatEngine(BaseChatEngine):
    """
    Multi-Step Context Chat Engine.

    Breaks down complex queries into sub-queries, retrieves context for each,
    and then uses an LLM to generate a comprehensive response.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: LLM,
        memory: BaseMemory,
        prefix_messages: List[ChatMessage],
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        context_template: Optional[str] = None,
        max_sub_queries: int = 3,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._memory = memory
        self._prefix_messages = prefix_messages
        self._node_postprocessors = node_postprocessors or []
        self._context_template = context_template or DEFAULT_CONTEXT_TEMPLATE
        self._max_sub_queries = max_sub_queries

        self.callback_manager = callback_manager or CallbackManager([])
        for node_postprocessor in self._node_postprocessors:
            node_postprocessor.callback_manager = self.callback_manager

    @classmethod
    def from_defaults(
        cls,
        retriever: BaseRetriever,
        service_context: Optional[ServiceContext] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        system_prompt: Optional[str] = None,
        prefix_messages: Optional[List[ChatMessage]] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        context_template: Optional[str] = None,
        max_sub_queries: int = 3,
        llm: Optional[LLM] = None,
        **kwargs: Any,
    ) -> "MultiStepContextChatEngine":
        """Initialize a MultiStepContextChatEngine from default parameters."""
        llm = llm or llm_from_settings_or_context(Settings, service_context)

        chat_history = chat_history or []
        memory = memory or ChatMemoryBuffer.from_defaults(
            chat_history=chat_history, token_limit=llm.metadata.context_window - 256
        )

        if system_prompt is not None:
            if prefix_messages is not None:
                raise ValueError(
                    "Cannot specify both system_prompt and prefix_messages"
                )
            prefix_messages = [
                ChatMessage(content=system_prompt, role=llm.metadata.system_role)
            ]

        prefix_messages = prefix_messages or []
        node_postprocessors = node_postprocessors or []

        return cls(
            retriever,
            llm=llm,
            memory=memory,
            prefix_messages=prefix_messages,
            node_postprocessors=node_postprocessors,
            callback_manager=callback_manager_from_settings_or_context(
                Settings, service_context
            ),
            context_template=context_template,
            max_sub_queries=max_sub_queries,
        )

    def _decompose_query(self, query_str: str) -> List[str]:
        """Break down a complex query into simpler sub-queries."""
        prompt = f"""Decompose the following complex query into {self._max_sub_queries} or fewer precise and focused sub-queries.  
        Each sub-query should:  
        - Target a specific aspect of the original question.  
        - Be self-contained and independently meaningful.  
        - Help retrieve relevant information efficiently.  

        ## Complex Query:
        {query_str}

        ## Output Format:
        Return the sub-queries as a **Python list of strings**, following this structure:  
        ["What is X?", "How does X relate to Y?", "What are the benefits of Z?"]

        ## Few-Shot Examples:

        ### Example 1:
        **Complex Query:**  
        "What are the causes, effects, and possible solutions to climate change?"  

        **Sub-Queries:**  
        ["What are the main causes of climate change?",  
        "What are the effects of climate change on the environment and human life?",  
        "What are the possible solutions to mitigate climate change?"]  

        ### Example 2:
        **Complex Query:**  
        "How does machine learning work, and what are its main applications in healthcare?"  

        **Sub-Queries:**  
        ["What is machine learning and how does it work?",  
        "What are the different types of machine learning algorithms?",  
        "What are the main applications of machine learning in healthcare?"]

        ### Example 3:
        **Complex Query:**  
        "What are the historical causes of economic recessions, and how do governments typically respond?"  

        **Sub-Queries:**  
        ["What are the common historical causes of economic recessions?",  
        "How do economic recessions impact businesses and employment?",  
        "What are the typical policy responses by governments to economic recessions?",  
        "How effective are government interventions in mitigating recessions?"]  

        ## Note:
        - If the query is already simple, you can return it as the only sub-query.
        - Ensure that the sub-queries are clear, concise, and focused on specific aspects of the original query.
        - Avoid sub-queries that are too broad or general, as they may not provide useful context.
        ## Now, generate sub-queries for the given complex query:

        """

        response = self._llm.complete(prompt)
        
        try:
            # Try to safely extract the list from the response
            text = response.text.strip()
            if text.startswith("[") and text.endswith("]"):
                # Clean up and parse the response as a Python list
                sub_queries = eval(text)
                if not isinstance(sub_queries, list):
                    raise ValueError("Response is not a list")
                # Filter out any non-string elements
                sub_queries = [q for q in sub_queries if isinstance(q, str)]
                # Limit the number of sub-queries
                return sub_queries[:self._max_sub_queries]
            else:
                # Fallback if the response isn't a well-formed list
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                sub_queries = []
                for line in lines:
                    # Remove numbered bullets or quotes if present
                    cleaned = line
                    if line.startswith('"') and line.endswith('"'):
                        cleaned = line[1:-1]
                    elif line.startswith("'") and line.endswith("'"):
                        cleaned = line[1:-1]
                    elif any(line.startswith(f"{i}.") for i in range(1, 10)):
                        cleaned = line[line.find(".")+1:].strip()
                    
                    if cleaned:
                        sub_queries.append(cleaned)
                
                return sub_queries[:self._max_sub_queries]
        except Exception:
            # If parsing fails, fall back to a simple approach - just return the original query
            return [query_str]

    def _generate_context(self, message: str) -> Tuple[str, List[NodeWithScore]]:
        """Generate context for a complex query using multi-step approach."""
        # Step 1: Decompose the query into sub-queries
        sub_queries = self._decompose_query(message)
        
        # Step 2: Retrieve context for each sub-query and combine results
        all_nodes = []
        seen_node_ids = set()
        sub_query_results = {}
        
        for sub_query in sub_queries:
            nodes = self._retriever.retrieve(sub_query)
            
            # Apply node postprocessors
            for postprocessor in self._node_postprocessors:
                nodes = postprocessor.postprocess_nodes(
                    nodes, query_bundle=QueryBundle(sub_query)
                )
            
            sub_query_results[sub_query] = nodes
            
            # Deduplicate nodes based on content hash
            for node in nodes:
                node_id = hash(node.node.get_content())
                if node_id not in seen_node_ids:
                    seen_node_ids.add(node_id)
                    all_nodes.append(node)
        
        # Generate context string from all nodes
        context_str = "\n\n".join(
            [n.node.get_content(metadata_mode=MetadataMode.LLM).strip() for n in all_nodes]
        )
        
        # Format the context template
        formatted_context = self._context_template.format(context_str=context_str)
        
        return formatted_context, all_nodes

    async def _agenerate_context(self, message: str) -> Tuple[str, List[NodeWithScore]]:
        """Generate context for a complex query using multi-step approach (async)."""
        # Step 1: Decompose the query into sub-queries
        sub_queries = self._decompose_query(message)
        
        # Step 2: Retrieve context for each sub-query and combine results
        all_nodes = []
        seen_node_ids = set()
        sub_query_results = {}
        
        # Process sub-queries concurrently
        async def process_sub_query(sub_q):
            nodes = await self._retriever.aretrieve(sub_q)
            for postprocessor in self._node_postprocessors:
                nodes = postprocessor.postprocess_nodes(
                    nodes, query_bundle=QueryBundle(sub_q)
                )
            return sub_q, nodes
        
        # Gather results from all sub-queries
        tasks = [process_sub_query(sq) for sq in sub_queries]
        results = await asyncio.gather(*tasks)
        
        # Process results
        for sub_q, nodes in results:
            sub_query_results[sub_q] = nodes
            
            # Deduplicate nodes based on content hash
            for node in nodes:
                node_id = hash(node.node.get_content())
                if node_id not in seen_node_ids:
                    seen_node_ids.add(node_id)
                    all_nodes.append(node)
        
        # Generate context string from all nodes
        context_str = "\n\n".join(
            [n.node.get_content(metadata_mode=MetadataMode.LLM).strip() for n in all_nodes]
        )
        
        # Format the context template
        formatted_context = self._context_template.format(context_str=context_str)
        
        return formatted_context, all_nodes

    def _get_prefix_messages_with_context(
        self, context_str: str, original_question: str, sub_queries: List[str]
    ) -> List[ChatMessage]:
        """Get the prefix messages with context and query decomposition info."""
        # Ensure we grab the user-configured system prompt
        system_prompt = ""
        prefix_messages = self._prefix_messages
        if (
            len(self._prefix_messages) != 0
            and self._prefix_messages[0].role == MessageRole.SYSTEM
        ):
            system_prompt = str(self._prefix_messages[0].content)
            prefix_messages = self._prefix_messages[1:]
        
        # Format sub-queries as a string
        sub_questions_str = "\n".join([f"- {sq}" for sq in sub_queries])
        
        # Create an enhanced system prompt that includes the decomposition information
        enhanced_system_prompt = DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(
            original_question=original_question,
            sub_questions=sub_questions_str
        )
        
        # Combine with user system prompt if any
        if system_prompt:
            enhanced_system_prompt = f"{system_prompt}\n\n{enhanced_system_prompt}"
        
        # Add context to system prompt
        context_str_w_sys_prompt = enhanced_system_prompt.strip() + "\n" + context_str
        
        return [
            ChatMessage(
                content=context_str_w_sys_prompt, role=self._llm.metadata.system_role
            ),
            *prefix_messages,
        ]

    @trace_method("chat")
    def chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        prev_chunks=None,
    ) -> AgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(ChatMessage(content=message, role="user"))

        # Decompose the query
        sub_queries = self._decompose_query(message)
        
        # Generate context using the multi-step approach
        context_str_template, nodes = self._generate_context(message)

        # If the fetched context is completely empty
        if len(nodes) == 0 and prev_chunks is not None:
            nodes = [j for i in prev_chunks for j in i]
            context_str = "\n\n".join(
                [
                    n.node.get_content(metadata_mode=MetadataMode.LLM).strip()
                    for n in nodes
                ]
            )

            # Create a new context string template by using previous nodes
            context_str_template = self._context_template.format(
                context_str=context_str
            )

        # Get enhanced prefix messages with query decomposition info
        prefix_messages = self._get_prefix_messages_with_context(
            context_str_template, message, sub_queries
        )
        
        prefix_messages_token_count = len(
            self._memory.tokenizer_fn(
                " ".join([(m.content or "") for m in prefix_messages])
            )
        )
        all_messages = prefix_messages + self._memory.get(
            initial_token_count=prefix_messages_token_count
        )
        chat_response = self._llm.chat(all_messages)
        ai_message = chat_response.message
        self._memory.put(ai_message)

        return AgentChatResponse(
            response=str(chat_response.message.content),
            sources=[
                ToolOutput(
                    tool_name="multi_step_retriever",
                    content=str(prefix_messages[0]),
                    raw_input={"message": message, "sub_queries": sub_queries},
                    raw_output=prefix_messages[0],
                )
            ],
            source_nodes=nodes,
        )

    @trace_method("chat")
    def stream_chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        prev_chunks=None,
    ) -> StreamingAgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)

        self._memory.put(ChatMessage(content=message, role="user"))

        # Decompose the query
        sub_queries = self._decompose_query(message)
        
        # Generate context using the multi-step approach
        context_str_template, nodes = self._generate_context(message)

        # If the fetched context is completely empty
        if len(nodes) == 0 and prev_chunks is not None:
            nodes = [j for i in prev_chunks for j in i]
            context_str = "\n\n".join(
                [
                    n.node.get_content(metadata_mode=MetadataMode.LLM).strip()
                    for n in nodes
                ]
            )

            # Create a new context string template by using previous nodes
            context_str_template = self._context_template.format(
                context_str=context_str
            )

        # Get enhanced prefix messages with query decomposition info
        prefix_messages = self._get_prefix_messages_with_context(
            context_str_template, message, sub_queries
        )
        
        initial_token_count = len(
            self._memory.tokenizer_fn(
                " ".join([(m.content or "") for m in prefix_messages])
            )
        )
        all_messages = prefix_messages + self._memory.get(
            initial_token_count=initial_token_count
        )

        chat_response = StreamingAgentChatResponse(
            chat_stream=self._llm.stream_chat(all_messages),
            sources=[
                ToolOutput(
                    tool_name="multi_step_retriever",
                    content=str(prefix_messages[0]),
                    raw_input={"message": message, "sub_queries": sub_queries},
                    raw_output=prefix_messages[0],
                )
            ],
            source_nodes=nodes,
        )
        thread = Thread(
            target=chat_response.write_response_to_history, args=(self._memory,)
        )
        thread.start()

        return chat_response

    @trace_method("chat")
    async def achat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        prev_chunks=None,
    ) -> AgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(ChatMessage(content=message, role="user"))

        # Decompose the query
        sub_queries = self._decompose_query(message)
        
        # Generate context using the multi-step approach
        context_str_template, nodes = await self._agenerate_context(message)

        # If the fetched context is completely empty
        if len(nodes) == 0 and prev_chunks is not None:
            nodes = [j for i in prev_chunks for j in i]
            context_str = "\n\n".join(
                [
                    n.node.get_content(metadata_mode=MetadataMode.LLM).strip()
                    for n in nodes
                ]
            )

            # Create a new context string template by using previous nodes
            context_str_template = self._context_template.format(
                context_str=context_str
            )

        # Get enhanced prefix messages with query decomposition info
        prefix_messages = self._get_prefix_messages_with_context(
            context_str_template, message, sub_queries
        )
        
        initial_token_count = len(
            self._memory.tokenizer_fn(
                " ".join([(m.content or "") for m in prefix_messages])
            )
        )
        all_messages = prefix_messages + self._memory.get(
            initial_token_count=initial_token_count
        )

        chat_response = await self._llm.achat(all_messages)
        ai_message = chat_response.message
        self._memory.put(ai_message)

        return AgentChatResponse(
            response=str(chat_response.message.content),
            sources=[
                ToolOutput(
                    tool_name="multi_step_retriever",
                    content=str(prefix_messages[0]),
                    raw_input={"message": message, "sub_queries": sub_queries},
                    raw_output=prefix_messages[0],
                )
            ],
            source_nodes=nodes,
        )

    @trace_method("chat")
    async def astream_chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        prev_chunks=None,
    ) -> StreamingAgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(ChatMessage(content=message, role="user"))

        # Decompose the query
        sub_queries = self._decompose_query(message)
        
        # Generate context using the multi-step approach
        context_str_template, nodes = await self._agenerate_context(message)

        # If the fetched context is completely empty
        if len(nodes) == 0 and prev_chunks is not None:
            nodes = [j for i in prev_chunks for j in i]
            context_str = "\n\n".join(
                [
                    n.node.get_content(metadata_mode=MetadataMode.LLM).strip()
                    for n in nodes
                ]
            )

            # Create a new context string template by using previous nodes
            context_str_template = self._context_template.format(
                context_str=context_str
            )

        # Get enhanced prefix messages with query decomposition info
        prefix_messages = self._get_prefix_messages_with_context(
            context_str_template, message, sub_queries
        )
        
        initial_token_count = len(
            self._memory.tokenizer_fn(
                " ".join([(m.content or "") for m in prefix_messages])
            )
        )
        all_messages = prefix_messages + self._memory.get(
            initial_token_count=initial_token_count
        )

        chat_response = StreamingAgentChatResponse(
            achat_stream=await self._llm.astream_chat(all_messages),
            sources=[
                ToolOutput(
                    tool_name="multi_step_retriever",
                    content=str(prefix_messages[0]),
                    raw_input={"message": message, "sub_queries": sub_queries},
                    raw_output=prefix_messages[0],
                )
            ],
            source_nodes=nodes,
        )
        asyncio.create_task(chat_response.awrite_response_to_history(self._memory))

        return chat_response

    def reset(self) -> None:
        self._memory.reset()

    @property
    def chat_history(self) -> List[ChatMessage]:
        """Get chat history."""
        return self._memory.get_all()