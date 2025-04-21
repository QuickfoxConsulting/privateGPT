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
from llama_index.core.tools import BaseTool, ToolMetadata

DEFAULT_CONTEXT_TEMPLATE = (
    "Context information is below."
    "\n--------------------\n"
    "{context_str}"
    "\n--------------------\n"
)

DEFAULT_SYSTEM_PROMPT_TEMPLATE = """You are an agentic retrieval assistant that breaks down complex questions, searches for information, and provides comprehensive answers.

Original question: {original_question}

This question was decomposed into the following sub-questions:
{sub_questions}

Using ONLY the provided context information, create a comprehensive answer to the original question.
If the context doesn't contain all necessary information, acknowledge what's missing.
"""

class AgenticRAGRetriever(BaseRetriever):
    """Agentic RAG Retriever that breaks down queries into sub-queries and retrieves relevant context."""
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm: LLM,
        max_sub_queries: int = 3,
        callback_manager: Optional[CallbackManager] = None,
    ):
        """Initialize the AgenticRAGRetriever.
        
        Args:
            base_retriever: The underlying retriever to use for fetching documents
            llm: LLM to use for query decomposition
            max_sub_queries: Maximum number of sub-queries to generate
            callback_manager: Callback manager for tracing
        """
        self._base_retriever = base_retriever
        self._llm = llm
        self._max_sub_queries = max_sub_queries
        self.callback_manager = callback_manager or CallbackManager([])
        super().__init__()
    
    @classmethod
    def from_defaults(
        cls,
        base_retriever: BaseRetriever,
        llm: Optional[LLM] = None,
        service_context: Optional[ServiceContext] = None,
        max_sub_queries: int = 3,
        **kwargs: Any,
    ) -> "AgenticRAGRetriever":
        """Create an AgenticRAGRetriever with default parameters."""
        llm = llm or llm_from_settings_or_context(Settings, service_context)
        
        return cls(
            base_retriever=base_retriever,
            llm=llm,
            max_sub_queries=max_sub_queries,
            callback_manager=callback_manager_from_settings_or_context(
                Settings, service_context
            ),
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

        ## Note:
        - If the query is already simple, you can return it as the only sub-query.
        - Ensure that the sub-queries are clear, concise, and focused on specific aspects of the original query.
        - Avoid sub-queries that are too broad or general, as they may not provide useful context.
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
    
    @trace_method("retrieve")
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes for a query by breaking it down into sub-queries."""
        query_str = query_bundle.query_str
        
        # Step 1: Decompose the query into sub-queries
        sub_queries = self._decompose_query(query_str)
        
        # Step 2: Retrieve context for each sub-query and combine results
        all_nodes = []
        seen_node_ids = set()
        sub_query_results = {}
        
        for sub_query in sub_queries:
            # Create a new query bundle for each sub-query
            sub_query_bundle = QueryBundle(sub_query)
            
            # Retrieve nodes for the sub-query
            nodes = self._base_retriever.retrieve(sub_query_bundle)
            
            sub_query_results[sub_query] = nodes
            
            # Deduplicate nodes based on content hash
            for node in nodes:
                node_id = hash(node.node.get_content())
                if node_id not in seen_node_ids:
                    seen_node_ids.add(node_id)
                    
                    # Add metadata about which sub-query retrieved this node
                    if not hasattr(node.node.metadata, "sub_query"):
                        node.node.metadata["sub_query"] = sub_query
                    
                    all_nodes.append(node)
        
        # Add metadata about the original query and sub-queries to the query bundle
        query_bundle.metadata = {
            "original_query": query_str,
            "sub_queries": sub_queries,
            "sub_query_results": sub_query_results
        }
        
        return all_nodes
    
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Asynchronously retrieve nodes for a query by breaking it down into sub-queries."""
        query_str = query_bundle.query_str
        
        # Step 1: Decompose the query into sub-queries
        sub_queries = self._decompose_query(query_str)
        
        # Step 2: Retrieve context for each sub-query and combine results
        all_nodes = []
        seen_node_ids = set()
        sub_query_results = {}
        
        # Process sub-queries concurrently
        async def process_sub_query(sub_q):
            sub_q_bundle = QueryBundle(sub_q)
            nodes = await self._base_retriever.aretrieve(sub_q_bundle)
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
                    
                    # Add metadata about which sub-query retrieved this node
                    if not hasattr(node.node.metadata, "sub_query"):
                        node.node.metadata["sub_query"] = sub_q
                    
                    all_nodes.append(node)
        
        # Add metadata about the original query and sub-queries to the query bundle
        query_bundle.metadata = {
            "original_query": query_str,
            "sub_queries": sub_queries,
            "sub_query_results": sub_query_results
        }
        
        return all_nodes


class AgenticRAGChatEngine(BaseChatEngine):
    """
    Agentic RAG Chat Engine.

    Uses an AgenticRAGRetriever to break down complex queries, retrieve context,
    and then uses an LLM to generate a comprehensive response.
    """

    def __init__(
        self,
        retriever: AgenticRAGRetriever,
        llm: LLM,
        memory: BaseMemory,
        prefix_messages: List[ChatMessage],
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        context_template: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        tools: Optional[List[BaseTool]] = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._memory = memory
        self._prefix_messages = prefix_messages
        self._node_postprocessors = node_postprocessors or []
        self._context_template = context_template or DEFAULT_CONTEXT_TEMPLATE
        self._tools = tools or []
        
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
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> "AgenticRAGChatEngine":
        """Initialize an AgenticRAGChatEngine from default parameters."""
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
        callback_manager = callback_manager_from_settings_or_context(
            Settings, service_context
        )
        
        # Convert base retriever to AgenticRAGRetriever if needed
        if not isinstance(retriever, AgenticRAGRetriever):
            retriever = AgenticRAGRetriever.from_defaults(
                base_retriever=retriever,
                llm=llm,
                service_context=service_context,
                max_sub_queries=max_sub_queries,
            )

        return cls(
            retriever,
            llm=llm,
            memory=memory,
            prefix_messages=prefix_messages,
            node_postprocessors=node_postprocessors,
            callback_manager=callback_manager,
            context_template=context_template,
            tools=tools,
        )

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
    ) -> AgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(ChatMessage(content=message, role="user"))
        
        # Use the agentic retriever to get context
        query_bundle = QueryBundle(message)
        nodes = self._retriever.retrieve(query_bundle)
        
        # Get sub-queries from the retriever's metadata
        sub_queries = query_bundle.metadata.get("sub_queries", [message])
        
        # Apply node postprocessors
        for postprocessor in self._node_postprocessors:
            nodes = postprocessor.postprocess_nodes(nodes, query_bundle=query_bundle)
            
        # Generate context string from nodes
        context_str = "\n\n".join(
            [n.node.get_content(metadata_mode=MetadataMode.LLM).strip() for n in nodes]
        )
        
        # Format the context template
        context_str_template = self._context_template.format(context_str=context_str)
        
        # Get enhanced prefix messages with query decomposition info
        prefix_messages = self._get_prefix_messages_with_context(
            context_str_template, message, sub_queries
        )
        
        # Add tool descriptions if there are tools
        if self._tools:
            tool_descriptions = "\n\nAvailable Tools:\n" + "\n".join([
                f"- {tool.metadata.name}: {tool.metadata.description}"
                for tool in self._tools
            ])
            prefix_messages[0].content += tool_descriptions
        
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
        
        # Execute tools if needed (parse the response for tool calls)
        # This is a simplified version - you may want to implement more sophisticated
        # tool execution logic based on your needs
        if self._tools and "I need to use a tool" in str(ai_message.content).lower():
            # Simplified tool execution logic
            for tool in self._tools:
                if tool.metadata.name.lower() in str(ai_message.content).lower():
                    # Extract potential arguments (this is very simplified)
                    # In a real implementation, you'd want to parse the tool calls properly
                    try:
                        # This is just a placeholder for proper tool execution logic
                        tool_result = tool.call({"query": message})
                        
                        # Update the response with tool result
                        tool_message = f"\n\nTool {tool.metadata.name} executed. Result: {tool_result}"
                        updated_content = str(ai_message.content) + tool_message
                        ai_message.content = updated_content
                        self._memory.put(ai_message)
                    except Exception as e:
                        print(f"Error executing tool {tool.metadata.name}: {e}")

        return AgentChatResponse(
            response=str(ai_message.content),
            sources=[
                ToolOutput(
                    tool_name="agentic_rag_retriever",
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
    ) -> StreamingAgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)
        self._memory.put(ChatMessage(content=message, role="user"))
        
        # Use the agentic retriever to get context
        query_bundle = QueryBundle(message)
        nodes = self._retriever.retrieve(query_bundle)
        
        # Get sub-queries from the retriever's metadata
        sub_queries = query_bundle.metadata.get("sub_queries", [message])
        
        # Apply node postprocessors
        for postprocessor in self._node_postprocessors:
            nodes = postprocessor.postprocess_nodes(nodes, query_bundle=query_bundle)
            
        # Generate context string from nodes
        context_str = "\n\n".join(
            [n.node.get_content(metadata_mode=MetadataMode.LLM).strip() for n in nodes]
        )
        
        # Format the context template
        context_str_template = self._context_template.format(context_str=context_str)
        
        # Get enhanced prefix messages with query decomposition info
        prefix_messages = self._get_prefix_messages_with_context(
            context_str_template, message, sub_queries
        )
        
        # Add tool descriptions if there are tools
        if self._tools:
            tool_descriptions = "\n\nAvailable Tools:\n" + "\n".join([
                f"- {tool.metadata.name}: {tool.metadata.description}"
                for tool in self._tools
            ])
            prefix_messages[0].content += tool_descriptions
        
        prefix_messages_token_count = len(
            self._memory.tokenizer_fn(
                " ".join([(m.content or "") for m in prefix_messages])
            )
        )
        all_messages = prefix_messages + self._memory.get(
            initial_token_count=prefix_messages_token_count
        )

        chat_response = StreamingAgentChatResponse(
            chat_stream=self._llm.stream_chat(all_messages),
            sources=[
                ToolOutput(
                    tool_name="agentic_rag_retriever",
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

    # Similar implementations for achat and astream_chat methods
    # (omitted for brevity but would follow same pattern as above with async operations)

    def reset(self) -> None:
        self._memory.reset()

    @property
    def chat_history(self) -> List[ChatMessage]:
        """Get chat history."""
        return self._memory.get_all()


