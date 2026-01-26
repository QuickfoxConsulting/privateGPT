import logging
import asyncio
from typing import List, Optional, Dict, Any, Union

from llama_index.core.chat_engine.types import (
    BaseChatEngine,
    AgentChatResponse,
    StreamingAgentChatResponse,
)
from llama_index.core.llms import LLM, ChatMessage
from llama_index.core.base.llms.types import MessageRole
from sqlalchemy.orm import Session

from private_gpt.server.agents.router_agent import RouterAgent
from private_gpt.server.agents.planner_agent import PlannerAgent
from private_gpt.server.agents.schemas import RouteType, RoutingResult
from private_gpt.server.tools.tool_registry import ToolRegistry
from private_gpt.server.chat.agentic_tool import AgenticRAGEngine, AgentStreamHandler
from private_gpt.server.chat.rag_config import RAG_CONFIG
from llama_index.core.callbacks import CallbackManager, CBEventType

logger = logging.getLogger(__name__)

class HierarchicalAgentEngine(BaseChatEngine):
    """
    Orchestrator engine that routes queries through a hierarchical agent structure.
    1. Router -> Classifies query (RAG, REACT, PLANNER)
    2. Execution -> Delegates to specialized engine or Planner
    """
    
    def __init__(
        self,
        llm: LLM,
        tool_registry: ToolRegistry,
        user_id: int,
        # Components needed for sub-engines
        index: Any,
        vector_store_component: Any,
        node_store_component: Any,
        node_postprocessors: Optional[List[Any]] = None,
        document_files: Optional[List[str]] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        self._llm = llm
        self._tool_registry = tool_registry
        self._user_id = user_id
        self.callback_manager = callback_manager or CallbackManager([])
        
        # Sub-agents
        self._router = RouterAgent(llm)
        self._planner = PlannerAgent(llm)
        
        # Components for execution engines
        self._index = index
        self._vector_store_component = vector_store_component
        self._node_store_component = node_store_component
        self._node_postprocessors = node_postprocessors
        self._document_files = document_files
        self._verbose = verbose
        self._system_prompt = system_prompt
        self._kwargs = kwargs
        
        # We'll initialize engines lazily if needed, but ReACT is the heavy lifter
        self._react_engine_cache = None

    @property
    def chat_history(self) -> List[ChatMessage]:
        # This will be tricky if we jump between engines, but let's assume a global history
        return [] # Placeholder

    def reset(self) -> None:
        pass

    async def _get_available_tools_metadata(self) -> List[Dict[str, Any]]:
        """Fetch metadata for tools installed by the user."""
        try:
            # Load actual tool instances to get accurate metadata including flattened tools
            tools = self._tool_registry.load_tools(self._user_id)
            return [
                {
                    "name": t.metadata.name,
                    "description": t.metadata.description,
                    "id": getattr(t, 'id', t.metadata.name)
                } for t in tools
            ]
        except Exception as e:
            logger.error(f"Failed to fetch tools for routing: {e}")
            return []

    async def achat(
        self, 
        message: str, 
        chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        """Main entry point for agentic chat."""
        
        # 1. Get available tools
        available_tools = await self._get_available_tools_metadata()
        
        # 2. Route the query
        routing = await self._router.route(message, available_tools, chat_history)
        
        # 3. Handle based on route
        if routing.route == RouteType.PLANNER:
            logger.info("Hierarchy: Routing to PLANNER")
            # For now, let's treat Planner as a sequence of ReACT calls
            # (In the future this will be more complex)
            return await self._execute_planner_flow(message, available_tools, chat_history)
        
        elif routing.route == RouteType.REACT:
            logger.info("Hierarchy: Routing to REACT")
            return await self._execute_react_flow(message, available_tools, chat_history)
            
        else: # Default or RAG
            logger.info("Hierarchy: Routing to RAG (fallback)")
            return await self._execute_react_flow(message, [], chat_history)

    async def _execute_react_flow(
        self,
        message: str,
        tools_metadata: List[Dict[str, Any]],
        chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        """Execute a single-step ReACT reasoning loop with specific tools."""
        
        # Load tool instances for the user
        tool_instances = self._tool_registry.load_tools(self._user_id)
        
        # Filter tools if the router suggested specific ones (optional optimization)
        
        # Initialize ReACT engine with system prompt
        engine = AgenticRAGEngine(
            llm=self._llm,
            index=self._index,
            vector_store_component=self._vector_store_component,
            node_store_component=self._node_store_component,
            node_postprocessors=self._node_postprocessors,
            document_files=self._document_files,
            external_tools=tool_instances,
            system_prompt=self._system_prompt,
            verbose=self._verbose,
            **self._kwargs
        )
        
        return await engine.achat(message, chat_history)

    async def _aexecute_react_flow_streaming(
        self,
        message: str,
        tools_metadata: List[Dict[str, Any]],
        chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        """Execute a single-step ReACT reasoning loop with specific tools (streaming)."""
        tool_instances = self._tool_registry.load_tools(self._user_id)
        
        engine = AgenticRAGEngine(
            llm=self._llm,
            index=self._index,
            vector_store_component=self._vector_store_component,
            node_store_component=self._node_store_component,
            node_postprocessors=self._node_postprocessors,
            document_files=self._document_files,
            external_tools=tool_instances,
            system_prompt=self._system_prompt,
            verbose=self._verbose,
            **self._kwargs
        )
        
        return await engine.astream_chat(message, chat_history)

    async def _execute_planner_flow(
        self,
        message: str,
        tools_metadata: List[Dict[str, Any]],
        chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        """Execute a multi-step plan."""
        
        # 1. Create plan
        plan = await self._planner.create_plan(message, tools_metadata)
        
        # 2. Execute each task
        # This is a simplified execution loop
        results = []
        context_from_tasks = ""
        
        for task in plan.tasks:
            logger.info(f"Hierarchy: Executing SubTask {task.id}: {task.description}")
            
            task_query = f"Task: {task.description}\nContext from previous steps: {context_from_tasks}"
            
            # For each subtask, we use the ReACT agent
            # We filter tools to only the ones hinted if possible
            task_response = await self._execute_react_flow(task_query, tools_metadata, chat_history)
            
            results.append(f"Task {task.id} Result: {task_response.response}")
            context_from_tasks += f"\nResult of '{task.description}': {task_response.response}\n"

        # 3. Final Synthesis
        synthesis_query = f"Original Goal: {message}\n\nExecution Results:\n{context_from_tasks}\n\nPlease provide a final comprehensive answer based on these results."
        
        # We can use the ReACT engine one last time for synthesis or just the LLM directly
        final_response = await self._execute_react_flow(synthesis_query, tools_metadata, chat_history)
        
        return final_response

    # Implement other methods as required by BaseChatEngine
    def chat(self, message: str, chat_history: Optional[List[ChatMessage]] = None) -> AgentChatResponse:
        # This sync method is problematic in async loops. 
        # In a real app, you'd use a dedicated loop or avoid sync calls.
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # This is still a risk, butachat is the primary path now
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(self.achat(message, chat_history))
            else:
                return loop.run_until_complete(self.achat(message, chat_history))
        except RuntimeError:
            return asyncio.run(self.achat(message, chat_history))

    def stream_chat(self, message: str, chat_history: Optional[List[ChatMessage]] = None) -> StreamingAgentChatResponse:
        response = self.chat(message, chat_history)
        
        def gen():
            yield response.response
            
        return StreamingAgentChatResponse(
            chat_stream=gen(),
            source_nodes=response.source_nodes,
            sources=response.sources
        )

    async def astream_chat(self, message: str, chat_history: Optional[List[ChatMessage]] = None) -> StreamingAgentChatResponse:
        handler = AgentStreamHandler()
        self.callback_manager.add_handler(handler)
        
        # Store source nodes to populate after streaming
        collected_source_nodes = []
        
        async def combined_gen():
            nonlocal collected_source_nodes
            try:
                # 1. Routing and Preparation Step (run in background to allow event fetching)
                async def prepare():
                    try:
                        handler._put_event({"thought": "🤔 Routing query to specialized engine..."})
                        available_tools = await self._get_available_tools_metadata()
                        routing = await self._router.route(message, available_tools, chat_history)
                        
                        final_stream_response = None
                        
                        if routing.route == RouteType.PLANNER:
                            handler._put_event({"thought": "📋 Complexity detected. Initializing multi-step plan..."})
                            plan = await self._planner.create_plan(message, available_tools)
                            
                            context_from_tasks = ""
                            for task in plan.tasks:
                                handler._put_event({"thought": f"🚀 Executing SubTask {task.id}: {task.description}"})
                                task_query = f"Task: {task.description}\nContext: {context_from_tasks}"
                                
                                # Streaming subtask execution is hard to nest, so we use achat but capturing events
                                task_response = await self._execute_react_flow(task_query, available_tools, chat_history)
                                context_from_tasks += f"\nResult: {task_response.response}\n"
                            
                            synthesis_query = f"Original Goal: {message}\n\nExecution Results:\n{context_from_tasks}\n\nPlease provide final answer."
                            handler._put_event({"thought": "✨ Synthesizing all results into a final response..."})
                            
                            # Final streaming response
                            final_stream_response = await self._aexecute_react_flow_streaming(synthesis_query, available_tools, chat_history)
                        
                        elif routing.route == RouteType.REACT:
                            handler._put_event({"thought": "🧠 Using ReACT reasoning to answer your query..."})
                            final_stream_response = await self._aexecute_react_flow_streaming(message, available_tools, chat_history)
                        
                        else:
                            handler._put_event({"thought": "🔎 Searching knowledge base for relevant information..."})
                            # Note: We still use the ReACT flow for RAG because it handles citations better
                            final_stream_response = await self._aexecute_react_flow_streaming(message, [], chat_history)
                        
                        return final_stream_response
                    except Exception as e:
                        logger.error(f"Error in preparation phase: {e}")
                        handler._put_event({"thought": f"❌ Error: {str(e)}"})
                        return None

                # Start preparation task
                prep_task = asyncio.create_task(prepare())
                
                # Consume events while preparing
                while not prep_task.done():
                    while not handler.queue.empty():
                        event = await handler.queue.get()
                        if isinstance(event, dict) and event.get("event") == "complete":
                            continue
                        logger.info(f"Orchestrator: Yielding routing event: {event}")
                        yield event
                    await asyncio.sleep(0.05)
                
                final_stream_response = await prep_task
                
                # Now consume the final_stream_response if preparation succeeded
                if final_stream_response:
                    # DON'T drain the queue here - the sub-engine's stream will include those events
                    # Draining here causes duplicates because the sub-engine already captured them

                    # To avoid duplicates, we'll temporarily remove OUR handler 
                    # before consuming the sub-engine's stream
                    self.callback_manager.remove_handler(handler)
                    
                    try:
                        # Extract source nodes before consuming stream
                        if hasattr(final_stream_response, 'source_nodes'):
                            collected_source_nodes.extend(final_stream_response.source_nodes or [])
                        
                        # Now stream the actual response
                        # StreamingAgentChatResponse has multiple possible generator attributes
                        # Helper to find the actual async generator among possible attributes
                        def get_async_gen(obj):
                            # Try each possible generator attribute
                            # IMPORTANT: Check chat_stream first as it's the custom wrapper from sub-engines
                            # Check if it's callable (method) - if so, call it to get the generator
                            for attr_name in ['chat_stream', 'async_response_gen', 'response_gen']:
                                val = getattr(obj, attr_name, None)
                                if val is None:
                                    continue
                                
                                # If callable, it's a method - call it to get the generator
                                if callable(val):
                                    try:
                                        logger.info(f"Orchestrator: Calling {attr_name}() to get generator")
                                        actual_gen = val()
                                        if actual_gen is not None:
                                            # Verify it's an async generator
                                            import inspect
                                            if inspect.isasyncgen(actual_gen) or hasattr(actual_gen, '__aiter__'):
                                                return actual_gen, attr_name
                                            else:
                                                logger.warning(f"Orchestrator: {attr_name}() returned sync generator")
                                    except Exception as e:
                                        logger.warning(f"Orchestrator: Failed to call {attr_name}(): {e}")
                                else:
                                    # It's already a generator - check if it's async
                                    import inspect
                                    if inspect.isasyncgen(val) or hasattr(val, '__aiter__'):
                                        logger.info(f"Orchestrator: Found async generator at {attr_name}")
                                        return val, attr_name
                                    else:
                                        logger.warning(f"Orchestrator: {attr_name} is a sync generator, need async")
                            
                            return None, None

                        active_generator, generator_name = get_async_gen(final_stream_response)
                        
                        if active_generator is not None:
                            logger.info(f"Orchestrator: Using {generator_name} for streaming")
                            logger.info(f"Orchestrator: Generator type: {type(active_generator)}")
                            
                            chunk_count = 0
                            async for chunk in active_generator:
                                chunk_count += 1
                                if chunk_count == 1:
                                    logger.info(f"Orchestrator: First chunk from sub-engine: {type(chunk)}, content: {chunk}")
                                elif chunk_count <= 5:
                                    logger.info(f"Orchestrator: Chunk {chunk_count}: {chunk}")
                                yield chunk
                            
                            logger.info(f"Orchestrator: Streamed {chunk_count} chunks from sub-engine")
                        else:
                            logger.error(f"Orchestrator: Could not find any valid async generator in {type(final_stream_response)}")
                            logger.error(f"Available attributes: {[attr for attr in dir(final_stream_response) if not attr.startswith('_')]}")
                            yield "Error: No streaming generator available"
                    finally:
                        # Re-add if needed for finally block, but usually fine
                        pass
                
            except Exception as e:
                logger.error(f"Error in Hierarchical agent stream: {e}", exc_info=True)
                yield {"thought": f"❌ Error: {str(e)}"}
            finally:
                # Ensure handler is removed if it wasn't already
                if handler in self.callback_manager.handlers:
                    self.callback_manager.remove_handler(handler)
        
        # Create the response object with a reference to source nodes that will be populated
        response = StreamingAgentChatResponse(
            chat_stream=combined_gen(),
            source_nodes=[],  # Will be populated during streaming
            sources=[]
        )
        
        # Attach the collected_source_nodes so they can be accessed after streaming
        response._collected_source_nodes = collected_source_nodes
        
        return response
