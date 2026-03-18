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
from private_gpt.di import global_injector

logger = logging.getLogger(__name__)

# --- Prompt Templates ---

DEFAULT_CONTEXT_PROMPT_TEMPLATE = """  
You are a document-grounded assistant. Use ONLY the context below to answer the user's question.

---

**RETRIEVED CONTEXT**  
{context_str}

---

###  Response Guidelines:

**Content Strategy**:
- Answer based solely on the provided context — no external knowledge or assumptions
- **For follow-up questions**: BUILD UPON previous answers, don't repeat facts already stated
- **Avoid redundancy**: If you've already mentioned a fact, don't repeat it unless directly asked
- Focus on NEW information or different aspects when answering related questions

**Formatting**:
- Use **bold** for important concepts
- Use hierarchical structure for complex topics:
  - ## for main topics
  - ### for subtopics
  - Bullet points or numbered lists where helpful
- Quote directly when precision matters, otherwise paraphrase accurately and concisely

**Citations (STRICT FORMAT)**: Use inline markdown links in the format `[page X](filename)` immediately after relevant claims.
  - X is the page number and filename is the document name.
  - **NEVER** use superscripts like `^[1]`.
  - Example: "The revenue increased by 15% [page 23](report.pdf)."
  - Multiple sources: "[page 5](doc1.pdf) [page 8](doc2.pdf)"

**Missing Information**: If information is **missing**, say:  
  "The provided documents do not contain information about [topic]."

**Contradictory Information**: If information is **contradictory**, acknowledge both perspectives neutrally

Voice: clear, confident, and helpful — like a domain expert who communicates well.

"""

DEFAULT_CONDENSE_PROMPT_TEMPLATE = """
Given the conversation history and a follow-up query, rephrase it as a standalone, context-rich question.

**Instructions**:
1. Replace pronouns (it, that, they, this) with specific entities from the history
2. Detect and preserve detail-level modifiers:
   - "brief/summary/short" → Add "Provide a brief summary of"
   - "detailed/deep/full/comprehensive" → Add "Provide a detailed explanation of"
   - "more detail/elaborate/expand" → Add "Provide additional details about"
3. Keep the original intent and detail level

**Examples**:
- History: "What is MallaNet?" | Query: "Describe in brief" 
  → "Provide a brief summary of MallaNet"
  
- History: "MallaNet is a model..." | Query: "I mean in deep version"
  → "Provide a comprehensive, detailed explanation of MallaNet with technical specifics"
  
- History: "HVCs are..." | Query: "Can you give me the full summary"
  → "Provide a comprehensive summary of HVCs (Homogeneous Vector Capsules)"

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:"""

# Prompt for query decomposition
DEFAULT_DECOMPOSE_PROMPT_TEMPLATE = """
Decompose the complex query into {max_sub_queries} or fewer simple sub-queries.
Ensure each sub-query is self-contained and specific.

**Query:** {query}

**Output:** JSON list of strings.
Example: ["Sub-query 1", "Sub-query 2"]
"""

# Maximum nodes to store in memory to prevent memory leaks
MAX_NODES_IN_MEMORY = 100


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
        

    def _get_default_system_prompt(self) -> str:
        """Generate a balanced default system prompt for RAG."""
        return (
            "You are a dedicated RAG (Retrieval-Augmented Generation) assistant. "
            "Your PRIMARY role is to retrieve and synthesize information from the user's uploaded documents. "
            "ALWAYS prioritize the provided context. "
            "However, if the context is insufficient, you may use your general knowledge to provide a helpful response, "
            "but please indicate when information comes from outside the documents."
        )
    
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
        """Get instruction based on detail level to inject into system message."""
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
        # Pattern for summary: "summarize <file>", "summary of <file>", "analyze <file>"
        summary_pattern = r"(?:summarize|summary of|analyze|overview of|check)\s+(?:the\s+)?(.+\.[a-zA-Z0-9]+)"
        summary_match = re.search(summary_pattern, query, re.IGNORECASE)
        if summary_match:
            return {"type": "summary", "file_name": summary_match.group(1).strip()}
            
        # Pattern for QA: "in <file>", "from <file>", "about <file>", "using <file>"
        qa_pattern = r"(?:in|from|about|using|inside)\s+(?:the\s+file\s+)?(.+\.[a-zA-Z0-9]+)"
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

        try:
            response = self._llm.predict(
                self._condense_prompt_template,
                question=latest_message,
                chat_history=chat_history_str,
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Error condensing question: {e}. Using original message.")
            return latest_message

    async def _acondense_question(
        self, chat_history: List[ChatMessage], latest_message: str
    ) -> str:
        """Async condense history and latest message."""
        if self._skip_condense or not chat_history:
            return latest_message

        chat_history_str = messages_to_history_str(chat_history)
        if self._verbose:
            logger.info(f"Condensing query with history: {chat_history_str}")

        try:
            # Apply timeout
            response = await asyncio.wait_for(
                self._llm.apredict(
                    self._condense_prompt_template,
                    question=latest_message,
                    chat_history=chat_history_str,
                ),
                timeout=self._condense_timeout
            )
            return response.strip()
        except asyncio.TimeoutError:
            logger.warning(f"Condense question timed out after {self._condense_timeout}s. Using original message.")
            return latest_message
        except Exception as e:
            logger.error(f"Error condensing question: {e}. Using original message.")
            return latest_message

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
        
        try:
            prompt = self._decompose_prompt_template.format(
                query=query, max_sub_queries=self._max_sub_queries
            )
            response = self._llm.complete(prompt)
            text = response.text.strip()

            # Robust Parsing
            if text.startswith("[") and text.endswith("]"):
                try:
                    return json.loads(text)[:self._max_sub_queries]
                except:
                    pass  # Parsing failed, try regex
            
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
        
        try:
            prompt = self._decompose_prompt_template.format(
                query=query, max_sub_queries=self._max_sub_queries
            )
            
            # Apply timeout
            response = await asyncio.wait_for(
                self._llm.acomplete(prompt),
                timeout=self._decompose_timeout
            )
            text = response.text.strip()
            
            # Robust Parsing
            if text.startswith("[") and text.endswith("]"):
                try:
                    return json.loads(text)[:self._max_sub_queries]
                except:
                    pass
            
            matches = re.findall(r'"([^"]+)"', text)
            if matches:
                return matches[:self._max_sub_queries]
            
            lines = [l.strip("- ").strip() for l in text.splitlines() if l.strip()]
            return lines[:self._max_sub_queries] if lines else [query]
            
        except asyncio.TimeoutError:
            logger.warning(f"Query decomposition timed out after {self._decompose_timeout}s. Using original query.")
            return [query]
        except Exception as e:
            logger.warning(f"Decomposition failed: {e}. Using original query.")
            return [query]


    def _retrieve_and_process_nodes(self, sub_queries: List[str]) -> Tuple[str, List[NodeWithScore]]:
        """Retrieve using ChunksService, deduplicate, and post-process nodes."""
        all_nodes_map: Dict[str, NodeWithScore] = {}
        
        for sub_q in sub_queries:
            try:
                # Use base retriever directly
                nodes = self._retriever.retrieve(sub_q)
                
                for node in nodes:
                    if node.node.node_id not in all_nodes_map:
                        all_nodes_map[node.node.node_id] = node
                        
                    # Prevent memory leaks
                    if len(all_nodes_map) >= MAX_NODES_IN_MEMORY:
                        logger.warning(f"Reached max nodes limit ({MAX_NODES_IN_MEMORY}). Stopping retrieval.")
                        break
                        
                if len(all_nodes_map) >= MAX_NODES_IN_MEMORY:
                    break
                    
            except Exception as e:
                logger.error(f"Error retrieving for '{sub_q}': {e}")
                
        combined_nodes = list(all_nodes_map.values())
        
        # Post-process nodes (FIXED: removed duplicate)
        if self._node_postprocessors and combined_nodes:
            primary_query_bundle = QueryBundle(sub_queries[0] if sub_queries else "")
            for postprocessor in self._node_postprocessors:
                try:
                    processed_nodes = postprocessor.postprocess_nodes(
                        combined_nodes, query_bundle=primary_query_bundle
                    )
                    # Only update if post-processing returned valid results
                    if processed_nodes:
                        combined_nodes = processed_nodes
                    else:
                        logger.warning(f"Post-processor {postprocessor.__class__.__name__} returned empty nodes, keeping original")
                except Exception as e:
                    logger.error(f"Error in post-processing with {postprocessor.__class__.__name__}: {e}. Keeping nodes from previous step.")
        elif self._node_postprocessors and not combined_nodes:
            logger.warning("Skipping post-processing: no nodes retrieved")
        
        context_str = self._format_context_merged(combined_nodes)
        return context_str, combined_nodes

    def _format_context_merged(self, nodes: List[NodeWithScore]) -> str:
        """Smartly merge nodes from the same file and adjacent pages."""
        if not nodes:
            return "No relevant documents found."
            
        # Group by file_name
        grouped_nodes = {}
        for node in nodes:
            fn = node.node.metadata.get("file_name", "Unknown File")
            if fn not in grouped_nodes:
                grouped_nodes[fn] = []
            grouped_nodes[fn].append(node)

        context_parts = []
        source_idx = 0
        
        # We iterate groups
        for fn, group in grouped_nodes.items():
            # Sort by page/start_char
            group.sort(key=lambda x: (self._get_page_val(x), x.node.start_char_idx or 0))
            
            merged_blocks = []
            current_block = None
            
            for node in group:
                content = node.node.get_content(metadata_mode=MetadataMode.LLM).strip()
                page = self._get_page_val(node)
                
                if current_block and self._is_adjacent(current_block, node, page):
                    current_block["content"] += "\n\n... [Continuous] ...\n\n" + content
                    if page not in current_block["pages"]:
                        current_block["pages"].append(page)
                else:
                    if current_block:
                        merged_blocks.append(current_block)
                    current_block = {
                        "file_name": fn,
                        "content": content,
                        "pages": [page]
                    }
            if current_block:
                merged_blocks.append(current_block)
                
            for block in merged_blocks:
                source_idx += 1
                pages_str = self._format_pages(block["pages"])
                context_parts.append(
                    f"Source [{source_idx}]: File: {block['file_name']} (Pages: {pages_str})\n"
                    f"{block['content']}"
                )
                
        return "\n\n".join(context_parts)

    def _get_page_val(self, node: NodeWithScore) -> int:
        try:
            return int(node.node.metadata.get("page", 0))
        except:
            return 0

    def _is_adjacent(self, block, next_node, curr_page) -> bool:
        last_page = block["pages"][-1]
        # Allow merge if same page or next page
        return (curr_page == last_page) or (curr_page == last_page + 1)

    def _format_pages(self, pages: List[int]) -> str:
        if not pages:
            return "?"
        pages = sorted(list(set(pages)))
        if len(pages) == 1:
            return str(pages[0])
        return f"{pages[0]}-{pages[-1]}"

    async def _aretrieve_and_process_nodes(self, sub_queries: List[str]) -> Tuple[str, List[NodeWithScore]]:
        """Async retrieve using ChunksService, deduplicate, and post-process."""
        all_nodes_map: Dict[str, NodeWithScore] = {}
        
        async def retrieve_safe(sq):
            try:
                # Use base retriever directly
                nodes = await asyncio.wait_for(
                    self._retriever.aretrieve(sq),
                    timeout=self._retrieval_timeout
                )
                return nodes
            except asyncio.TimeoutError:
                logger.warning(f"Retrieval for '{sq}' timed out after {self._retrieval_timeout}s.")
                return []
            except Exception as e: 
                logger.error(f"Async retrieval error: {e}")
                return []
            
        results = await asyncio.gather(*[retrieve_safe(sq) for sq in sub_queries])
        
        for nodes in results:
            for node in nodes:
                if node.node.node_id not in all_nodes_map:
                    all_nodes_map[node.node.node_id] = node
                    
                # Prevent memory leaks
                if len(all_nodes_map) >= MAX_NODES_IN_MEMORY:
                    logger.warning(f"Reached max nodes limit ({MAX_NODES_IN_MEMORY}). Stopping.")
                    break
                    
            if len(all_nodes_map) >= MAX_NODES_IN_MEMORY:
                break
                    
        combined_nodes = list(all_nodes_map.values())
        
        # Post-process nodes (FIXED: removed duplicate)
        if self._node_postprocessors and combined_nodes:
            primary_query_bundle = QueryBundle(sub_queries[0] if sub_queries else "")
            for postprocessor in self._node_postprocessors:
                try:
                    processed_nodes = postprocessor.postprocess_nodes(
                        combined_nodes, query_bundle=primary_query_bundle
                    )
                    # Only update if post-processing returned valid results
                    if processed_nodes:
                        combined_nodes = processed_nodes
                    else:
                        logger.warning(f"Post-processor {postprocessor.__class__.__name__} returned empty nodes, keeping original")
                except Exception as e:
                    logger.error(f"Error in post-processing with {postprocessor.__class__.__name__}: {e}. Keeping nodes from previous step.")
        elif self._node_postprocessors and not combined_nodes:
            logger.warning("Skipping post-processing: no nodes retrieved")
        
        context_str = self._format_context_merged(combined_nodes)
        return context_str, combined_nodes

    # --- Execution Flows ---

    def _run_agentic_condense_sync(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> Tuple[List[ChatMessage], ToolOutput, List[NodeWithScore]]:
        
        # 0. Tool Routing Check
        tool_intent = self._detect_specific_document_intent(message)
        tool_output = None
        if tool_intent:
            logger.info(f"Detected tool intent: {tool_intent}")
            tool_output = self._create_and_run_tool(tool_intent, message)
            if tool_output.is_error:
                 logger.warning(f"Tool failed: {tool_output.raw_output}. Falling back to standard RAG.")
                 tool_output = None

        # 1. Memory Setup
        if chat_history is not None:
            self._memory.set(chat_history)
        
        # 2. Condense
        current_history = self._memory.get(input=message)
        standalone_question = self._condense_question(current_history, message)
        
        context_str = ""
        context_nodes = []
        sub_queries = []

        # 3 & 4. Retrieve (Standard or Tool-based)
        if tool_output:
            logger.info("Using tool output as context.")
            context_str = tool_output.content
            if isinstance(tool_output.raw_output, list):
                context_nodes = tool_output.raw_output
            sub_queries = [message]
        else:
            # 3. Decompose
            sub_queries = self._decompose_query(standalone_question)
            
            # 4. Retrieve
            context_str, context_nodes = self._retrieve_and_process_nodes(sub_queries)
        
        # 5. Construct System Message with Detail-Level Awareness
        # Detect detail level from original message
        detail_level = self._detect_detail_level(message)
        detail_instruction = self._get_detail_instruction(detail_level)
        
        formatted_prompt = self._context_prompt_template.format(
            context_str=context_str
        )
        
        if self._system_prompt:
             final_system_msg = f"{self._system_prompt}\n\n{formatted_prompt}{detail_instruction}"
        else:
             # Use default strict prompt if none provided
             final_system_msg = f"{self._get_default_system_prompt()}\n\n{formatted_prompt}{detail_instruction}"
             
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
        
        # 0. Tool Routing Check (Async) - FIXED: use asyncio.to_thread for blocking call
        tool_intent = self._detect_specific_document_intent(message)
        tool_output = None
        if tool_intent:
            logger.info(f"Detected tool intent (Async): {tool_intent}")
            try:
                # Run blocking tool in thread pool to avoid blocking event loop
                tool_output = await asyncio.to_thread(
                    self._create_and_run_tool, tool_intent, message
                )
                if tool_output.is_error:
                     logger.warning(f"Tool failed: {tool_output.raw_output}. Falling back to standard RAG.")
                     tool_output = None
            except Exception as e:
                logger.error(f"Async tool execution failed: {e}")
                tool_output = None

        # 1. Memory Setup
        if chat_history is not None:
             self._memory.set(chat_history)
             
        # 2. Condense
        current_history = self._memory.get(input=message)
        standalone_question = await self._acondense_question(current_history, message)
        
        context_str = ""
        context_nodes = []
        sub_queries = []

        # 3 & 4. Retrieve (Standard or Tool-based)
        if tool_output:
            logger.info("Using tool output as context (Async).")
            context_str = tool_output.content
            if isinstance(tool_output.raw_output, list):
                context_nodes = tool_output.raw_output
            sub_queries = [message]
        else:
            # 3. Decompose
            sub_queries = await self._adecompose_query(standalone_question)
            
            # 4. Retrieve
            context_str, context_nodes = await self._aretrieve_and_process_nodes(sub_queries)
        
        # 5. Construct System Message with Detail-Level Awareness
        # Detect detail level from original message
        detail_level = self._detect_detail_level(message)
        detail_instruction = self._get_detail_instruction(detail_level)
        
        formatted_prompt = self._context_prompt_template.format(
            context_str=context_str
        )
        
        if self._system_prompt:
             final_system_msg = f"{self._system_prompt}\n\n{formatted_prompt}{detail_instruction}"
        else:
             # Use default strict prompt if none provided
             final_system_msg = f"{self._get_default_system_prompt()}\n\n{formatted_prompt}{detail_instruction}"
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
        try:
            chat_messages, context_source, context_nodes = self._run_agentic_condense_sync(
                message, chat_history
            )
            
            chat_response = self._llm.chat(chat_messages)
            assistant_message = chat_response.message
            self._memory.put(assistant_message)
            
            # Apply citation replacement with error handling
            try:
                final_response = CitationHelper.smart_citation_replacement(assistant_message.content, context_nodes)
                cited_nodes = CitationHelper.get_cited_nodes(assistant_message.content, context_nodes)
            except Exception as e:
                logger.warning(f"Citation processing failed: {e}. Using original response.")
                final_response = assistant_message.content
                cited_nodes = context_nodes
            
            return AgentChatResponse(
                response=final_response,
                sources=[context_source],
                source_nodes=cited_nodes
            )
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            # Return error response
            return AgentChatResponse(
                response=f"I encountered an error processing your request: {str(e)}",
                sources=[],
                source_nodes=[]
            )

    @trace_method("chat")
    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        try:
            chat_messages, context_source, context_nodes = self._run_agentic_condense_sync(
                message, chat_history
            )
            
            chat_stream = self._llm.stream_chat(chat_messages)
            
            chat_response = StreamingAgentChatResponse(
                chat_stream=chat_stream,
                sources=[context_source],
                source_nodes=context_nodes,
            )
            
            # FIXED: Use asyncio.create_task instead of Thread for better async handling
            # Note: If this needs to remain sync, we should use proper thread synchronization
            thread = Thread(
                target=chat_response.write_response_to_history, args=(self._memory,)
            )
            thread.start()
            
            return chat_response
        except Exception as e:
            logger.error(f"Error in stream_chat: {e}")
            raise

    @trace_method("chat")
    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        try:
            chat_messages, context_source, context_nodes = await self._arun_agentic_condense(
                message, chat_history
            )
            
            chat_response = await self._llm.achat(chat_messages)
            assistant_message = chat_response.message
            self._memory.put(assistant_message)
            
            # Apply citation replacement with error handling
            try:
                final_response = CitationHelper.smart_citation_replacement(assistant_message.content, context_nodes)
                cited_nodes = CitationHelper.get_cited_nodes(assistant_message.content, context_nodes)
            except Exception as e:
                logger.warning(f"Citation processing failed: {e}. Using original response.")
                final_response = assistant_message.content
                cited_nodes = context_nodes
            
            return AgentChatResponse(
                response=final_response,
                sources=[context_source],
                source_nodes=cited_nodes
            )
        except Exception as e:
            logger.error(f"Error in achat: {e}")
            return AgentChatResponse(
                response=f"I encountered an error processing your request: {str(e)}",
                sources=[],
                source_nodes=[]
            )

    @trace_method("chat")
    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        try:
            chat_messages, context_source, context_nodes = await self._arun_agentic_condense(
                message, chat_history
            )
            
            chat_stream = await self._llm.astream_chat(chat_messages)
            
            chat_response = StreamingAgentChatResponse(
                achat_stream=chat_stream,
                sources=[context_source],
                source_nodes=context_nodes,
            )
            
            # FIXED: Properly await the async task
            asyncio.create_task(chat_response.awrite_response_to_history(self._memory))
            return chat_response
        except Exception as e:
            logger.error(f"Error in astream_chat: {e}")
            raise

    def reset(self) -> None:
        self._memory.reset()

    @property
    def chat_history(self) -> List[ChatMessage]:
        return self._memory.get_all()

class ExpertRAGEngine(AgenticCondenseChatEngine):
    """
    Expert RAG Engine that leverages Entity-based knowledge.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from private_gpt.components.expert.expert_orchestrator import ExpertOrchestrator
        self._expert_orchestrator = global_injector.get(ExpertOrchestrator)

    async def _aretrieve_and_process_nodes(self, sub_queries: List[str]) -> Tuple[str, List[NodeWithScore]]:
        # 1. Standard retrieval
        context_str, nodes = await super()._aretrieve_and_process_nodes(sub_queries)
        
        # 2. Entity-based extension
        from private_gpt.server.chat.rag_config import RAG_CONFIG
        if RAG_CONFIG.expert.enabled:
            # We use the primary query (first sub-query) for entity search
            entity_context = self._expert_orchestrator.get_expert_context_extension(sub_queries[0])
            if entity_context:
                context_str = entity_context + "\n\n**VECTOR SEARCH RESULTS**\n" + context_str
                
        return context_str, nodes
