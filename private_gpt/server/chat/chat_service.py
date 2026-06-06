import logging
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from injector import inject, singleton
from llama_index.core.chat_engine import (
    SimpleChatEngine,
    CondensePlusContextChatEngine,
    ContextChatEngine,
)
from llama_index.core.chat_engine.types import (
    BaseChatEngine,
)
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.postprocessor import SimilarityPostprocessor, rankGPT_rerank
from llama_index.core.storage import StorageContext
from llama_index.core.types import TokenGen, TokenAsyncGen
from private_gpt.server.cache.faq_service import FAQService
from private_gpt.utils.chat_enums import ChatMode
from private_gpt.components.retriever.metadata_retriever import MetadataFilterRetriever

from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine

from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.server.ingest.model import Chunk
from private_gpt.settings.settings import Settings

from private_gpt.paths import models_path

from llama_index.core.postprocessor import LongContextReorder
from private_gpt.server.chat.agentic_rag import AgenticCondenseChatEngine
from private_gpt.server.chat.agentic_tool import AgenticRAGEngine
from private_gpt.server.chat.search_tool import SearchRAGEngine
from private_gpt.components.postprocessor.PrevNext import (
    DocumentAwarePrevNextPostprocessor,
)

from private_gpt.server.agents.orchestrator_engine import HierarchicalAgentEngine
from private_gpt.server.tools.tool_registry import ToolRegistry
from private_gpt.server.tools.document_catalog_tool import DocumentCatalogTool

from private_gpt.server.cache.cache_service import CacheService
from private_gpt.users.services.prompt_service import prompt_service
from sqlalchemy.orm import Session
from private_gpt.server.chat.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    RETRIEVAL_SYSTEM_PROMPT,
    AGENTIC_SYSTEM_PROMPT,
    ENHANCED_QA_TEMPLATE,
    resolve_system_prompt,
)
from private_gpt.server.utils.notifications import NotificationService
import re


logger = logging.getLogger(__name__)


class Completion(BaseModel):
    cache_id: Optional[str] = None
    response: str
    sources: list[Chunk] | None = None


class CompletionGen(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    response: TokenGen | TokenAsyncGen
    sources: list[Chunk] | None = None


class TitleGeneration(BaseModel):
    title: str


reranker_path = models_path / "reranker"

CONTEXT_PROMPT_TEMPLATE = """  
You are a document-grounded assistant. Use ONLY the context below to answer the user's question.

---

**RETRIEVED CONTEXT**  
{context_str}

---

###  Response Guidelines:

- Answer based solely on the provided context — no external knowledge or assumptions
- Format using Markdown:
  - Use **bold** for important concepts
  - Bullet points or lists where helpful
  - Headings (##, ###) for structure in longer answers
- Quote directly when precision matters, otherwise paraphrase accurately and concisely
- **Citations (STRICT FORMAT)**: Use inline markdown links in the format `[page X](filename)` immediately after relevant claims.
  - X is the page number and filename is the document name.
  - **NEVER** use superscripts like `^[1]`.
  - Example: "The revenue increased by 15% [page 23](report.pdf)."
  - Multiple sources: "[page 5](doc1.pdf) [page 8](doc2.pdf)"
- If information is **missing**, say:  
  "The provided documents do not contain information about [topic]."
- If information is **contradictory**, acknowledge both perspectives neutrally
- Be concise, informative, and natural — no apologies unless truly warranted
Voice: clear, confident, and helpful — like a domain expert who communicates well.


"""

# CONDENSE_PROMPT_TEMPLATE = """
# You transform conversational follow-up questions into comprehensive, standalone queries optimized for document retrieval.

# **Chat History:**
# {chat_history}

# **Follow-Up Question:**
# {question}

# **Transformation Guidelines:**
# 1. Create a complete, self-contained question that incorporates all necessary context from the chat history
# 2. Replace all pronouns (it, they, these, etc.) with their explicit referents
# 3. Preserve all entities, dates, time periods, specific terminology, and contextual details
# 4. Include implied constraints or parameters from earlier conversation
# 5. Maintain the original intent while optimizing for accurate document retrieval
# 6. Write as a natural, fluent question — not as keywords or a search query

# **Output Instructions:**
# - Return ONLY the rewritten standalone question without explanation or commentary
# - If the original question is already standalone or if chat history is empty, optimize only for clarity and specificity
# - Ensure the output is clean and ready for direct use in retrieval

# The ideal rewritten question should retrieve all relevant document passages without requiring prior chat context.

# Standalone question:
# """

CONDENSE_PROMPT_TEMPLATE = """
You transform conversational follow-up questions into comprehensive, standalone questions optimized for high-recall document retrieval.  
You may internally leverage Hypothetical Document Expansion (HyDE): reason about what an ideal answer would likely contain, and use that reasoning to enrich the rewritten question with missing but implied context. Do NOT output a hypothetical answer.

**Chat History:**  
{chat_history}

**Follow-Up Question:**  
{question}

**Transformation Guidelines:**
1. Produce a single, fully self-contained natural-language question that requires no prior chat context
2. Resolve all pronouns and vague references by replacing them with explicit entities or concepts
3. Preserve and restate all relevant entities, systems, technologies, dates, constraints, and assumptions from the chat history
4. Infer and include implicit context, scope, or constraints that an expert answer would reasonably address (HyDE principle)
5. Expand underspecified questions to reflect the full informational intent, without introducing new facts
6. Optimize the question to maximize retrieval of all relevant document passages, not just a narrow answer
7. Maintain the original user intent and tone, while improving clarity, specificity, and completeness

**Output Instructions:**
- Output ONLY the rewritten standalone question
- Do NOT include explanations, analysis, or hypothetical answers
- If the question is already standalone or chat history is empty, refine only for clarity and retrieval specificity
- Ensure the result is fluent, precise, and suitable for direct use in a retrieval pipeline

Standalone question:
"""


@dataclass
class ChatEngineInput:
    system_message: ChatMessage | None = None
    last_message: ChatMessage | None = None
    chat_history: list[ChatMessage] | None = None
    last_image: str | None = None  # Add image support

    @classmethod
    def from_messages(cls, messages: list[ChatMessage]) -> "ChatEngineInput":
        system_message = (
            messages[0]
            if len(messages) > 0 and messages[0].role == MessageRole.SYSTEM
            else None
        )
        last_message = (
            messages[-1]
            if len(messages) > 0 and messages[-1].role == MessageRole.USER
            else None
        )
        last_image = getattr(messages[-1], "image", None) if last_message else None
        if system_message:
            messages.pop(0)
        if last_message:
            messages.pop(-1)
        chat_history = messages if len(messages) > 0 else None

        return cls(
            system_message=system_message,
            last_message=last_message,
            chat_history=chat_history,
            last_image=last_image,
        )


@singleton
class ChatService:
    settings: Settings

    @inject
    def __init__(
        self,
        settings: Settings,
        llm_component: LLMComponent,
        vector_store_component: VectorStoreComponent,
        embedding_component: EmbeddingComponent,
        node_store_component: NodeStoreComponent,
    ) -> None:
        self.settings = settings
        self.llm_component = llm_component
        self.embedding_component = embedding_component
        self.vector_store_component = vector_store_component
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store_component.vector_store,
            docstore=node_store_component.doc_store,
            index_store=node_store_component.index_store,
        )
        self.index = VectorStoreIndex.from_vector_store(
            vector_store_component.vector_store,
            storage_context=self.storage_context,
            llm=llm_component.llm,
            embed_model=embedding_component.embedding_model,
            show_progress=False,
        )
        self.node_store = node_store_component

        # Track when the index was last built for efficient refreshing
        import time

        self._index_last_built: float = time.time()
        self._index_check_interval: float = 3.0  # seconds
        self.document_catalog_tool = DocumentCatalogTool()

    def _get_index_mtime(self) -> float:
        """Get the latest modification time of the docstore/index files."""
        from private_gpt.paths import local_data_path

        max_mtime = 0.0
        for fname in ["docstore.json", "index_store.json"]:
            fpath = local_data_path / fname
            if fpath.exists():
                max_mtime = max(max_mtime, fpath.stat().st_mtime)
        return max_mtime

    def _refresh_index(self) -> None:
        """Refresh index only when backing files have changed.

        Prevents unnecessary rebuilds while ensuring cross-worker consistency.
        """
        import time

        now = time.time()

        # Throttle: don't even check more than once every N seconds
        if now - self._index_last_built < self._index_check_interval:
            return

        latest_mtime = self._get_index_mtime()

        # Only rebuild if files changed since we last built
        if latest_mtime <= self._index_last_built:
            # Update check time even if not rebuilt to maintain throttle
            self._index_last_built = now
            return

        logger.info("Detected index file change, rebuilding VectorStoreIndex...")

        from llama_index.core.storage.docstore import SimpleDocumentStore
        from llama_index.core.storage.index_store import SimpleIndexStore
        from private_gpt.paths import local_data_path

        try:
            # 1. Hot-reload Docstore if it's Simple (file-based)
            if isinstance(self.storage_context.docstore, SimpleDocumentStore):
                self.storage_context.docstore = SimpleDocumentStore.from_persist_dir(
                    persist_dir=str(local_data_path)
                )

            # 2. Hot-reload Indexstore if it's Simple (file-based)
            if isinstance(self.storage_context.index_store, SimpleIndexStore):
                self.storage_context.index_store = SimpleIndexStore.from_persist_dir(
                    persist_dir=str(local_data_path)
                )

            # 3. Re-instantiate the VectorStoreIndex
            # to ensure it uses the newly loaded metadata/nodes.
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store_component.vector_store,
                storage_context=self.storage_context,
                llm=self.llm_component.llm,
                embed_model=self.embedding_component.embedding_model,
                show_progress=False,
            )
            self._index_last_built = now
            logger.info("Index successfully rebuilt after file change.")
        except Exception as e:
            logger.error(f"Failed to rebuild index: {e}", exc_info=True)

    def _should_trigger_notification(
        self, sources: list[Chunk] | None, response_text: str
    ) -> bool:
        """
        Check if the system failed to retrieve info AND the LLM confirmed it.

        Using Hybrid Logic:
        1. Sources Check: If sources exist, we found info -> No Email.
        2. Text Check: If sources are empty (could be greeting or failure),
           verify the LLM is explicitly stating that info is missing.
        """
        if sources:
            return False

        # If no sources, check if it's a "failure" response (not just a greeting)
        patterns = [
            r"following question detail is not provided in docs",
            r"provided documents do not contain information",
            r"cannot find information about this in the provided documents",
            r"context does not contain",
            r"answer is not in the context provided",
        ]
        return any(re.search(p, response_text, re.IGNORECASE) for p in patterns)

    def _get_qa_template(self, db: Session, user_id: int | None, mode: str) -> str:
        """Document-grounded QA template with strict context usage and markdown citations."""
        if db and user_id:
            qa_prompt = prompt_service.get_resolved_prompt(db, user_id, mode, "qa")
            if qa_prompt:
                return qa_prompt

        return ENHANCED_QA_TEMPLATE

    def _check_faq_cache(
        self, cache_service: CacheService, question: str
    ) -> dict | None:
        """Check if the question matches a frequently asked question in cache using vector similarity.

        For general users, we prioritize FAQ responses to ensure consistency across all users.
        We use a more aggressive matching strategy to catch similar questions.

        Args:
            question: The user's question

        Returns:
            Cached answer if found, None otherwise
        """
        logger.info(f"Checking FAQ cache for question: '{question}'")
        # Check if cache service is connected
        if not cache_service.is_connected:
            logger.info("Cache service not connected, skipping FAQ cache check")
            return None
        try:
            # Search for similar questions in FAQ cache using vector similarity
            # For general users, we use a lower threshold to catch more matches
            # and increase the limit to get more potential matches
            logger.info("Calling FAQ cache search with threshold=0.9, limit=3")
            search_results = cache_service.search_faqs(
                question, limit=3, similarity_threshold=0.9
            )
            logger.info(f"FAQ cache search returned {len(search_results)} results")

            if search_results:
                best_match = search_results[0]
                faq_id = getattr(getattr(best_match, "faq", None), "id", None)
                similarity_info = getattr(best_match, "similarity", "unknown")
                logger.info(
                    f"FAQ match found - Question: {best_match.faq.question[:50]}... Similarity: {similarity_info}"
                )

                answer_dict = best_match.faq.answer
                if isinstance(answer_dict, dict):
                    content = answer_dict.get("content", "")
                    sources = answer_dict.get("sources", [])
                    logger.info(f"Returning FAQ answer: {content[:100]}...")
                    return {"id": str(faq_id), "content": content, "sources": sources}
                else:
                    # backward compatibility if stored as plain string
                    logger.info(
                        f"Returning FAQ answer (string): {str(answer_dict)[:100]}..."
                    )
                    return {
                        "id": str(faq_id),
                        "content": str(answer_dict),
                        "sources": [],
                    }
            else:
                logger.info({"content": "No FAQ match found"})

        except Exception as e:
            logger.error(f"Error checking FAQ cache: {e}", exc_info=True)
        return None

    def _answer_document_catalog_query(
        self,
        db: Session | None,
        user_id: int | None,
        question: str,
        conversation_context: list[str] | None = None,
    ) -> str | None:
        if not db or not user_id or not question:
            return None

        try:
            result = self.document_catalog_tool.answer(
                db=db,
                user_id=user_id,
                question=question,
                conversation_context=conversation_context,
            )
            return result.answer if result else None
        except Exception as e:
            logger.error("Document catalog tool failed", exc_info=True)
            return (
                "I could not retrieve the document inventory right now. "
                "Please try again."
            )

    @staticmethod
    def _message_content_to_text(content: object) -> str:
        if isinstance(content, dict):
            return str(content.get("text", ""))
        if isinstance(content, str):
            return content
        return str(content or "")

    def _conversation_context_text(self, messages: list[ChatMessage]) -> list[str]:
        context = []
        for message in messages[:-1]:
            content = self._message_content_to_text(message.content)
            if content:
                context.append(content)
        return context

    async def _chat_engine(
        self,
        use_context: ChatMode.CHAT.value,
        system_prompt: str | None = None,
        context_filter: ContextFilter | None = None,
        file_list: List[str] = None,
        chat_history: list[ChatMessage] | None = None,
        user_id: int | None = None,
        db: Session | None = None,
    ) -> BaseChatEngine:
        settings = self.settings
        logger.info(f"Chat mode: {use_context}")

        # Resolve dynamic prompts
        resolved_system_prompt = system_prompt
        resolved_qa_template = None
        resolved_condense_prompt = None
        resolved_decompose_prompt = None
        resolved_context_prompt = None

        if db and user_id:
            # System Prompt
            db_system_prompt = prompt_service.get_resolved_prompt(
                db, user_id, use_context, "system"
            )
            if db_system_prompt:
                resolved_system_prompt = db_system_prompt

            # QA Template
            resolved_qa_template = prompt_service.get_resolved_prompt(
                db, user_id, use_context, "qa"
            )

            # Condense Prompt
            resolved_condense_prompt = prompt_service.get_resolved_prompt(
                db, user_id, use_context, "condense"
            )

            # Decompose Prompt
            resolved_decompose_prompt = prompt_service.get_resolved_prompt(
                db, user_id, use_context, "decompose"
            )

            # Context Prompt
            resolved_context_prompt = prompt_service.get_resolved_prompt(
                db, user_id, use_context, "context"
            )

        if use_context == ChatMode.AGENTIC.value:
            # 1. Base Processors
            node_postprocessors = [
                MetadataReplacementPostProcessor(target_metadata_key="window"),
            ]

            # 2. Similarity Filtering (dial back if reranker is doing the heavy lifting)
            node_postprocessors.append(
                SimilarityPostprocessor(
                    similarity_cutoff=settings.rag.similarity_value,
                    filter_empty=True,
                    filter_duplicates=True,
                    filter_similar=True,
                )
            )

            # 3. Context Expansion (changed to "both" to catch leading context)
            node_postprocessors.append(
                DocumentAwarePrevNextPostprocessor(
                    docstore=self.storage_context.docstore,
                    prev_pages=1,
                    next_pages=1,
                    mode="both",
                )
            )

            # 4. Reranking (must happen before LongContextReorder)
            if settings.rag.rerank.enabled:
                rerank_postprocessor = rankGPT_rerank.RankGPTRerank(
                    llm=self.llm_component.llm, top_n=5, verbose=True
                )
                node_postprocessors.append(rerank_postprocessor)

            # 5. Long Context Reorder (must be absolute last)
            node_postprocessors.append(LongContextReorder())

            # Use HierarchicalAgentEngine instead of raw AgenticRAGEngine
            tool_registry = ToolRegistry(db)
            # Pass UNRESOLVED system prompt - let AgenticRAGEngine resolve it with tool descriptions
            return HierarchicalAgentEngine(
                llm=self.llm_component.llm,
                tool_registry=tool_registry,
                user_id=user_id,
                index=self.index,
                vector_store_component=self.vector_store_component,
                node_store_component=self.node_store,
                node_postprocessors=node_postprocessors,
                document_files=file_list,
                system_prompt=resolved_system_prompt
                or AGENTIC_SYSTEM_PROMPT,  # Pass unresolved
                qa_template_str=resolved_qa_template,
                max_iterations=20,
                verbose=True,
            )

        elif use_context == ChatMode.SEARCH.value:
            vector_index_retriever = self.vector_store_component.get_retriever(
                index=self.index,
                context_filter=context_filter,
                similarity_top_k=self.settings.rag.similarity_top_k,
            )

            # 1. Base Processors
            node_postprocessors = [
                MetadataReplacementPostProcessor(target_metadata_key="window"),
            ]

            # 2. Similarity Filtering (dial back if reranker is doing the heavy lifting)
            node_postprocessors.append(
                SimilarityPostprocessor(
                    similarity_cutoff=settings.rag.similarity_value,
                    filter_empty=True,
                    filter_duplicates=True,
                    filter_similar=True,
                )
            )

            # 3. Context Expansion (changed to "both" to catch leading context)
            node_postprocessors.append(
                DocumentAwarePrevNextPostprocessor(
                    docstore=self.storage_context.docstore,
                    prev_pages=1,  # Fixed from 0
                    next_pages=1,
                    mode="both",  # Fixed from "next"
                )
            )

            # 4. Reranking (must happen before LongContextReorder)
            if settings.rag.rerank.enabled:
                rerank_postprocessor = rankGPT_rerank.RankGPTRerank(
                    llm=self.llm_component.llm,
                    top_n=settings.rag.rerank.top_n,
                    verbose=True,
                )
                node_postprocessors.append(rerank_postprocessor)

            # 5. Long Context Reorder (must be absolute last)
            node_postprocessors.append(LongContextReorder())

            # Map resolved QA template to context_prompt for custom citation rules
            final_context_prompt = (
                resolved_qa_template
                or resolved_context_prompt
                or self._get_qa_template(db, user_id, use_context)
            )
            final_system_prompt = resolve_system_prompt(
                resolved_system_prompt or RETRIEVAL_SYSTEM_PROMPT
            )

            return AgenticCondenseChatEngine.from_defaults(
                retriever=vector_index_retriever,
                llm=self.llm_component.llm,
                node_postprocessors=node_postprocessors,
                condense_prompt=resolved_condense_prompt,
                decompose_prompt=resolved_decompose_prompt,
                context_prompt=final_context_prompt,  # Fixed: inject QA/citation instructions
                system_prompt=final_system_prompt,
                skip_condense=False,  # Fixed: enable chat history condensation
                verbose=True,
            )
        else:
            return SimpleChatEngine.from_defaults(
                system_prompt=resolve_system_prompt(
                    resolved_system_prompt or DEFAULT_SYSTEM_PROMPT
                ),
                llm=self.llm_component.llm,
                streaming=True,
            )

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        use_context: ChatMode.CHAT.value,
        file_list: List[str] = None,
        context_filter: ContextFilter | None = None,
        cache_service: CacheService | None = None,
        user_id: int | None = None,
        db: Session | None = None,
    ) -> CompletionGen:
        logger.info(
            f"Starting stream_chat with mode: {use_context}, user_id: {user_id}"
        )

        # Ensure we have the latest index data (for multi-worker sync)
        self._refresh_index()

        # Debug logging for message content types
        for i, msg in enumerate(messages):
            logger.info(
                f"Message {i} ({msg.role}): content type={type(msg.content)}, content_preview={str(msg.content)[:100]}"
            )

        chat_engine_input = ChatEngineInput.from_messages(list(messages))
        last_message_content = (
            chat_engine_input.last_message.content
            if chat_engine_input.last_message
            else None
        )

        # Handle dict vs string content
        last_message = ""
        if isinstance(last_message_content, dict):
            last_message = last_message_content.get("text", "")
        elif isinstance(last_message_content, str):
            last_message = last_message_content

        logger.info(
            f"Extracted last_message: '{last_message[:100]}...' (original type: {type(last_message_content)})"
        )

        catalog_answer = self._answer_document_catalog_query(
            db=db,
            user_id=user_id,
            question=last_message,
            conversation_context=self._conversation_context_text(messages),
        )
        if catalog_answer:
            logger.info("Answering streaming chat with DocumentCatalogTool")

            async def catalog_stream():
                yield catalog_answer

            return CompletionGen(response=catalog_stream(), sources=[])

        # Check FAQ cache for streaming mode too (performance optimization)
        if cache_service and last_message and use_context == ChatMode.SEARCH.value:
            cache_answer = self._check_faq_cache(cache_service, last_message)
            if cache_answer:
                logger.info(f"Using cached FAQ answer for streaming")

                # Convert cached response to streaming format
                async def cached_stream():
                    yield cache_answer["content"]

                return CompletionGen(
                    response=cached_stream(), sources=cache_answer["sources"]
                )

        # Normalize chat_history to ensure all messages have string content
        chat_history = []
        if chat_engine_input.chat_history:
            for msg in chat_engine_input.chat_history:
                content = msg.content
                if isinstance(content, dict):
                    content = content.get("text", "")
                chat_history.append(ChatMessage(role=msg.role, content=content))

        logger.info(
            f"Final last_message: '{last_message[:50]}...' (original type: {type(last_message_content)})"
        )
        logger.info(f"Final chat_history length: {len(chat_history)}")

        chat_engine = await self._chat_engine(
            use_context=use_context,
            system_prompt=None,  # System prompt handled in _chat_engine
            file_list=file_list,
            context_filter=context_filter,
            user_id=user_id,
            db=db,
        )

        logger.info(f"Chat engine created: {type(chat_engine)}, starting astream_chat")
        streaming_response = await chat_engine.astream_chat(
            message=last_message,
            chat_history=chat_history,
        )

        # Debug: Inspect the streaming response object
        logger.info(f"astream_chat returned type: {type(streaming_response)}")
        logger.info(
            f"Streaming response attributes: chat_stream={getattr(streaming_response, 'chat_stream', 'NOT_FOUND')}, response_gen={getattr(streaming_response, 'response_gen', 'NOT_FOUND')}"
        )

        # Try to extract initial sources if available
        initial_sources = []
        if (
            hasattr(streaming_response, "source_nodes")
            and streaming_response.source_nodes
        ):
            initial_sources = [
                Chunk.from_node(node) for node in streaming_response.source_nodes
            ]
        elif hasattr(streaming_response, "sources") and streaming_response.sources:
            initial_sources = streaming_response.sources
        # Check for collected sources from nested engines
        elif (
            hasattr(streaming_response, "_collected_source_nodes")
            and streaming_response._collected_source_nodes
        ):
            initial_sources = [
                Chunk.from_node(node)
                for node in streaming_response._collected_source_nodes
            ]
        elif (
            hasattr(streaming_response, "_collected_sources")
            and streaming_response._collected_sources
        ):
            initial_sources = streaming_response._collected_sources

        logger.info(
            f"Got streaming response with {len(initial_sources)} initial sources"
        )

        # Wrap generator to log chunks and handle errors
        async def logged_gen():
            full_response = ""
            first_chunk = True
            chunk_count = 0

            try:
                # Helper to find the actual async generator among possible attributes
                def get_async_gen(obj):
                    # Try each possible generator attribute
                    # achat_stream is used by async astream_chat, chat_stream by sync stream_chat
                    for attr_name in [
                        "achat_stream",
                        "chat_stream",
                        "async_response_gen",
                        "response_gen",
                    ]:
                        val = getattr(obj, attr_name, None)
                        if val is None:
                            continue

                        if callable(val):
                            try:
                                logger.info(f"Calling {attr_name}() to get generator")
                                actual_gen = val()
                                if actual_gen is not None:
                                    import inspect

                                    if inspect.isasyncgen(actual_gen) or hasattr(
                                        actual_gen, "__aiter__"
                                    ):
                                        return actual_gen, attr_name
                                    else:
                                        logger.warning(
                                            f"{attr_name}() returned sync generator, need async"
                                        )
                            except Exception as e:
                                logger.warning(f"Failed to call {attr_name}(): {e}")
                        else:
                            import inspect

                            if inspect.isasyncgen(val) or hasattr(val, "__aiter__"):
                                return val, attr_name
                            else:
                                logger.warning(
                                    f"{attr_name} is a sync generator, need async"
                                )

                    return None, None

                active_generator, generator_name = get_async_gen(streaming_response)

                if active_generator is not None:
                    logger.info(
                        f"Using {generator_name} generator from {type(streaming_response)}"
                    )
                    logger.info(f"Generator type: {type(active_generator)}")

                    async for chunk in active_generator:
                        chunk_count += 1
                        if first_chunk:
                            # Extract text from chunk if it's an object
                            preview = (
                                str(getattr(chunk, "delta", chunk))[:100]
                                if hasattr(chunk, "delta")
                                else str(chunk)[:100]
                            )
                            logger.info(
                                f"First chunk type: {type(chunk)}, preview: {preview}"
                            )
                            first_chunk = False
                        yield chunk
                        if hasattr(chunk, "delta"):
                            full_response += str(chunk.delta)
                        else:
                            full_response += str(chunk)

                    logger.info(
                        f"Streaming completed with {chunk_count} chunks via {generator_name}"
                    )
                else:
                    logger.error(
                        f"Could not find any valid generator in {type(streaming_response)}"
                    )
                    logger.error(
                        f"Available attributes: {[attr for attr in dir(streaming_response) if not attr.startswith('_')]}"
                    )
                    yield "Error: No valid streaming generator found"

            except Exception as e:
                logger.error(f"Error during streaming: {e}", exc_info=True)
                yield f"Error during streaming: {str(e)}"

            if self._should_trigger_notification(initial_sources, full_response):
                logger.info(
                    "Triggering vendor notification: No sources + Refusal detected"
                )
                try:
                    NotificationService.send_vendor_notification_email(
                        last_message, full_response
                    )
                except Exception as e:
                    logger.error(f"Error triggering notification: {e}")

        completion_gen = CompletionGen(response=logged_gen(), sources=initial_sources)
        return completion_gen

    async def chat(
        self,
        messages: list[ChatMessage],
        use_context: ChatMode.CHAT.value,
        file_list: List[str] = None,
        context_filter: ContextFilter | None = None,
        cache_service: CacheService | None = None,
        user_id: int | None = None,
        db: Session | None = None,
    ) -> Completion:
        self._refresh_index()

        chat_engine_input = ChatEngineInput.from_messages(list(messages))
        last_message_content = (
            chat_engine_input.last_message.content
            if chat_engine_input.last_message
            else None
        )

        # Normalize dict vs string content (same logic as stream_chat)
        last_message = ""
        if isinstance(last_message_content, dict):
            last_message = last_message_content.get("text", "")
        elif isinstance(last_message_content, str):
            last_message = last_message_content

        catalog_answer = self._answer_document_catalog_query(
            db=db,
            user_id=user_id,
            question=last_message,
            conversation_context=self._conversation_context_text(messages),
        )
        if catalog_answer:
            logger.info("Answering chat with DocumentCatalogTool")
            return Completion(response=catalog_answer, sources=[], cache_id=None)

        # Check FAQ cache for non-streaming mode too
        if cache_service and last_message and use_context == ChatMode.SEARCH.value:
            cache_answer = self._check_faq_cache(cache_service, last_message)
            if cache_answer:
                logger.info(f"Using cached FAQ answer for non-streaming chat")
                return Completion(
                    response=cache_answer["content"],
                    sources=[
                        Chunk(**s) if isinstance(s, dict) else s
                        for s in cache_answer["sources"]
                    ],
                    cache_id=cache_answer.get("id"),
                )

        # Normalize chat_history content types
        chat_history = []
        if chat_engine_input.chat_history:
            for msg in chat_engine_input.chat_history:
                content = msg.content
                if isinstance(content, dict):
                    content = content.get("text", "")
                chat_history.append(ChatMessage(role=msg.role, content=content))

        chat_engine = await self._chat_engine(
            system_prompt=RETRIEVAL_SYSTEM_PROMPT,
            use_context=use_context,
            context_filter=context_filter,
            file_list=file_list,
            chat_history=chat_history if chat_history else None,
            user_id=user_id,
            db=db,
        )
        wrapped_response = await chat_engine.achat(
            message=last_message,
            chat_history=chat_history if chat_history else None,
        )
        sources = [Chunk.from_node(node) for node in wrapped_response.source_nodes]
        completion = Completion(
            response=wrapped_response.response, sources=sources, cache_id=None
        )
        if self._should_trigger_notification(completion.sources, completion.response):
            logger.info("Triggering vendor notification: No sources + Refusal detected")
            try:
                NotificationService.send_vendor_notification_email(
                    last_message, completion.response
                )
            except Exception as e:
                logger.error(f"Error triggering notification: {e}")

        return completion

    async def generate_title(
        self,
        messages: list[ChatMessage],
    ) -> TitleGeneration:
        """Generates a concise, 3-5 word title with an emoji summarizing the chat history."""
        DEFAULT_TITLE_GENERATION_PROMPT_TEMPLATE = """### Task: You are a title generator.
            Generate a concise, 3-5 word title summarizing the chat history.
            
            ### Guidelines:
            - The title should clearly represent the main theme or subject of the conversation.
            - Write the title in the chat's primary language; default to English if multilingual.
            - Prioritize accuracy over excessive creativity; keep it clear and simple.
            doc
            ### Output:
            Strict follow JSON format: { "title": "your concise title here" }
            
            ### Examples:
            - { "title": "Stock Market Trends" },
            - { "title": "Perfect Chocolate Chip Recipe" },
            - { "title": "Evolution of Music Streaming" },
            - { "title": "Remote Work Productivity Tips" },
            - { "title": "Artificial Intelligence in Healthcare" },
            - { "title": "Video Game Development Insights" }
            
            ### Chat History:
            <chat_history>
            {{MESSAGES:END:2}}
            </chat_history>"""

        if not messages:
            return TitleGeneration(title="No messages provided")

        chat_history = "\n".join(
            [
                msg.content.get("text", str(msg.content))
                if isinstance(msg.content, dict)
                else str(msg.content)
                for msg in messages
            ]
        )
        prompt = DEFAULT_TITLE_GENERATION_PROMPT_TEMPLATE.replace(
            "{{MESSAGES:END:2}}", chat_history
        )

        chat_engine = SimpleChatEngine.from_defaults(
            system_prompt=prompt,
            llm=self.llm_component.llm,
        )
        try:
            response = await chat_engine.achat(chat_history)
            import json

            try:
                import re

                match = re.search(r'{\s*"title"\s*:\s*".+?"\s*}', response.response)
                if match:
                    title_data = json.loads(match.group(0))
                    return TitleGeneration(title=title_data["title"])
                else:
                    # Fallback: try naive string stripping
                    stripped = (
                        response.response.strip("{}")
                        .replace('"title":', "")
                        .strip()
                        .strip('"')
                    )
                    return TitleGeneration(title=stripped)
            except json.JSONDecodeError:
                return TitleGeneration(title="Invalid title format")
        except Exception as e:
            return TitleGeneration(title=f"Error generating title: {str(e)}")
