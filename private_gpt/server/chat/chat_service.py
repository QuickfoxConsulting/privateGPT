import logging
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from injector import inject, singleton
from llama_index.core.chat_engine import SimpleChatEngine, CondensePlusContextChatEngine, ContextChatEngine
from llama_index.core.chat_engine.types import (
    BaseChatEngine,
)
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.llms import ChatMessage, MessageRole 
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    rankGPT_rerank
)
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
from private_gpt.server.chunks.chunks_service import Chunk
from private_gpt.settings.settings import Settings

from private_gpt.paths import models_path

from llama_index.core.postprocessor import LongContextReorder
from private_gpt.server.chat.agentic_rag import AgenticCondenseChatEngine
from private_gpt.server.chat.agentic_tool import AgenticRAGEngine
from private_gpt.server.chat.search_tool import SearchRAGEngine
from private_gpt.components.postprocessor.PrevNext import DocumentAwarePrevNextPostprocessor

from private_gpt.server.agents.orchestrator_engine import HierarchicalAgentEngine
from private_gpt.server.tools.tool_registry import ToolRegistry

from private_gpt.server.cache.cache_service import CacheService
from private_gpt.users.services.prompt_service import prompt_service
from sqlalchemy.orm import Session
from private_gpt.server.chat.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    RETRIEVAL_SYSTEM_PROMPT,
    AGENTIC_SYSTEM_PROMPT,
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


reranker_path = models_path / 'reranker'

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
        last_image = getattr(messages[-1], 'image', None) if last_message else None
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
            show_progress=True,
        )
        self.node_store = node_store_component

    def _should_trigger_notification(self, sources: list[Chunk] | None, response_text: str) -> bool:
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
            r"answer is not in the context provided"
        ]
        return any(re.search(p, response_text, re.IGNORECASE) for p in patterns)
        
    def _get_qa_template(self, db: Session, user_id: int | None, mode: str) -> str:
        """Document-grounded QA template with strict context usage and markdown citations."""
        if db and user_id:
            qa_prompt = prompt_service.get_resolved_prompt(db, user_id, mode, "qa")
            if qa_prompt:
                return qa_prompt

        return """
            You are a document-grounded assistant. Answer the query using **only** the information
            contained in the provided context. Do not rely on prior knowledge.

            CONTEXT:
            ---------------------
            {context_str}
            ---------------------

            INSTRUCTIONS:
            1. **Strict Grounding**: Base your answer exclusively on the provided context documents.
            Do NOT add assumptions, interpretations, or external information.

            2. **Answer Scope**:
            - If the answer is present, respond clearly and concisely.
            - If the context does NOT contain sufficient information, state exactly:
                "I cannot find information about this in the provided documents."

            3. **Accuracy**:
            - Preserve original wording when quoting.
            - Maintain exact numerical values, units, and dates.
            - Do not paraphrase if precision would be lost.

            4. **Quotations**:
            - Quote relevant passages directly using quotation marks when appropriate.
            - Place citations immediately after the quoted or referenced content.

            5. **Citations (MANDATORY - STRICT FORMAT)**:
            - Use inline markdown citations in the form `[page X](filename.pdf)`.
            - Place citations at the end of the sentence, paragraph, or list they support.
            - Group citations when multiple claims share the same source.
            - Multiple sources must be listed sequentially:
                `[page 5](doc1.pdf) [page 8](doc2.pdf)`
            - **NEVER** use formats such as `^[1]`, `(Source 1)`, tool names, or fabricated page numbers.

            6. **Formatting and Tone**:
            - Use Markdown where it improves clarity.
            - Maintain a neutral, professional, and factual tone.
            - Be concise and avoid redundancy.

            QUERY:
            {query_str}

            Answer in the same language as the query.
            """


    def _check_faq_cache(self, cache_service: CacheService, question: str) -> dict | None:
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
            search_results = cache_service.search_faqs(question, limit=3, similarity_threshold=0.9)
            logger.info(f"FAQ cache search returned {len(search_results)} results")

            if search_results:
                best_match = search_results[0]
                faq_id = getattr(getattr(best_match, "faq", None), "id", None)
                similarity_info = getattr(best_match, 'similarity', 'unknown')
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
                    logger.info(f"Returning FAQ answer (string): {str(answer_dict)[:100]}...")
                    return {"id": str(faq_id), "content": str(answer_dict), "sources": []}
            else:
                logger.info({"content": "No FAQ match found"})

        except Exception as e:
            logger.error(f"Error checking FAQ cache: {e}", exc_info=True)
        return None

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
            db_system_prompt = prompt_service.get_resolved_prompt(db, user_id, use_context, "system")
            if db_system_prompt:
                resolved_system_prompt = db_system_prompt
            
            # QA Template
            resolved_qa_template = prompt_service.get_resolved_prompt(db, user_id, use_context, "qa")
            
            # Condense Prompt
            resolved_condense_prompt = prompt_service.get_resolved_prompt(db, user_id, use_context, "condense")

            # Decompose Prompt
            resolved_decompose_prompt = prompt_service.get_resolved_prompt(db, user_id, use_context, "decompose")

            # Context Prompt
            resolved_context_prompt = prompt_service.get_resolved_prompt(db, user_id, use_context, "context")

        if use_context == ChatMode.AGENTIC.value:
            node_postprocessors = [
                MetadataReplacementPostProcessor(target_metadata_key="window"),
                SimilarityPostprocessor(
                    similarity_cutoff=settings.rag.similarity_value,
                    filter_empty=True,
                    filter_duplicates=True,
                    filter_similar=True
                ),
                DocumentAwarePrevNextPostprocessor(
                    docstore=self.storage_context.docstore,
                    prev_pages=1,
                    next_pages=1,
                    mode="both"
                ),
                LongContextReorder(),
            ]

            if settings.rag.rerank.enabled:
                rerank_postprocessor = rankGPT_rerank.RankGPTRerank(
                    llm=self.llm_component.llm, 
                    top_n=5,
                    verbose=True
                )
                node_postprocessors.append(rerank_postprocessor)

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
                system_prompt=resolved_system_prompt or AGENTIC_SYSTEM_PROMPT,  # Pass unresolved
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
            node_postprocessors = [
                MetadataReplacementPostProcessor(target_metadata_key="window"),
                SimilarityPostprocessor(
                    similarity_cutoff=settings.rag.similarity_value,
                    filter_empty=True,
                    filter_duplicates=True,
                    filter_similar=True
                ),
                DocumentAwarePrevNextPostprocessor(
                    docstore=self.storage_context.docstore,
                    prev_pages=0,
                    next_pages=1,
                    mode="next"
                ),
                LongContextReorder(),
            ]
            if settings.rag.rerank.enabled:
                rerank_postprocessor = rankGPT_rerank.RankGPTRerank(
                    llm=self.llm_component.llm, 
                    top_n=settings.rag.rerank.top_n,
                    verbose=True
                )
                node_postprocessors.append(rerank_postprocessor)

            response_synthesizer = get_response_synthesizer(
                response_mode="tree_summarize",
                llm=self.llm_component.llm,
                structured_answer_filtering=True,
                text_qa_template=resolved_qa_template or self._get_qa_template(db, user_id, use_context),
                streaming=True  # Enable streaming for better responsiveness
            )
            
            # return CondensePlusContextChatEngine.from_defaults(
            #     retriever=vector_index_retriever,
            #     llm=self.llm_component.llm,
            #     node_postprocessors=node_postprocessors,
            #     system_prompt=resolve_system_prompt(resolved_system_prompt or RETRIEVAL_SYSTEM_PROMPT),
            #     condense_prompt=resolved_condense_prompt,
            #     context_prompt=resolved_context_prompt,
            #     streaming=True,
            #     verbose=True,
            # )
            return AgenticCondenseChatEngine.from_defaults(
                retriever=vector_index_retriever,
                llm=self.llm_component.llm, 
                node_postprocessors=node_postprocessors,
                condense_prompt=resolved_condense_prompt,
                decompose_prompt=resolved_decompose_prompt,
                context_prompt=resolved_context_prompt,
                system_prompt=resolved_system_prompt,
                skip_condense=True,
                verbose=True,
            )
        else:
            return SimpleChatEngine.from_defaults(
                system_prompt=resolve_system_prompt(resolved_system_prompt or DEFAULT_SYSTEM_PROMPT),
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
        logger.info(f"Starting stream_chat with mode: {use_context}, user_id: {user_id}")
        
        # Debug logging for message content types
        for i, msg in enumerate(messages):
            logger.info(f"Message {i} ({msg.role}): content type={type(msg.content)}, content_preview={str(msg.content)[:100]}")

        chat_engine_input = ChatEngineInput.from_messages(messages)
        last_message_content = (
            chat_engine_input.last_message.content
            if chat_engine_input.last_message
            else None
        )
        
        # Handle dict vs string content
        last_message = ""
        if isinstance(last_message_content, dict):
            last_message = last_message_content.get('text', '')
        elif isinstance(last_message_content, str):
            last_message = last_message_content
        
        logger.info(f"Extracted last_message: '{last_message[:100]}...' (original type: {type(last_message_content)})")
        
        # Check FAQ cache for streaming mode too (performance optimization)
        if cache_service and last_message and use_context == ChatMode.SEARCH.value:
            cache_answer = self._check_faq_cache(cache_service, last_message)
            if cache_answer:
                logger.info(f"Using cached FAQ answer for streaming")
                # Convert cached response to streaming format
                async def cached_stream():
                    yield cache_answer["content"]
                
                return CompletionGen(
                    response=cached_stream(),
                    sources=cache_answer["sources"]
                )
        
        # Normalize chat_history to ensure all messages have string content
        chat_history = []
        if chat_engine_input.chat_history:
            for msg in chat_engine_input.chat_history:
                content = msg.content
                if isinstance(content, dict):
                    content = content.get('text', '')
                chat_history.append(ChatMessage(role=msg.role, content=content))
        
        logger.info(f"Final last_message: '{last_message[:50]}...' (original type: {type(last_message_content)})")
        logger.info(f"Final chat_history length: {len(chat_history)}")
        
        chat_engine = await self._chat_engine(
            use_context=use_context,
            system_prompt=None, # System prompt handled in _chat_engine
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
        logger.info(f"Streaming response attributes: chat_stream={getattr(streaming_response, 'chat_stream', 'NOT_FOUND')}, response_gen={getattr(streaming_response, 'response_gen', 'NOT_FOUND')}")
        
        # Try to extract initial sources if available
        initial_sources = []
        if hasattr(streaming_response, 'source_nodes') and streaming_response.source_nodes:
            initial_sources = [Chunk.from_node(node) for node in streaming_response.source_nodes]
        elif hasattr(streaming_response, 'sources') and streaming_response.sources:
            initial_sources = streaming_response.sources
        # Check for collected sources from nested engines
        elif hasattr(streaming_response, '_collected_source_nodes') and streaming_response._collected_source_nodes:
            initial_sources = [Chunk.from_node(node) for node in streaming_response._collected_source_nodes]
        elif hasattr(streaming_response, '_collected_sources') and streaming_response._collected_sources:
            initial_sources = streaming_response._collected_sources
        
        logger.info(f"Got streaming response with {len(initial_sources)} initial sources")
        
        # Wrap generator to log chunks and handle errors
        async def logged_gen():
            full_response = ""
            first_chunk = True
            chunk_count = 0
            
            try:
                # Helper to find the actual async generator among possible attributes
                def get_async_gen(obj):
                    # Try each possible generator attribute
                    # IMPORTANT: Check chat_stream first as it's the custom wrapper from our engines
                    # Check if it's callable (method) - if so, call it to get the generator
                    for attr_name in ['chat_stream', 'async_response_gen', 'response_gen']:
                        val = getattr(obj, attr_name, None)
                        if val is None:
                            continue
                        
                        # If callable, it's a method - call it to get the generator
                        if callable(val):
                            try:
                                logger.info(f"Calling {attr_name}() to get generator")
                                actual_gen = val()
                                if actual_gen is not None:
                                    # Verify it's an async generator
                                    import inspect
                                    if inspect.isasyncgen(actual_gen) or hasattr(actual_gen, '__aiter__'):
                                        return actual_gen, attr_name
                                    else:
                                        logger.warning(f"{attr_name}() returned sync generator, need async")
                            except Exception as e:
                                logger.warning(f"Failed to call {attr_name}(): {e}")
                        else:
                            # It's already a generator - check if it's async
                            import inspect
                            if inspect.isasyncgen(val) or hasattr(val, '__aiter__'):
                                return val, attr_name
                            else:
                                logger.warning(f"{attr_name} is a sync generator, need async")
                    
                    return None, None

                active_generator, generator_name = get_async_gen(streaming_response)
                
                if active_generator is not None:
                    logger.info(f"Using {generator_name} generator from {type(streaming_response)}")
                    logger.info(f"Generator type: {type(active_generator)}")
                    
                    async for chunk in active_generator:
                        chunk_count += 1
                        if first_chunk:
                            # Extract text from chunk if it's an object
                            preview = str(getattr(chunk, 'delta', chunk))[:100] if hasattr(chunk, 'delta') else str(chunk)[:100]
                            logger.info(f"First chunk type: {type(chunk)}, preview: {preview}")
                            first_chunk = False
                        yield chunk
                        if hasattr(chunk, 'delta'):
                            full_response += str(chunk.delta)
                        else:
                            full_response += str(chunk)
                    
                    logger.info(f"Streaming completed with {chunk_count} chunks via {generator_name}")
                else:
                    logger.error(f"Could not find any valid generator in {type(streaming_response)}")
                    logger.error(f"Available attributes: {[attr for attr in dir(streaming_response) if not attr.startswith('_')]}")
                    yield "Error: No valid streaming generator found"
                    
            except Exception as e:
                logger.error(f"Error during streaming: {e}", exc_info=True)
                yield f"Error during streaming: {str(e)}"

            if self._should_trigger_notification(initial_sources, full_response):
                logger.info("Triggering vendor notification: No sources + Refusal detected")
                try:
                    NotificationService.send_vendor_notification_email(last_message, full_response)
                except Exception as e:
                    logger.error(f"Error triggering notification: {e}")
        
        
        completion_gen = CompletionGen(
            response=logged_gen(), sources=initial_sources
        )
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
        # Check FAQ cache for all user messages in the conversation for general user queries
        chat_engine_input = ChatEngineInput.from_messages(messages)
        last_message = (
            chat_engine_input.last_message.content
            if chat_engine_input.last_message
            else None
        )
        chat_history = (
            chat_engine_input.chat_history if chat_engine_input.chat_history else None
        )
        chat_engine = await self._chat_engine(
            system_prompt=RETRIEVAL_SYSTEM_PROMPT,
            use_context=use_context,
            context_filter=context_filter,
            file_list=file_list,
            chat_history=chat_history,
            user_id=user_id,
            db=db,
        )
        wrapped_response = await chat_engine.achat(
            message=last_message if last_message is not None else "",
            chat_history=chat_history,
        )
        sources = [Chunk.from_node(node) for node in wrapped_response.source_nodes]
        completion = Completion(response=wrapped_response.response, sources=sources, cache_id=None)
        if self._should_trigger_notification(completion.sources, completion.response):
                    logger.info("Triggering vendor notification: No sources + Refusal detected")
                    try:
                        NotificationService.send_vendor_notification_email(last_message, completion.response)
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

        chat_history = "\n".join([msg.content['text'] for msg in messages])
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
                    stripped = response.response.strip('{}').replace('"title":', '').strip().strip('"')
                    return TitleGeneration(title=stripped)
            except json.JSONDecodeError:
                return TitleGeneration(title="Invalid title format")
        except Exception as e:
            return TitleGeneration(title=f"Error generating title: {str(e)}")
