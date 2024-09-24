from dataclasses import dataclass

from injector import inject, singleton
from llama_index.core.chat_engine import SimpleChatEngine, ContextChatEngine
from llama_index.core.chat_engine.types import (
    BaseChatEngine,
)
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor, TimeWeightedPostprocessor
from llama_index.core.llms import ChatMessage, MessageRole 
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    rankGPT_rerank
)
from llama_index.core.storage import StorageContext
from llama_index.core.types import TokenGen
from pydantic import BaseModel

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

from typing import List, Optional
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore


class Completion(BaseModel):
    response: str
    sources: list[Chunk] | None = None
class CompletionGen(BaseModel):
    response: TokenGen
    sources: list[Chunk] | None = None

reranker_path = models_path / 'reranker'


CONDENSE_PROMPT_TEMPLATE = """
    Given the following conversation between a user and an AI assistant, along with a follow-up question from the user, rephrase the follow-up question into a standalone query. The new query should:

    1. Capture the core intent of the user's follow-up question
    2. Incorporate relevant context from the conversation history
    3. Be self-contained and understandable without requiring knowledge of the previous conversation
    4. Be concise and focused

    Conversation History:
    {chat_history}

    Follow-up Question: {question}

    Standalone Query:"""

class SimilarityPostprocessorWithAtLeastOneResult(SimilarityPostprocessor):
    """Similarity-based Node processor. Return always one result if result is empty"""

    @classmethod
    def class_name(cls) -> str:
        return "SimilarityPostprocessorWithAtLeastOneResult"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        new_nodes = super()._postprocess_nodes(nodes, query_bundle)

        if not new_nodes: 
            return [max(nodes, key=lambda x: x.score)] if nodes else []

        return new_nodes

@dataclass
class ChatEngineInput:
    system_message: ChatMessage | None = None
    last_message: ChatMessage | None = None
    chat_history: list[ChatMessage] | None = None

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
        if system_message:
            messages.pop(0)
        if last_message:
            messages.pop(-1)
        chat_history = messages if len(messages) > 0 else None

        return cls(
            system_message=system_message,
            last_message=last_message,
            chat_history=chat_history,
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

    def _chat_engine(
        self,
        system_prompt: str | None = None,
        use_context: bool = False,
        context_filter: ContextFilter | None = None,
    ) -> BaseChatEngine:
        settings = self.settings
        if use_context:
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
                TimeWeightedPostprocessor(time_decay=0.5, time_access_refresh=False)
            ]
            if settings.rag.rerank.enabled:
                rerank_postprocessor = rankGPT_rerank.RankGPTRerank(
                    llm=self.llm_component.llm, 
                    top_n=settings.rag.rerank.top_n,
                    # verbose=True
                )
                # rerank_postprocessor = SentenceTransformerRerank(
                #     model=settings.rag.rerank.model, top_n=settings.rag.rerank.top_n
                # )
                node_postprocessors.append(rerank_postprocessor)
            
            response_synthesizer = get_response_synthesizer(
                response_mode="compact",
                llm=self.llm_component.llm,
                structured_answer_filtering=True,
                # streaming=True  # Enable streaming for better responsiveness
            )
            
            custom_query_engine = RetrieverQueryEngine.from_args(
                retriever=vector_index_retriever,
                llm=self.llm_component.llm,
                response_synthesizer=response_synthesizer,
                # node_postprocessors=node_postprocessors,
                verbose=True  # For debugging and understanding the process
            )
            
            return ContextChatEngine.from_defaults(
                system_prompt=system_prompt,
                retriever=custom_query_engine,
                llm=self.llm_component.llm,  # Takes no effect at the moment
                node_postprocessors=node_postprocessors,
                # condense_prompt=CONDENSE_PROMPT_TEMPLATE,
            )
        else:
            return SimpleChatEngine.from_defaults(
                system_prompt=system_prompt,
                llm=self.llm_component.llm,
            )

    def stream_chat(
        self,
        messages: list[ChatMessage],
        use_context: bool = False,
        context_filter: ContextFilter | None = None,
    ) -> CompletionGen:
        chat_engine_input = ChatEngineInput.from_messages(messages)
        last_message = (
            chat_engine_input.last_message.content
            if chat_engine_input.last_message
            else None
        )
        system_prompt = (
            chat_engine_input.system_message.content
            if chat_engine_input.system_message
            else None
        )
        chat_history = (
            chat_engine_input.chat_history if chat_engine_input.chat_history else None
        )

        chat_engine = self._chat_engine(
            system_prompt=system_prompt,
            use_context=use_context,
            context_filter=context_filter,
        )
        streaming_response = chat_engine.stream_chat(
            message=last_message if last_message is not None else "",
            chat_history=chat_history,
        )
        sources = [Chunk.from_node(node) for node in streaming_response.source_nodes]
        completion_gen = CompletionGen(
            response=streaming_response.response_gen, sources=sources
        )
        return completion_gen

    def chat(
        self,
        messages: list[ChatMessage],
        use_context: bool = False,
        context_filter: ContextFilter | None = None,
    ) -> Completion:
        chat_engine_input = ChatEngineInput.from_messages(messages)
        last_message = (
            chat_engine_input.last_message.content
            if chat_engine_input.last_message
            else None
        )
        system_prompt = """
            You are an AI assistant designed to provide accurate and concise answers based on retrieved bank documents. 
            Analyze and respond to queries using only the information from the given context.

            Rules:
            1. Use ONLY the information in the provided context to answer questions. Do not use prior knowledge.
            2. If the answer is not in the context, state "The provided bank documents don't contain enough information to answer this question accurately."
            3. Aim for responses no longer than 3-4 sentences unless the query explicitly requires more detail.
            4. Use simple language and explain any necessary banking terms.
            5. If the query is unclear or not related to banking, politely ask for clarification.
            6. Present the most relevant information first for complex queries.
            7. Be cautious with potentially sensitive financial information.
            8. For potential follow-up questions, offer to provide more details on specific aspects if needed.
            9. Consider and mention the date of the information when relevant.

            Context documents:
            {context_str}

            Guidelines for responses:
            - Base your answers exclusively on the provided bank-related context.
            - Provide clear, concise answers that directly address the query about banking information.
            - When referencing financial data or tables, be specific and accurate.
            - If the context includes multiple bank documents or sections, cite the relevant sources in your response.
            - Interpret and explain information from financial tables or charts if relevant to the query.
            - If there are limitations or ambiguities in the available banking information, state them clearly.
            - Avoid making assumptions beyond what is explicitly stated in the context.
            - Keep responses brief and to the point, avoiding unnecessary elaboration.
                    

            Remember, provide helpful, accurate, and concise information based solely on the retrieved bank documents. Ensure your responses are clear, direct, and focused on the banking-related query at hand.
        """
        chat_history = (
            chat_engine_input.chat_history if chat_engine_input.chat_history else None
        )

        chat_engine = self._chat_engine(
            system_prompt=system_prompt,
            use_context=use_context,
            context_filter=context_filter,
        )
        wrapped_response = chat_engine.chat(
            message=last_message if last_message is not None else "",
            chat_history=chat_history,
        )
        # print("---------------------------------------------------------")
        # for node in wrapped_response.source_nodes:
        #     print("*********************************************")
        #     print(node)
        #     print("*********************************************")
        #     # print(Chunk.from_node(node))
        # print("---------------------------------------------------------")

        sources = [Chunk.from_node(node) for node in wrapped_response.source_nodes]
        completion = Completion(response=wrapped_response.response, sources=sources)
        return completion
