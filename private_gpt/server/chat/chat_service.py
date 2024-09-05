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
    rankGPT_rerank,
)
from llama_index.core.storage import StorageContext
from llama_index.core.types import TokenGen
from pydantic import BaseModel

from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine

# HYBRID GRAPH RAG
from llama_index.core.retrievers.router_retriever import RouterRetriever
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools.retriever_tool import RetrieverTool
from private_gpt.components.graph_store.graph_store_component import GraphStoreComponent


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
        graph_store_component: GraphStoreComponent,
    ) -> None:
        self.settings = settings
        self.llm_component = llm_component
        self.embedding_component = embedding_component
        self.vector_store_component = vector_store_component
        self.graph_store_component = graph_store_component
        self.knowledge_graph_index = graph_store_component.graph_store

        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store_component.vector_store,
            graph_store=graph_store_component.graph_store,
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
            graph_knowledge_retrevier = self.graph_store_component.get_knowledge_graph(
                llm=self.llm_component.llm,
                storage_context=self.storage_context,
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
            
            retrievers = [
                r for r in [vector_index_retriever, graph_knowledge_retrevier] if r
            ]
            response_synthesizer = get_response_synthesizer(
                response_mode="tree_summarize", 
                llm=self.llm_component.llm,
                # structured_answer_filtering=True
            )
            
            # custom_query_engine = RetrieverQueryEngine(
            #     retriever=vector_index_retriever,
            #     response_synthesizer=response_synthesizer
            # )
            retriever = RouterRetriever.from_defaults(
                retriever_tools=[
                    RetrieverTool.from_defaults(retriever) for retriever in retrievers
                ],
                llm=self.llm_component.llm,
                selector=LLMSingleSelector.from_defaults(
                    llm=self.llm_component.llm
                ),  # TODO: Could be LLMMultiSelector if needed
                select_multi=len(retrievers) > 1,
            )
            return ContextChatEngine.from_defaults(
                system_prompt=system_prompt,
                retriever=retriever,
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
        # system_prompt = """
        #     You are a helpful AI assistant named QuickGPT, created by Quickfox Consulting. Your primary function is to provide comprehensive answers based solely on the information contained in the given context documents.

        #     Please adhere to the following guidelines:

        #     1. Answer questions truthfully based only on the provided context documents.
        #     2. For each document, check whether it is related to the question. Only use relevant documents to answer.
        #     3. Ignore documents that are not related to the question.
        #     4. If the answer exists in several documents, summarize the information from all relevant sources.
        #     5. Do not make up information or use external knowledge. Only answer based on the provided documents.
        #     6. Always use references in the form [NUMBER] when citing information from a document, e.g., [3] for the third document.
        #     7. The reference should only include the document number in square brackets.
        #     8. Provide comprehensive answers, but ensure they are concise and directly relevant to the question asked.
        #     9. If the documents cannot answer the question or you are unsure, state: "The answer cannot be found in the provided context."
        #     10. If more specific information is needed to answer the question, you may ask the user to provide a more specific query.

        #     Context documents:
        #     {context_str}

        #     Your task is to provide detailed answers to user questions based exclusively on the above documents. Remember, if the information isn't in the context, simply state that you don't know or cannot find the answer in the provided information.
        #     """
        system_prompt = (
            """
            You are QuickGPT, an AI assistant created by Quickfox Consulting. 
            Your sole purpose is to provide answers based strictly on the given context documents.
            Adhere to these rules without exception:
            1. Use ONLY the information in the provided context and not prior knowledge, answer the question.
            2. If the exact answer is not in the context, state "I don't have enough information to answer that question."
            3. If asked about topics not covered in the context, respond with "That topic is not covered in the available information."

            Context documents:
            {context_str}

            Your responses must be based exclusively on the above documents. Do not deviate from this context under any circumstances.
            """
        )
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
