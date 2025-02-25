from dataclasses import dataclass

from injector import inject, singleton
from llama_index.core.chat_engine import SimpleChatEngine, ContextChatEngine, CondensePlusContextChatEngine
from llama_index.core.chat_engine.types import (
    BaseChatEngine,
)
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor, TimeWeightedPostprocessor, SentenceTransformerRerank
from llama_index.core.llms import ChatMessage, MessageRole 
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    rankGPT_rerank
)
from llama_index.core.storage import StorageContext
from llama_index.core.types import TokenGen
from private_gpt.components.retriever.metadata_retriever import MetadataFilterRetriever
from private_gpt.server.chat.query_expansion import QueryExpander
from private_gpt.server.chat.self_retriever import SelfRAGRetriever
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

from typing import List, Optional, Tuple
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore


from llama_index.core import PromptTemplate
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import BaseNode
from llama_index.core.query_engine import RetrieverQueryEngine



class Completion(BaseModel):
    response: str
    sources: list[Chunk] | None = None
class CompletionGen(BaseModel):
    response: TokenGen
    sources: list[Chunk] | None = None

class TitleGeneration(BaseModel):
    title: str

reranker_path = models_path / 'reranker'


CONDENSE_PROMPT_TEMPLATE = """Your task is to refine a query to ensure it is highly effective for retrieving relevant search results.
        Analyze the given input to grasp the core semantic intent or meaning. Identify the key concepts and technical terms. If the query is not in English, translate it while preserving any technical terms or proper nouns.
        Original Query:
        ------- 
        {question}
        ------- 

        Guidelines for optimization:
        - Remove filler words, unnecessary context, and redundancies
        - Preserve specific technical terms or unique identifiers
        - Ensure the query is specific enough to return relevant results
        - Limit the optimized query to 10-15 words when possible
        - For ambiguous queries, choose the most likely intent based on context

        If the original query is already optimal, return it unchanged.

        Respond with the optimized query only, without explanations or additional text.
        Standalone question:"""

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

    def _detect_language(self, text: str) -> str:
        """Detect language using LLM"""
        prompt = f"Detect the language of this text whether it is nepali or english. Respond only with the language name in English. Text: {text}"
        response = self.llm_component.llm.complete(prompt).text.strip().lower()
        return response

    def _translate_to_english(self, text: str) -> str:
        """Translate text to English using LLM"""
        prompt = f"Translate the following text to English. Text: {text}"
        return self.llm_component.llm.complete(prompt).text.strip()

    def _chat_engine(
        self,
        system_prompt: str | None = None,
        use_context: bool = False,
        context_filter: ContextFilter | None = None,
    ) -> BaseChatEngine:
        settings = self.settings
        if use_context:
            # vector_index_retriever = self.vector_store_component.get_retriever(
            #     index=self.index,
            #     context_filter=context_filter,
            #     similarity_top_k=self.settings.rag.similarity_top_k,
            # )
            
            # node_postprocessors = [
            #     MetadataReplacementPostProcessor(target_metadata_key="window"),
            #     SimilarityPostprocessor(
            #         similarity_cutoff=settings.rag.similarity_value,
            #         filter_empty=True,
            #         filter_duplicates=True,
            #         filter_similar=True
            #     ),
            #     TimeWeightedPostprocessor(time_decay=0.5, time_access_refresh=False)
            # ]
            # if settings.rag.rerank.enabled:
            #     rerank_postprocessor = rankGPT_rerank.RankGPTRerank(
            #         llm=self.llm_component.llm, 
            #         top_n=settings.rag.rerank.top_n,
            #         verbose=True
            #     )
            #     # rerank_postprocessor = SentenceTransformerRerank(
            #     #     model=settings.rag.rerank.model, top_n=settings.rag.rerank.top_n
            #     # )
            #     node_postprocessors.append(rerank_postprocessor)
            
            # response_synthesizer = get_response_synthesizer(
            #     response_mode="compact_accumulate",
            #     llm=self.llm_component.llm,
            #     structured_answer_filtering=True,
            #     # streaming=True  # Enable streaming for better responsiveness
            # )
            
            # custom_query_engine = RetrieverQueryEngine.from_args(
            #     retriever=vector_index_retriever,
            #     llm=self.llm_component.llm,
            #     response_synthesizer=response_synthesizer,
            #     # node_postprocessors=node_postprocessors,
            #     verbose=True  # For debugging and understanding the process
            # )
            
            # return ContextChatEngine.from_defaults(
            #     system_prompt=system_prompt,
            #     retriever=custom_query_engine,
            #     llm=self.llm_component.llm,  # Takes no effect at the moment
            #     node_postprocessors=node_postprocessors,
            #     # condense_prompt=CONDENSE_PROMPT_TEMPLATE,
            # )
            
            base_retriever = self.vector_store_component.get_retriever(
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

            if settings.rag.query_expansion_enabled:
                base_retriever = self._wrap_retriever_with_translation(base_retriever)
                query_expander = QueryExpander(
                    llm=self.llm_component.llm,
                    embed_model=self.embedding_component.embedding_model,
                )
                base_retriever = self._wrap_retriever_with_expansion(
                    base_retriever, query_expander
                )
            if settings.rag.self_rag_enabled:
                base_retriever = SelfRAGRetriever(
                    base_retriever=base_retriever,
                    llm=self.llm_component.llm,
                    node_postprocessors=node_postprocessors,
                )
            response_synthesizer = get_response_synthesizer(
                response_mode="compact",
                llm=self.llm_component.llm,
                structured_answer_filtering=True,
                # streaming=True  # Enable streaming for better responsiveness
            )
            
            custom_query_engine = RetrieverQueryEngine.from_args(
                retriever=base_retriever,
                llm=self.llm_component.llm,
                response_synthesizer=response_synthesizer,
                # node_postprocessors=node_postprocessors,
                verbose=True  # For debugging and understanding the process
            )

            return CondensePlusContextChatEngine.from_defaults(
                system_prompt=system_prompt,
                retriever=custom_query_engine,
                llm=self.llm_component.llm,
                condense_prompt=CONDENSE_PROMPT_TEMPLATE,
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
        '''
                    ### **Example of Citation**  
            - If a document "whitepaper.pdf" contains the information and has a {file_name}, cite as:  
            *"The proposed method increases efficiency by 20% [whitepaper.pdf]."*  
            - If no {file_name} is present, **omit citations**. 
        '''
    
        system_prompt = """
            You are a precise and helpful AI assistant designed to retrieve and communicate information from a given set of documents using Retrieval-Augmented Generation (RAG). Your primary goal is to provide **accurate, context-aware, and well-structured responses** based strictly on retrieved information.  

            ### **Guidelines**  
            - **Use only retrieved information** to answer questions. If unsure, state that clearly.  
            - **If uncertain, ask for clarification.**  
            - **Respond in the same language** as the user's query.  
            - If the context is **poor quality or unreadable**, inform the user and provide the best possible answer.  
            - **If the answer isn't in the context but you possess the knowledge,** explain this and provide an answer using your understanding.  
            - **Only include inline citations ([file_name]) when a {file_name} tag is explicitly provided in the context.** Do not cite otherwise.  
            - **Do not use XML tags in responses.**  
            - Ensure citations are **concise and directly related** to the information provided.  

            ### **Response Principles**  
            - **Structure responses clearly** using markdown:  
            - **Bold** for emphasis  
            - *Italics* for explanations  
            - `Code` for technical terms  
            - Bullet points and lists for clarity  
            - **Highlight conflicting information** when applicable.  
            - **Do not fabricate information.** If the retrieved documents do not contain the answer, state so.  

            ### **Error Handling**  
            - If no relevant documents are found, respond:  
            _"The provided documents do not contain enough information to answer this question."_  
            - If a retrieval error occurs, suggest alternative approaches (e.g., rephrasing the query).  

            **Your primary responsibility is to be a reliable, context-aware retrieval assistant.** Prioritize **accuracy, clarity, and appropriate citation** in all responses.
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
        sources = [Chunk.from_node(node) for node in wrapped_response.source_nodes]
        completion = Completion(response=wrapped_response.response, sources=sources)
        return completion

    def _wrap_retriever_with_translation(self, base_retriever: BaseRetriever) -> BaseRetriever:
        """Wrap retriever with query translation to English"""
        class TranslatedRetriever(BaseRetriever):
            def __init__(self, base: BaseRetriever, svc: ChatService):
                self.base = base
                self.svc = svc

            def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
                original_query = query_bundle.query_str
                lang = self.svc._detect_language(original_query)
                if lang != "english":
                    translated = self.svc._translate_to_english(original_query)
                    new_bundle = QueryBundle(query_str=translated)
                    return self.base.retrieve(new_bundle)
                return self.base.retrieve(query_bundle)

        return TranslatedRetriever(base_retriever, self)

    def _wrap_retriever_with_expansion(
        self, retriever: BaseRetriever, query_expander: QueryExpander
    ) -> BaseRetriever:
        """Wrap retriever with query expansion capabilities"""
        class ExpandedRetriever(BaseRetriever):
            def __init__(self, base: BaseRetriever, expander: QueryExpander):
                self.base = base
                self.expander = expander

            def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
                expanded_query = self.expander.expand(query_bundle.query_str)
                new_bundle = QueryBundle(
                    query_str=expanded_query,
                    embedding=query_bundle.embedding,
                )
                return self.base.retrieve(new_bundle)

        return ExpandedRetriever(retriever, query_expander)

    async def generate_title(
        self,
        messages: list[ChatMessage],
    ) -> TitleGeneration:
        """Generates a concise, 3-5 word title with an emoji summarizing the chat history."""
        DEFAULT_TITLE_GENERATION_PROMPT_TEMPLATE = """### Task: You are a title generator.
            Generate a concise, 3-5 word title with an emoji summarizing the chat history.
            
            ### Guidelines:
            - The title should clearly represent the main theme or subject of the conversation.
            - Use emojis that enhance understanding of the topic, but avoid quotation marks or special formatting.
            - Write the title in the chat's primary language; default to English if multilingual.
            - Prioritize accuracy over excessive creativity; keep it clear and simple.
            
            ### Output:
            Strict JSON format: { "title": "your concise title here" }
            
            ### Examples:
            - { "title": "📉 Stock Market Trends" },
            - { "title": "🍪 Perfect Chocolate Chip Recipe" },
            - { "title": "Evolution of Music Streaming" },
            - { "title": "Remote Work Productivity Tips" },
            - { "title": "Artificial Intelligence in Healthcare" },
            - { "title": "🎮 Video Game Development Insights" }
            
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
                title_data = json.loads(response.response)
                return TitleGeneration(title=title_data["title"])
            except json.JSONDecodeError:
                return TitleGeneration(title=response.response.strip('{}').replace('"title":', '').strip().strip('"'))
        except Exception as e:
            return TitleGeneration(title=f"Error generating title: {str(e)}")