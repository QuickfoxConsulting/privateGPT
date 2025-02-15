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

class SelfRAGRetriever(BaseRetriever):
    """Retriever with Self-RAG capabilities"""
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm: LLMComponent,
        critique_prompt: str,
        **kwargs
    ) -> None:
        self.base_retriever = base_retriever
        self.llm = llm
        self.critique_prompt_template = PromptTemplate(critique_prompt)
        super().__init__(**kwargs)

    def _should_retrieve(self, query: str) -> Tuple[bool, str]:
        """Determine if retrieval is needed using LLM self-reflection"""
        prompt = f"""Evaluate if this query requires factual information retrieval. 
        Respond ONLY with 'YES' or 'NO':
        Query: {query}
        Answer:"""
        
        response = self.llm.complete(prompt).text.strip().upper()
        return response == "YES", response

    def _critique_node(self, node: BaseNode, query: str) -> bool:
        """Evaluate if node is relevant using LLM"""
        prompt = self.critique_prompt_template.format(
            context=node.get_content(),
            query=query
        )
        response = self.llm.complete(prompt).text.strip().upper()
        return "YES" in response

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # First decide if retrieval is needed
        should_retrieve, reason = self._should_retrieve(query_bundle.query_str)
        if not should_retrieve:
            return []
            
        # Perform base retrieval
        nodes = self.base_retriever.retrieve(query_bundle)
        
        # Critique and filter nodes
        filtered_nodes = []
        for node in nodes:
            if self._critique_node(node.node, query_bundle.query_str):
                filtered_nodes.append(node)
        
        return filtered_nodes

class QueryExpander:
    """Query expansion with synonym generation and LLM-based rewriting"""
    
    def __init__(self, llm: LLMComponent, embed_model: any):
        self.llm = llm
        self.embed_model = embed_model

    def expand(self, query: str) -> str:
        """Expand query using multiple techniques"""
        # Synonym expansion
        synonyms = self._generate_synonyms(query)
        
        # LLM-based expansion
        expanded = self._llm_expansion(query)
        
        # Combine all terms
        return f"{query} {' '.join(synonyms)} {expanded}"

    def _generate_synonyms(self, query: str) -> List[str]:
        """Generate synonyms using embedding similarity"""
        query_embed = self.embed_model.get_query_embedding(query)
        # This would normally query a synonym database, simplified here
        return ["related terms", "similar concepts", "associated ideas"]

    def _llm_expansion(self, query: str) -> str:
        """Use LLM to rewrite and expand the query"""
        prompt = f"""Expand this search query while maintaining its core meaning. Also translate into english if query is in another language.
        Include related terms and alternative phrasings. 
        Keep it concise.
        Query: {query}
        Expanded:"""
        
        return self.llm.complete(prompt).text

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
            # if settings.rag.rerank.enabled:
            #     rerank_postprocessor = rankGPT_rerank.RankGPTRerank(
            #         llm=self.llm_component.llm, 
            #         top_n=settings.rag.rerank.top_n,
            #         # verbose=True
            #     )
            #     # rerank_postprocessor = SentenceTransformerRerank(
            #     #     model=settings.rag.rerank.model, top_n=settings.rag.rerank.top_n
            #     # )
            #     node_postprocessors.append(rerank_postprocessor)
            
            if settings.rag.query_expansion_enabled:
                base_retriever = self._wrap_retriever_with_translation(base_retriever)
                query_expander = QueryExpander(
                    llm=self.llm_component.llm,
                    embed_model=self.embedding_component.embedding_model
                )
                base_retriever = self._wrap_retriever_with_expansion(
                    base_retriever, query_expander)

            if settings.rag.self_rag_enabled:
                critique_prompt = """Evaluate if this passage is relevant to answering the query. 
                Consider:
                - Directly answers the question
                - Provides supporting evidence
                - Contains factual information related to the query
                Respond ONLY with 'RELEVANT: YES' or 'RELEVANT: NO'
                Passage: {context}
                Query: {query}
                Judgment:"""
                
                base_retriever = SelfRAGRetriever(
                    base_retriever=base_retriever,
                    llm=self.llm_component.llm,
                    critique_prompt=critique_prompt
                )
            
            query_engine = RetrieverQueryEngine(
                retriever=base_retriever,
                response_synthesizer=get_response_synthesizer(
                    llm=self.llm_component.llm,
                    response_mode="compact",
                    verbose=True,
                ),
                node_postprocessors=node_postprocessors
            )
            
            return CondensePlusContextChatEngine.from_defaults(
                system_prompt=system_prompt,
                retriever=query_engine,
                llm=self.llm_component.llm,  # Takes no effect at the moment
                node_postprocessors=node_postprocessors,
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
        system_prompt = """
            You are a precise and helpful AI assistant designed to retrieve and communicate information from a given set of documents using Retrieval-Augmented Generation (RAG). Your primary goal is to provide **accurate, context-aware, and well-structured responses** based strictly on retrieved information.  

            ### **Guidelines**  
            - **Use only retrieved information** to answer questions. If unsure, state that clearly.  
            - **If uncertain, ask for clarification.**  
            - **Respond in the same language** as the user's query.  
            - If the context is **poor quality or unreadable**, inform the user and provide the best possible answer.  
            - **If the answer isn't in the context but you possess the knowledge,** explain this and provide an answer using your understanding.  
            - **Only include inline citations ([file_name]) when a <file_name> tag is explicitly provided in the context.** Do not cite otherwise.  
            - **Do not use XML tags in responses.**  
            - Ensure citations are **concise and directly related** to the information provided.  

            ### **Example of Citation**  
            - If a document "whitepaper.pdf" contains the information and has a <file_name>, cite as:  
            *"The proposed method increases efficiency by 20% [whitepaper.pdf]."*  
            - If no <file_name> is present, **omit citations**.  

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
                if lang != 'english':
                    translated = self.svc._translate_to_english(original_query)
                    new_bundle = QueryBundle(query_str=translated)
                    return self.base.retrieve(new_bundle)
                return self.base.retrieve(query_bundle)

        return TranslatedRetriever(base_retriever, self)
    

    def _wrap_retriever_with_expansion(
        self, 
        retriever: BaseRetriever,
        query_expander: QueryExpander
    ) -> BaseRetriever:
        """Wrap retriever with query expansion capabilities"""
        class ExpandedRetriever(BaseRetriever):
            def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
                expanded_query = query_expander.expand(query_bundle.query_str)
                new_bundle = QueryBundle(
                    query_str=expanded_query,
                    embedding=query_bundle.embedding
                )
                return retriever.retrieve(new_bundle)
                
        return ExpandedRetriever()


    def generate_title(
        self,
        messages: str,
    ) -> TitleGeneration:
        '''
        Generates a concise, 3-5 word title with an emoji summarizing the chat history.
        '''
        DEFAULT_TITLE_GENERATION_PROMPT_TEMPLATE = """### Task:
            Generate a concise, 3-5 word title with an emoji summarizing the chat history.
            
            ### Guidelines:
            - The title should clearly represent the main theme or subject of the conversation.
            - Use emojis that enhance understanding of the topic, but avoid quotation marks or special formatting.
            - Write the title in the chat's primary language; default to English if multilingual.
            - Prioritize accuracy over excessive creativity; keep it clear and simple.
            
            ### Output:
            JSON format: { "title": "your concise title here" }
            
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
        
        chat_history = "\n".join([msg.content for msg in messages[-2:]])
        prompt = DEFAULT_TITLE_GENERATION_PROMPT_TEMPLATE.replace("{{MESSAGES:END:2}}", chat_history)
        
        chat_engine = SimpleChatEngine.from_defaults(
            system_prompt=prompt,
            llm=self.llm_component.llm,
        )
        try:
            response = chat_engine.chat(chat_history)
            return TitleGeneration(title=response.response)
        except Exception as e:
            return TitleGeneration(title=f"Error generating title: {str(e)}")
