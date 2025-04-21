from dataclasses import dataclass

from injector import inject, singleton
from llama_index.core.chat_engine import SimpleChatEngine, ContextChatEngine, CondensePlusContextChatEngine
from llama_index.core.chat_engine.types import (
    BaseChatEngine,
)
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor, TimeWeightedPostprocessor, AutoPrevNextNodePostprocessor
from llama_index.core.llms import ChatMessage, MessageRole 
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    rankGPT_rerank
)
from llama_index.core.storage import StorageContext
from llama_index.core.types import TokenGen
from private_gpt.components.retriever.metadata_retriever import MetadataFilterRetriever
from private_gpt.server.chat.query_expansion import QueryExpander
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


from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import BaseNode
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.core.postprocessor import LongContextReorder


class Completion(BaseModel):
    response: str
    sources: list[Chunk] | None = None
class CompletionGen(BaseModel):
    response: TokenGen
    sources: list[Chunk] | None = None



class TitleGeneration(BaseModel):
    title: str

reranker_path = models_path / 'reranker'

SYSTEM_PROMPT = """
QuickREF is a specialized retrieval-augmented AI assistant designed by Quickfox Consulting for answering queries related to given context documents.

**Core Purpose:**
You are a helpful, conversational assistant whose knowledge is grounded exclusively in the provided context documents. Your goal is to make this information accessible and useful while maintaining the accuracy and integrity of the source material.

**Fundamental Guidelines:**

1. **Document-Grounded Knowledge:**
* Base your responses solely on information explicitly present in the provided context documents.
* Do not introduce external knowledge or make assumptions beyond what's in the documents.
* When information is unavailable in the documents, acknowledge this limitation naturally: "The documents don't appear to cover that specific point. Would you like me to share what they do mention about [related topic]?"

2. **Conversation Quality:**
* Maintain a warm, helpful tone that feels like talking with a knowledgeable colleague.
* Use natural language transitions rather than mechanical references to "the documents."
* Ask clarifying questions when the user's query could be interpreted in multiple ways.
* Personalize responses by referring to previous exchanges in the conversation.

3. **Information Presentation:**
* Synthesize information from multiple document sections into cohesive, flowing responses.
* Begin with the most relevant information that directly addresses the user's question.
* Organize longer responses with a clear structure - main point first, followed by supporting details.
* Use natural paragraph breaks that follow conversational rhythm rather than rigid formatting.

4. **Handling Incomplete Information:**
* When documents provide partial information, share what is available while acknowledging limitations.
* Offer related information that might be helpful: "While the documents don't specify X, they do mention Y, which might be relevant."
* When appropriate, suggest more specific questions the user could ask that would be answerable based on the documents.

Remember: Your value comes from making document information accessible through natural conversation, not from appearing knowledgeable beyond your sources. Build trust through transparency about what you know from the documents and what you don't. 
"""


CONTEXT_PROMPT_TEMPLATE = """
You are a knowledgeable assistant delivering precise, contextually-grounded responses.

CONTEXT: 
{context_str}

When crafting your response:
- Draw exclusively from the provided context
- Quote specific passages when it strengthens your answer
- Acknowledge directly if the context lacks sufficient information
- Prioritize clarity and relevance over comprehensiveness
- Connect related concepts from different parts of the context when appropriate
- Use a conversational yet professional tone that builds rapport

FORMAT YOUR RESPONSE:
- Begin with the most relevant point that directly addresses the question
- Use markdown formatting for readability (headings, bullet points, bold for key concepts)
- Include brief quotations when they provide specific value
- Structure longer answers with natural paragraph breaks

Remember: Your value comes from making this specific context accessible and useful, not from demonstrating general knowledge.
"""

CONDENSE_PROMPT_TEMPLATE = """
Transform the following conversation and new question into a single, self-contained search query that will retrieve the most relevant context.

Chat history:
{chat_history}

Follow Up question: {question}

Your task:
1. Identify the core information need in the new question
2. Incorporate essential context from the conversation history if needed
3. Include specific terminology, identifiers, or constraints that would help retrieve relevant information
4. Formulate a precise, information-dense query that stands alone

The query should:
- Capture the user's current information need completely
- Include relevant context without unnecessary details
- Preserve technical terms exactly as mentioned
- Be clear and specific enough to guide accurate retrieval

If the new question is already optimal (contains all necessary context and is precisely formulated), return it unchanged.
Don't always try to incorporate previous context; only do so if it adds value to the new question.

Standalone Question:
"""

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

    def _get_qa_template(self) -> str:
        """Custom QA template with better context integration."""
        return """Context information is below:
            ---------------------
            {context_str}
            ---------------------

            Given the context documents and not prior knowledge:
            1. Answer the query based ONLY on the provided context.
            2. If the context does not contain the answer, state clearly "I cannot find information about this in the provided documents."
            3. Be concise and do not add information not present in the context.
            4. Quote relevant passages directly using quotation marks when possible.
            5. Cite the source document filename using [filename] format after the relevant sentence or paragraph. If page number is available in metadata, use [filename, p. N].

            Query: {query_str}

            Answer in the same language as the query. Maintain original numerical values and dates. Use markdown formatting where appropriate:
            """

    async def _chat_engine(
        self,
        system_prompt: str | None = None,
        use_context: bool = False,
        context_filter: ContextFilter | None = None,
        chat_history: list[ChatMessage] | None = None,
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
                LongContextReorder(),
                AutoPrevNextNodePostprocessor(
                    docstore=self.storage_context.docstore,
                    llm=self.llm_component.llm
                ),
                # TimeWeightedPostprocessor(time_decay=0.5, time_access_refresh=False)
            ]

            if settings.rag.rerank.enabled:
                rerank_postprocessor = rankGPT_rerank.RankGPTRerank(
                    llm=self.llm_component.llm, 
                    top_n=settings.rag.rerank.top_n,
                    verbose=True
                )
                node_postprocessors.append(rerank_postprocessor)

            # if settings.rag.query_expansion_enabled:
            #     query_expander = QueryExpander(
            #         llm=self.llm_component.llm,
            #         embed_model=self.embedding_component.embedding_model,
            #         language="en",
            #     )
            #     vector_index_retriever = self._wrap_retriever_with_expansion(
            #         vector_index_retriever, query_expander, chat_history
            #     )   

            response_synthesizer = get_response_synthesizer(
                response_mode="compact",
                llm=self.llm_component.llm,
                structured_answer_filtering=True,
                text_qa_template=self._get_qa_template(),
                # streaming=True  # Enable streaming for better responsiveness
            )

            custom_query_engine = RetrieverQueryEngine.from_args(
                retriever=vector_index_retriever,
                llm=self.llm_component.llm,
                response_synthesizer=response_synthesizer,
                verbose=True  # For debugging and understanding the process
            )
            
            return CondensePlusContextChatEngine.from_defaults(
                system_prompt=system_prompt,
                retriever=custom_query_engine,
                llm=self.llm_component.llm,  # Takes no effect at the moment
                node_postprocessors=node_postprocessors,
                condense_prompt=CONDENSE_PROMPT_TEMPLATE,
                context_prompt=CONTEXT_PROMPT_TEMPLATE,
                verbose=True,
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
        # sources = [Chunk.from_node(node) for node in streaming_response.source_nodes]
        sources = []
        seen_nodes = set()

        for node in streaming_response.source_nodes:
            # This example uses the node's content as the identifier
            # Replace with whatever makes nodes "the same" in your context
            node_key = hash(node.content)  # or whatever identifies duplicates
            
            if node_key not in seen_nodes:
                seen_nodes.add(node_key)
                sources.append(Chunk.from_node(node))


        completion_gen = CompletionGen(
            response=streaming_response.response_gen, sources=sources
        )
        return completion_gen

    async def chat(
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
        
        chat_history = (
            chat_engine_input.chat_history if chat_engine_input.chat_history else None
        )

        chat_engine = await self._chat_engine(
            system_prompt=SYSTEM_PROMPT,
            use_context=use_context,
            context_filter=context_filter,
            chat_history=chat_history
        )
        wrapped_response = chat_engine.chat(
            message=last_message if last_message is not None else "",
            chat_history=chat_history,
        )
        sources = [Chunk.from_node(node) for node in wrapped_response.source_nodes]
        completion = Completion(response=wrapped_response.response, sources=sources)
        return completion

    def _wrap_retriever_with_expansion(
        self, 
        retriever: BaseRetriever, 
        query_expander: QueryExpander, 
        chat_history: list[ChatMessage] | None = None
    ) -> BaseRetriever:
        """Wrap retriever with query expansion capabilities"""
        class ExpandedRetriever(BaseRetriever):
            def __init__(self, base: BaseRetriever, expander: QueryExpander, history: list[ChatMessage] | None):
                self.base = base
                self.expander = expander
                self.history = history

            def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
                expanded_query = self.expander.expand(query_bundle.query_str, self.history)
                new_bundle = QueryBundle(
                    query_str=expanded_query,
                    embedding=query_bundle.embedding,
                )
                return self.base.retrieve(new_bundle)

        return ExpandedRetriever(retriever, query_expander, chat_history)


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
            Strict follow JSON format: { "title": "your concise title here" }
            
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
