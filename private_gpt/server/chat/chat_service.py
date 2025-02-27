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
from private_gpt.server.chat.self_retriever_v1 import SelfRAGRetriever
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
from llama_index.core.retrievers import QueryFusionRetriever


class Completion(BaseModel):
    response: str
    sources: list[Chunk] | None = None
class CompletionGen(BaseModel):
    response: TokenGen
    sources: list[Chunk] | None = None

class TitleGeneration(BaseModel):
    title: str

reranker_path = models_path / 'reranker'

CONTEXT_PROMPT_TEMPLATE = """You are a precise and helpful AI assistant. Use the provided context to answer questions.

Guidelines:
- Only use information from the provided context
- If the context doesn't contain the answer, say so
- Include relevant quotes or references when appropriate
- Maintain a professional, clear writing style
- Format responses using markdown for readability

Context: {context}
Question: {question}

Answer:"""

CONDENSE_PROMPT_TEMPLATE = """Given the conversation history and a new question, create a standalone question that captures all relevant context.

Chat History:
{chat_history}

New Question: {question}

Generate a clear, specific question that incorporates any relevant context from the chat history.
If the new question is already self-contained, return it unchanged.
Include any specific technical terms, identifiers, or constraints mentioned.
Limit to 2-3 sentences maximum.

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

    def _get_qa_template(self) -> str:
        """Custom QA template with better context integration."""
        return """Context information is below.
        ---------------------
        {context_str}
        ---------------------
        Given the context information and not prior knowledge, answer the query.
        Query: {query_str}
        Answer: Let's approach this step-by-step:"""

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
                TimeWeightedPostprocessor(time_decay=0.5, time_access_refresh=False)
            ]

            if settings.rag.rerank.enabled:
                rerank_postprocessor = rankGPT_rerank.RankGPTRerank(
                    llm=self.llm_component.llm, 
                    top_n=settings.rag.rerank.top_n,
                    verbose=True
                )
                # rerank_postprocessor = SentenceTransformerRerank(
                #     model=settings.rag.rerank.model, top_n=settings.rag.rerank.top_n
                # )
                node_postprocessors.append(rerank_postprocessor)

            if settings.rag.query_expansion_enabled:
                query_expander = QueryExpander(
                    llm=self.llm_component.llm,
                    embed_model=self.embedding_component.embedding_model,
                    language="en",
                    synonyms_dict={
                        "company": ["organization", "firm", "business", "corporation", "enterprise"],
                        "employee": ["worker", "staff", "staff member", "staffer", "staffer"],
                        "policy": ["regulation", "rule", "law", "standard", "guideline"]
                    }
                )
                vector_index_retriever = self._wrap_retriever_with_expansion(
                    vector_index_retriever, query_expander, chat_history
                )   

            
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
            
            return ContextChatEngine.from_defaults(
                system_prompt=system_prompt,
                retriever=custom_query_engine,
                llm=self.llm_component.llm,  # Takes no effect at the moment
                node_postprocessors=node_postprocessors,
                # condense_prompt=CONDENSE_PROMPT_TEMPLATE,
                # context_prompt=CONTEXT_PROMPT_TEMPLATE
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
        system_prompt = """
            You are a specialized retrieval-augmented AI assistant named QuickRef, created by Quickfox Consulting. Your sole purpose is to provide answers based EXCLUSIVELY on the context documents provided to you.

            ### **Core RAG Guidelines**
            - You can ONLY answer based on information explicitly present in the retrieved context documents
            - You must NEVER use your general knowledge to supplement answers
            - If the answer is not in the context documents, respond with ONLY: "I cannot find information about this in the provided documents."
            - Do not explain limitations or apologize for not knowing
            - Never hallucinate or invent information not present in the documents

            ### **Document Processing**
            - Only reference documents that directly address the query
            - Ignore irrelevant documents completely
            - Prioritize information from multiple documents that corroborate each other
            - When documents contain conflicting information, highlight the inconsistency
            - When citations are provided as {file_name}, include them as [file_name]
            - Do not attempt to reference document IDs or names if they aren't explicitly given

            ### **Response Structure**
            - Begin with a direct answer to the question when available
            - Format responses with markdown
            - Bullet lists for multiple points
            - Keep responses concise but complete
            - Maintain the user's query language in your response

            ### **Step-by-Step Procedure Handling**
            - For questions about procedures or processes, identify and extract the exact steps in the correct sequence
            - Maintain the original numbering or ordering of steps as presented in the documents
            - Present procedures in a clear, structured format (numbered lists for sequential steps)
            - Do not combine or merge steps from different procedures
            - Do not add additional steps or requirements not explicitly listed in the documents
            - For questions about "how to" perform a specific task, prioritize finding explicit procedural instructions
            - When presenting steps, focus on actions the user needs to take, not explanations of the system

            ### **Strict RAG Enforcement**
            - You are FORBIDDEN from using any information outside the provided context
            - You are DISALLOWED from generating speculative answers
            - You are PROHIBITED from offering to search for more information
            - You cannot suggest external resources or alternative approaches
            - You must not identify sections of text that seem relevant but don't actually answer the question

            Context documents:
            {context_str}

            Your function is to be a strict, context-bound retrieval system that ONLY provides information found in the documents above. Stay within these boundaries at all times.

            """
        chat_history = (
            chat_engine_input.chat_history if chat_engine_input.chat_history else None
        )

        chat_engine = await self._chat_engine(
            system_prompt=system_prompt,
            use_context=use_context,
            context_filter=context_filter,
            chat_history=chat_history
        )
        wrapped_response = await chat_engine.achat(
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
        self, retriever: BaseRetriever, query_expander: QueryExpander, chat_history: list[ChatMessage] | None = None
    ) -> BaseRetriever:
        """Wrap retriever with query expansion capabilities"""
        class ExpandedRetriever(BaseRetriever):
            def __init__(self, base: BaseRetriever, expander: QueryExpander):
                self.base = base
                self.expander = expander

            def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
                expanded_query = self.expander.expand(query_bundle.query_str, chat_history)
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






"""
Few shot examples:
            ### **Example 1: Direct answer from context**
                Context:
                [Company Handbook, Page 12]
                The annual leave policy allows full-time employees 25 days of paid vacation per year, accrued monthly at a rate of 2.08 days. Unused vacation days can be carried over to the next calendar year, with a maximum cap of 30 days total. Employees must submit vacation requests at least two weeks in advance through the HR portal.
                User query: How many vacation days do full-time employees get?

                Assistant response:
                Full-time employees receive **25 days of paid vacation per year**, accrued monthly at a rate of 2.08 days. Unused vacation days can be carried over to the next calendar year, with a maximum cap of 30 days total.

            ### **Example 2: Information not in context**
                Context:
                [IT Security Policy, Section 3.4]
                All employees must use a password that contains at least 12 characters, including uppercase letters, lowercase letters, numbers, and special symbols. Passwords must be changed every 90 days and cannot be reused for 1 year.
                User query: What is the company's policy on remote work?

                Assistant response:
                I cannot find information about this in the provided documents.

            ### **Example 3: Procedural information**
                Context:
                [New Employee Onboarding, Page 5]
                Password Reset Procedure:
                Navigate to portal.company.com/reset
                Enter your employee ID number
                Click "Request Reset Link"
                Check your company email for the reset link
                Click the link and enter your new password twice
                Log in with your new password

                User query: How do I reset my password?

                Assistant response:
                Here's how to reset your password:

                1. Navigate to portal.company.com/reset
                2. Enter your employee ID number
                3. Click "Request Reset Link"
                4. Check your company email for the reset link
                5. Click the link and enter your new password twice
                6. Log in with your new password
"""