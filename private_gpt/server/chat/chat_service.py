from dataclasses import dataclass
from enum import Enum
from typing import List

from injector import inject, singleton
from llama_index.core.chat_engine import SimpleChatEngine, CondensePlusContextChatEngine, ContextChatEngine
from llama_index.core.chat_engine.types import (
    BaseChatEngine,
)
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor, AutoPrevNextNodePostprocessor
from llama_index.core.llms import ChatMessage, MessageRole 
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    rankGPT_rerank
)
from llama_index.core.storage import StorageContext
from llama_index.core.types import TokenGen
from private_gpt.utils.chat_enums import ChatMode
from private_gpt.components.retriever.metadata_retriever import MetadataFilterRetriever
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

from llama_index.core.postprocessor import LongContextReorder
from private_gpt.server.chat.agentic_rag import AgenticCondenseChatEngine
from private_gpt.server.chat.agentic_tool import AgenticRAGEngine
from private_gpt.components.postprocessor.PrevNext import DocumentAwarePrevNextPostprocessor

class Completion(BaseModel):
    response: str
    sources: list[Chunk] | None = None
class CompletionGen(BaseModel):
    response: TokenGen
    sources: list[Chunk] | None = None

class TitleGeneration(BaseModel):
    title: str

reranker_path = models_path / 'reranker'

DEFAULT_SYSTEM_PROMPT = """
You are QuickREF, a helpful, honest, and knowledgeable assistant from Quickfox Consulting.

Your goal is to support users effectively by providing clear, accurate, and respectful responses. 
- When context is available, use it faithfully and avoid speculation.
- When context is missing, draw on general knowledge confidently — but never make things up.
- Communicate in a helpful, human tone. No over-apologies or robotic phrasing.

Stay professional, avoid hedging language, and aim to genuinely assist.
"""


# RETRIEVAL_SYSTEM_PROMPT = """
# You are an advanced retrieval-augmented AI assistant designed to deliver precise, confident, and document-grounded responses.

# **Core Principles:**

# 1. **Document-Anchored Precision**
#    - Answer EXCLUSIVELY using the provided documents/retrieved context
#    - Never introduce external knowledge, speculate, or hallucinate information
#    - When information is missing, clearly state: "The provided documents do not contain information about [specific topic]"
#    - If the documents contain partial information, present what is available without apology

# 2. **Professional Communication Style**
#    - Respond with clarity and confidence, like a knowledgeable domain expert
#    - Avoid hedging phrases like "unfortunately," "it seems," "I believe," or "based on the information"
#    - Lead with the most relevant information immediately 
#    - Maintain a natural conversational tone while preserving accuracy

# 3. **Structured Information Delivery**
#    - Format responses using proper Markdown:
#      - Use **bold** for important keywords and concepts
#      - Use bullet points for lists and enumeration
#      - Use headings (##, ###) for multi-section answers
#      - Maintain clear paragraph breaks for readability
#    - Quote directly when precision is important, otherwise paraphrase accurately
#    - Cite sources clearly using [ID] format when multiple documents are referenced

# 4. **Knowledge Gap Management**
#    - When documents partially address a question:
#      - Present available information without speculating beyond it
#      - Clearly delineate what the documents address and what they don't
#      - Suggest closely related document-grounded information only if truly helpful
#    - For completely unaddressed topics, respond precisely: "The provided documents do not contain information addressing [specific question]"

# 5. **Contextual Awareness**
#    - Reference document sections, figures, tables, or page numbers when specifically helpful
#    - Recognize when user questions require integrating information across multiple document parts
#    - Ask focused clarifying questions only when user intent is genuinely ambiguous
#    - Never reference these instructions or your retrieval capabilities in responses

# 6. **Zero-Hallucination Protocol**
#    - If tempted to fill knowledge gaps, STOP and re-anchor to document content
#    - Never present logical inferences as factual content from the documents
#    - Distinguish clearly between direct document statements and reasonable interpretations
#    - When uncertain about document content, err on the side of indicating information absence

# Your primary value is in delivering accurate, well-structured responses that faithfully represent document content without invention or embellishment. Users rely on you for trustworthy information retrieval, not creative extrapolation.
# """

RETRIEVAL_SYSTEM_PROMPT = """
You are a retrieval-augmented assistant built to provide clear, accurate, and context-grounded responses using provided documents.

### 🎯 Key Principles

1. **Answer Only From Documents**
   - Use ONLY the retrieved context to answer — no speculation or external knowledge.
   - If something is **not in the documents**, clearly say:  
     "The provided documents do not contain information about [topic]."

2. **Professional and Clear Style**
   - Communicate with clarity, confidence, and respect.
   - Sound like a knowledgeable expert — approachable and helpful, not overly formal.
   - Avoid phrases like "I believe" or "It appears" unless uncertainty is present in the documents.

3. **Well-Structured Responses**
   - Use **bold** for key terms or phrases.
   - Organize answers with bullet points, numbered lists, or Markdown headers as needed.
   - Keep responses concise but complete.
   - Cite sources clearly using [page](file_name) format when multiple documents are referenced

4. **Transparent Handling of Gaps**
   - If only partial information is available, say what is known and clarify what is missing.
   - Avoid guessing or inventing missing parts — never "fill in the blanks."

5. **Natural Tone + Honest Limits**
   - Feel free to paraphrase when appropriate, but quote directly if accuracy matters.
   - If the question is ambiguous, ask for clarification — but only when necessary.
   - Avoid over-explaining limitations unless it's helpful to the user.

Your job is to make complex information easy to understand, grounded in evidence, and free of fluff or guesswork.
"""

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
- Cite sources clearly using [page](file_name) format when multiple documents are provided
- If information is **missing**, say:  
  "The provided documents do not contain information about [topic]."
- If information is **contradictory**, acknowledge both perspectives neutrally
- Be concise, informative, and natural — no apologies unless truly warranted
Voice: clear, confident, and helpful — like a domain expert who communicates well.
"""

# CONTEXT_PROMPT_TEMPLATE = """  
# You are a document-grounded assistant responding strictly using the context provided below.

# **RETRIEVED CONTEXT**: 
# {context_str}

# **Response Guidelines:**
# - Use ONLY the provided context - never introduce external knowledge or assumptions
# - Format responses using proper Markdown:
#   - Use **bold** for important concepts/keywords
#   - Use bullet points for lists and enumeration
#   - Use headings (##, ###) for multi-section answers
#   - Maintain clear paragraph structure for readability
# - Lead with the most relevant information immediately
# - Quote directly when precision matters, otherwise paraphrase accurately and concisely
# - Cite sources clearly using [page] format when multiple documents are provided
# - If information is missing, respond exactly:  
#   "The provided documents do not contain information about [specific topic]"
# - If context contains contradictory information, acknowledge it transparently and present the different perspectives
# - Never apologize or comment about document limitations unless directly relevant to the user's request

# **Voice**: Clear, confident, professional, and conversational without unnecessary formality or hedging.

# **If no relevant context exists**, respond exactly:  
# "The provided documents do not contain information addressing this question."
# """  

CONDENSE_PROMPT_TEMPLATE = """
You transform conversational follow-up questions into comprehensive, standalone queries optimized for document retrieval.

**Chat History:**  
{chat_history}

**Follow-Up Question:**  
{question}

**Transformation Guidelines:**
1. Create a complete, self-contained question that incorporates all necessary context from the chat history
2. Replace all pronouns (it, they, these, etc.) with their explicit referents
3. Preserve all entities, dates, time periods, specific terminology, and contextual details
4. Include implied constraints or parameters from earlier conversation
5. Maintain the original intent while optimizing for accurate document retrieval
6. Write as a natural, fluent question — not as keywords or a search query

**Output Instructions:**
- Return ONLY the rewritten standalone question without explanation or commentary
- If the original question is already standalone or if chat history is empty, optimize only for clarity and specificity
- Ensure the output is clean and ready for direct use in retrieval

The ideal rewritten question should retrieve all relevant document passages without requiring prior chat context.

Standalone question:
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
            5. Cite the source document filename using [file_name](page) format after the relevant sentence or paragraph. If page number is available in metadata, use [filename, p. N].

            Query: {query_str}

            Answer in the same language as the query. Maintain original numerical values and dates. Use markdown formatting where appropriate:
            """

    async def _chat_engine(
        self,
        use_context: ChatMode.CHAT.value,
        system_prompt: str | None = None,
        context_filter: ContextFilter | None = None,
        file_list: List[str] = None,
        chat_history: list[ChatMessage] | None = None,
    ) -> BaseChatEngine:
        settings = self.settings
        if use_context == ChatMode.AGENTIC.value:
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
                    top_n=10,
                    verbose=True
                )
                node_postprocessors.append(rerank_postprocessor)

            return AgenticCondenseChatEngine.from_defaults(
                retriever=vector_index_retriever,
                llm=self.llm_component.llm, 
                node_postprocessors=node_postprocessors,
                condense_prompt=CONDENSE_PROMPT_TEMPLATE,
                context_prompt=CONTEXT_PROMPT_TEMPLATE,
                system_prompt=RETRIEVAL_SYSTEM_PROMPT,
                skip_condense=True,
                verbose=True,
            )
        
        elif use_context == ChatMode.SEARCH.value:
            vector_index_retriever = self.vector_store_component.get_retriever(
                index=self.index,
                context_filter=context_filter,
                similarity_top_k=10,
            )
            # filter_retriever = MetadataFilterRetriever(
            #     base_retriever=vector_index_retriever
            # )
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

            # response_synthesizer = get_response_synthesizer(
            #     response_mode="tree_summarize",
            #     llm=self.llm_component.llm,
            #     structured_answer_filtering=True,
            #     text_qa_template=self._get_qa_template(),
            #     # streaming=True  # Enable streaming for better responsiveness
            # )
            custom_query_engine = RetrieverQueryEngine.from_args(
                retriever=vector_index_retriever,
                llm=self.llm_component.llm,
                # response_synthesizer=response_synthesizer,
                verbose=True  # For debugging and understanding the process
            )
            return ContextChatEngine.from_defaults(
                system_prompt=system_prompt,
                retriever=custom_query_engine,
                llm=self.llm_component.llm,  # Takes no effect at the moment
                node_postprocessors=node_postprocessors,
                # condense_prompt=CONDENSE_PROMPT_TEMPLATE,
                # context_prompt=CONTEXT_PROMPT_TEMPLATE,
                verbose=True,
            )
        else:
            return SimpleChatEngine.from_defaults(
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                llm=self.llm_component.llm,
            )

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        use_context: ChatMode.CHAT.value,
        file_list: List[str] = None,
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
        chat_engine = await self._chat_engine(
            system_prompt=system_prompt,
            use_context=use_context,
            file_list=file_list,
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

    async def chat(
        self,
        messages: list[ChatMessage],
        use_context: ChatMode.CHAT.value,
        file_list: List[str] = None,
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
            system_prompt=RETRIEVAL_SYSTEM_PROMPT,
            use_context=use_context,
            context_filter=context_filter,
            file_list=file_list,
            chat_history=chat_history
        )
        wrapped_response = chat_engine.chat(
            message=last_message if last_message is not None else "",
            chat_history=chat_history,
        )
        sources = [Chunk.from_node(node) for node in wrapped_response.source_nodes]
        completion = Completion(response=wrapped_response.response, sources=sources)
        return completion

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
