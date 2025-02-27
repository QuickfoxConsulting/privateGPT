
import asyncio
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple, Dict, Any, Union

from injector import inject, singleton
from llama_index.core.chat_engine import SimpleChatEngine, ContextChatEngine
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor, TimeWeightedPostprocessor, SentenceTransformerRerank
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.postprocessor import SimilarityPostprocessor, rankGPT_rerank
from llama_index.core.storage import StorageContext
from llama_index.core.types import TokenGen
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore, BaseNode
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from pydantic import BaseModel, Field

from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import VectorStoreComponent
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.server.chunks.chunks_service import Chunk
from private_gpt.settings.settings import Settings


class QueryIntent(Enum):
    """Enum representing different query intents for specialized handling."""
    FACTUAL = auto()
    PROCEDURAL = auto()
    EXPLORATORY = auto()
    CLARIFICATION = auto()
    COMPARISON = auto()
    UNKNOWN = auto()


class RetrievalStrategy(Enum):
    """Enum representing different retrieval strategies."""
    SEMANTIC = auto()
    KEYWORD = auto()
    HYBRID = auto()
    MULTI_HOP = auto()
    RECURSIVE = auto()


class Completion(BaseModel):
    response: str
    sources: list[Chunk] | None = None
    reasoning_trace: list[str] | None = None
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


class CompletionGen(BaseModel):
    response: TokenGen
    sources: list[Chunk] | None = None
    reasoning_trace: list[str] | None = None


class TitleGeneration(BaseModel):
    title: str


class ReasoningStep(BaseModel):
    """Model for tracking the agent's reasoning process."""
    step_type: str
    description: str
    result: Any
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


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


class QueryPlanner:
    """Responsible for analyzing the query and planning the retrieval strategy."""
    
    def __init__(self, llm_component: LLMComponent):
        self.llm = llm_component.llm
    
    def analyze_query(self, query: str, chat_history: List[ChatMessage] = None) -> Dict[str, Any]:
        """Analyze the query to determine intent and optimal retrieval strategy."""
        prompt = f"""
        Analyze the following query and determine:
        1. The primary intent (FACTUAL, PROCEDURAL, EXPLORATORY, CLARIFICATION, COMPARISON)
        2. Key entities or concepts that need retrieval
        3. Optimal retrieval strategy (SEMANTIC, KEYWORD, HYBRID, MULTI_HOP, RECURSIVE)
        4. Whether the query requires reasoning beyond simple retrieval
        5. Required context window size (SMALL, MEDIUM, LARGE)
        
        Format your response as a JSON object.
        
        Query: {query}
        
        Chat History: {self._format_chat_history(chat_history) if chat_history else "None"}
        """
        
        response = self.llm.complete(prompt)
        import json
        try:
            analysis = json.loads(response.text)
        except json.JSONDecodeError:
            # Fallback to default analysis if parsing fails
            analysis = {
                "intent": "UNKNOWN",
                "entities": [],
                "strategy": "HYBRID",
                "requires_reasoning": True,
                "context_size": "MEDIUM"
            }
        
        return analysis
    
    def _format_chat_history(self, chat_history: List[ChatMessage]) -> str:
        """Format chat history for analysis."""
        formatted = []
        for msg in chat_history:
            role = "User" if msg.role == MessageRole.USER else "Assistant"
            formatted.append(f"{role}: {msg.content}")
        return "\n".join(formatted)


class ContextProcessor:
    """Responsible for processing and transforming retrieved contexts."""
    
    def __init__(self, llm_component: LLMComponent):
        self.llm = llm_component.llm
    
    def summarize_contexts(self, nodes: List[NodeWithScore], query: str) -> List[NodeWithScore]:
        """Summarize long contexts to extract the most relevant information."""
        processed_nodes = []
        
        for node in nodes:
            if len(node.get_content()) > 1000:  # Only summarize long texts
                summary = self._generate_focused_summary(node.get_content(), query)
                # Create a new node with the summary but preserve metadata
                new_node = type(node)(
                    text=summary,
                    metadata=node.metadata,
                    excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
                    excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
                    relationships=node.relationships,
                )
                new_node.score = node.score
                processed_nodes.append(new_node)
            else:
                processed_nodes.append(node)
        
        return processed_nodes
    
    def _generate_focused_summary(self, content: str, query: str) -> str:
        """Generate a summary focused on answering the query."""
        prompt = f"""
        Summarize the following content, focusing specifically on information relevant to this query:
        
        Query: {query}
        
        Content:
        {content[:3000]}  # Limit content length to avoid token issues
        
        Provide a concise summary (200-300 words) that preserves all information needed to answer the query.
        """
        
        summary = self.llm.complete(prompt).text
        return summary
    
    def identify_contradictions(self, nodes: List[NodeWithScore]) -> List[Dict[str, Any]]:
        """Identify contradictions between different contexts."""
        if len(nodes) <= 1:
            return []
        
        # Compare pairs of nodes for contradictions
        contradictions = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                comparison = self._compare_nodes(nodes[i], nodes[j])
                if comparison["has_contradiction"]:
                    contradictions.append(comparison)
        
        return contradictions
    
    def _compare_nodes(self, node1: NodeWithScore, node2: NodeWithScore) -> Dict[str, Any]:
        """Compare two nodes for contradictions."""
        prompt = f"""
        Compare these two texts and identify if they contain contradictory information.
        
        Text 1:
        {node1.get_content()[:1000]}
        
        Text 2:
        {node2.get_content()[:1000]}
        
        Format your response as a JSON object with the following fields:
        - has_contradiction: boolean
        - contradiction_description: string (if has_contradiction is true)
        - reconciliation: string (suggested way to reconcile the contradiction if possible)
        """
        
        response = self.llm.complete(prompt).text
        import json
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            result = {
                "has_contradiction": False,
                "contradiction_description": "",
                "reconciliation": ""
            }
        
        # Add node identifiers
        result["node1_id"] = node1.node_id if hasattr(node1, "node_id") else "unknown"
        result["node2_id"] = node2.node_id if hasattr(node2, "node_id") else "unknown"
        
        return result


class SelfAwareRetriever(BaseRetriever):
    """Retriever that can self-evaluate and adapt its retrieval strategy."""
    
    def __init__(
        self,
        base_retrievers: Dict[RetrievalStrategy, BaseRetriever],
        llm_component: LLMComponent,
        query_planner: QueryPlanner,
        default_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        adaptive: bool = True
    ):
        super().__init__()
        self.base_retrievers = base_retrievers
        self.llm = llm_component.llm
        self.query_planner = query_planner
        self.default_strategy = default_strategy
        self.adaptive = adaptive
        self.retrieval_history = []
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Self-aware retrieval with strategy selection and evaluation."""
        if self.adaptive:
            # Analyze query to determine optimal strategy
            analysis = self.query_planner.analyze_query(query_bundle.query_str)
            try:
                strategy = RetrievalStrategy[analysis.get("strategy", "HYBRID")]
            except (KeyError, TypeError):
                strategy = self.default_strategy
        else:
            strategy = self.default_strategy
        
        # Use the selected retriever
        selected_retriever = self.base_retrievers.get(strategy, self.base_retrievers[self.default_strategy])
        nodes = selected_retriever.retrieve(query_bundle)
        
        # Evaluate retrieval quality
        evaluation = self._evaluate_retrieval(nodes, query_bundle.query_str)
        
        # If quality is poor and we haven't tried alternative strategies yet, try a different strategy
        if evaluation["quality_score"] < 0.6 and len(self.retrieval_history) < 2:
            fallback_strategy = self._get_fallback_strategy(strategy)
            fallback_retriever = self.base_retrievers.get(fallback_strategy, self.base_retrievers[self.default_strategy])
            
            # Log the retrieval attempt
            self.retrieval_history.append({
                "strategy": strategy.name,
                "query": query_bundle.query_str,
                "quality_score": evaluation["quality_score"],
                "node_count": len(nodes)
            })
            
            # Try with fallback strategy
            nodes = fallback_retriever.retrieve(query_bundle)
        
        # If still no results, try hybrid search as last resort
        if not nodes and RetrievalStrategy.HYBRID in self.base_retrievers:
            nodes = self.base_retrievers[RetrievalStrategy.HYBRID].retrieve(query_bundle)
        
        return nodes
    
    def _evaluate_retrieval(self, nodes: List[NodeWithScore], query: str) -> Dict[str, Any]:
        """Evaluate the quality of retrieved nodes in relation to the query."""
        if not nodes:
            return {"quality_score": 0.0, "feedback": "No documents retrieved"}
        
        # Calculate average relevance score
        avg_score = sum(node.score for node in nodes if hasattr(node, "score")) / len(nodes) if nodes else 0
        
        # Get a sample of content to evaluate (first 3 nodes)
        sample_nodes = nodes[:3]
        sample_content = "\n\n".join([node.get_content()[:500] + "..." for node in sample_nodes])
        
        # Have LLM evaluate relevance more deeply
        prompt = f"""
        Evaluate how well these retrieved documents address this query:
        
        Query: {query}
        
        Retrieved content samples:
        {sample_content}
        
        Rate the relevance on a scale of 0.0 to 1.0, where:
        0.0 = Completely irrelevant
        0.5 = Somewhat relevant but missing key information
        1.0 = Highly relevant and contains all necessary information
        
        Also provide brief feedback on the retrieval quality.
        
        Format your response as a JSON object with these fields:
        - quality_score: float
        - feedback: string
        """
        
        response = self.llm.complete(prompt).text
        import json
        try:
            evaluation = json.loads(response)
        except json.JSONDecodeError:
            # Fallback if parsing fails
            evaluation = {
                "quality_score": min(0.7, avg_score),  # Conservative estimate
                "feedback": "Unable to parse LLM evaluation"
            }
        
        return evaluation
    
    def _get_fallback_strategy(self, current_strategy: RetrievalStrategy) -> RetrievalStrategy:
        """Get fallback retrieval strategy based on the current one."""
        strategy_fallbacks = {
            RetrievalStrategy.SEMANTIC: RetrievalStrategy.HYBRID,
            RetrievalStrategy.KEYWORD: RetrievalStrategy.HYBRID,
            RetrievalStrategy.HYBRID: RetrievalStrategy.MULTI_HOP,
            RetrievalStrategy.MULTI_HOP: RetrievalStrategy.RECURSIVE,
            RetrievalStrategy.RECURSIVE: RetrievalStrategy.SEMANTIC
        }
        
        return strategy_fallbacks.get(current_strategy, RetrievalStrategy.HYBRID)


class ReasoningEngine:
    """Engine that performs reasoning over retrieved context to generate answers."""
    
    def __init__(self, llm_component: LLMComponent):
        self.llm = llm_component.llm
    
    def reason(self, query: str, nodes: List[NodeWithScore], contradictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform reasoning over retrieved contexts."""
        # Extract content from nodes
        context_texts = [f"[Document {i+1}]: {node.get_content()}" for i, node in enumerate(nodes)]
        full_context = "\n\n".join(context_texts[:5])  # Limit to top 5 to avoid token issues
        
        # Handle contradictions if any
        contradiction_text = ""
        if contradictions:
            contradiction_list = []
            for i, contra in enumerate(contradictions):
                contradiction_list.append(
                    f"Contradiction {i+1}: {contra['contradiction_description']}\n"
                    f"Reconciliation: {contra['reconciliation']}"
                )
            contradiction_text = "The following contradictions were detected:\n" + "\n\n".join(contradiction_list)
        
        # Generate reasoning trace
        trace_prompt = f"""
        You are an advanced reasoning engine tasked with answering a query based on provided context.
        
        Query: {query}
        
        Context:
        {full_context}
        
        {contradiction_text}
        
        Think through this step by step:
        1. What are the key pieces of information needed to answer this query?
        2. Which documents provide this information?
        3. Is there missing information that prevents a complete answer?
        4. How should contradictions be resolved (if any)?
        5. What is the most accurate answer based on the provided context?
        
        For each step, provide your reasoning clearly.
        """
        
        reasoning_trace = self.llm.complete(trace_prompt).text
        
        # Now generate the final answer
        answer_prompt = f"""
        Based on your reasoning:
        
        {reasoning_trace}
        
        Provide a final answer to the original query: {query}
        
        Also provide a confidence score between 0.0 and 1.0, where:
        0.0 = No confidence (pure speculation)
        0.5 = Moderate confidence (some evidence but inconclusive)
        1.0 = High confidence (strong evidence in context)
        
        Format as JSON with fields:
        - answer: string
        - confidence_score: float
        """
        
        response = self.llm.complete(answer_prompt).text
        import json
        try:
            result = json.loads(response)
            result["reasoning_trace"] = reasoning_trace.split("\n")
        except json.JSONDecodeError:
            # Fallback if parsing fails
            result = {
                "answer": response,
                "confidence_score": 0.7,  # Default moderate confidence
                "reasoning_trace": reasoning_trace.split("\n")
            }
        
        return result


class AgenticRAG:
    """Main class that orchestrates the agentic RAG pipeline."""
    
    def __init__(
        self,
        llm_component: LLMComponent,
        vector_store_component: VectorStoreComponent,
        embedding_component: EmbeddingComponent,
        node_store_component: NodeStoreComponent,
        settings: Settings
    ):
        self.llm = llm_component.llm
        self.embedding = embedding_component.embedding_model
        self.vector_store = vector_store_component.vector_store
        self.settings = settings
        
        # Create storage context
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store_component.vector_store,
            docstore=node_store_component.doc_store,
            index_store=node_store_component.index_store,
        )
        
        # Create index
        self.index = VectorStoreIndex.from_vector_store(
            vector_store_component.vector_store,
            storage_context=self.storage_context,
            llm=llm_component.llm,
            embed_model=embedding_component.embedding_model,
            show_progress=True,
        )
        
        # Initialize components
        self.query_planner = QueryPlanner(llm_component)
        self.context_processor = ContextProcessor(llm_component)
        self.reasoning_engine = ReasoningEngine(llm_component)
        
        # Setup retrievers
        self.retrievers = self._setup_retrievers(vector_store_component)
        
        # Create self-aware retriever
        self.self_aware_retriever = SelfAwareRetriever(
            base_retrievers=self.retrievers,
            llm_component=llm_component,
            query_planner=self.query_planner,
            adaptive=True
        )
    
    def _setup_retrievers(self, vector_store_component: VectorStoreComponent) -> Dict[RetrievalStrategy, BaseRetriever]:
        """Setup different retriever strategies."""
        retrievers = {}
        
        # Basic semantic search retriever
        retrievers[RetrievalStrategy.SEMANTIC] = vector_store_component.get_retriever(
            index=self.index,
            similarity_top_k=self.settings.rag.similarity_top_k
        )
        
        # Hybrid retriever (combine semantic and keyword search)
        # For simplicity, we'll use the same retriever but in real implementation
        # you would create a proper hybrid retriever
        retrievers[RetrievalStrategy.HYBRID] = retrievers[RetrievalStrategy.SEMANTIC]
        
        # Multi-hop retriever 
        # Simplified implementation - in real code you'd implement a proper multi-hop retriever
        retrievers[RetrievalStrategy.MULTI_HOP] = retrievers[RetrievalStrategy.SEMANTIC]
        
        # Recursive retriever
        # Simplified implementation - in real code you'd implement a proper recursive retriever
        retrievers[RetrievalStrategy.RECURSIVE] = retrievers[RetrievalStrategy.SEMANTIC]
        
        return retrievers
    
    async def process_query(
        self,
        query: str,
        chat_history: List[ChatMessage] = None,
        context_filter: ContextFilter = None
    ) -> Completion:
        """Process query using the agentic RAG pipeline."""
        # Step 1: Plan query approach
        query_analysis = self.query_planner.analyze_query(query, chat_history)
        
        # Step 2: Retrieve relevant documents
        query_bundle = QueryBundle(query_str=query)
        nodes = self.self_aware_retriever.retrieve(query_bundle)
        
        # Step 3: Process and transform contexts
        processed_nodes = self.context_processor.summarize_contexts(nodes, query)
        contradictions = self.context_processor.identify_contradictions(processed_nodes)
        
        # Step 4: Perform reasoning
        reasoning_result = self.reasoning_engine.reason(query, processed_nodes, contradictions)
        
        # Step 5: Generate final answer
        sources = [Chunk.from_node(node) for node in processed_nodes]
        
        # Remove duplicates by content
        unique_sources = []
        seen_contents = set()
        for source in sources:
            if source.content not in seen_contents:
                seen_contents.add(source.content)
                unique_sources.append(source)
        
        return Completion(
            response=reasoning_result["answer"],
            sources=unique_sources,
            reasoning_trace=reasoning_result["reasoning_trace"],
            confidence_score=reasoning_result["confidence_score"]
        )
    
    async def stream_process_query(
        self,
        query: str,
        chat_history: List[ChatMessage] = None,
        context_filter: ContextFilter = None
    ) -> CompletionGen:
        """Process query and stream the response."""
        # This is simplified - in a real implementation you would implement proper streaming
        # For now, we'll generate the full response first and then simulate streaming
        completion = await self.process_query(query, chat_history, context_filter)
        
        async def token_generator():
            # Simulate token-by-token streaming
            tokens = completion.response.split(" ")
            for token in tokens:
                yield token + " "
                await asyncio.sleep(0.05)  # Simulate delay
        
        return CompletionGen(
            response=token_generator(),
            sources=completion.sources,
            reasoning_trace=completion.reasoning_trace
        )


@singleton
class ChatService:
    """Main service for handling chat interactions."""
    
    @inject
    def __init__(
        self,
        settings: Settings,
        llm_component: LLMComponent,
        vector_store_component: VectorStoreComponent,
        embedding_component: EmbeddingComponent,
        node_store_component: NodeStoreComponent,
    ):
        self.settings = settings
        self.llm_component = llm_component
        self.embedding_component = embedding_component
        self.vector_store_component = vector_store_component
        
        # Initialize AgenticRAG
        self.agentic_rag = AgenticRAG(
            llm_component=llm_component,
            vector_store_component=vector_store_component,
            embedding_component=embedding_component,
            node_store_component=node_store_component,
            settings=settings
        )
        
        # Initialize storage context and index (for backward compatibility)
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
        """Detect language using LLM."""
        prompt = f"Detect the language of this text whether it is nepali or english. Respond only with the language name in English. Text: {text}"
        response = self.llm_component.llm.complete(prompt).text.strip().lower()
        return response
    
    def _translate_to_english(self, text: str) -> str:
        """Translate text to English using LLM."""
        prompt = f"Translate the following text to English. Text: {text}"
        return self.llm_component.llm.complete(prompt).text.strip()
    
    async def _chat_engine(
        self,
        system_prompt: str | None = None,
        use_context: bool = False,
        context_filter: ContextFilter | None = None,
    ) -> BaseChatEngine:
        """Legacy method for backward compatibility."""
        if use_context:
            return ContextChatEngine.from_defaults(
                system_prompt=system_prompt,
                retriever=self.agentic_rag.self_aware_retriever,
                llm=self.llm_component.llm,
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
        """Stream chat responses using the agentic RAG pipeline."""
        chat_engine_input = ChatEngineInput.from_messages(messages)
        last_message = (
            chat_engine_input.last_message.content
            if chat_engine_input.last_message
            else None
        )
        chat_history = (
            chat_engine_input.chat_history if chat_engine_input.chat_history else None
        )
        
        if use_context:
            # Use the agentic RAG pipeline
            completion_gen = self.agentic_rag.stream_process_query(
                query=last_message if last_message is not None else "",
                chat_history=chat_history,
                context_filter=context_filter
            )
        else:
            # Use simple LLM generation
            chat_engine = self._chat_engine(
                system_prompt=chat_engine_input.system_message.content if chat_engine_input.system_message else None,
                use_context=False
            )
            streaming_response = chat_engine.stream_chat(
                message=last_message if last_message is not None else "",
                chat_history=chat_history,
            )
            completion_gen = CompletionGen(
                response=streaming_response.response_gen,
                sources=[]
            )
        
        return completion_gen
    
    async def chat(
        self,
        messages: list[ChatMessage],
        use_context: bool = False,
        context_filter: ContextFilter | None = None,
    ) -> Completion:
        """Chat using the agentic RAG pipeline."""
        chat_engine_input = ChatEngineInput.from_messages(messages)
        last_message = (
            chat_engine_input.last_message.content
            if chat_engine_input.last_message
            else None
        )
        system_prompt = (
            chat_engine_input.system_message.content
            if chat_engine_input.system_message
            else """
            You are a specialized retrieval-augmented AI assistant named QuickRef, created by Quickfox Consulting. Your sole purpose is to provide answers based EXCLUSIVELY on the context documents provided to you.
            """
        )
        chat_history = (
            chat_engine_input.chat_history if chat_engine_input.chat_history else None
        )
        
        if use_context:
            # Use the agentic RAG pipeline
            completion = await self.agentic_rag.process_query(
                query=last_message if last_message is not None else "",
                chat_history=chat_history,
                context_filter=context_filter
            )
        else:
            # Use simple LLM generation
            chat_engine = await self._chat_engine(
                system_prompt=system_prompt,
                use_context=False
            )
            response = await chat_engine.achat(
                message=last_message if last_message is not None else "",
                chat_history=chat_history,
            )
            completion = Completion(response=response.response, sources=[])
        
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
            return TitleGeneration(title="No title found")