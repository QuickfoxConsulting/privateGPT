from typing import List, Optional, Dict
from llama_index.core.chat_engine.types import ChatMessage
from private_gpt.components.llm.llm_component import LLMComponent

class QueryExpander:
    """Query expansion with synonym generation and LLM-based rewriting"""
    
    def __init__(
        self, 
        llm: LLMComponent,
        embed_model: any,
        language: str = "en",
        synonyms_dict: Optional[Dict[str, List[str]]] = None,
    ):
        self.llm = llm
        self.embed_model = embed_model
        self.language = language
        self.synonyms_dict = synonyms_dict or {} 
    
    def expand(self, query: str, chat_history: list[ChatMessage] | None = None) -> str:
        """Expand query using multiple techniques"""
        # synonyms = self._generate_synonyms(query)        
        expanded = self._llm_expansion(query, chat_history)        
        all_terms = f"{query} {expanded}"
        print('Expanded query:', all_terms)
        return all_terms
    
    def _generate_synonyms(self, query: str) -> List[str]:
        """Generate synonyms for key terms in the query"""
        words = query.lower().split()
        synonyms = []        
        query_embed = self.embed_model.get_query_embedding(query)        
        for word in words:
            if word in self.synonyms_dict:
                synonyms.extend(self.synonyms_dict[word])
                
        if not synonyms:
            synonyms_prompt = f"""Generate 3-5 synonyms or related terms for this search query: "{query}"
            Return only the synonyms as a comma-separated list without explanations."""
            synonyms_text = self.llm.complete(synonyms_prompt).text.strip()
            synonyms = [term.strip() for term in synonyms_text.split(',')]
            
        return synonyms
    
    def _llm_expansion(self, query_str: str, chat_history: list[ChatMessage] | None = None) -> str:
        """Use LLM to rewrite and expand the query while considering chat history context.
        
        Args:
            query_str: The original query to expand
            chat_history: Optional list of previous chat messages for context
        """
        # Format chat history if available
        history_context = ""
        if chat_history and len(chat_history) > 0:
            history_messages = "\n".join([
                f"{'user' if msg.role == 'user' else 'assistant'}: {msg.content}"
                for msg in chat_history[-3:]  
            ])
            history_context = f"""Previous conversation context:
            {history_messages}
            """

        prompt = f"""You are a query optimization expert. Your task is to enhance the given query for better vector search results.

        {history_context}

        Original Query: {query_str}

        Instructions:
        1. Analyze the query carefully
        2. Identify key concepts and related terms
        3. ONLY include context from previous conversation if it's directly relevant to current query
        4. If the query represents a new topic unrelated to previous conversation, ignore the conversation history
        5. Create a search-optimized version that will find relevant documents

        Requirements:
        - Maintain the original intent
        - Be specific and focused
        - Keep the query concise (max 2-3 sentences)
        - The expanded query must relate ONLY to the original query's intent

        Your response must contain NOTHING except the optimized query itself - no explanations, no notes, no headers, no formatting:"""

        expanded_query = self.llm.complete(prompt).text.strip()        
        expanded_query = expanded_query.strip('"').strip("'")
        
        return expanded_query
    