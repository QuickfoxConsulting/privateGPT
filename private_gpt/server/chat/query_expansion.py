from typing import List, Optional, Dict
from dataclasses import dataclass
from enum import Enum
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
    
    def expand(self, query: str) -> str:
        """Expand query using multiple techniques"""
        # synonyms = self._generate_synonyms(query)        
        expanded = self._llm_expansion(query)        
        all_terms = f"{query} {expanded}"
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
    
    def _llm_expansion(self, query_str: str) -> str:
        """Use LLM to rewrite and expand the query"""
        # prompt = f"You are given a user query that should be answered by looking up documents that from a document store using a distance based similarity measure. The documents fetched from the document store were found to be irrelevant to answer the question. Rewrite the following question into an alternative that increases the likelihood of finding relevant documents from the database. You may only answer with the exact rephrasing.  If the query is not in {self.language}, translate it. The original question is: {query} Expanded:"
        prompt = f"""Your task is to refine a query to ensure it is highly effective for retrieving relevant search results. \n
        Analyze the given input to grasp the core semantic intent or meaning. \n
        Original Query:
        \n ------- \n
        {query_str}
        \n ------- \n
        Your goal is to rephrase or enhance this query to improve its search performance. Ensure the revised query is concise and directly aligned with the intended search objective. \n
        Respond with the optimized query only:"""
        return self.llm.complete(prompt).text.strip()
    
    def _deduplicate_terms(self, terms_string: str) -> str:
        """Remove duplicate terms while preserving original query"""
        terms = terms_string.split()
        original_query_words = terms[:len(terms_string.split(' ', 1)[0].split())]
        expanded_terms = terms[len(original_query_words):]
        
        seen = set(word.lower() for word in original_query_words)
        result = list(original_query_words)
        
        for word in expanded_terms:
            if word.lower() not in seen:
                result.append(word)
                seen.add(word.lower())
        
        return ' '.join(result)