import logging
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from llama_index.core.llms import ChatMessage

from private_gpt.server.cache.cache_service import CacheService
from private_gpt.users import crud, models, schemas

logger = logging.getLogger(__name__)


class FAQAnalyzer:
    """Analyzes chat conversations to identify potential frequently asked questions."""
    
    def __init__(self, cache_service: CacheService):
        # Use injected cache service instead of getting it from global injector
        self.cache_service = cache_service
        logger.debug(f"FAQAnalyzer initialized with cache service: {self.cache_service is not None}")
        logger.debug(f"FAQAnalyzer cache status: {self.cache_service.cache_status}")
    
    def analyze_conversation(self, db: Session, messages: List[ChatMessage], user_id: int = None) -> bool:
        """Analyze a conversation to identify potential FAQs.
        
        Args:
            db: Database session
            messages: List of chat messages
            user_id: ID of the user who initiated the conversation
            
        Returns:
            bool: True if potential FAQ was identified and added, False otherwise
        """
        try:
            logger.debug(f"Analyzing conversation with {len(messages)} messages")
            
            # Extract user questions and AI responses
            conversation_pairs = self._extract_qa_pairs(messages)
            logger.debug(f"Extracted {len(conversation_pairs)} Q&A pairs")
            
            # For each question-answer pair, determine if it should be a FAQ
            for question, answer in conversation_pairs:
                logger.debug(f"Checking if Q&A pair should be FAQ: {question[:50]}...")
                if self._should_be_faq(question, answer):
                    logger.debug(f"Q&A pair qualifies as FAQ candidate: {question[:50]}...")
                    # Check if this FAQ already exists
                    if not self._faq_exists(db, question):
                        logger.debug(f"FAQ does not exist, creating new one: {question[:50]}...")
                        # Create new FAQ
                        faq_in = schemas.FAQCreate(
                            question=question,
                            answer=answer,
                            category="General"  # Default category, could be improved with classification
                        )
                        faq = crud.faq.create_with_user(db, obj_in=faq_in, user_id=user_id or 1)
                        logger.debug(f"Created FAQ with ID: {faq.id}")
                        
                        # Add to cache using the cache service
                        try:
                            cache_result = self.cache_service.add_faq(faq)
                            logger.debug(f"Added FAQ to cache, result: {cache_result}")
                        except Exception as e:
                            logger.error(f"Failed to add FAQ to cache: {e}", exc_info=True)
                        
                        logger.info(f"Added new FAQ: {question}")
                    else:
                        logger.debug(f"FAQ already exists for: {question[:50]}...")
                else:
                    logger.debug(f"Q&A pair does not qualify as FAQ: {question[:50]}...")
            
            return True
        except Exception as e:
            logger.error(f"Failed to analyze conversation for FAQs: {e}", exc_info=True)
            return False
    
    def _extract_qa_pairs(self, messages: List[ChatMessage]) -> List[tuple]:
        """Extract question-answer pairs from chat messages.
        
        Args:
            messages: List of chat messages
            
        Returns:
            List of (question, answer) tuples
        """
        qa_pairs = []
        user_question = None
        
        for message in messages:
            if message.role == "user":
                user_question = message.content
            elif message.role == "assistant" and user_question:
                # This is an answer to the previous user question
                qa_pairs.append((user_question, message.content))
                user_question = None
                
        return qa_pairs
    
    def _should_be_faq(self, question: str, answer: str) -> bool:
        """Determine if a question-answer pair should be a FAQ.
        
        Args:
            question: The user's question
            answer: The AI's answer
            
        Returns:
            bool: True if this should be a FAQ, False otherwise
        """
        # Simple heuristics for now - could be enhanced with ML models
        question = question.lower().strip()
        answer = answer.lower().strip()
        
        # Skip if question is too short or answer is too short
        if len(question) < 10 or len(answer) < 20:
            return False
            
        # Skip if answer indicates uncertainty
        uncertainty_phrases = [
            "i don't know", "i'm not sure", "i cannot", "not found", 
            "no information", "cannot find", "does not contain"
        ]
        
        for phrase in uncertainty_phrases:
            if phrase in answer:
                return False
                
        # Common FAQ patterns
        faq_indicators = [
            "what is", "how to", "how do i", "can you", "is it", 
            "why is", "where is", "when is", "who is", "which is"
        ]
        
        for indicator in faq_indicators:
            if indicator in question:
                return True
                
        # If answer is long and detailed, it might be a good FAQ
        if len(answer) > 100:
            return True
            
        return False
    
    def _faq_exists(self, db: Session, question: str) -> bool:
        """Check if a FAQ with similar question already exists.
        
        Args:
            db: Database session
            question: The question to check
            
        Returns:
            bool: True if similar FAQ exists, False otherwise
        """
        # First check in cache
        try:
            cached_results = self.cache_service.search_faqs(question, limit=3, similarity_threshold=0.9)
            for result in cached_results:
                if self._questions_similar(result.faq.question, question):
                    return True
        except Exception as e:
            logger.warning(f"Failed to search FAQs in cache: {e}")
        
        # Then check in database
        db_matches = crud.faq.search(db, query=question, limit=5)
        for db_faq in db_matches:
            if self._questions_similar(db_faq.question, question):
                return True
                
        return False
    
    def _questions_similar(self, q1: str, q2: str) -> bool:
        """Check if two questions are similar.
        
        Args:
            q1: First question
            q2: Second question
            
        Returns:
            bool: True if questions are similar, False otherwise
        """
        # Simple string similarity - could be enhanced with more sophisticated algorithms
        q1 = q1.lower().strip()
        q2 = q2.lower().strip()
        
        # Exact match
        if q1 == q2:
            return True
            
        # One question contains the other
        if q1 in q2 or q2 in q1:
            return True
            
        # Calculate simple similarity ratio
        words1 = set(q1.split())
        words2 = set(q2.split())
        
        if len(words1) == 0 or len(words2) == 0:
            return False
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union)
        return similarity > 0.5  # 50% word overlap

