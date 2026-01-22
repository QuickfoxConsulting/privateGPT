import logging
from typing import List, Optional, Dict, Any

from llama_index.core.llms import LLM
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.program import LLMTextCompletionProgram
from pydantic import BaseModel

from private_gpt.server.agents.schemas import RouteType, RoutingResult
from private_gpt.server.chat.prompts import ROUTER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class RouterAgent:
    """Agent that classifies user queries and routes them to the appropriate engine."""
    
    def __init__(self, llm: LLM):
        self.llm = llm
        self.prompt = PromptTemplate(ROUTER_SYSTEM_PROMPT)
    
    async def route(
        self, 
        query: str, 
        available_tools: List[Dict[str, Any]],
        chat_history: Optional[List[Any]] = None
    ) -> RoutingResult:
        """Analyze query and decide on the best route."""
        
        # Prepare tool descriptions for the prompt
        tools_str = "\n".join([
            f"- {t['name']}: {t['description']}" 
            for t in available_tools
        ])
        
        # Use structured LLM program if supported, or simple completion + parser
        try:
            from llama_index.core.program import MultiModalLLMCompletionProgram
            # For simplicity, we'll use LLMTextCompletionProgram for structured output
            program = LLMTextCompletionProgram.from_defaults(
                output_cls=RoutingResult,
                prompt=self.prompt,
                llm=self.llm,
                verbose=True
            )
            
            result = await program.acall(
                query=query,
                tools=tools_str,
                history=str(chat_history) if chat_history else "No history"
            )
            
            logger.info(f"Router decided: {result.route} (Reason: {result.reasoning})")
            return result
            
        except Exception as e:
            logger.error(f"Error in RouterAgent: {e}")
            # Fallback to RAG if something goes wrong
            return RoutingResult(
                route=RouteType.RAG,
                reasoning=f"Error in routing layer: {str(e)}. Falling back to default RAG.",
                suggested_tools=[]
            )
