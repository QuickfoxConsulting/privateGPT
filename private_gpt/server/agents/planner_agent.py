import logging
import datetime
from typing import List, Optional, Dict, Any

from llama_index.core.llms import LLM
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.program import LLMTextCompletionProgram

from private_gpt.server.agents.schemas import Plan, SubTask
from private_gpt.server.chat.prompts import PLANNER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class PlannerAgent:
    """Agent that decomposes complex tasks into a multi-step plan."""
    
    def __init__(self, llm: LLM):
        self.llm = llm
        self.prompt = PromptTemplate(PLANNER_SYSTEM_PROMPT)
    
    async def create_plan(
        self, 
        query: str, 
        available_tools: List[Dict[str, Any]]
    ) -> Plan:
        """Generate a multi-step plan for the query."""
        
        tools_str = "\n".join([
            f"- {t['name']}: {t['description']}" 
            for t in available_tools
        ])
        
        try:
            program = LLMTextCompletionProgram.from_defaults(
                output_cls=Plan,
                prompt=self.prompt,
                llm=self.llm,
                verbose=True
            )
            
            plan = await program.acall(
                query=query,
                tools=tools_str
            )
            
            # Add metadata if missing from LLM response
            plan.planned_at = datetime.datetime.now().isoformat()
            
            logger.info(f"Planner generated {len(plan.tasks)} tasks for query: {query[:50]}...")
            return plan
            
        except Exception as e:
            logger.error(f"Error in PlannerAgent: {e}")
            # Fallback: simple one-step plan
            return Plan(
                tasks=[
                    SubTask(
                        id=1,
                        description=f"Execute the query using available tools: {query}",
                        expected_output="Result of the query",
                        dependencies=[],
                        tool_hints=[t['name'] for t in available_tools[:3]]
                    )
                ],
                overall_goal=query,
                planned_at=datetime.datetime.now().isoformat()
            )
