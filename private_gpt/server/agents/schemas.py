from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field

class RouteType(str, Enum):
    RAG = "rag"              # Simple document search
    REACT = "react"          # Direct tool use (single step reasoning)
    PLANNER = "planner"      # Complex multi-step task
    WORKFLOW = "workflow"    # Predefined automated process

class RoutingResult(BaseModel):
    route: RouteType
    reasoning: str
    suggested_tools: List[str] = Field(default_factory=list)

class SubTask(BaseModel):
    id: int
    description: str
    expected_output: str
    dependencies: List[int] = Field(default_factory=list)
    tool_hints: List[str] = Field(default_factory=list)
    status: str = "pending" # pending, in_progress, completed, failed
    result: Optional[str] = None

class Plan(BaseModel):
    tasks: List[SubTask]
    overall_goal: str
    planned_at: str
