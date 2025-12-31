from datetime import datetime
from typing import Optional
from pydantic import BaseModel

class PromptBase(BaseModel):
    name: str
    content: str
    description: Optional[str] = None
    type: str  # "system", "qa", "condense", "decompose"
    mode: str  # "chat", "search", "agentic"
    is_active: bool = True
    is_default: bool = False

class PromptCreate(PromptBase):
    pass

class PromptUpdate(BaseModel):
    name: Optional[str] = None
    content: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None
    is_default: Optional[bool] = None

class PromptSchema(PromptBase):
    id: int
    user_id: Optional[int]
    version: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True
