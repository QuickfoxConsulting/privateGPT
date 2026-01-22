from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class ToolCategory(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    icon: Optional[str] = None
    
    class Config:
        from_attributes = True

class ToolDefinition(BaseModel):
    id: int
    name: str
    display_name: Optional[str] = None
    category_id: int
    version: Optional[str] = None
    description: Optional[str] = None
    author: Optional[str] = None
    icon: Optional[str] = None
    config_schema: Optional[Dict[str, Any]] = None
    dependencies: Optional[Dict[str, Any]] = None
    is_official: bool = False
    is_verified: bool = False
    created_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class UserToolInstallation(BaseModel):
    id: int
    user_id: int
    tool_id: int
    config: Optional[Dict[str, Any]] = None
    is_enabled: bool = True
    installed_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    usage_count: Optional[int] = 0
    
    tool: Optional[ToolDefinition] = None
    
    class Config:
        from_attributes = True

class ToolInstallRequest(BaseModel):
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ToolConfigUpdate(BaseModel):
    config: Dict[str, Any]
    is_enabled: Optional[bool] = None

class ToolListResponse(BaseModel):
    tools: List[ToolDefinition]
    total: int
    page: int
    page_size: int

class InstalledToolsResponse(BaseModel):
    max_count: int
    tools: List[UserToolInstallation]

class ToolAuthUrlRequest(BaseModel):
    redirect_uri: str
    state: Optional[str] = None

class ToolAuthUrlResponse(BaseModel):
    auth_url: str
    state: str

class ToolAuthExchangeRequest(BaseModel):
    code: str
    redirect_uri: str
    state: Optional[str] = None
