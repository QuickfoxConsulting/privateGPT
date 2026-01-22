from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type
from enum import Enum
from pydantic import BaseModel
from llama_index.core.tools import BaseTool, ToolMetadata

class ToolCapability(str, Enum):
    """Standard tool capabilities"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    SEARCH = "search"
    CREATE = "create"

class ToolAuthConfig(BaseModel):
    """Authentication configuration"""
    auth_type: str  # oauth2, api_key, basic
    required_params: Dict[str, str]
    oauth_scopes: Optional[List[str]] = None

class BaseMCPTool(BaseTool, ABC):
    """Enhanced base tool with MCP-like capabilities"""
    
    def __init__(
        self,
        user_id: int,
        config: Dict[str, Any],
        **kwargs
    ):
        self.user_id = user_id
        self.config = config
        
        # Initialize BaseTool with metadata
        # Concrete classes should set self._metadata via property or in __init__
        self._tool_metadata = self._create_metadata()
        super().__init__()
    
    @property
    def metadata(self) -> ToolMetadata:
        return self._tool_metadata
    
    @abstractmethod
    def _create_metadata(self) -> ToolMetadata:
        """Create tool metadata"""
        pass
        
    @classmethod
    @abstractmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """Return JSON schema for tool configuration"""
        pass
    
    @classmethod
    @abstractmethod
    def get_capabilities(cls) -> List[ToolCapability]:
        """Return list of tool capabilities"""
        pass
    
    @classmethod
    @abstractmethod
    def get_auth_config(cls) -> Optional[ToolAuthConfig]:
        """Return authentication requirements"""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate user-provided configuration"""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test if tool is properly configured and accessible"""
        pass
