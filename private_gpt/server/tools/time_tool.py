from datetime import datetime
from typing import Any
from llama_index.core.tools import BaseTool
from llama_index.core.tools.types import ToolMetadata, ToolOutput

class TimeTool(BaseTool):
    """Tool to get the current time."""
    
    def __init__(self) -> None:
        self.name = "get_current_time"
        self.description = "Get the current time in UTC. Useful for time-sensitive operations or when you need to know the current time."
        self.fn = self._get_time
        super().__init__()
    
    def _get_time(self) -> str:
        """Get the current time in UTC."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    def __call__(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """Call the tool."""
        return ToolOutput(
            content=self._get_time(),
            tool_name=self.name,
            raw_input={},
            raw_output=self._get_time(),
            is_error=False
        )
        
    @property
    def metadata(self) -> ToolMetadata:
        """Get tool metadata."""
        return ToolMetadata(
            name=self.name,
            description=self.description
        )
