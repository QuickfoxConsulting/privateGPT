from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Optional, Dict, Any

class IntegrationMetadata(BaseModel):
    id: str
    name: str
    description: str
    icon: str  # URL or icon name
    version: str = "1.0.0"
    is_installed: bool = False
    is_configurable: bool = False
    settings_path: Optional[str] = None  # Frontend path to clean settings
    
class BaseIntegration(ABC):
    """Abstract base class for all integrations."""
    
    @property
    @abstractmethod
    def metadata(self) -> IntegrationMetadata:
        """Return static metadata about the integration."""
        pass

    @abstractmethod
    def install(self, user_id: int) -> bool:
        """
        Perform installation steps. 
        Return True if successful.
        """
        pass

    @abstractmethod
    def uninstall(self, user_id: int) -> bool:
        """
        Perform uninstallation/cleanup steps.
        Return True if successful.
        """
        pass
    
    @abstractmethod
    def is_installed(self, user_id: int) -> bool:
        """Check if integration is active for user."""
        pass

    @abstractmethod
    def get_config(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get current configuration if applicable."""
        pass
