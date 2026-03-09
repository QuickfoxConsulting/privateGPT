import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

class DiscoveryType(enum.Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CHART = "chart"
    UNKNOWN = "unknown"

@dataclass
class DiscoveryElement:
    """Represents a discrete piece of content found during discovery."""
    element_type: DiscoveryType
    content: Any  # str for text, PIL.Image for images/visuals
    page_number: int
    bbox: Optional[List[float]] = None  # [x0, y0, x1, y1] if available
    metadata: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class DiscoveryResult:
    """The unified output of a discovery process."""
    file_path: Path
    file_type: str
    elements: List[DiscoveryElement]
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseDiscoveryProvider:
    """Abstract base class for format-specific discovery logic."""
    
    def __init__(self):
        pass

    def discover(self, file_path: Path) -> DiscoveryResult:
        """
        Cracks open the file and extracts text and visual elements.
        Must be implemented by child providers.
        """
        raise NotImplementedError("Subclasses must implement discover()")

    @property
    def supported_extensions(self) -> List[str]:
        """Returns a list of extensions this provider can handle."""
        raise NotImplementedError("Subclasses must implement supported_extensions")
