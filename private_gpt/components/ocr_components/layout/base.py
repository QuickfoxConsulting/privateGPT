import enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

class LayoutType(enum.Enum):
    TEXT_LINE = "text_line"
    TEXT_BLOCK = "text_block"
    TABLE = "table"
    IMAGE = "image"
    UNKNOWN = "unknown"

@dataclass
class LayoutBlock:
    """Represents a spatial region found by the Visual Layout Engine."""
    block_type: LayoutType
    bbox: List[int]  # [x, y, w, h] format standard for OpenCV
    confidence: float = 1.0 # 1.0 for mathematical extraction
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def coordinates(self) -> List[int]:
        """Returns [x0, y0, x1, y1] format."""
        x, y, w, h = self.bbox
        return [x, y, x + w, y + h]
