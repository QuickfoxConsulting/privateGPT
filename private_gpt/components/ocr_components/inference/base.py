from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class OcrResult:
    """The final structured output of a VLM OCR inference."""
    raw_text: str
    analysis: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
