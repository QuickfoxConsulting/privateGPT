from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class OcrResult:
    """The final structured output of a VLM OCR inference."""
    raw_text: str
    analysis: Optional[str] = None
    metadata: Dict[str, any] = field(default_factory=dict)
    
@dataclass
class VlmInferenceConfig:
    """Configuration for VLM inference."""
    base_url: str = "http://192.168.1.135:8000/v1"
    model_name: str = "nanonets/Nanonets-OCR2-3B"
    temperature: float = 0.05
    max_tokens: int = 1536
    timeout: int = 60
    enable_cot: bool = True
