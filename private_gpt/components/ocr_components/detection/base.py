import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

@dataclass
class DetectionBox:
    """Represents a detected text region."""
    bbox: List[float]  # [x0, y0, x1, y1] (normalized 0-1)
    confidence: float
    label: str = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DetectionResult:
    """The output of the text detection process."""
    boxes: List[DetectionBox]
    original_size: Tuple[int, int]  # (width, height)
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseDetectionProvider:
    """Abstract base class for text detection providers."""
    
    def detect(self, image: Image.Image) -> DetectionResult:
        """
        Detects text regions in the given image.
        Must be implemented by child providers.
        """
        raise NotImplementedError("Subclasses must implement detect()")

    @property
    def provider_name(self) -> str:
        """Returns the name of the provider."""
        return self.__class__.__name__
