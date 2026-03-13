import logging
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from private_gpt.components.ocr_components.detection.base import (
    BaseDetectionProvider,
    DetectionResult,
)

logger = logging.getLogger(__name__)

class DetectionEngine:
    """
    Orchestrates text detection across different providers.
    
    Found boxes are used to align VLM text or provide spatial context.
    """

    def __init__(self, providers: Optional[List[BaseDetectionProvider]] = None):
        self._providers: List[BaseDetectionProvider] = providers or []
        if not self._providers:
            logger.warning("DetectionEngine initialized with no providers.")

    def register_provider(self, provider: BaseDetectionProvider) -> None:
        """Adds a detection provider to the engine."""
        self._providers.append(provider)
        logger.info("Registered detection provider: %s", provider.provider_name)

    def detect(self, image: Image.Image) -> DetectionResult:
        """
        Runs detection using the first available provider.
        """
        if not self._providers:
            raise RuntimeError("No detection providers registered.")

        # For now, we just use the first provider. 
        # In the future, we could have complex fallback/concurrence.
        provider = self._providers[0]
        logger.debug("Running detection using provider: %s", provider.provider_name)
        return provider.detect(image)
