import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
from injector import inject, singleton

from private_gpt.components.ocr_components.reconstruction.base import (
    BaseReconstructionProvider,
    ReconstructionResult
)

logger = logging.getLogger(__name__)

@singleton
class ReconstructionEngine:
    """
    Orchestrates the reconstruction of documents from OCR results.
    Can generate searchable PDFs or other visual documents.
    """

    @inject
    def __init__(self, providers: List[BaseReconstructionProvider]) -> None:
        self._providers = providers
        logger.info("ReconstructionEngine initialized with %d providers", len(self._providers))

    def reconstruct(
        self, 
        original_images: List[Image.Image],
        text_locations: List[List[Dict[str, Any]]],
        output_name: str,
        original_path: Optional[Path] = None
    ) -> ReconstructionResult:
        """
        Uses the first registered provider to reconstruct the document.
        """
        if not self._providers:
            return ReconstructionResult(
                success=False, 
                error="No reconstruction providers registered"
            )
        
        # We use the first provider as the default for now
        provider = self._providers[0]
        logger.info("Starting reconstruction using %s", provider.__class__.__name__)
        
        try:
            return provider.reconstruct(original_images, text_locations, output_name, original_path)
        except Exception as e:
            logger.error("Reconstruction failed: %s", str(e))
            return ReconstructionResult(success=False, error=str(e))
