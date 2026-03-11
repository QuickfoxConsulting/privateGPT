import hashlib
import logging
import os
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import fitz  # PyMuPDF
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

@dataclass
class DocumentVisualElement:
    """Represents a visual element extracted from a document with spatial context."""
    element_type: str
    page_number: int
    bbox: List[float]
    content: Any = None
    metadata: Dict[str, Any] = None

class VisualProcessingOptions(BaseModel):
    """Configuration for visual extraction and rendering."""
    dpi: int = Field(default=300, description="DPI for page rendering")
    extract_images: bool = Field(default=True, description="Whether to extract raw image objects")
    render_pages: bool = Field(default=True, description="Whether to render full pages as images")
    image_format: str = Field(default="png", description="Format for extracted/rendered images")
    annotate_layout: bool = Field(default=False, description="Manual layout annotation is disabled (Layout Engine Removed)")

class VisualDocumentEngine:
    """
    The High-Fidelity Technical Eye of the System (Pure Discovery & Inference mode).
    
    Layout analysis has been removed as per User feedback. 
    This engine now performs pure One-Shot Page Discovery.
    """

    def __init__(self, options: Optional[VisualProcessingOptions] = None):
        self.options = options or VisualProcessingOptions()
        logger.info("Initializing VisualDocumentEngine (No-Layout Mode) with DPI=%d", self.options.dpi)

    def open_document(self, file_path: Union[str, Path]) -> fitz.Document:
        path_str = str(file_path)
        try:
            return fitz.open(path_str)
        except Exception as e:
            logger.error("Failed to open document %s: %s", path_str, str(e))
            raise RuntimeError(f"PyMuPDF failed: {str(e)}")

    def render_page_as_image(self, page: fitz.Page) -> Tuple[Image.Image, Optional[np.ndarray]]:
        logger.info("-> DISCOVERY: Rendering page %d to image (DPI=%d)...", page.number + 1, self.options.dpi)
        zoom = self.options.dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        logger.info("-> DISCOVERY: Page %d rendered successfully (%dx%d px)", 
                    page.number + 1, img.width, img.height)
        return img, None

    def create_visual_summary(self, file_path: Union[str, Path]) -> Iterator[Tuple[int, Image.Image, List[Any], str]]:
        """
        Processes a document page by page.
        Yields: (page_num, page_image, empty_list, ocr_prompt)
        """
        from private_gpt.components.ocr_components.inference.prompts import VLM_OCR_COT_PROMPT
        
        logger.info("-> DISCOVERY: Starting visual summary for %s", file_path)
        with self.open_document(file_path) as doc:
            total_pages = len(doc)
            logger.info("-> DISCOVERY: Document has %d pages", total_pages)
            for page_num in range(total_pages):
                page = doc[page_num]
                logger.info("-> DISCOVERY: Processing page %d/%d", page_num + 1, total_pages)
                page_img, transform = self.render_page_as_image(page)
                
                # Layout Engine is REMOVED. Bypassing granular block detection. 
                # We send the RAW page to the VLM once.
                
                logger.info("-> DISCOVERY: Page %d discovery complete. Yielding vision data.", page_num + 1)
                yield (page_num + 1, page_img, [], VLM_OCR_COT_PROMPT)

    @staticmethod
    def get_engine_version() -> str:
        return fitz.version[0]
