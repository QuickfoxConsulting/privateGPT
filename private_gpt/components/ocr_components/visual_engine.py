import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import fitz  # PyMuPDF
from PIL import Image
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

@dataclass
class DocumentVisualElement:
    """Represents a visual element extracted from a document with spatial context."""
    element_type: str  # 'image', 'table', 'chart', 'text_block'
    page_number: int    # 1-indexed
    bbox: List[float]   # [x0, y0, x1, y1] in PDF points
    content: Any = None  # Could be bytes, PIL.Image, or text
    metadata: Dict[str, Any] = None

class VisualProcessingOptions(BaseModel):
    """Configuration for visual extraction and rendering."""
    dpi: int = Field(default=300, description="DPI for page rendering")
    extract_images: bool = Field(default=True, description="Whether to extract raw image objects")
    render_pages: bool = Field(default=True, description="Whether to render full pages as images")
    image_format: str = Field(default="png", description="Format for extracted/rendered images")
    upscale_small_images: bool = Field(default=False, description="Upscale small images for better OCR")

class VisualDocumentEngine:
    """
    The High-Fidelity Technical Eye of the System.
    
    Provides high-performance PDF rendering, spatial coordinate mapping,
    and visual element extraction using PyMuPDF (fitz).
    
    This engine is designed to be the foundation for VLM-based OCR workflows
    where precise positioning and image quality are critical.
    """

    def __init__(self, options: Optional[VisualProcessingOptions] = None):
        self.options = options or VisualProcessingOptions()
        logger.info("Initializing VisualDocumentEngine with DPI=%d", self.options.dpi)

    def open_document(self, file_path: Union[str, Path]) -> fitz.Document:
        """
        Safely opens a document and returns the fitz Document handle.
        Caller is responsible for closing or using as a context manager.
        """
        path_str = str(file_path)
        if not os.path.exists(path_str):
            logger.error("File not found: %s", path_str)
            raise FileNotFoundError(f"Could not find document at {path_str}")
        
        try:
            doc = fitz.open(path_str)
            logger.debug("Successfully opened document: %s (Pages: %d)", path_str, len(doc))
            return doc
        except Exception as e:
            logger.error("Failed to open document %s: %s", path_str, str(e))
            raise RuntimeError(f"PyMuPDF failed to process document: {str(e)}")

    def render_page_as_image(self, page: fitz.Page) -> Image.Image:
        """
        Renders a specific page to a high-DPI PIL Image.
        """
        zoom = self.options.dpi / 72  # PDF points are 72 dpi
        matrix = fitz.Matrix(zoom, zoom)
        
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        logger.debug("Rendered Page %d at %d DPI (%dx%d)", 
                     page.number + 1, self.options.dpi, pix.width, pix.height)
        return img

    def extract_visual_elements(self, doc: fitz.Document) -> List[DocumentVisualElement]:
        """
        Scans the document for visual elements (images) and records their spatial coordinates.
        This provides the mapping needed for 'Lossless' re-insertion.
        """
        elements = []
        
        for page_num, page in enumerate(doc):
            # 1. Extract Images
            image_list = page.get_images(full=True)
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                
                # Get spatial coordinates of where the image is displayed on the page
                # Note: One xref might be displayed multiple times; we want all instances
                inst_list = page.get_image_info(xrefs=True)
                for inst in inst_list:
                    if inst['xref'] == xref:
                        # Ensure coordinates are a list of standard floats (not fitz objects/tuples)
                        bbox = [float(v) for v in inst['bbox']] 
                        
                        img_data = base_image['image']
                        img_digest = hashlib.md5(img_data).hexdigest()
                        
                        elements.append(DocumentVisualElement(
                            element_type='image',
                            page_number=page_num + 1,
                            bbox=bbox,
                            content=img_data, 
                            metadata={
                                'xref': xref,
                                'ext': base_image['ext'],
                                'width': base_image['width'],
                                'height': base_image['height'],
                                'colorspace': base_image['colorspace'],
                                'digest': img_digest # Deterministic MD5
                            }
                        ))
        
        logger.info("Extracted %d visual elements from document.", len(elements))
        return elements

    def get_page_layout_map(self, page: fitz.Page) -> Dict[str, Any]:
        """
        Returns a rich dictionary representing the visual layout of a page.
        Includes text blocks, image blocks, and their precise coordinates.
        """
        return page.get_text("dict")

    def create_visual_summary(self, file_path: Union[str, Path]) -> Iterator[Tuple[int, Image.Image, List[DocumentVisualElement]]]:
        """
        A high-level generator that processes a document page by page.
        Yields: (page_number, page_image, list_of_visual_elements_on_page)
        """
        with self.open_document(file_path) as doc:
            all_elements = self.extract_visual_elements(doc)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_img = self.render_page_as_image(page)
                
                page_elements = [e for e in all_elements if e.page_number == page_num + 1]
                
                yield (page_num + 1, page_img, page_elements)

    @staticmethod
    def get_engine_version() -> str:
        """Returns the underlying MuPDF version."""
        return fitz.version[0]

# --- Expert Usage Implementation ---
# This static method provides a senior-level entry point for one-off processing
def process_document_visually(path: Union[str, Path], dpi: int = 300) -> List[Dict[str, Any]]:
    """
    Expert utility to quickly get a visual inventory of a document.
    """
    engine = VisualDocumentEngine(VisualProcessingOptions(dpi=dpi))
    results = []
    
    for page_num, img, elements in engine.create_visual_summary(path):
        results.append({
            "page": page_num,
            "dimensions": img.size,
            "element_count": len(elements),
            "elements": [vars(e) for e in elements]
        })
    
    return results
