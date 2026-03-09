import hashlib
import logging
import os
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import fitz  # PyMuPDF
from PIL import Image
from pydantic import BaseModel, Field

from private_gpt.components.ocr_components.preprocessing.pipeline import PreprocessingPipeline
from private_gpt.components.ocr_components.preprocessing.steps.grayscale import GrayscaleStep
from private_gpt.components.ocr_components.preprocessing.steps.deskew import DeskewStep
from private_gpt.components.ocr_components.preprocessing.steps.enhancer import CLAHEEnhancerStep
from private_gpt.components.ocr_components.preprocessing.steps.denoise import DenoiseStep
from private_gpt.components.ocr_components.layout.engine import VisualLayoutEngine
from private_gpt.components.ocr_components.layout.base import LayoutBlock

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
    
    # Preprocessing Settings
    enable_preprocessing: bool = Field(default=True, description="Enable the modular preprocessing pipeline")
    preprocessing_steps: List[str] = Field(
        default=["grayscale", "deskew", "clahe", "denoise"],
        description="Ordered list of preprocessing steps to run"
    )
    denoise_h: int = Field(default=10, description="Denoising strength (h parameter)")
    enable_layout_extraction: bool = Field(default=True, description="Enable mathematical bounding box extraction")

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
        
        # Initialize Preprocessing Pipeline if enabled
        self.preprocessing_pipeline = None
        if self.options.enable_preprocessing:
            steps = []
            step_map = {
                "grayscale": GrayscaleStep(),
                "deskew": DeskewStep(),
                "clahe": CLAHEEnhancerStep(),
                "denoise": DenoiseStep()
            }
            for step_name in self.options.preprocessing_steps:
                if step_name in step_map:
                    steps.append(step_map[step_name])
            
            self.preprocessing_pipeline = PreprocessingPipeline(steps)
            logger.info("Preprocessing Pipeline initialized with steps: %s", self.options.preprocessing_steps)

        # Initialize Layout Engine if enabled
        self.layout_engine = None
        if self.options.enable_layout_extraction:
            self.layout_engine = VisualLayoutEngine()
            logger.info("Visual Layout Engine initialized for morphological bounding box extraction.")

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

    def render_page_as_image(self, page: fitz.Page) -> Tuple[Image.Image, Optional[np.ndarray]]:
        """
        Renders a specific page to a high-DPI PIL Image and applies preprocessing.
        Returns: (PIL Image, Optional Transformation Matrix)
        """
        zoom = self.options.dpi / 72  # PDF points are 72 dpi
        matrix = fitz.Matrix(zoom, zoom)
        
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        transformation_matrix = None
        if self.preprocessing_pipeline:
            logger.debug("Applying preprocessing to page %d", page.number + 1)
            # Pass options like denoise_h down if needed (pipeline currently uses dict)
            task = self.preprocessing_pipeline.run(img, {"denoise_h": self.options.denoise_h})
            # Convert OpenCV back to PIL if needed, or update doc to use CV
            # For now, let's keep the engine returning PIL for compatibility
            processed_cv = task.image
            if len(processed_cv.shape) == 2: # Grayscale
                img = Image.fromarray(processed_cv)
            else: # BGR to RGB
                img = Image.fromarray(processed_cv[:, :, ::-1])
            
            transformation_matrix = task.transformation_matrix

        logger.debug("Rendered and Preprocessed Page %d at %d DPI (%dx%d)", 
                     page.number + 1, self.options.dpi, img.width, img.height)
        return img, transformation_matrix

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

    def create_visual_summary(self, file_path: Union[str, Path]) -> Iterator[Tuple[int, Image.Image, List[DocumentVisualElement], Optional[np.ndarray], List[LayoutBlock]]]:
        """
        A high-level generator that processes a document page by page.
        Yields: (page_number, page_image, list_of_visual_elements_on_page, transformation_matrix, layout_blocks)
        """
        with self.open_document(file_path) as doc:
            all_elements = self.extract_visual_elements(doc)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_img, transform = self.render_page_as_image(page)
                
                page_elements = [e for e in all_elements if e.page_number == page_num + 1]
                
                layout_blocks = []
                if self.layout_engine:
                    layout_blocks = self.layout_engine.extract_layout(page_img)
                
                yield (page_num + 1, page_img, page_elements, transform, layout_blocks)

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
    
    for page_num, img, elements, transform, layout_blocks in engine.create_visual_summary(path):
        results.append({
            "page": page_num,
            "dimensions": img.size,
            "element_count": len(elements),
            "elements": [vars(e) for e in elements],
            "transformation_matrix": transform.tolist() if transform is not None else None,
            "layout_blocks": [vars(b) for b in layout_blocks] if layout_blocks else []
        })
    
    return results
