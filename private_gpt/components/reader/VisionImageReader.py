import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from private_gpt.di import global_injector
from private_gpt.components.ocr_components.visual_engine import VisualDocumentEngine

logger = logging.getLogger(__name__)

class VisionImageReader(BaseReader):
    """
    Vision-Aware Image Reader.
    
    Treats a standalone image as a 'one-page document'.
    Prepares images for OCR/Transcription with proper visual handling.
    """

    def __init__(self, 
                 visual_engine: Optional[VisualDocumentEngine] = None):
        self._visual_engine = visual_engine or global_injector.get(VisualDocumentEngine)

    def load_data(self, file_path: Union[str, Path], extra_info: Optional[Dict] = None) -> List[Document]:
        file_path = Path(file_path)
        
        # Captures the image context and metadata.
        try:
            # 1. Process through the Visual Engine (Applies Preprocessing automatically)
            logger.info("VisionImageReader: Processing standalone image %s", file_path.name)
            
            # create_visual_summary yields (page_num, image, elements, transform)
            # For standalone images, there is only one page.
            summary = list(self._visual_engine.create_visual_summary(file_path))
            if not summary:
                raise ValueError("Could not process image through Visual Engine")
            
            page_num, processed_img, visual_elements, transform, layout_blocks = summary[0]
            
            # OCR is temporarily disabled
            transcription_text = ""
            
            # Store transformation matrix for possible coordinate mapping
            transform_list = transform.tolist() if transform is not None else None
            
            metadata = {
                "file_name": file_path.name,
                "document_id": str(uuid.uuid4()),
                "page": 1,
                "document_class": "scanned",
                "render_dpi": self._visual_engine.options.dpi,
                "page_width_px": processed_img.width,
                "page_height_px": processed_img.height,
                "has_visuals": True,
                "element_type": "standalone_image",
                "transformation_matrix": transform_list,
                "layout_blocks": [vars(b) for b in layout_blocks] if layout_blocks else [],
                **(extra_info or {})
            }

            llama_doc = Document(
                text=transcription_text, 
                metadata=metadata,
                id_=f"{metadata['document_id']}_img"
            )
            
            logger.info("VisionImageReader captured visual context and applied preprocessing for %s", file_path.name)
            return [llama_doc]

        except Exception as e:
            logger.error("VisionImageReader failed for %s: %s", file_path, str(e))
            raise RuntimeError(f"Image visual processing failed: {str(e)}")
