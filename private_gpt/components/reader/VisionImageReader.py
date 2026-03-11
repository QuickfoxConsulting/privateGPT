import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from private_gpt.di import global_injector
from private_gpt.components.ocr_components.visual_engine import VisualDocumentEngine
from private_gpt.components.ocr_components.inference.vlm_engine import VlmOcrEngine

logger = logging.getLogger(__name__)

class VisionImageReader(BaseReader):
    """
    Vision-Aware Image Reader (Discovery & Inference Mode).
    
    Treats a standalone image as a 'one-page document'.
    Uses VLM Inference for high-fidelity OCR.
    """

    def __init__(self, 
                 visual_engine: Optional[VisualDocumentEngine] = None,
                 vlm_engine: Optional[VlmOcrEngine] = None):
        self._visual_engine = visual_engine or global_injector.get(VisualDocumentEngine)
        self._vlm_engine = vlm_engine or global_injector.get(VlmOcrEngine)

    def load_data(self, file_path: Union[str, Path], extra_info: Optional[Dict] = None) -> List[Document]:
        file_path = Path(file_path)
        
        try:
            logger.info("VisionImageReader: Processing standalone image %s", file_path.name)
            
            # 1. Image Discovery
            summary = list(self._visual_engine.create_visual_summary(file_path))
            if not summary:
                raise ValueError("Could not process image through Visual Engine")
            
            page_num, page_image, _, ocr_prompt = summary[0]
            
            # 2. VLM Inference
            logger.info("VisionImageReader: Running VLM Inference...")
            result = self._vlm_engine.infer_strip(page_image, custom_prompt=ocr_prompt)
            transcription_text = result.raw_text or ""
            
            # 3. Metadata
            metadata = {
                "file_name": file_path.name,
                "document_id": str(uuid.uuid4()),
                "page": 1,
                "document_class": "scanned",
                "render_dpi": self._visual_engine.options.dpi,
                "page_width_px": page_image.width,
                "page_height_px": page_image.height,
                "element_type": "standalone_image",
                **(extra_info or {})
            }

            llama_doc = Document(
                text=transcription_text, 
                metadata=metadata,
                id_=f"{metadata['document_id']}_img"
            )
            
            logger.info("VisionImageReader completed OCR for %s", file_path.name)
            return [llama_doc]

        except Exception as e:
            logger.error("VisionImageReader failed for %s: %s", file_path, str(e))
            raise RuntimeError(f"Image visual processing failed: {str(e)}")
