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

class VisionPDFReader(BaseReader):
    """
    Vision-Aware PDF Reader (Discovery & Inference Mode).
    
    Uses One-Shot High-Fidelity Rendering and VLM Inference for OCR.
    """

    def __init__(self, 
                 visual_engine: Optional[VisualDocumentEngine] = None,
                 vlm_engine: Optional[VlmOcrEngine] = None):
        """
        Initialize the Vision Ingestion Pipeline.
        """
        self._visual_engine = visual_engine or global_injector.get(VisualDocumentEngine)
        self._vlm_engine = vlm_engine or global_injector.get(VlmOcrEngine)
        logger.debug("VisionPDFReader initialized (Clean Discovery Mode)")

    def load_data(self, file_path: Union[str, Path], extra_info: Optional[Dict] = None) -> List[Document]:
        """
        Synchronously loads and processes a PDF with high-fidelity discovery.
        """
        file_path = Path(file_path)
        documents = []
        
        try:
            with self._visual_engine.open_document(file_path) as doc:
                doc_metadata = {
                    "file_name": file_path.name,
                    "total_pages": len(doc),
                    "document_id": str(uuid.uuid4()),
                    **(extra_info or {})
                }

                # Process Page by Page
                for page_num, page_image, _, ocr_prompt in self._visual_engine.create_visual_summary(file_path):
                    page_index = page_num - 1
                    page_obj = doc[page_index]
                    
                    # 1. Digital Text Extraction
                    blocks = page_obj.get_text("blocks")
                    digital_text = "\n".join([b[4] for b in blocks if b[4].strip()])
                    
                    # 2. Vision/OCR Discovery
                    page_text = digital_text
                    document_class = "digital"
                    
                    if not digital_text.strip():
                        # Scanned Page -> Trigger VLM Inference
                        document_class = "scanned"
                        logger.info("Page %d: Scanned content detected. Running VLM Inference...", page_num)
                        result = self._vlm_engine.infer_strip(page_image, custom_prompt=ocr_prompt)
                        page_text = result.raw_text or ""
                    
                    # 3. Construct Metadata
                    metadata = {
                        **doc_metadata,
                        "page": page_num,
                        "document_class": document_class,
                        "render_dpi": self._visual_engine.options.dpi,
                        "page_width_px": page_image.width,
                        "page_height_px": page_image.height,
                    }

                    llama_doc = Document(
                        text=page_text,
                        metadata=metadata,
                        id_=f"{doc_metadata['document_id']}_p{page_num}"
                    )
                    documents.append(llama_doc)

            logger.info("VisionPDFReader completed processing %s (%d pages)", 
                        file_path.name, len(documents))
            return documents

        except Exception as e:
            logger.error("VisionPDFReader failed to process %s: %s", file_path, str(e))
            raise RuntimeError(f"Optical processing failed: {str(e)}")
