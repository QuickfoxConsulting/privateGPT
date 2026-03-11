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
        logger.info("-> READER: VisionPDFReader [HYBRID_ENGINE_V1] initialized successfully")

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
                logger.info("-> READER: Starting high-fidelity processing of %d pages for %s", len(doc), file_path.name)
                for page_num, page_image, _, ocr_prompt in self._visual_engine.create_visual_summary(file_path):
                    page_index = page_num - 1
                    page_obj = doc[page_index]
                    
                    # 1. Digital Text Extraction
                    logger.debug("-> READER: Processing Page %d. Attempting digital text extraction...", page_num)
                    blocks = page_obj.get_text("blocks")
                    digital_text = "\n".join([b[4] for b in blocks if b[4].strip()])
                    
                    # 2. Vision/OCR Discovery (Detection of Hybrid/Scanned)
                    images = page_obj.get_images()
                    drawings = page_obj.get_drawings()
                    
                    has_text = bool(digital_text.strip())
                    has_images = len(images) > 0
                    has_drawings = len(drawings) > 0 # Even 1 drawing can be a diagram/chart
                    
                    if not has_text:
                        document_class = "scanned"
                    elif has_images or has_drawings:
                        document_class = "hybrid"
                    else:
                        document_class = "digital"
                    
                    reason = []
                    if has_images: reason.append(f"{len(images)} images")
                    if has_drawings: reason.append(f"{len(drawings)} drawings")
                    if has_text: reason.append("digital text")
                    
                    logger.info("-> READER: [DISCOVERY] Page %d detected as %s (%s)", 
                                page_num, document_class.upper(), ", ".join(reason))
                    
                    # 3. Decision Logic
                    page_text = digital_text
                    if document_class in ["scanned", "hybrid"]:
                        logger.info("-> READER: Page %d is %s. Triggering VLM OCR for visual extraction...", 
                                    page_num, document_class.upper())
                        
                        result = self._vlm_engine.infer_strip(page_image, custom_prompt=ocr_prompt)
                        vlm_text = result.raw_text or ""
                        
                        if document_class == "hybrid":
                            # Merge digital text with VLM findings
                            page_text = (
                                f"{digital_text}\n\n"
                                f"--- VISUAL CONTENT DESCRIPTION (from VLM) ---\n"
                                f"{vlm_text}"
                            )
                            logger.info("-> READER: Page %d (HYBRID) merged digital text (%d chars) with VLM text (%d chars)", 
                                        page_num, len(digital_text), len(vlm_text))
                        else:
                            page_text = vlm_text
                            logger.info("-> READER: Page %d (SCANNED) OCR completed. Total: %d chars", 
                                        page_num, len(page_text))
                    else:
                        logger.info("-> READER: Page %d (DIGITAL) skipping OCR. Pure text extraction used (%d chars)", 
                                    page_num, len(digital_text))
                    
                    # 4. Construct Metadata
                    metadata = {
                        **doc_metadata,
                        "page": page_num,
                        "document_class": document_class,
                        "has_visuals": has_images or has_drawings,
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

            logger.info("-> READER: VisionPDFReader completed processing %s (%d documents created)", 
                        file_path.name, len(documents))
            return documents

        except Exception as e:
            logger.error("VisionPDFReader failed to process %s: %s", file_path, str(e))
            raise RuntimeError(f"Optical processing failed: {str(e)}")
