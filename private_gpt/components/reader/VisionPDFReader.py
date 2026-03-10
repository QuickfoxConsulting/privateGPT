import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from private_gpt.di import global_injector
from private_gpt.components.ocr_components.visual_engine import VisualDocumentEngine

logger = logging.getLogger(__name__)

class VisionPDFReader(BaseReader):
    """
    Vision-Aware PDF Reader.
    
    Uses Physical/Visual Coordinate Mapping via VisualDocumentEngine to 
    extract spatial metadata for layout processing.
    """

    def __init__(self, 
                 visual_engine: Optional[VisualDocumentEngine] = None):
        """
        Initialize the Vision Ingestion Pipeline.
        """
        self._visual_engine = visual_engine or global_injector.get(VisualDocumentEngine)
        logger.debug("VisionPDFReader initialized with engine version %s", 
                     self._visual_engine.get_engine_version())

    def load_data(self, file_path: Union[str, Path], extra_info: Optional[Dict] = None) -> List[Document]:
        """
        Synchronously loads and processes a PDF with visual awareness.
        """
        file_path = Path(file_path)
        documents = []
        
        try:
            with self._visual_engine.open_document(file_path) as doc:
                # 1. Capture Document-Level Metadata
                doc_metadata = {
                    "file_name": file_path.name,
                    "total_pages": len(doc),
                    "document_id": str(uuid.uuid4()),
                    **(extra_info or {})
                }

                # 2. Process Page by Page (The Visual Loop)
                for page_num, page_img, visual_elements, transform, layout_blocks in self._visual_engine.create_visual_summary(file_path):
                    page_index = page_num - 1
                    page_obj = doc[page_index]
                    
                    # Store transformation matrix in metadata
                    transform_list = transform.tolist() if transform is not None else None
                    
                    # High-fidelity text extraction using PyMuPDF blocks
                    blocks = page_obj.get_text("blocks")
                    page_text = "\n".join([b[4] for b in blocks if b[4].strip()])
                    
                    # 3. Vision Decision Engine (OCR vs Layout)
                    document_class = "digital"  # Default: pure digital text
                    
                    # Case A: Scanned Document or Pure Image (No Text Layer)
                    if not page_text.strip():
                        document_class = "scanned"
                        logger.info("Page %d: Scanned content detected. OCR disabled for now.", page_num)
                    
                    # Case B: Hybrid Document (Text + Images)
                    elif visual_elements:
                        document_class = "hybrid"
                        logger.info("Page %d: Hybrid content detected.", page_num)
                    
                    # 4. Construct the Vision-Aware Metadata
                    metadata = {
                        **doc_metadata,
                        "page": page_num,
                        "document_class": document_class,
                        "render_dpi": self._visual_engine.options.dpi,
                        "page_width_px": page_img.width,
                        "page_height_px": page_img.height,
                        "has_visuals": len(visual_elements) > 0,
                        "transformation_matrix": transform_list,
                        "layout_blocks": [vars(b) for b in layout_blocks] if layout_blocks else []
                    }

                    # Create LlamaIndex Document object
                    llama_doc = Document(
                        text=page_text,
                        metadata=metadata,
                        id_=f"{doc_metadata['document_id']}_p{page_num}"
                    )
                    
                    # Exclude heavy visual metadata from LLM/Embedding context 
                    # but keep it for retrieval/rendering
                    llama_doc.excluded_embed_metadata_keys.extend(["layout_blocks"])
                    llama_doc.excluded_llm_metadata_keys.extend(["layout_blocks"])
                    
                    documents.append(llama_doc)

            logger.info("VisionPDFReader completed processing %s (%d pages)", 
                        file_path.name, len(documents))
            return documents

        except Exception as e:
            logger.error("VisionPDFReader failed to process %s: %s", file_path, str(e))
            raise RuntimeError(f"Optical processing failed: {str(e)}")
