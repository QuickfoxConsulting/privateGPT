import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from private_gpt.di import global_injector
from private_gpt.components.ocr_components.visual_engine import VisualDocumentEngine
from private_gpt.components.ocr_components.vlm_client import NuMarkdownClient

logger = logging.getLogger(__name__)

class VisionPDFReader(BaseReader):
    """
    Vision-Aware PDF Reader.
    
    Uses Physical/Visual Coordinate Mapping via VisualDocumentEngine to 
    ensure high-fidelity ingestion. This reader is the 'Optical Eye' that 
    prepares documents for VLM-based OCR (NuMarkdown).
    """

    def __init__(self, 
                 visual_engine: Optional[VisualDocumentEngine] = None,
                 vlm_client: Optional[NuMarkdownClient] = None):
        """
        Initialize the Vision Ingestion Pipeline.
        """
        self._visual_engine = visual_engine or global_injector.get(VisualDocumentEngine)
        self._vlm_client = vlm_client or global_injector.get(NuMarkdownClient)
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
                for page_num, page_img, visual_elements in self._visual_engine.create_visual_summary(file_path):
                    page_index = page_num - 1
                    page_obj = doc[page_index]
                    
                    # High-fidelity text extraction using PyMuPDF blocks
                    blocks = page_obj.get_text("blocks")
                    page_text = "\n".join([b[4] for b in blocks if b[4].strip()])
                    
                    # 3. Vision Decision Engine (OCR vs Layout)
                    final_visual_elements = []
                    
                    # Case A: Scanned Document (No Text Layer)
                    if not page_text.strip() and visual_elements:
                        logger.info("Page %d: Scanned content detected. Triggering Full-Page VLM OCR.", page_num)
                        full_transcription = self._vlm_client.transcribe_image(page_img)
                        page_text = full_transcription.text
                        
                        # Create a virtual 'full_page_ocr' element to ensure the reconstruction engine
                        # has a target to stamp the full-page text into.
                        page_rect = page_obj.rect
                        virtual_element_bbox = [float(page_rect.x0), float(page_rect.y0), 
                                                float(page_rect.x1), float(page_rect.y1)]
                        
                        final_visual_elements = [{
                            "type": "full_page_ocr",
                            "bbox": virtual_element_bbox,
                            "metadata": {
                                "transcription": full_transcription.text,
                                "confidence": full_transcription.confidence,
                                "reflection": full_transcription.reflection
                            }
                        }]
                        
                        # Tag confidence for RAG transparency
                        page_text += f"\n\n[OCR_CONFIDENCE: {full_transcription.confidence:.2f}]"
                    
                    # Case B: Hybrid Document (Text + Images)
                    elif visual_elements:
                        logger.info("Page %d: Hybrid content detected. Transcribing %d visual elements.", 
                                    page_num, len(visual_elements))
                        # Describe images to provide context to the RAG
                        element_transcriptions = self._vlm_client.describe_visual_elements(
                            page_img, 
                            [{"type": e.element_type, "bbox": e.bbox} for e in visual_elements]
                        )
                        visual_context = "\n\n### Visual Context ###\n" + "\n".join(
                            [f"{t.text} [CONF: {t.confidence:.2f}]" for t in element_transcriptions]
                        )
                        page_text += visual_context

                        # Decorate the visual element metadata and prepare for serialization
                        for i, e in enumerate(visual_elements):
                            if i < len(element_transcriptions):
                                e.metadata["transcription"] = element_transcriptions[i].text
                                e.metadata["confidence"] = element_transcriptions[i].confidence
                                e.metadata["reflection"] = element_transcriptions[i].reflection
                                
                            final_visual_elements.append({
                                "type": e.element_type,
                                "bbox": [float(val) for val in e.bbox],
                                "metadata": e.metadata
                            })
                    
                    # 4. Construct the Vision-Aware Metadata
                    metadata = {
                        **doc_metadata,
                        "page": page_num,
                        "has_visuals": len(final_visual_elements) > 0,
                        "visual_element_count": len(final_visual_elements),
                        "visual_elements": final_visual_elements
                    }

                    # Create LlamaIndex Document object
                    llama_doc = Document(
                        text=page_text,
                        metadata=metadata,
                        id_=f"{doc_metadata['document_id']}_p{page_num}"
                    )
                    
                    # Exclude heavy visual metadata from LLM/Embedding context 
                    # but keep it for retrieval/rendering
                    llama_doc.excluded_embed_metadata_keys.extend(["visual_elements", "visual_element_count"])
                    llama_doc.excluded_llm_metadata_keys.extend(["visual_elements", "visual_element_count"])
                    
                    documents.append(llama_doc)

            logger.info("VisionPDFReader completed processing %s (%d pages)", 
                        file_path.name, len(documents))
            return documents

        except Exception as e:
            logger.error("VisionPDFReader failed to process %s: %s", file_path, str(e))
            raise RuntimeError(f"Optical processing failed: {str(e)}")
