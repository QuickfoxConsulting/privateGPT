import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from private_gpt.di import global_injector
from private_gpt.components.ocr_components.visual_engine import VisualDocumentEngine
from private_gpt.components.ocr_components.inference.vlm_engine import VlmOcrEngine
from private_gpt.components.ocr_components.reconstruction.engine import ReconstructionEngine

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
                
                all_pages_images: List[Image.Image] = []
                all_pages_locations: List[List[Dict[str, Any]]] = []
                
                for page_num, page_image, detected_boxes, ocr_prompt in self._visual_engine.create_visual_summary(file_path):
                    page_index = page_num - 1
                    page_obj = doc[page_index]
                    
                    # 1. Digital Text Extraction
                    logger.debug("-> READER: Processing Page %d. Attempting digital text extraction...", page_num)
                    blocks = page_obj.get_text("blocks")
                    digital_text = "\n".join([b[4] for b in blocks if b[4].strip()])
                    
                    # 2. Vision/OCR Discovery (Detection of Hybrid/Scanned)
                    # Only count images LARGER than passport-size (~100x127 PDF pts ≈ 3.5x4.5cm)
                    # This filters out small logos, icons, divider lines, and decorative elements.
                    MIN_IMAGE_WIDTH_PT = 100   # ~35mm in PDF points
                    MIN_IMAGE_HEIGHT_PT = 127  # ~45mm in PDF points
                    
                    images = page_obj.get_images(full=True)
                    significant_images = []
                    for img in images:
                        # get_images(full=True) returns: (xref, smask, width, height, bpc, colorspace, alt_colorspace, name, filter, referencer)
                        img_width = img[2]   # pixel width
                        img_height = img[3]  # pixel height
                        # Estimate PDF-point size using page DPI (images are embedded at ~72-150 DPI typically)
                        # A safer approach: find the image bbox on the page
                        try:
                            img_rects = page_obj.get_image_rects(img[0])  # xref
                            for rect in img_rects:
                                rect_w = rect.width   # in PDF points
                                rect_h = rect.height  # in PDF points
                                if rect_w >= MIN_IMAGE_WIDTH_PT and rect_h >= MIN_IMAGE_HEIGHT_PT:
                                    significant_images.append(img)
                                    logger.debug("-> READER: Page %d significant image: %.0fx%.0f pts", page_num, rect_w, rect_h)
                                    break
                        except Exception:
                            # Fallback: if we can't get rect, check raw pixel size (assume ~150 DPI)
                            pt_w = img_width * 72 / 150
                            pt_h = img_height * 72 / 150
                            if pt_w >= MIN_IMAGE_WIDTH_PT and pt_h >= MIN_IMAGE_HEIGHT_PT:
                                significant_images.append(img)
                    
                    has_text = bool(digital_text.strip())
                    has_significant_images = len(significant_images) > 0
                    # NOTE: has_drawings removed — PDF drawings (table borders, lines, underlines) 
                    # are NOT visual content worth triggering VLM OCR for.
                    
                    if not has_text:
                        document_class = "scanned"
                    elif has_significant_images:
                        document_class = "hybrid"
                    else:
                        document_class = "digital"
                    
                    logger.info("-> READER: [DISCOVERY] Page %d detected as %s. Images: %d total, %d significant (>passport-size). CNN boxes: %d.", 
                                page_num, document_class.upper(), len(images), len(significant_images), len(detected_boxes))
                    
                    # 3. Decision Logic
                    page_text = digital_text
                    
                    # Store original digital text blocks for perfect reconstruction
                    # Blocks format: (x0, y0, x1, y1, "text", block_no, block_type)
                    text_locations = [
                        {
                            "text": b[4].strip(), 
                            "bbox": [b[0], b[1], b[2], b[3]] # Absolute PyMuPDF points
                        } 
                        for b in blocks if b[4].strip()
                    ]

                    if document_class == "scanned":
                        logger.info("-> READER: Page %d is SCANNED. Triggering VLM OCR on entire page...", page_num)
                        
                        result = self._vlm_engine.infer_strip(page_image, custom_prompt=ocr_prompt)
                        vlm_text = result.raw_text or ""
                        page_text = vlm_text
                        
                    elif document_class == "hybrid":
                        logger.info("-> READER: Page %d is HYBRID. Extracting and performing OCR only on %d significant images...", 
                                    page_num, len(significant_images))
                        vlm_text_parts = []
                        import io
                        for img_info in significant_images:
                            xref = img_info[0]
                            try:
                                base_image = doc.extract_image(xref)
                                image_bytes = base_image["image"]
                                pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                                
                                logger.info("-> READER: Submitting cropped image object (xref=%s, size=%dx%d) to VLM OCR...", 
                                            xref, pil_img.width, pil_img.height)
                                result = self._vlm_engine.infer_strip(pil_img, custom_prompt=ocr_prompt)
                                if result.raw_text:
                                    vlm_text_parts.append(result.raw_text)
                                    
                                    # PERFECT MAPPING: the text was found precisely in this physical image bounding box!
                                    try:
                                        img_rects = page_obj.get_image_rects(xref)
                                        if img_rects:
                                            r = img_rects[0]
                                            text_locations.append({
                                                "text": result.raw_text,
                                                "bbox": [r.x0, r.y0, r.x1, r.y1],
                                                "is_vlm": True
                                            })
                                    except Exception as rect_e:
                                        logger.warning("-> READER: Could not map image rect for xref %s: %s", xref, rect_e)
                            except Exception as e:
                                logger.warning("-> READER: Failed to extract/OCR image xref %s: %s", xref, e)

                        vlm_text = "\n\n".join(vlm_text_parts)
                        page_text = (
                            f"{digital_text}\n\n"
                            f"--- VISUAL CONTENT DESCRIPTION (from VLM) ---\n"
                            f"{vlm_text}"
                        )
                        
                    elif document_class == "scanned":
                        logger.info("-> READER: Page %d is SCANNED. Triggering VLM OCR on entire page...", page_num)
                        
                        result = self._vlm_engine.infer_strip(page_image, custom_prompt=ocr_prompt)
                        vlm_text = result.raw_text or ""
                        page_text = vlm_text
                        
                        if detected_boxes and vlm_text:
                            vlm_locations = self._align_text_to_boxes(vlm_text, detected_boxes)
                            for v in vlm_locations:
                                v["is_vlm"] = True
                            text_locations.extend(vlm_locations)
                        else:
                            text_locations.append({
                                "text": vlm_text,
                                "bbox": [0, 0, 1.0, 1.0], # Fallback to normalized full-page bounds
                                "is_vlm": True
                            })
                    
                    # Collect page image and aligned locations for reconstruction
                    all_pages_images.append(page_image)
                    all_pages_locations.append(text_locations)

                    # 4. Construct Metadata
                    metadata = {
                        **doc_metadata,
                        "page": page_num,
                        "document_class": document_class,
                        "render_dpi": self._visual_engine.options.dpi,
                        "text_locations": text_locations, # Store all boxes + text mappings
                        "detected_box_count": len(detected_boxes)
                    }

                    llama_doc = Document(
                        text=page_text,
                        metadata=metadata,
                        id_=f"{doc_metadata['document_id']}_p{page_num}"
                    )
                    documents.append(llama_doc)

            # 5. Optional: Trigger Reconstruction (Searchable PDF)
            try:
                # We use the injector to get the engine, or fallback to direct init if needed
                try:
                    recon_engine = global_injector.get(ReconstructionEngine)
                except Exception:
                    from private_gpt.components.ocr_components.reconstruction.providers.fitz_reconstructor import FitzReconstructionProvider
                    recon_engine = ReconstructionEngine([FitzReconstructionProvider()])
                
                output_name = file_path.stem
                recon_result = recon_engine.reconstruct(
                    original_images=all_pages_images, 
                    text_locations=all_pages_locations, 
                    output_name=output_name,
                    original_path=file_path
                )
                
                if recon_result.success:
                    logger.info("-> READER: Enhanced (Searchable) PDF created successfully: %s", recon_result.output_path)
                    # Attach to metadata of all docs in this file
                    for llama_doc in documents:
                        llama_doc.metadata["enhanced_pdf_path"] = str(recon_result.output_path)
            except Exception as e:
                logger.warning("-> READER: Optional reconstruction failed: %s", str(e))

            logger.info("-> READER: VisionPDFReader completed processing %s (%d documents created)", 
                        file_path.name, len(documents))
            return documents

        except Exception as e:
            logger.error("VisionPDFReader failed to process %s: %s", file_path, str(e))
            raise RuntimeError(f"Optical processing failed: {str(e)}")

    def _align_text_to_boxes(self, text: str, boxes: List) -> List[Dict]:
        """
        Heuristic alignment of VLM text lines to CNN bounding boxes.
        
        Assumes both follow a roughly top-to-bottom reading order.
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        # Sort boxes by Y then X
        sorted_boxes = sorted(boxes, key=lambda b: (b.bbox[1], b.bbox[0]))
        
        aligned = []
        # Simple greedy alignment: map N-th line to N-th box
        # Real-world logic would use spatial overlap or fuzzy matching, 
        # but for a "Google Lens" first-pass, this is the lightest path.
        for i, line in enumerate(lines):
            if i < len(sorted_boxes):
                box = sorted_boxes[i]
                aligned.append({
                    "text": line,
                    "bbox": box.bbox,
                    "confidence": box.confidence
                })
        return aligned
