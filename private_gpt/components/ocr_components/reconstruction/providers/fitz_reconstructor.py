import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import fitz  # PyMuPDF

from private_gpt.components.ocr_components.reconstruction.base import (
    BaseReconstructionProvider,
    ReconstructionResult
)
from private_gpt.constants import UPLOAD_DIR

logger = logging.getLogger(__name__)

class FitzReconstructionProvider(BaseReconstructionProvider):
    """
    Uses PyMuPDF (fitz) to create searchable PDFs by layering text on images.
    This replaces the need for ReportLab as Fitz is already in the project.
    """

    def reconstruct(
        self, 
        original_images: List[Image.Image], 
        text_locations: List[List[Dict[str, Any]]],
        output_name: str,
        original_path: Optional[Path] = None
    ) -> ReconstructionResult:
        """
        Creates a searchable PDF. If original_path is provided, it non-destructively 
        appends OCR text to the original PDF. Otherwise, it rebuilds the PDF from 
        images (legacy mode).
        """
        try:
            non_destructive = original_path is not None and original_path.exists()
            doc = fitz.open(str(original_path)) if non_destructive else fitz.open()
            
            # Legacy constructive mode: rebuild from images
            if not non_destructive:
                for i, img in enumerate(original_images):
                    img_path = Path(UPLOAD_DIR) / f"temp_recon_{i}.png"
                    img.save(img_path)
                    w, h = img.size
                    page = doc.new_page(width=w, height=h)
                    page.insert_image(page.rect, filename=str(img_path))
                    if img_path.exists():
                        img_path.unlink()
            
            # Text Injection Phase
            for i, page_locations in enumerate(text_locations):
                if i >= len(doc):
                    break
                    
                page = doc[i]
                w, h = page.rect.width, page.rect.height
                
                for segment in page_locations:
                    text = segment.get('text', '')
                    bbox = segment.get('bbox', [])
                    
                    if len(bbox) == 4 and text:
                        x0, y0, x1, y1 = bbox
                        
                        is_vlm = segment.get("is_vlm", False)
                        
                        if non_destructive and not is_vlm:
                            # Skip double-injecting original text, it's already perfectly there
                            continue
                            
                        # Format check: scanned pages using normalized < 1.0 fallbacks must be scaled up.
                        # However, perfectly mapped absolute coordinates (from hybrids) should be kept verbatim!
                        is_normalized_fallback = (x1 <= 1.5 and y1 <= 1.5)
                        rect = fitz.Rect(x0 * w, y0 * h, x1 * w, y1 * h) if is_normalized_fallback else fitz.Rect(x0, y0, x1, y1)
                        
                        try:
                            # Use insert_textbox instead of insert_text!
                            # insert_textbox automatically wraps text at word boundaries 
                            # and scales font sizes appropriately to perfectly fit the mapped bounding box.
                            page.insert_textbox(
                                rect,
                                text,
                                fontsize=12,            # Starter max-size
                                fontname="helv",
                                align=0,                # 0 = Left Align
                                render_mode=3           # 3 = Invisible (searchable) text
                            )
                        except Exception as text_err:
                            logger.warning("Failed to insert textbox segment on page %d: %s", i + 1, str(text_err))

            # Save the final PDF
            output_path = Path(UPLOAD_DIR) / f"{output_name}_enhanced.pdf"
            page_count = len(doc)
            doc.save(str(output_path))
            doc.close()
            
            logger.info("Reconstructed PDF saved to %s (Non-Destructive: %s)", output_path, non_destructive)
            return ReconstructionResult(
                success=True, 
                output_path=output_path,
                metadata={"page_count": page_count, "backend": "fitz", "non_destructive": non_destructive}
            )
            
        except Exception as e:
            logger.error("Fitz reconstruction failed: %s", str(e))
            return ReconstructionResult(success=False, error=str(e))
