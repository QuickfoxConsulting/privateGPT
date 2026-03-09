import logging
import io
from pathlib import Path
from typing import List, Optional

import fitz
from PIL import Image

from private_gpt.components.ocr_components.discovery.base import (
    BaseDiscoveryProvider,
    DiscoveryElement,
    DiscoveryResult,
    DiscoveryType,
)

logger = logging.getLogger(__name__)

class PDFDiscoveryProvider(BaseDiscoveryProvider):
    """
    Discovery Provider for Adobe PDF files.
    
    Acts as the 'Surveyor' for PDFs, extracting digital text blocks,
    embedded image objects, and identifying scanned pages.
    """

    def __init__(self, dpi: int = 300):
        super().__init__()
        self.dpi = dpi

    @property
    def supported_extensions(self) -> List[str]:
        return ["pdf"]

    def discover(self, file_path: Path) -> DiscoveryResult:
        logger.info("Surveying PDF for discovery: %s", file_path.name)
        doc = fitz.open(file_path)
        elements = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            current_page = page_num + 1
            
            # 1. Extract Digital Text Blocks
            blocks = page.get_text("blocks")
            for block in blocks:
                text = block[4].strip()
                if text:
                    elements.append(DiscoveryElement(
                        element_type=DiscoveryType.TEXT,
                        content=text,
                        page_number=current_page,
                        bbox=list(block[:4]),
                        metadata={"block_no": block[6]}
                    ))

            # 2. Extract Embedded Image Objects (Hybrid Mode)
            image_list = page.get_images(full=True)
            for img_info in image_list:
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image['image']
                    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    
                    # We need the spatial position of where this image is on the page
                    # Note: An image xref can appear multiple times. We look for instances.
                    # For discovery, we just take the first display instance of this xref on this page.
                    img_info_list = page.get_image_info(xrefs=True)
                    for info in img_info_list:
                        if info['xref'] == xref:
                            bbox = list(info['bbox'])
                            elements.append(DiscoveryElement(
                                element_type=DiscoveryType.IMAGE,
                                content=img,
                                page_number=current_page,
                                bbox=bbox,
                                metadata={
                                    "xref": xref,
                                    "ext": base_image['ext'],
                                    "width": base_image['width'],
                                    "height": base_image['height']
                                }
                            ))
                            break # Only record one instance for discovery purposes
                except Exception as e:
                    logger.warning("Failed to extract PDF image xref %d: %s", xref, str(e))

            # 3. Scanned Content Detection
            # If no text was found, we flag the whole page for Full-Page Rendering
            # (The Discovery Engine records this as a specialized request)
            if not any(e.element_type == DiscoveryType.TEXT and e.page_number == current_page for e in elements):
                logger.info("Page %d of %s appears to be scanned. Flagging as Visual Element.", current_page, file_path.name)
                
                # Render the page as a high-DPI image for the VLM/Preprocssing
                zoom = self.dpi / 72
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                elements.append(DiscoveryElement(
                    element_type=DiscoveryType.IMAGE, # Full-page image
                    content=img,
                    page_number=current_page,
                    bbox=list(page.rect),
                    metadata={"is_full_page_scan": True}
                ))

        doc.close()
        logger.info("Discovery complete for %s. Found %d elements.", file_path.name, len(elements))
        return DiscoveryResult(
            file_path=file_path,
            file_type="pdf",
            elements=elements
        )

# Add missing import to docx.py too in next step if needed
import io 
