import io
import logging
from pathlib import Path
from typing import List

from docx import Document as DocxDocument
from PIL import Image

from private_gpt.components.ocr_components.discovery.base import (
    BaseDiscoveryProvider,
    DiscoveryElement,
    DiscoveryResult,
    DiscoveryType,
)

logger = logging.getLogger(__name__)

class DOCXDiscoveryProvider(BaseDiscoveryProvider):
    """
    Discovery Provider for Microsoft Word (.docx) files.
    
    Extracts text paragraphs and embedded images from the document structure.
    """

    @property
    def supported_extensions(self) -> List[str]:
        return ["docx"]

    def discover(self, file_path: Path) -> DiscoveryResult:
        logger.info("Opening DOCX for discovery: %s", file_path.name)
        doc = DocxDocument(file_path)
        elements = []

        # 1. Extract Text Paragraphs
        # Note: In Word, "pages" are dynamic. We record page=1 for now,
        # or we could try to estimate pages if needed.
        for para in doc.paragraphs:
            if para.text.strip():
                elements.append(DiscoveryElement(
                    element_type=DiscoveryType.TEXT,
                    content=para.text,
                    page_number=1,
                    metadata={"style": para.style.name}
                ))

        # 2. Extract Embedded Images
        # Every image in a docx is stored in the 'related_parts' of the document.
        # We look for parts that have 'image' in their content type.
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                try:
                    image_part = rel.target_part
                    image_bytes = image_part.blob
                    image_ext = image_part.content_type.split('/')[-1]
                    
                    # Convert to PIL Image for the unified Discovery interface
                    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    
                    elements.append(DiscoveryElement(
                        element_type=DiscoveryType.IMAGE,
                        content=img,
                        page_number=1,
                        metadata={
                            "extension": image_ext,
                            "rel_id": rel.rId,
                            "part_name": image_part.partname
                        }
                    ))
                    logger.debug("Discovered image in DOCX: %s", image_part.partname)
                except Exception as e:
                    logger.warning("Failed to extract image from DOCX relation %s: %s", rel.rId, str(e))

        logger.info("Discovery complete for %s. Found %d elements.", file_path.name, len(elements))
        return DiscoveryResult(
            file_path=file_path,
            file_type="docx",
            elements=elements
        )
