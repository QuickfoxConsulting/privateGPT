import logging
from pathlib import Path
from typing import List

from PIL import Image

from private_gpt.components.ocr_components.discovery.base import (
    BaseDiscoveryProvider,
    DiscoveryElement,
    DiscoveryResult,
    DiscoveryType,
)

logger = logging.getLogger(__name__)

class ImageDiscoveryProvider(BaseDiscoveryProvider):
    """
    Discovery Provider for standalone Image files (JPG, PNG).
    
    Treats an image as a single-page 'scanned' document elements.
    """

    @property
    def supported_extensions(self) -> List[str]:
        return ["jpg", "jpeg", "png", "webp"]

    def discover(self, file_path: Path) -> DiscoveryResult:
        logger.info("Surveying Image for discovery: %s", file_path.name)
        
        try:
            img = Image.open(file_path).convert("RGB")
            width, height = img.size
            
            # For standalone images, the 'discovery' is the image itself.
            element = DiscoveryElement(
                element_type=DiscoveryType.IMAGE,
                content=img,
                page_number=1,
                bbox=[0, 0, width, height],
                metadata={"is_full_page_scan": True, "source": "standalone_image"}
            )
            
            return DiscoveryResult(
                file_path=file_path,
                file_type="image",
                elements=[element]
            )
        except Exception as e:
            logger.error("Failed to discover image %s: %s", file_path.name, str(e))
            raise RuntimeError(f"Image discovery failed: {str(e)}")
