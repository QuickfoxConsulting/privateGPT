import logging
import re
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

class MDDiscoveryProvider(BaseDiscoveryProvider):
    """
    Discovery Provider for Markdown (.md) files.
    
    Extracts text and finds image links pointing to local files.
    """

    @property
    def supported_extensions(self) -> List[str]:
        return ["md", "txt"]

    def discover(self, file_path: Path) -> DiscoveryResult:
        logger.info("Reading Markdown for discovery: %s", file_path.name)
        text_content = file_path.read_text(encoding='utf-8')
        elements = []

        # 1. Add Full Text
        elements.append(DiscoveryElement(
            element_type=DiscoveryType.TEXT,
            content=text_content,
            page_number=1
        ))

        # 2. Discover Images via Markdown Syntax ![](/path/to/img)
        # We look for local image paths.
        img_pattern = r'!\[.*?\]\((.*?)\)'
        matches = re.finditer(img_pattern, text_content)
        
        for i, match in enumerate(matches):
            img_path_str = match.group(1).strip()
            
            # Resolve relative paths
            img_path = file_path.parent / img_path_str
            if img_path.exists() and img_path.suffix.lower().lstrip('.') in ["jpg", "png", "jpeg", "tif"]:
                try:
                    img = Image.open(img_path).convert("RGB")
                    elements.append(DiscoveryElement(
                        element_type=DiscoveryType.IMAGE,
                        content=img,
                        page_number=1,
                        metadata={"original_path": str(img_path), "discovery_index": i}
                    ))
                    logger.debug("Discovered local image in MD: %s", img_path_str)
                except Exception as e:
                    logger.warning("Failed to load discovered image %s: %s", img_path_str, str(e))

        return DiscoveryResult(
            file_path=file_path,
            file_type="md",
            elements=elements
        )
