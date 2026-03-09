import logging
import re
from pathlib import Path
from typing import List

from striprtf.striprtf import rtf_to_text

from private_gpt.components.ocr_components.discovery.base import (
    BaseDiscoveryProvider,
    DiscoveryElement,
    DiscoveryResult,
    DiscoveryType,
)

logger = logging.getLogger(__name__)

class RTFDiscoveryProvider(BaseDiscoveryProvider):
    """
    Discovery Provider for Rich Text Format (.rtf) files.
    
    Extracts text and identifies embedded OLE objects (visuals).
    Note: RTF image extraction is historically complex; this implementation
    focuses on text and marks if visuals are found.
    """

    @property
    def supported_extensions(self) -> List[str]:
        return ["rtf"]

    def discover(self, file_path: Path) -> DiscoveryResult:
        logger.info("Reading RTF for discovery: %s", file_path.name)
        with open(file_path, 'r', encoding='ascii', errors='ignore') as f:
            rtf_content = f.read()

        # 1. Extract Plain Text
        text = rtf_to_text(rtf_content).strip()
        elements = []
        if text:
            elements.append(DiscoveryElement(
                element_type=DiscoveryType.TEXT,
                content=text,
                page_number=1
            ))

        # 2. Extract Embedded Images
        # RTF images start with '{\pict' or '{\object'
        # We look for hex data blocks.
        image_matches = re.finditer(r'\\pict(.*?)\}', rtf_content, re.DOTALL)
        for i, match in enumerate(image_matches):
            # This is a placeholder for actual Hex -> Image conversion
            # Extraction from RTF hex is brittle without complex parsers.
            # We flag it so the Discovery engine knows visuals exist.
            elements.append(DiscoveryElement(
                element_type=DiscoveryType.UNKNOWN,
                content=f"[IMAGE_PLACEHOLDER_{i}]",
                page_number=1,
                metadata={"type": "embedded_pict", "index": i}
            ))
            logger.debug("Discovered embedded picture in RTF at index %d", i)

        return DiscoveryResult(
            file_path=file_path,
            file_type="rtf",
            elements=elements
        )
