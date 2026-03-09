import io
import logging
from pathlib import Path
from typing import List

from openpyxl import load_workbook
from PIL import Image

from private_gpt.components.ocr_components.discovery.base import (
    BaseDiscoveryProvider,
    DiscoveryElement,
    DiscoveryResult,
    DiscoveryType,
)

logger = logging.getLogger(__name__)

class ExcelDiscoveryProvider(BaseDiscoveryProvider):
    """
    Discovery Provider for Microsoft Excel (.xlsx) files.
    
    Extracts cell data strings and floating images/charts from sheets.
    """

    @property
    def supported_extensions(self) -> List[str]:
        return ["xlsx"]

    def discover(self, file_path: Path) -> DiscoveryResult:
        logger.info("Opening Excel for discovery: %s", file_path.name)
        # Using data_only=True to get values instead of formulas
        wb = load_workbook(file_path, data_only=True)
        elements = []

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            
            # 1. Extract Cell Text
            # We aggregate cells into rows to provide context
            for row in ws.iter_rows(values_only=True):
                row_text = " | ".join([str(cell) for cell in row if cell is not None])
                if row_text.strip():
                    elements.append(DiscoveryElement(
                        element_type=DiscoveryType.TEXT,
                        content=row_text,
                        page_number=1,
                        metadata={"sheet_name": sheet_name, "type": "table_row"}
                    ))

            # 2. Extract Floating Images
            # openpyxl stores images in the worksheet's _images list
            if hasattr(ws, "_images"):
                for idx, img_obj in enumerate(ws._images):
                    try:
                        # get_contents() returns the raw image bytes
                        image_bytes = img_obj.ref.data
                        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                        
                        elements.append(DiscoveryElement(
                            element_type=DiscoveryType.IMAGE,
                            content=img,
                            page_number=1,
                            metadata={
                                "sheet_name": sheet_name,
                                "anchor": str(img_obj.anchor),
                                "index": idx
                            }
                        ))
                    except Exception as e:
                        logger.warning("Failed to extract image %d from sheet %s: %s", idx, sheet_name, str(e))

        logger.info("Discovery complete for %s. Found %d elements.", file_path.name, len(elements))
        return DiscoveryResult(
            file_path=file_path,
            file_type="xlsx",
            elements=elements
        )
