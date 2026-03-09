import logging
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from private_gpt.components.ocr_components.layout.base import LayoutBlock, LayoutType

logger = logging.getLogger(__name__)

class VisualLayoutEngine:
    """
    Analyzes document structures using mathematical morphological operations.
    Extracts high-precision bounding boxes for text lines and structural elements
    without needing a Neural Network.
    """

    def __init__(self, dilation_kernel_size: Tuple[int, int] = (60, 5),
                 min_block_area: int = 200,
                 min_width_ratio: float = 0.03,
                 min_height_px: int = 8):
        """
        Args:
            dilation_kernel_size: (width, height) - Controls how aggressively 
                characters are merged. (60,5) merges words into full sentence lines.
            min_block_area: Minimum pixel area a block must have to survive filtering.
            min_width_ratio: Minimum width as a fraction of the image width (0.10 = 10%).
                Blocks narrower than this are treated as noise or stray words.
            min_height_px: Minimum height in pixels. Removes thin horizontal artifacts.
        """
        self.dilation_kernel_size = dilation_kernel_size
        self.min_block_area = min_block_area
        self.min_width_ratio = min_width_ratio
        self.min_height_px = min_height_px

    def extract_layout(self, image: Image.Image) -> List[LayoutBlock]:
        """
        Executes the "Strikethrough" logic to find bounding boxes.
        Assumes the input image has already been deskewed by the Preprocessing Engine.
        """
        logger.info("Starting Morphological Layout Extraction.")
        
        # 1. Convert to OpenCV Format
        img_np = np.array(image)
        # Handle RGB vs Grayscale input
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np.copy()

        img_height, img_width = gray.shape[:2]
        min_width_px = int(img_width * self.min_width_ratio)

        # 2. Adaptive Binarization (Background Removal)
        # We invert it (THRESH_BINARY_INV) so text is White (255) and bg is Black (0)
        # This is strictly required for dilation to "smear" the ink
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15,  # Block size
            10   # C constant
        )

        # Remove small noise dots before dilation
        kernel_noise = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_noise, iterations=1)

        # 3. Morphological Dilation ("Strikethrough" Logic)
        # A wide kernel (60x5) merges adjacent characters into full sentence-level bars.
        # This is much more aggressive than word-level (25x3), ensuring small
        # fragments are absorbed into the nearest sentence line.
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, self.dilation_kernel_size)
        dilated = cv2.dilate(binary, kernel_line, iterations=2)

        # Light vertical closing to merge lines that are very close together
        # (e.g., superscripts, subscripts touching the main line)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        # 4. Coordinate Extraction (Contour Finding)
        # RETR_EXTERNAL gets only the outer boundaries, ignoring loops inside words
        contours, hierarchy = cv2.findContours(
            dilated, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )

        blocks = []
        rejected = 0

        for idx, contour in enumerate(contours):
            # Get the exact bounding box of the solid "strikethrough" shape
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # === STRICT FILTERING (Sentence-Level Only) ===
            # Filter 1: Minimum area - removes tiny dust specks
            if area < self.min_block_area:
                rejected += 1
                continue

            # Filter 2: Minimum width - removes isolated words or short fragments
            # A real sentence line should span at least 10% of the page width
            if w < min_width_px:
                rejected += 1
                continue

            # Filter 3: Minimum height - removes thin horizontal artifacts (rules, lines)
            if h < self.min_height_px:
                rejected += 1
                continue

            # Classification based on shape
            block_type = LayoutType.TEXT_LINE
            aspect_ratio = w / float(h)
            
            # Very tall blocks (multi-line paragraphs merged together) = TEXT_BLOCK
            if h > 100: 
                block_type = LayoutType.TEXT_BLOCK
            
            # Very wide AND very tall = likely a TABLE or IMAGE region
            if w > img_width * 0.5 and h > img_height * 0.15:
                block_type = LayoutType.TABLE

            blocks.append(LayoutBlock(
                block_type=block_type,
                bbox=[x, y, w, h],
                metadata={
                    "contour_index": idx,
                    "area": area,
                    "aspect_ratio": round(aspect_ratio, 2)
                }
            ))

        # Sort blocks top-to-bottom, left-to-right
        blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
        
        # Assign reading order and page dimensions to each block
        for order_idx, block in enumerate(blocks):
            block.metadata["reading_order"] = order_idx
            block.metadata["page_width_px"] = img_width
            block.metadata["page_height_px"] = img_height
        
        logger.info("Layout extraction complete. Found %d structural blocks (%d noise blocks rejected).", 
                     len(blocks), rejected)
        return blocks
