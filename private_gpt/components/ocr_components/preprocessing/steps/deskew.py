import cv2
import numpy as np
import logging
from typing import Any, Dict

from private_gpt.components.ocr_components.preprocessing.base import (
    BasePreprocessingStep,
    PreprocessingTask,
)

logger = logging.getLogger(__name__)

class DeskewStep(BasePreprocessingStep):
    """
    Detects document orientation using Hough Line Transform
    and rotates it to zero-degree level.
    """
    
    @property
    def name(self) -> str:
        return "deskew"

    def run(self, task: PreprocessingTask, options: Dict[str, Any]) -> PreprocessingTask:
        img = task.image
        
        # 1. Edge detection for line finding
        # If already grayscale, use as-is; otherwise convert
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Threshold to binary for better edge finding
        # We use Otsu's thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 2. Find all lines using Hough Transform
        lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is None:
            logger.info("Deskew: No lines detected. Skipping deskewing.")
            return task

        # 3. Calculate median angle of detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Filter for near-horizontal lines (-45 to 45 degree range)
            if -45 < angle < 45:
                angles.append(angle)
                
        if not angles:
            logger.info("Deskew: No near-horizontal lines found. Skipping.")
            return task
            
        median_angle = np.median(angles)
        
        # Threshold: if skew is minimal, don't rotate (avoid interpolation artifacts)
        if abs(median_angle) < 0.2:
            logger.info("Deskew: Minimal skew detected (%.2f°). Skipping.", median_angle)
            return task
            
        logger.info("Deskew: Correcting skew of %.2f degrees.", median_angle)
        
        # 4. Perform Rotation
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        
        # Rotation Matrix (2x3)
        rot_mat = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        
        # Rotate image (using white border for new pixels)
        rotated_img = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        
        # 5. Spatial Tracking: Update Transformation Matrix
        # Convert 2x3 Affine matrix to 3x3 for composition
        affine_3x3 = np.eye(3)
        affine_3x3[0:2, 0:3] = rot_mat
        
        task.update_image(rotated_img, matrix=affine_3x3)
        
        return task
