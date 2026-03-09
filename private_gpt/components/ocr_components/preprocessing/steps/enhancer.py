import cv2
import numpy as np
from typing import Any, Dict

from private_gpt.components.ocr_components.preprocessing.base import (
    BasePreprocessingStep,
    PreprocessingTask,
)

class CLAHEEnhancerStep(BasePreprocessingStep):
    """
    Contrast Limited Adaptive Histogram Equalization.
    Enhances local contrast without over-amplifying noise.
    """
    
    @property
    def name(self) -> str:
        return "clahe"

    def run(self, task: PreprocessingTask, options: Dict[str, Any]) -> PreprocessingTask:
        img = task.image
        
        # CLAHE requires a 1-channel grayscale image
        is_color = len(img.shape) == 3
        if is_color:
            # If color, convert to Lab and apply to L channel (lightness)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)
        else:
            # Direct grayscale application
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(img)
            
        task.update_image(enhanced)
        return task
