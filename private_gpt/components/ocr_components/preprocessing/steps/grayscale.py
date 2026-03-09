import cv2
from typing import Any, Dict

from private_gpt.components.ocr_components.preprocessing.base import (
    BasePreprocessingStep,
    PreprocessingTask,
)

class GrayscaleStep(BasePreprocessingStep):
    """Simple step to convert BGR/Color image to Grayscale."""
    
    @property
    def name(self) -> str:
        return "grayscale"

    def run(self, task: PreprocessingTask, options: Dict[str, Any]) -> PreprocessingTask:
        # Avoid double-grayscaling
        if len(task.image.shape) == 2:
            return task
            
        gray = cv2.cvtColor(task.image, cv2.COLOR_BGR2GRAY)
        task.update_image(gray)
        return task
