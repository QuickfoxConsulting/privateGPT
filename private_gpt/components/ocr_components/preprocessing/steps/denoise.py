import cv2
from typing import Any, Dict

from private_gpt.components.ocr_components.preprocessing.base import (
    BasePreprocessingStep,
    PreprocessingTask,
)

class DenoiseStep(BasePreprocessingStep):
    """
    Removes digital and scan noise using Non-Local Means Denoising.
    Preserves edges while smoothing background grain.
    """
    
    @property
    def name(self) -> str:
        return "denoise"

    def run(self, task: PreprocessingTask, options: Dict[str, Any]) -> PreprocessingTask:
        img = task.image
        
        # h=10 is a good default for "typical" scan noise
        # Higher h = more aggressive smoothing
        h = options.get("denoise_h", 10)
        
        if len(img.shape) == 2:
            # Grayscale fastNLMeans
            denoised = cv2.fastNlMeansDenoising(img, None, h, 7, 21)
        else:
            # Colored fastNLMeans (preserves hue better than Gaussian blur)
            denoised = cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)
            
        task.update_image(denoised)
        return task
