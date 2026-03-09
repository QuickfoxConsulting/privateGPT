import logging
from typing import Any, Dict, List, Optional

from PIL import Image
import numpy as np

from private_gpt.components.ocr_components.preprocessing.base import (
    BasePreprocessingStep,
    PreprocessingTask,
)

logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    """
    Orchestrates the document image enhancement process.
    """

    def __init__(self, steps: List[BasePreprocessingStep]):
        self.steps = steps
        logger.info("Initialized PreprocessingPipeline with %d steps: %s", 
                    len(steps), [s.name for s in steps])

    def run(self, pil_image: Image.Image, options: Optional[Dict[str, Any]] = None) -> PreprocessingTask:
        """
        Runs the full pipeline on a PIL Image.
        Returns a PreprocessingTask containing the final image and spatial metadata.
        """
        options = options or {}
        
        # 1. Convert PIL to OpenCV (NumPy) format
        opencv_img = np.array(pil_image.convert("RGB"))
        # RGB -> BGR for OpenCV
        opencv_img = opencv_img[:, :, ::-1].copy()
        
        task = PreprocessingTask(
            image=opencv_img,
            original_size=pil_image.size
        )
        
        # 2. Sequential Execution
        for step in self.steps:
            logger.debug("Running preprocessing step: %s", step.name)
            task = step.run(task, options)
            
        return task
