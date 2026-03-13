import logging
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from typing import List, Tuple
from private_gpt.components.ocr_components.detection.base import (
    BaseDetectionProvider,
    DetectionBox,
    DetectionResult,
)

logger = logging.getLogger(__name__)

class OnnxDetectorProvider(BaseDetectionProvider):
    """
    Minimalist Text Detection Provider using a DBNet ONNX model.
    
    This is the "Finder" in our Google Lens flow.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._session = None
        self._input_name = None
        self._output_name = None
        logger.info("Initializing OnnxDetectorProvider with model: %s", model_path)

    def _ensure_session(self):
        if self._session is None:
            # We prefer GPU but fallback to CPU
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self._session = ort.InferenceSession(self.model_path, providers=providers)
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            logger.info("ONNX Inference session established for detection")

    def _preprocess(self, image: Image.Image) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
        """Prepares image for detection (Grayscale, Resize to 600x800)."""
        # Convert to Grayscale as expected by robertknight/ocrs model
        img = np.array(image.convert("L"))
        h, w = img.shape
        
        # Model expects fixed 800x600 (H x W)
        target_w, target_h = 600, 800
        scale_x = target_w / w
        scale_y = target_h / h
        
        resized = cv2.resize(img, (target_w, target_h))
        
        # Model expects [1, 1, 800, 600] float32
        blob = (resized.astype(np.float32) / 255.0)
        blob = np.expand_dims(blob, axis=0) # [800, 600] -> [1, 800, 600]
        blob = np.expand_dims(blob, axis=0) # [1, 800, 600] -> [1, 1, 800, 600]
        
        return blob, (scale_x, scale_y), (0, 0)

    def detect(self, image: Image.Image) -> DetectionResult:
        self._ensure_session()
        
        orig_w, orig_h = image.size
        blob, (scale_x, scale_y), _ = self._preprocess(image)
        
        outputs = self._session.run([self._output_name], {self._input_name: blob})
        heatmap = outputs[0][0][0] # First batch, first channel
        
        # Post-process: Thresholding to create binary map
        # Slightly lowered threshold to 0.25 to catch faint signals
        threshold = 0.25
        binary_map = (heatmap > threshold).astype(np.uint8) * 255
        
        # This model uses fixed 800x600, so binary_map is already 600x800
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for i, cnt in enumerate(contours):
            # x, y, w, h in the 800x600 space
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Normalize to 0-1 (relative to 600x800)
            target_w = 600
            target_h = 800
            box = [
                x / target_w,
                y / target_h,
                (x + w) / target_w,
                (y + h) / target_h
            ]
            
            # Filter small noise
            if w * h < 20: 
                continue
                
            # Confidence estimation from heatmap
            conf = float(np.mean(heatmap[y:y+h, x:x+w])) if w > 0 and h > 0 else 0.0
            
            boxes.append(DetectionBox(
                bbox=box,
                confidence=conf,
                label="text",
                metadata={"index": i}
            ))
            
        logger.info("Detection complete: Found %d text blocks", len(boxes))
        return DetectionResult(
            boxes=boxes,
            original_size=(orig_w, orig_h)
        )
