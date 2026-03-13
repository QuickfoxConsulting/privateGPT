import logging
import os
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

# Minimalist DBNet ONNX model from Hugging Face (robertknight/ocrs)
# This is a robust and fast model for general text detection.
DEFAULT_MODEL_URL = "https://huggingface.co/robertknight/ocrs/resolve/main/text-detection-ssfbcj81.onnx"

def ensure_detection_model(target_dir: str = "models/ocr") -> str:
    """
    Ensures that the text detection model is present.
    Downloads it if missing.
    """
    os.makedirs(target_dir, exist_ok=True)
    model_path = os.path.join(target_dir, "en_PP-OCRv3_det_infer.onnx")
    
    if not os.path.exists(model_path):
        logger.info("Text detection model missing. Downloading from %s...", DEFAULT_MODEL_URL)
        try:
            response = requests.get(DEFAULT_MODEL_URL, stream=True, timeout=30)
            response.raise_for_status()
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("Detection model downloaded successfully to %s", model_path)
        except Exception as e:
            logger.error("Failed to download detection model: %s", str(e))
            # Fallback or re-raise
            raise RuntimeError(f"Could not obtain detection model: {e}")
            
    return os.path.abspath(model_path)
