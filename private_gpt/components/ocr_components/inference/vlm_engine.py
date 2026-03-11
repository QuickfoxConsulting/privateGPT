import base64
import io
import logging
import time
from typing import Dict, List, Optional, Tuple

import httpx
from PIL import Image

from private_gpt.components.ocr_components.inference.base import OcrResult, VlmInferenceConfig
from private_gpt.components.ocr_components.inference.prompts import VLM_OCR_COT_PROMPT, VLM_OCR_SIMPLE_PROMPT

logger = logging.getLogger(__name__)

class VlmOcrEngine:
    """
    Inference engine that connects to a vLLM server (OpenAI-compatible)
    to perform high-precision OCR using Vision-Language Models.
    """

    def __init__(self, config: Optional[VlmInferenceConfig] = None):
        self.config = config or VlmInferenceConfig()
        logger.info("Initializing VlmOcrEngine at %s using model: %s", 
                    self.config.base_url, self.config.model_name)

    def _encode_image(self, image: Image.Image) -> str:
        """Converts a PIL image to a base64 encoded string."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def infer_strip(self, image: Image.Image, custom_prompt: Optional[str] = None) -> OcrResult:
        """
        Performs OCR on a single image strip.
        """
        b64_image = self._encode_image(image)
        
        # Determine the prompt
        if custom_prompt:
            prompt = custom_prompt
        elif self.config.enable_cot:
            prompt = VLM_OCR_COT_PROMPT
        else:
            prompt = VLM_OCR_SIMPLE_PROMPT

        payload = {
            "model": self.config.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_image}"}
                        }
                    ]
                }
            ],
            "temperature": 0.0,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.5,
            "max_tokens": self.config.max_tokens
        }

        start_time = time.time()
        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                response = client.post(
                    f"{self.config.base_url}/chat/completions",
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                
            elapsed = time.time() - start_time
            content = data["choices"][0]["message"]["content"].strip()
            
            # For 3B models, return the raw content if no specific markers are found
            raw_text = content
            analysis = None
            
            transcription_markers = [
                "**2. Raw Transcription:**", 
                "**Transcription:**",
                "Raw Transcription:"
            ]
            
            for marker in transcription_markers:
                if marker in content:
                    parts = content.split(marker)
                    raw_text = parts[1].strip()
                    break
            
            logger.info("VLM Inference successful (Time: %.2fs)", elapsed)
            return OcrResult(
                raw_text=raw_text,
                analysis=analysis,
                metadata={
                    "tokens": data.get("usage", {}).get("total_tokens", 0), 
                    "time": elapsed,
                    "raw_content": content
                }
            )

        except Exception as e:
            logger.error("VLM Inference failed: %s", str(e))
            return OcrResult(
                raw_text="",
                analysis=None,
                metadata={"error": str(e)}
            )

    def batch_infer_strips(self, strips: List[Image.Image]) -> List[OcrResult]:
        """Processes multiple strips sequentially."""
        results = []
        for i, strip in enumerate(strips):
            logger.info("Processing strip %d/%d...", i + 1, len(strips))
            results.append(self.infer_strip(strip))
        return results
