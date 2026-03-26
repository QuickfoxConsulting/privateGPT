import base64
import io
import logging
import time
from typing import Dict, List, Optional, Tuple
from injector import inject, singleton

import httpx
from PIL import Image

from private_gpt.settings.settings import Settings

from private_gpt.components.ocr_components.inference.base import OcrResult
from private_gpt.components.ocr_components.inference.prompts import VLM_OCR_COT_PROMPT, VLM_OCR_SIMPLE_PROMPT

logger = logging.getLogger(__name__)

@singleton
class VlmOcrEngine:
    """
    Inference engine that connects to a vLLM server (OpenAI-compatible)
    to perform high-precision OCR using Vision-Language Models.

    When settings.ocr.use_gemini is True, delegates all inference to
    GeminiOcrEngine instead.
    """

    @inject
    def __init__(self, settings: Settings):
        self.config = settings.ocr
        self._gemini_delegate = None

        if self.config.use_gemini:
            from private_gpt.components.ocr_components.inference.gemini_ocr_engine import GeminiOcrEngine
            logger.info("-> OCR: use_gemini=True. Initializing Gemini delegate...")
            self._gemini_delegate = GeminiOcrEngine(settings)
        else:
            logger.info("-> OCR: Initializing VlmOcrEngine at %s using model: %s", 
                        self.config.base_url, self.config.model_name)

    def _encode_image(self, image: Image.Image) -> str:
        """Converts a PIL image to a base64 encoded string, with resizing to fit VLM context."""
        # VLM models (like Nanonets 3B) usually have a 4k context window.
        # A 2550x3300 image is too big. 1600px width is optimal for OCR fidelity vs context.
        MAX_WIDTH = 1600
        if image.width > MAX_WIDTH:
            scale = MAX_WIDTH / image.width
            new_size = (MAX_WIDTH, int(image.height * scale))
            logger.info("-> OCR: Resizing image from %dx%d to %dx%d for VLM context limits", 
                        image.width, image.height, new_size[0], new_size[1])
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def infer_strip(self, image: Image.Image, custom_prompt: Optional[str] = None) -> OcrResult:
        """
        Performs OCR on a single image.
        """
        # Delegate to Gemini if configured
        if self._gemini_delegate:
            return self._gemini_delegate.infer_strip(image, custom_prompt)

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
            "max_tokens": getattr(self.config, "max_tokens", 1024) # Reduced to 1024 for safety
        }

        logger.info("-> OCR: Sending request to VLM at %s", self.config.base_url)
        logger.info("-> OCR: Model: %s, Max Tokens: %s", self.config.model_name, payload["max_tokens"])
        
        start_time = time.time()
        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                response = client.post(
                    f"{self.config.base_url}/chat/completions",
                    json=payload
                )
                if response.status_code != 200:
                    logger.error("-> OCR: VLM Server returned %d: %s", response.status_code, response.text)
                response.raise_for_status()
                data = response.json()
                
            elapsed = time.time() - start_time
            if "choices" not in data or not data["choices"]:
                logger.error("-> OCR: Invalid VLM response structure: %s", data)
                raise ValueError("No choices in VLM response")

            content = data["choices"][0]["message"]["content"].strip()
            
            logger.info("-> OCR: Received response from VLM (Time: %.2fs)", elapsed)
            logger.debug("-> OCR: Raw VLM Content: %s", content)
            
            # For 3B models, return the raw content if no specific markers are found
            raw_text = content
            analysis = None
            
            transcription_markers = [
                "**2. Raw Transcription:**", 
                "**Transcription:**",
                "Raw Transcription:"
            ]
            
            found_marker = False
            for marker in transcription_markers:
                if marker in content:
                    parts = content.split(marker)
                    raw_text = parts[1].strip()
                    logger.info("-> OCR: Found marker '%s', extracted %d chars of text.", marker, len(raw_text))
                    found_marker = True
                    break
            
            if not found_marker:
                logger.info("-> OCR: No markers found. Using raw content (%d chars).", len(raw_text))

            logger.info("-> OCR: Extracting exact text block below:\n" + "="*40 + "\n" + raw_text + "\n" + "="*40)

            return OcrResult(
                raw_text=raw_text.strip(),
                analysis=analysis,
                metadata={
                    "tokens": data.get("usage", {}).get("total_tokens", 0), 
                    "time": elapsed,
                    "raw_content": content
                }
            )

        except Exception as e:
            logger.error("-> OCR: VLM Inference failed: %s", str(e))
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
