import io
import logging
import time
from typing import Optional

from PIL import Image

from private_gpt.settings.settings import Settings
from private_gpt.components.ocr_components.inference.base import OcrResult
from private_gpt.components.ocr_components.inference.prompts import VLM_OCR_COT_PROMPT, VLM_OCR_SIMPLE_PROMPT

logger = logging.getLogger(__name__)


class GeminiOcrEngine:
    """
    OCR inference engine that uses Google Gemini's vision capabilities.

    Implements the same infer_strip() interface as VlmOcrEngine so it can
    be used as a drop-in delegate.
    """

    def __init__(self, settings: Settings):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai is required for Gemini OCR. "
                "Install it with: pip install google-generativeai"
            )

        self.config = settings.ocr
        api_key = settings.gemini.api_key
        if not api_key:
            raise ValueError("Gemini API key is not configured. Set gemini.api_key in settings.")

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(self.config.gemini_model)
        logger.info("-> OCR: [GEMINI] Initialized GeminiOcrEngine with model: %s", self.config.gemini_model)

    def infer_strip(self, image: Image.Image, custom_prompt: Optional[str] = None) -> OcrResult:
        """
        Performs OCR on a single image using Gemini vision.
        """
        # Determine the prompt
        if custom_prompt:
            prompt = custom_prompt
        elif self.config.enable_cot:
            prompt = VLM_OCR_COT_PROMPT
        else:
            prompt = VLM_OCR_SIMPLE_PROMPT

        logger.info("-> OCR: [GEMINI] Sending image (%dx%d) to model %s",
                     image.width, image.height, self.config.gemini_model)

        start_time = time.time()
        try:
            response = self._model.generate_content(
                [prompt, image],
                generation_config={
                    "temperature": 0.0,
                    "max_output_tokens": getattr(self.config, "max_tokens", 1536),
                }
            )

            elapsed = time.time() - start_time
            content = response.text.strip()

            logger.info("-> OCR: [GEMINI] Received response (Time: %.2fs, Length: %d chars)", elapsed, len(content))
            logger.debug("-> OCR: [GEMINI] Raw content: %s", content)

            # Parse transcription markers (same logic as VlmOcrEngine)
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
                    logger.info("-> OCR: [GEMINI] Found marker '%s', extracted %d chars.", marker, len(raw_text))
                    found_marker = True
                    break

            if not found_marker:
                logger.info("-> OCR: [GEMINI] No markers found. Using raw content (%d chars).", len(raw_text))

            return OcrResult(
                raw_text=raw_text.strip(),
                analysis=analysis,
                metadata={
                    "provider": "gemini",
                    "model": self.config.gemini_model,
                    "time": elapsed,
                    "raw_content": content
                }
            )

        except Exception as e:
            logger.error("-> OCR: [GEMINI] Inference failed: %s", str(e))
            return OcrResult(
                raw_text="",
                analysis=None,
                metadata={"error": str(e), "provider": "gemini"}
            )
