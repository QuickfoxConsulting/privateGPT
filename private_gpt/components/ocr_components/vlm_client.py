import base64
import io
import logging
import json
import re
from typing import List, Optional, Union, Dict, Any

from openai import OpenAI
from PIL import Image
from pydantic import BaseModel

from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)

class VLMTranscription(BaseModel):
    """The structured result of a VLM transcription."""
    text: str
    model: str
    tokens_used: int
    confidence: float = 1.0
    reflection: Optional[str] = None

class NuMarkdownClient:
    """
    The Thinking Brain of the System: NuMarkdown-8B-Thinking Client.
    
    This client communicates with a vLLM server to perform high-fidelity OCR,
    image transcription, and document layout understanding.
    """

    def __init__(self, settings: Settings):
        self._vlm_settings = settings.vlm_ocr
        
        if not self._vlm_settings.enabled:
            logger.warning("VLM OCR is disabled in settings.")
            self._client = None
            return

        logger.info("Initializing NuMarkdownClient for model %s at %s", 
                    self._vlm_settings.model, self._vlm_settings.base_url)
        
        self._client = OpenAI(
            base_url=self._vlm_settings.base_url,
            api_key=self._vlm_settings.api_key
        )

    def _parse_reflection_json(self, text: str) -> Dict[str, Any]:
        """Attempts to extract JSON from the VLM reflection response."""
        # Strip <think> blocks which models like NuMarkdown-8B-Thinking often include
        clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        
        try:
            # Extract JSON from markdown code blocks if present
            match = re.search(r'```json\s*(.*?)\s*```', clean_text, re.DOTALL)
            json_str = match.group(1) if match else clean_text
            # Clean potential trailing commas or junk
            data = json.loads(json_str)
            return data
        except Exception as e:
            logger.warning("Failed to parse VLM reflection JSON: %s. Using default confidence.", str(e))
            # Return a default structure to avoid downstream errors
            return {"confidence": 0.5, "is_accurate": True, "reflection": f"Parsing failed: {str(e)}"}

    def _clean_vlm_output(self, text: str) -> str:
        """
        Removes <think> blocks and extracts content from <answer> tags if they exist.
        Ensures the final text is clean Markdown and NOT a system status message.
        """
        if not text:
            return ""
            
        # 1. Try to extract from <answer> tags
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if answer_match:
            text = answer_match.group(1).strip()
        else:
            # 2. Otherwise, strip <think> blocks
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            
        # 3. Status Message Protection:
        # If the output looks like a meta-commentary instead of document content,
        # we return an empty string or None to signal fallback.
        status_patterns = [
            r"the provided markdown is correct",
            r"no corrections are needed",
            r"transcription is accurate",
            r"i have reviewed",
            r"everything looks correct"
        ]
        text_lower = text.lower()
        if any(re.search(pattern, text_lower) for pattern in status_patterns) and len(text) < 150:
            logger.info("VLM output detected as status message, discarding: %s", text)
            print(f"[LightOnOCR] ⚠️  FILTERED: Discarding system status message ('{text[:30]}...')")
            return ""
            
        return text

    def _encode_image(self, image: Image.Image) -> str:
        """Converts a PIL image to a base64 string for vLLM transmission."""
        buffered = io.BytesIO()
        # Using PNG to preserve lossless quality for OCR
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def transcribe_image(self, image: Image.Image, prompt: Optional[str] = None) -> VLMTranscription:
        """
        Transcribes an image using NuMarkdown-8B.
        Default prompt is tuned for high-fidelity Markdown OCR.
        """
        if not self._client:
            raise RuntimeError("VLM Client is disabled or not initialized.")

        # Senior-level default prompt for NuMarkdown-8B-Thinking
        # Ultra-strict prompt for NuMarkdown-8B-Thinking to prevent 'summarization' behavior
        default_prompt = (
            "You are a professional document extractor. "
            "Transcribe exactly what you see in this image into clean, "
            "well-formatted Markdown. Include all text, tables, and structural elements. "
            "Do not add any preamble or commentary."
        )
        
        final_prompt = prompt or default_prompt
        base64_image = self._encode_image(image)

        try:
            # 1. Initial High-Fidelity Transcription
            response = self._client.chat.completions.create(
                model=self._vlm_settings.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": final_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self._vlm_settings.max_tokens,
                temperature=self._vlm_settings.temperature
            )

            initial_text = response.choices[0].message.content
            total_tokens = response.usage.total_tokens

            # Enhanced Terminal Logging: FULL OUTPUT for Monitoring
            print(f"\n[LightOnOCR] 📝 FULL TRANSCRIPTION EXTRACTED:")
            print("=" * 60)
            print(initial_text)
            print("=" * 60)

            # Pre-clean initial text for the final result
            cleaned_initial_text = self._clean_vlm_output(initial_text)

            # 2. Intellectual Guard: Self-Reflection Pass
            # We ask the VLM to criticize its own output to catch hallucinations
            reflection_prompt = (
                "Review your previous transcription of this image. "
                "Are there any errors in numbers, names, or formatting? "
                "Check for hallucinations. "
                "Respond ONLY with a JSON object in this format: "
                "```json\n"
                "{\n"
                "  \"confidence\": 0.95,\n"
                "  \"is_accurate\": true,\n"
                "  \"reflection\": \"brief analysis of errors\",\n"
                "  \"corrected_markdown\": \"...full corrected markdown ONLY if is_accurate is false...\"\n"
                "}\n"
                "```"
            )

            reflection_response = self._client.chat.completions.create(
                model=self._vlm_settings.model,
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": final_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]},
                    {"role": "assistant", "content": initial_text},
                    {"role": "user", "content": reflection_prompt}
                ],
                max_tokens=1024,
                temperature=0.0  # Force maximum deterministic reflection
            )

            total_tokens += reflection_response.usage.total_tokens
            reflection_data = self._parse_reflection_json(reflection_response.choices[0].message.content)

            # INTELLIGENT LOGIC FIX: 
            # Only switch to corrected_markdown if the model explicitly says it's NOT accurate.
            # This prevents "The provided Markdown is correct" messages from replacing the text.
            is_accurate = reflection_data.get("is_accurate", True)
            corrected_markdown = reflection_data.get("corrected_markdown")
            
            if not is_accurate and corrected_markdown and len(corrected_markdown.strip()) > 50:
                final_text = self._clean_vlm_output(corrected_markdown)
            else:
                final_text = cleaned_initial_text

            confidence = reflection_data.get("confidence", 0.8)
            reflection_note = reflection_data.get("reflection", "No issues identified.")

            logger.info("VLM Reflection completed (Confidence: %.2f). Total Tokens: %d", 
                        confidence, total_tokens)

            print(f"[LightOnOCR] ✅ STAGE: Verification Complete")
            print(f"[LightOnOCR] 📊 METRICS | Confidence: {confidence:.2f} | Total Tokens: {total_tokens}")
            if reflection_note:
                print(f"[LightOnOCR] 🧠 MODEL LOGIC: {reflection_note}\n")

            return VLMTranscription(
                text=final_text,
                model=self._vlm_settings.model,
                tokens_used=total_tokens,
                confidence=confidence,
                reflection=reflection_note
            )

        except Exception as e:
            logger.error("VLM Transcription failed: %s", str(e))
            raise RuntimeError(f"NuMarkdown-8B communication failure: {str(e)}")

    def describe_visual_elements(self, page_image: Image.Image, elements: List[dict]) -> List[VLMTranscription]:
        """
        Takes a full page and a list of coordinate elements, crops them,
        and transcribes each one individually using the 'Think' capability.
        """
        transcriptions = []
        for el in elements:
            bbox = el.get('bbox')
            if not bbox:
                continue
            
            # Crop the image (Mapping Points -> Pixels is handled here)
            # PDF Points (72 DPI) -> Pixel (300 DPI)
            scale = 300 / 72
            left, top, right, bottom = [v * scale for v in bbox]
            
            # Ensure crop is within bounds
            img_w, img_h = page_image.size
            crop_box = (
                max(0, left), 
                max(0, top), 
                min(img_w, right), 
                min(img_h, bottom)
            )
            
            cropped_img = page_image.crop(crop_box)
            print(f"[LightOnOCR] 🖼️  Processing Visual Element: {el['type']} at {bbox}")
            transcription = self.transcribe_image(
                cropped_img, 
                prompt=f"Transcribe this {el['type']} exactly. If it contains text, provide it verbatim. If a table, format as Markdown. No summary."
            )
            transcriptions.append(transcription)
            
        return transcriptions
