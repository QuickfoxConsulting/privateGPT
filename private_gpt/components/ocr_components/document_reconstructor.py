import logging
import os
from pathlib import Path
from typing import List, Union

import re
import unicodedata
import fitz  # PyMuPDF
from llama_index.core.schema import Document
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)

class DocumentReconstructor:
    """
    The 'Stamper' Module: Reconstructs PDFs with searchable VLM data.
    
    This professional component takes the original PDF and the visual transcriptions
    from the VLM "Brain" and merges them into a single, high-fidelity 'Enhanced PDF'.
    
    It preserves the visual integrity of the original file while adding a hidden, 
    searchable text layer behind every visual element.
    """

    def __init__(self, settings: Settings):
        self._vlm_settings = settings.vlm_ocr
        self._output_folder = Path(self._vlm_settings.reconstructed_folder)
        
        if not self._output_folder.exists():
            logger.info("Creating enhanced documents folder at %s", self._output_folder)
            self._output_folder.mkdir(parents=True, exist_ok=True)

    def reconstruct_pdf(self, 
                        original_path: Union[str, Path], 
                        docs: List[Document]) -> Path:
        """
        Creates a new PDF with invisible text layers for all VLM transcriptions.
        
        Args:
            original_path: Path to the original 'dumb' PDF.
            docs: List of LlamaIndex Documents containing VLM metadata.
            
        Returns:
            The Path to the newly generated 'Enhanced' PDF.
        """
        original_path = Path(original_path)
        # Using a deterministic naming convention for easy retrieval
        output_path = self._output_folder / f"enhanced_{original_path.name}"
        
        if not self._vlm_settings.enable_reconstruction:
            logger.debug("PDF Reconstruction is disabled in settings. Skipping.")
            return original_path

        logger.info("Starting High-Fidelity PDF Reconstruction for %s", original_path.name)
        
        try:
            doc = fitz.open(str(original_path))
            
            # Map page numbers to documents for efficient lookup
            # page_num in metadata is 1-indexed
            page_docs = {d.metadata.get('page'): d for d in docs}
            
            reconstructed_count = 0
            
            for page_num in range(1, len(doc) + 1):
                page = doc[page_num - 1]
                
                # --- Multi-Language Font Support (Nepali Optimized) ---
                # We use Noto Sans Devanagari for perfect rendering.
                font_name = "helv" 
                # Pointing to the font inside the private_gpt folder (which is mounted in Docker)
                current_dir = Path(__file__).parent
                font_path = current_dir.parent.parent / "NotoSansDevanagari-Regular.ttf"
                
                logger.debug("Checking for font at: %s", font_path)
                
                try:
                    if font_path.exists():
                        # Register the specialized Devanagari font with complex script support
                        font_name = "nepali-font"
                        # set_simple=False is crucial for Devanagari ligatures
                        font_ref = page.insert_font(fontname=font_name, fontfile=str(font_path), set_simple=False)
                        logger.info("Successfully registered font '%s' (ref: %s) from %s", 
                                    font_name, font_ref, font_path)
                    else:
                        logger.warning("Font file NOT FOUND at %s. Falling back to serif.", font_path)
                        page.insert_font(fontname="serif", fontbuffer=None)
                        font_name = "serif"
                except Exception as font_e:
                    logger.error("Failed to register font: %s. Last resort fallback to helv.", str(font_e))
                    font_name = "helv"
                
                llama_doc = page_docs.get(page_num)
                
                if not llama_doc:
                    continue
                
                visual_elements = llama_doc.metadata.get('visual_elements', [])
                for el in visual_elements:
                    transcription = el.get('metadata', {}).get('transcription')
                    bbox = el.get('bbox')
                    el_type = el.get('type')
                    
                    if transcription and bbox:
                        # 1. Clean the text: Remove Markdown artifacts and normalize Unicode
                        # We want the content, not the syntax, inside the PDF.
                        
                        # Strip Markdown Images and Links: ![alt](url) -> ""
                        clean_text = re.sub(r'!\[.*?\]\(.*?\)', '', transcription)
                        # Strip Markdown Links: [text](url) -> "text"
                        clean_text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', clean_text)
                        
                        # Strip Table structural characters
                        clean_text = re.sub(r'\|', ' ', clean_text)
                        clean_text = re.sub(r'^[ \t]*[-: ]+[-| :]*$', '', clean_text, flags=re.MULTILINE)
                        
                        # Basic Cleanup
                        clean_text = re.sub(r'#+\s*', '', clean_text) # Remove headers
                        clean_text = re.sub(r'\*\*|__', '', clean_text)  # Remove bold
                        
                        # Normalize Unicode (NFKC) to handle combining characters correctly
                        clean_text = unicodedata.normalize('NFKC', clean_text)
                        
                        # Remove non-printable control characters BUT PRESERVE NEWLINES (\n is Cc)
                        # This was the cause of the 'clumping' bug.
                        clean_text = "".join(ch for ch in clean_text if unicodedata.category(ch)[0] != "C" or ch in "\n\r\t")
                        clean_text = clean_text.strip()

                        if not clean_text:
                            continue

                        try:
                            rect = fitz.Rect(bbox)
                            
                            # 2. Advanced Positioning: Prevent "Top Clumping" & "Line Misalignment"
                            if el_type == "full_page_ocr":
                                # Split by individual lines for granular vertical distribution
                                lines = [l.strip() for l in clean_text.split("\n") if l.strip()]
                                
                                logger.debug("Processing full_page_ocr: %d lines found. Snippet: %.20s", 
                                             len(lines), clean_text[:20])

                                if lines:
                                    # We start at 5% from the top and end at 5% from the bottom
                                    # This handles the natural margins of the document
                                    top_margin = rect.height * 0.05
                                    usable_height = rect.height * 0.90
                                    
                                    # Calculate height per line based on total lines
                                    # Add a small buffer between lines
                                    line_step = usable_height / max(len(lines), 1)
                                    
                                    # Ensure fontsize isn't too large for the step
                                    calc_fontsize = min(11, max(7, int(line_step * 0.8)))

                                    for i, line in enumerate(lines):
                                        # Calculate exact Y coordinate for this specific line
                                        line_y0 = rect.y0 + top_margin + (i * line_step)
                                        
                                        # Create a surgical rectangle for this single line
                                        # Width remains the page width (minus small margins)
                                        line_rect = fitz.Rect(rect.x0 + 40, line_y0, 
                                                              rect.x1 - 40, line_y0 + line_step)
                                        
                                        page.insert_textbox(
                                            line_rect, 
                                            line, 
                                            fontname=font_name, # Use our registered font
                                            fontsize=calc_fontsize,
                                            render_mode=3,
                                            align=fitz.TEXT_ALIGN_LEFT
                                        )
                            else:
                                # Standard Stamping for small visual elements
                                logger.debug("Stamping small element [%s]: %.30s...", el_type, clean_text)
                                page.insert_textbox(
                                    rect, 
                                    clean_text, 
                                    fontname=font_name,
                                    fontsize=10, 
                                    render_mode=3,
                                    align=fitz.TEXT_ALIGN_LEFT
                                )
                            reconstructed_count += 1
                        except Exception as inner_e:
                            logger.warning("Failed to stamp element on page %d: %s", page_num, str(inner_e))
            
            # Save the new PDF
            # Use garbage=4 (deflate) to keep file size optimized
            doc.save(str(output_path), garbage=4, deflate=True)
            doc.close()
            
            logger.info("PDF Reconstruction complete. Stamped %d segments. File: %s", 
                        reconstructed_count, output_path)
            return output_path
            
        except Exception as e:
            logger.error("Failed to reconstruct PDF %s: %s", original_path, str(e))
            # We return the original path as a fallback so the ingestion doesn't crash
            return original_path

    @staticmethod
    def get_enhanced_path_for_original(original_name: str, settings: Settings) -> Path:
        """Utility to predict the enhanced path for a given filename."""
        vlm_settings = settings.vlm_ocr
        return Path(vlm_settings.reconstructed_folder) / f"enhanced_{original_name}"
