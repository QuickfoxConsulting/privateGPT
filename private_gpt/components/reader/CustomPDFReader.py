
import os
from typing import Dict, List, Optional, Set
import fitz  # PyMuPDF
import pymupdf4llm
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from chonkie import LateChunker
import unicodedata
import re
from docling.document_converter import DocumentConverter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat

@dataclass
class TextBlock:
    text: str
    page_num: int
    block_index: int
    metadata: Dict = None

chunker = LateChunker.from_recipe(
        name="markdown",
        chunk_size=512,
        min_characters_per_chunk=100,
        embedding_model="all-MiniLM-L6-v2", 
        lang="en"
    )

pipeline_options = PdfPipelineOptions(
            # artifacts_path=artifacts_path,
            # do_ocr=True,
            do_table_structure=True,
            # do_code_enrichment=True,          # Enable code enrichment for code snippets
            do_formula_enrichment=True,         # Enable formula enrichment for mathematical formulas
            # do_picture_classification=True,   # Classify images if present
            do_picture_description=True,        # Generate descriptive captions for images
            # generate_page_images=True,        # Capture page images for visual context
            # images_scale=0.8,                 # Adjust the scale of generated images
            table_structure_options=dict(
                mode=TableFormerMode.FAST   # Use an accurate mode for table extraction
            ),
            enable_remote_services=False
        )
CONVERTER = DocumentConverter(
    format_options={
    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
class TextMatcher:
    def __init__(self, similarity_threshold: float = 0.9, overlap_threshold: float = 0.9):
        self.similarity_threshold = similarity_threshold
        self.overlap_threshold = overlap_threshold
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            min_df=1,
            strip_accents='unicode'
        )
    
    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing."""
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'-+\n', ' ', text) 
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip().lower()
    
    def calculate_text_overlap(self, text1: str, text2: str) -> float:
        """Calculate character-level overlap between two texts."""
        text1_chars = set(self.preprocess_text(text1))
        text2_chars = set(self.preprocess_text(text2))
        overlap = len(text1_chars.intersection(text2_chars))
        total = len(text1_chars.union(text2_chars))
        return overlap / total if total > 0 else 0

    def fill_small_gaps(self,  pages: List[int], max_gap: int = 2) -> List[int]:
        if not pages:
            return []
        filled = set()
        for i in range(len(pages) - 1):
            filled.update(range(pages[i], pages[i+1] + 1) if pages[i+1] - pages[i] <= max_gap else [pages[i]])
        filled.add(pages[-1])
        return sorted(filled)

    def find_page_ranges(self, chunk_text: str, text_blocks: List[TextBlock]) -> List[int]:
        """Find page ranges for a chunk using multiple matching strategies."""
        chunk_text = self.preprocess_text(chunk_text)
        
        # Strategy 1: Exact substring matching (highest priority)
        matched_pages = set()
        chunk_sentences = chunk_text.split('.')
        for sentence in chunk_sentences:
            if len(sentence.strip()) < 10: 
                continue
            for block in text_blocks:
                if sentence.strip() in self.preprocess_text(block.text):
                    matched_pages.add(block.page_num)
        
        if matched_pages:
            # Fill gaps in page ranges
            page_list = sorted(matched_pages)
            if len(page_list) > 1:
                matched_pages = set(self.fill_small_gaps(page_list))
            return sorted(matched_pages)
        
        # Strategy 2: TF-IDF similarity + overlap threshold with sliding window
        block_texts = [self.preprocess_text(block.text) for block in text_blocks]
        if not block_texts or not chunk_text:
            return [1]

        # Create overlapping windows of text to better match chunks that cross page boundaries
        window_size = 3
        windowed_texts = []
        windowed_pages = []
        for i in range(len(block_texts)):
            window_text = ' '.join(block_texts[max(0, i - window_size):min(len(block_texts), i + window_size + 1)])
            windowed_texts.append(window_text)
            windowed_pages.append(text_blocks[i].page_num)

        tfidf_matrix = self.vectorizer.fit_transform(windowed_texts + [chunk_text])
        similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

        # Find matches that pass BOTH similarity and overlap thresholds
        matched_indices = []
        for idx, sim in enumerate(similarities):
            if sim >= self.similarity_threshold:
                overlap = self.calculate_text_overlap(chunk_text, windowed_texts[idx])
                if overlap >= self.overlap_threshold:
                    matched_indices.append(idx)

        if matched_indices:
            matched_pages = {windowed_pages[i] for i in matched_indices}
            page_list = sorted(matched_pages)
            if len(page_list) > 1:
                full_range = set(range(min(page_list), max(page_list) + 1))
                matched_pages.update(full_range)
            return sorted(matched_pages)

        # Fallback: best match and neighbors
        best_match_idx = int(np.argmax(similarities))
        matched_pages = {windowed_pages[best_match_idx]}
        if best_match_idx > 0:
            matched_pages.add(windowed_pages[best_match_idx - 1])
        if best_match_idx < len(windowed_pages) - 1:
            matched_pages.add(windowed_pages[best_match_idx + 1])

        return sorted(matched_pages)
    
    
class CustomPDFReader(BaseReader):

    HEADERS_TO_SPLIT_ON = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        # ("####", "Header 4"),
        # ("#####", "Header 5"),
    ]
    def __init__(self, chunk_size: int = 512, similarity_threshold: float = 0.95):
        self.text_matcher = TextMatcher(similarity_threshold=similarity_threshold)

    def _extract_pdf_metadata(self, doc: fitz.Document) -> Dict:
        """Extract comprehensive metadata from PDF."""
        metadata = {
            'title': doc.metadata.get('title', ''),
            'author': doc.metadata.get('author', ''),
            'subject': doc.metadata.get('subject', ''),
            'keywords': doc.metadata.get('keywords', ''),
            'total_pages': len(doc),
        }
        
        # Add creation and modification dates if available
        if doc.metadata.get('creationDate'):
            metadata['creation_date'] = doc.metadata['creationDate']
        if doc.metadata.get('modDate'):
            metadata['modification_date'] = doc.metadata['modDate']
            
        return metadata

    def _extract_page_metadata(self, page: fitz.Page) -> Dict:
        """Extract metadata for a specific page."""
        return {
            'page_number': page.number + 1,
            'width': page.rect.width,
            'height': page.rect.height,
            'rotation': page.rotation,
            'has_images': len(page.get_images()) > 0,
            'has_tables': bool(page.find_tables()), # if using PyMuPDF >= 1.23.0
        }

    def _extract_block_metadata(self, block: tuple, page_num: int) -> Dict:
        """Extract metadata from a text block."""
        x0, y0, x1, y1, text, block_type, block_no = block
        return {
            'block_type': block_type,
            'block_number': block_no,
            'position': {
                'x0': x0, 'y0': y0,
                'x1': x1, 'y1': y1
            },
            'location': 'top' if y0 < 100 else 'bottom' if y1 > 700 else 'middle',
            'is_header': y0 < 100 and len(text.strip()) < 200,
            'is_footer': y1 > 700 and len(text.strip()) < 200,
        }

    def _extract_pdf_text_with_pages(self, pdf_path: str) -> List[TextBlock]:
        """Extracts text and metadata from a PDF while preserving page numbers."""
        text_blocks = []
        doc = fitz.open(pdf_path)
        pdf_metadata = self._extract_pdf_metadata(doc)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_metadata = self._extract_page_metadata(page)
            blocks = page.get_text("blocks")
            
            for idx, block in enumerate(blocks):
                block_metadata = self._extract_block_metadata(block, page_num)
                text_blocks.append(
                    TextBlock(
                        text=block[4].strip(),
                        page_num=page_num + 1,
                        block_index=idx,
                        metadata={
                            **pdf_metadata,
                            **page_metadata,
                            **block_metadata
                        }
                    )
                )
        doc.close()
        return text_blocks

    def load_data(self, pdf_path: str, extra_info: Optional[Dict] = None) -> List[Document]:
        text_blocks = self._extract_pdf_text_with_pages(pdf_path)
        filename = os.path.basename(pdf_path)
        
        try:
            result = CONVERTER.convert(pdf_path)
            md_text = result.document.export_to_markdown()
            # md_text = pymupdf4llm.to_markdown(pdf_path)

        except Exception:
            md_text = "\n\n".join(block.text for block in text_blocks)
        
        blocks_by_page = defaultdict(list)
        for block in text_blocks:
            blocks_by_page[block.page_num].append(block)
        
        late_chunks = chunker(md_text)
        chunks = []
        for chunk in late_chunks:
            text = chunk.text
            page_numbers = self.text_matcher.find_page_ranges(text, text_blocks)
            chunk_metadata = {
                'filename': filename,
                'page': page_numbers,
                'total_pages': max(block.metadata['total_pages'] for block in text_blocks),
                'title': next(iter(text_blocks)).metadata.get('title', ''), 
                'author': next(iter(text_blocks)).metadata.get('author', ''),
            }
            doc = Document( 
                text=text,
                metadata=chunk_metadata
            )
            chunks.append(doc)
        return chunks

