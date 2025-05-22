import uuid
import re
from pathlib import Path
from typing import Dict, List, Optional
import fitz  # PyMuPDF
from langchain_text_splitters import MarkdownHeaderTextSplitter
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from collections import defaultdict
from dataclasses import dataclass
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser

@dataclass
class TextBlock:
    text: str
    page_num: int
    block_index: int
    metadata: Dict = None

config = {
    "output_format": "markdown",
    "paginate_output": True,
    "force_ocr": True,
    "strip_existing_ocr": True,
}
config_parser = ConfigParser(config)
marker_converter = PdfConverter(
    artifact_dict=create_model_dict(),
    config=config_parser.generate_config_dict(),
    processor_list=config_parser.get_processors(),
    renderer=config_parser.get_renderer(),
)

class CustomMarkerPDFReader(BaseReader):
    HEADERS_TO_SPLIT_ON = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    def __init__(self, chunk_size: int = 512):
        self.chunk_size = chunk_size
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.HEADERS_TO_SPLIT_ON,
            return_each_line=True,
            strip_headers=True,
        )

    def _extract_pdf_metadata(self, doc: fitz.Document) -> Dict:
        """Extract comprehensive metadata from PDF."""
        metadata = {
            'title': doc.metadata.get('title', ''),
            'author': doc.metadata.get('author', ''),
            'subject': doc.metadata.get('subject', ''),
            'keywords': doc.metadata.get('keywords', ''),
            'total_pages': len(doc),
        }        
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
            'has_tables': bool(page.find_tables()), 
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
    
    def _extract_headers(self, md_text: str) -> List[Dict[str, str]]:
        md_header_splits = self.markdown_splitter.split_text(md_text)
        seen = set()
        cleaned_headers = []

        for chunk in md_header_splits:
            if hasattr(chunk, 'metadata') and chunk.metadata:
                for level, header_text in chunk.metadata.items():
                    key = f"{level}:{header_text.strip()}"
                    if key not in seen:
                        seen.add(key)
                        cleaned_headers.append({level: header_text.strip()})
        return cleaned_headers
           
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

    def _split_markdown_by_pages(self, md_text: str) -> List[Dict]:
        """Split markdown text into pages based on page markers."""
        # Pattern to match page markers like {0}---------- or {15}----------
        page_pattern = r'\{(\d+)\}-+\n\n'
        
        page_markers = [(int(match.group(1)), match.start(), match.end()) 
                        for match in re.finditer(page_pattern, md_text)]
        if not page_markers:
            return [{"page_num": 1, "text": md_text}]
        
        page_chunks = []
        if page_markers[0][1] > 0:
            page_chunks.append({
                "page_num": 1,  # Default to page 1 for content before first marker
                "text": md_text[:page_markers[0][1]]
            })
        
        for i, (page_num, start_pos, end_pos) in enumerate(page_markers):
            next_start = page_markers[i+1][1] if i < len(page_markers)-1 else len(md_text)            
            page_text = md_text[end_pos:next_start]            
            page_chunks.append({
                "page_num": page_num + 1,  # Convert from 0-based to 1-based page numbers
                "text": page_text
            })
            
        return page_chunks

    def load_data(self, pdf_path: str, extra_info: Optional[Dict] = None) -> List[Document]:
        text_blocks = self._extract_pdf_text_with_pages(pdf_path)
        filename = Path(pdf_path).name
        
        try:
            rendered = marker_converter(str(pdf_path))
            md_text = rendered.markdown
        except Exception as e:
            print(f"Marker conversion failed: {e}. Falling back to pymupdf4llm.")
            from pymupdf4llm import LlamaMarkdownReader
            return LlamaMarkdownReader().load_data(pdf_path)
        
        page_chunks = self._split_markdown_by_pages(md_text)        
        document_id = str(uuid.uuid4())
        documents = []
        
        for chunk_idx, page_chunk in enumerate(page_chunks):
            page_num = page_chunk["page_num"]
            text = page_chunk["text"]
            
            headers = self._extract_headers(text)
            
            chunk_metadata = {
                'filename': filename,
                'page': [page_num],  
                'total_pages': max(block.metadata['total_pages'] for block in text_blocks),
                'title': next(iter(text_blocks)).metadata.get('title', ''),
                'author': next(iter(text_blocks)).metadata.get('author', ''),
                'headers': headers,
                'document_id': document_id,
                'chunk_id': f"{document_id}_{chunk_idx}",
                'chunk_index': chunk_idx,
                'total_chunks': len(page_chunks),
                'prev_chunk_id': f"{document_id}_{chunk_idx-1}" if chunk_idx > 0 else None,
                'next_chunk_id': f"{document_id}_{chunk_idx+1}" if chunk_idx < len(page_chunks)-1 else None,
            }
            doc = Document(
                text=text,
                metadata=chunk_metadata
            )
            documents.append(doc)
        
        return documents
    
