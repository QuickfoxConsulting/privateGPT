import os
import re
from typing import Dict, List, Optional, Tuple

import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, TokenTextSplitter
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from pathlib import Path

class CustomPDFReader(BaseReader):
    """
    A class to convert PDF files to markdown chunks with metadata.
    
    Attributes:
        CHUNK_SIZE (int): Size of text chunks.
        CHUNK_OVERLAP_SIZE (int): Overlap between text chunks.
        HEADERS_TO_SPLIT_ON (List[Tuple[str, str]]): Markdown headers to split on.
    """
    
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP_SIZE = 100
    HEADERS_TO_SPLIT_ON = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP_SIZE):
        """
        Initialize the PDF to Markdown chunker.
        
        Args:
            chunk_size (int, optional): Size of text chunks. Defaults to 250.
            chunk_overlap (int, optional): Overlap between text chunks. Defaults to 50.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _preprocess_markdown(self, md_text: str) -> str:
        """
        Preprocess the markdown text by cleaning and standardizing.
        
        Args:
            md_text (str): Raw markdown text.
        
        Returns:
            str: Preprocessed markdown text.
        """
        md_text = re.sub(r'\n+', '\n', md_text)
        md_text = '\n'.join(line.strip() for line in md_text.split('\n'))
        return md_text

    def _get_page_indexes(self, md_text: str, page_break: str = '-----') -> List[Tuple[int, int]]:
        """
        Get page index ranges in the markdown text.
        
        Args:
            md_text (str): Preprocessed markdown text.
            page_break (str, optional): Page break separator. Defaults to '-----'.
        
        Returns:
            List[Tuple[int, int]]: List of page start and end indexes.
        """
        page_split_indexes = [0]
        page_split_indexes.extend(
            [match.start() + len(page_break) for match in re.finditer(re.escape(page_break), md_text)]
        )
        
        page_indexes = []
        for n, idx in enumerate(page_split_indexes):
            try:
                page_indexes.append((idx, page_split_indexes[n+1]-1))
            except IndexError:
                page_indexes.append((idx, len(md_text)))
        
        return page_indexes

    def _assign_page_metadata(self, chunks: List[Document], md_text: str, page_indexes: List[Tuple[int, int]], filename: str) -> List[Document]:
        """
        Assign page metadata to markdown chunks.
        
        Args:
            chunks (List[Document]): List of markdown chunks.
            md_text (str): Full markdown text.
            page_indexes (List[Tuple[int, int]]): Page index ranges.
            filename (str): PDF filename.
        
        Returns:
            List[Document]: Chunks with page metadata added.
        """
        for chunk in chunks:
            start = md_text.find(chunk.text)
            end = start + len(chunk.text)
            
            page_coverage = []
            for page_number, (page_start_idx, page_end_idx) in enumerate(page_indexes, start=1):
                if page_start_idx <= start <= page_end_idx:
                    page_coverage.append(page_number)
                    break
            
            for page_number, (page_start_idx, page_end_idx) in enumerate(page_indexes, start=1):
                if page_start_idx <= end <= page_end_idx:
                    if page_number not in page_coverage:
                        page_coverage.append(page_number)
                    break
            
            if len(page_coverage) > 1:
                page_coverage = list(range(page_coverage[0], page_coverage[1] + 1))            
            chunk.metadata['page'] = page_coverage
        
        return chunks

    def load_data(self, pdf_path: str, extra_info: Optional[Dict] = None) -> List[Document]:
        """
        Load and chunk PDF data into markdown documents.
        
        Args:
            pdf_path (str): Path to the PDF file.
            extra_info (Optional[Dict], optional): Additional metadata. Defaults to None.
        
        Returns:
            List[Document]: List of markdown chunks with metadata.
        """
        # Validate extra_info
        if extra_info is not None and not isinstance(extra_info, dict):
            raise TypeError("extra_info must be a dictionary.")
        
        # Convert PDF to markdown
        filename = os.path.basename(pdf_path)
        md_text = pymupdf4llm.to_markdown(pdf_path)
        
        # Optional: Save markdown to file
        # Path("internship.md").write_bytes(md_text.encode())
        
        md_text = self._preprocess_markdown(md_text)
        page_indexes = self._get_page_indexes(md_text)
        
        markdown_splitter = MarkdownHeaderTextSplitter(
            self.HEADERS_TO_SPLIT_ON, 
            strip_headers=False
        )
        md_header_splits = markdown_splitter.split_text(md_text)
        
        text_splitter = TokenTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        
        chunks = []
        for chunk in md_header_splits:
            metadata = chunk.metadata
            for chunk_split in text_splitter.split_text(chunk.page_content):
                doc = Document(
                    text=chunk_split,
                    metadata=metadata
                )
                chunks.append(doc)
        
        return self._assign_page_metadata(chunks, md_text, page_indexes, filename)