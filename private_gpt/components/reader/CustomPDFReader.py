import os
from typing import Dict, List, Optional, Tuple
import fitz  # PyMuPDF
import pymupdf4llm
from difflib import SequenceMatcher
from langchain_text_splitters import MarkdownHeaderTextSplitter
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from chonkie import LateChunker

class CustomPDFReader(BaseReader):
    """
    A class to convert PDF files to markdown chunks with accurate page mapping,
    using LateChunker for intelligent chunking.
    """
    
    HEADERS_TO_SPLIT_ON = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
    ]

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        mode: str = "sentence",
        chunk_size: int = 1024,
        min_sentences_per_chunk: int = 1,
        min_characters_per_sentence: int = 12
    ):
        """
        Initialize the PDF reader with LateChunker configuration.
        
        Args:
            embedding_model (str): Name of the embedding model to use
            mode (str): Chunking mode ("sentence" or "word")
            chunk_size (int): Maximum size of chunks
            min_sentences_per_chunk (int): Minimum number of sentences per chunk
            min_characters_per_sentence (int): Minimum characters per sentence
        """
        self.chunker = LateChunker(
            embedding_model=embedding_model,
            mode=mode,
            chunk_size=chunk_size,
            min_sentences_per_chunk=min_sentences_per_chunk,
            min_characters_per_sentence=min_characters_per_sentence,
        )

    def _extract_pdf_text_with_pages(self, pdf_path: str) -> List[Dict]:
        """
        Extract text content with page numbers, maintaining sequence for multi-page detection.
        
        Args:
            pdf_path (str): Path to the PDF file.
            
        Returns:
            List[Dict]: List of text blocks with their page numbers and positions.
        """
        text_blocks = []
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("blocks")
            
            for block in blocks:
                # Each block contains (x0, y0, x1, y1, text, block_no, block_type)
                text_blocks.append({
                    'page_num': page_num + 1,
                    'text': block[4].strip(),
                    'position': (block[0], block[1], block[2], block[3])
                })
        
        doc.close()
        return text_blocks

    def _find_matching_pages(self, chunk_text: str, text_blocks: List[Dict], 
                           min_match_length: int = 50) -> List[int]:
        """
        Find all pages that contain parts of the given chunk text.
        
        Args:
            chunk_text (str): The text chunk to locate.
            text_blocks (List[Dict]): List of text blocks with page numbers.
            min_match_length (int): Minimum length of text to consider a match.
            
        Returns:
            List[int]: List of page numbers containing the chunk text.
        """
        matching_pages = set()
        chunk_text = chunk_text.strip()
        
        # Split chunk into smaller segments for better matching
        sentences = chunk_text.split('.')
        segments = [sent.strip() for sent in sentences if len(sent.strip()) > min_match_length]
        
        for segment in segments:
            for block in text_blocks:
                block_text = block['text']
                
                # Check if this block contains the segment
                if (segment in block_text or 
                    SequenceMatcher(None, segment, block_text).ratio() > 0.8):
                    matching_pages.add(block['page_num'])
                    
                    # Check adjacent blocks for continuation
                    block_idx = text_blocks.index(block)
                    if block_idx > 0:
                        matching_pages.add(text_blocks[block_idx-1]['page_num'])
                    if block_idx < len(text_blocks) - 1:
                        matching_pages.add(text_blocks[block_idx+1]['page_num'])
        
        # If we found no matches but the chunk is substantial, 
        # try a more lenient matching approach
        if not matching_pages and len(chunk_text) > min_match_length:
            for block in text_blocks:
                if any(phrase in block['text'] for phrase in chunk_text.split('\n') 
                       if len(phrase.strip()) > min_match_length):
                    matching_pages.add(block['page_num'])
        
        return sorted(list(matching_pages))

    def load_data(self, pdf_path: str, extra_info: Optional[Dict] = None) -> List[Document]:
        """
        Load and chunk PDF data into markdown documents with accurate page mapping.
        
        Args:
            pdf_path (str): Path to the PDF file.
            extra_info (Optional[Dict]): Additional metadata.
            
        Returns:
            List[Document]: List of markdown chunks with metadata.
        """
        if extra_info is not None and not isinstance(extra_info, dict):
            raise TypeError("extra_info must be a dictionary.")
        
        # Extract text blocks with page numbers
        text_blocks = self._extract_pdf_text_with_pages(pdf_path)
        
        # Convert PDF to markdown
        filename = os.path.basename(pdf_path)
        md_text = pymupdf4llm.to_markdown(pdf_path)
        
        # Split by headers
        markdown_splitter = MarkdownHeaderTextSplitter(
            self.HEADERS_TO_SPLIT_ON, 
            strip_headers=False
        )
        md_header_splits = markdown_splitter.split_text(md_text)
        
        chunks = []
        for header_chunk in md_header_splits:
            metadata = header_chunk.metadata
            
            # Use LateChunker to split the text
            text_chunks = self.chunker(header_chunk.page_content)
            
            for chunk in text_chunks:
                # Find all pages containing parts of this chunk
                page_numbers = self._find_matching_pages(chunk.text, text_blocks)
                
                # Ensure we always have at least one page number
                if not page_numbers:
                    # If no match found, use textual analysis to estimate page range
                    text_position = md_text.find(chunk.text)
                    if text_position != -1:
                        # Estimate page based on position in full text
                        estimated_page = (text_position // 3000) + 1  # Rough estimate
                        page_numbers = [min(estimated_page, len(text_blocks))]
                
                # Create document with combined metadata
                chunk_metadata = {
                    **metadata,
                    'page': page_numbers,
                    'filename': filename
                }
                if extra_info:
                    chunk_metadata.update(extra_info)
                    
                doc = Document(
                    text=chunk.text,
                    metadata=chunk_metadata
                )
                chunks.append(doc)
        
        return chunks