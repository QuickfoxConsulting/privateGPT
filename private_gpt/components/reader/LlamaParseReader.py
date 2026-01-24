import os
import uuid
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from llama_parse import LlamaParse
from llama_index.core.schema import Document
from llama_index.core.readers.base import BaseReader
from private_gpt.users.core.config import settings  # Assuming this is correctly imported
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

@dataclass
class TextBlock:
    doc_id: str  
    text: str
    page_num: int
    block_index: int
    metadata: Dict = None  

class LlamaParseReader(BaseReader):
    """
    A Document reader using LlamaParse.
    
    This class parses PDF files using LlamaParse and extracts text blocks into 
    llama_index Document objects, adding metadata such as previous/next page IDs.
    """
    
    # Maximum file size: 100MB
    MAX_FILE_SIZE = 100 * 1024 * 1024
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".ppt", ".xlsx", ".xls"}
    
    def __init__(
        self,
        num_workers: int = 2,
        result_type: str = "markdown",
        verbose: bool = True,
        page_prefix: str = "START OF PAGE: {pageNumber}\n",
        page_suffix: str = "\nEND OF PAGE: {pageNumber}",
        spreadsheet_extract_sub_tables: bool = True,
        preserve_layout_alignment_across_pages: bool = True,
        preserve_very_small_text: bool = True,
        merge_tables_across_pages_in_markdown: bool = True,
        presentation_out_of_bounds_content: bool = True,
        output_tables_as_HTML: bool = True,
        join_pages: bool = True,
    ):
        """
        Initialize the LlamaParseReader with customizable parsing options.
        """
        self.api_key = settings.LLAMA_PARSE
        if not self.api_key:
            raise ValueError("LLAMA_PARSE API key is not set in settings.")
        
        self.num_workers = num_workers
        self.result_type = result_type
        self.verbose = verbose
        self.join_pages = join_pages
        
        self.parser = LlamaParse(
            api_key=self.api_key,
            num_workers=self.num_workers,
            result_type=self.result_type,
            verbose=self.verbose,
            page_prefix=page_prefix,
            page_suffix=page_suffix,
            spreadsheet_extract_sub_tables=spreadsheet_extract_sub_tables,
            preserve_layout_alignment_across_pages=preserve_layout_alignment_across_pages,
            preserve_very_small_text=preserve_very_small_text,
            merge_tables_across_pages_in_markdown=merge_tables_across_pages_in_markdown,
            presentation_out_of_bounds_content=presentation_out_of_bounds_content,
            output_tables_as_HTML=output_tables_as_HTML,
        )

    def _validate_file(self, file_path: Path) -> None:
        """Validate file exists, has supported extension, and is within size limits."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        file_size = file_path.stat().st_size
        if file_size > self.MAX_FILE_SIZE:
            raise ValueError(
                f"File too large: {file_size} bytes (max: {self.MAX_FILE_SIZE} bytes)"
            )
        
        logger.debug(f"File validation passed for {file_path.name} ({file_size} bytes)")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        reraise=True
    )
    async def _call_llama_parse_async(self, file_path: str) -> List[Document]:
        """Call LlamaParse API with retry logic for transient failures."""
        logger.debug(f"Calling LlamaParse API for {file_path}")
        return await self.parser.aload_data(file_path)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        reraise=True
    )
    def _call_llama_parse_sync(self, file_path: str) -> List[Document]:
        """Call LlamaParse API with retry logic for transient failures."""
        logger.debug(f"Calling LlamaParse API for {file_path}")
        return self.parser.load_data(file_path)

    def _extract_text_blocks(self, llama_docs: List[Document]) -> List[TextBlock]:
        """
        Extract TextBlock objects from parsed Llama documents.
        """
        text_blocks = []
        for idx, doc in enumerate(llama_docs):
            page_label = doc.metadata.get("page_label", idx + 1)
            try:
                page_num = int(page_label)
            except (ValueError, TypeError):
                page_num = idx + 1
            
            doc_id = doc.id_ if doc.id_ else str(uuid.uuid4())
            
            text_blocks.append(
                TextBlock(
                    doc_id=doc_id,
                    text=doc.text,
                    page_num=page_num,
                    block_index=idx,
                    metadata=doc.metadata or {},
                )
            )
        return text_blocks

    def _process_documents(self, llama_docs: List[Document], filename: str, extra_info: Optional[Dict] = None) -> List[Document]:
        """Process Llama documents, joining pages if configured."""
        if not llama_docs:
            return []

        if self.join_pages:
            combined_text = ""
            page_map = []
            char_offset = 0
            
            for idx, doc in enumerate(llama_docs):
                page_label = doc.metadata.get("page_label", idx + 1)
                try:
                    page_num = int(page_label)
                except (ValueError, TypeError):
                    page_num = idx + 1
                
                page_text = doc.text
                if idx > 0:
                    # Add a separator if joining
                    separator = "\n\n"
                    page_text = separator + page_text
                
                start_idx = char_offset
                combined_text += page_text
                char_offset = len(combined_text)
                
                page_map.append({
                    "page": page_num,
                    "start_char_idx": start_idx,
                    "end_char_idx": char_offset
                })

            # Create a single document
            main_doc = Document(
                text=combined_text,
                id_=str(uuid.uuid4()),
                metadata={
                    "filename": filename,
                    "file_name": filename, # Add for frontend compatibility
                    "total_pages": len(llama_docs),
                    "page_map": page_map,
                    "joined_pages": True
                },
                excluded_embed_metadata_keys=["page_map"],
                excluded_llm_metadata_keys=["page_map"]
            )
            if extra_info:
                main_doc.metadata.update(extra_info)
            return [main_doc]

        # Standard behavior: one document per page
        text_blocks = self._extract_text_blocks(llama_docs)
        result_docs = []
        
        for idx, doc in enumerate(llama_docs):
            if doc.metadata is None:
                doc.metadata = {}
            
            block = text_blocks[idx]
            current_doc_id = block.doc_id
            prev_doc_id = text_blocks[idx - 1].doc_id if idx > 0 else None
            next_doc_id = text_blocks[idx + 1].doc_id if idx < len(text_blocks) - 1 else None
            
            doc.metadata.update({
                "filename": filename,
                "file_name": filename, # Add for frontend compatibility
                "page": block.page_num,
                "document_id": current_doc_id,
                "total_pages": len(llama_docs),
                "prev_page": prev_doc_id,
                "next_page": next_doc_id,
            })
            
            if extra_info:
                doc.metadata.update(extra_info)
            
            result_docs.append(doc)
        return result_docs

    async def aload_data(self, file_path: Union[str, Path], extra_info: Optional[Dict] = None) -> List[Document]:
        """
        Asynchronously load a PDF and return a list of Document objects with metadata.
        """
        file_path = Path(file_path)
        self._validate_file(file_path)
        
        try:
            filename = file_path.name
            llama_docs = await self._call_llama_parse_async(str(file_path))
            return self._process_documents(llama_docs, filename, extra_info)
        except Exception as e:
            raise RuntimeError(f"Error parsing file '{file_path}': {str(e)}") from e

    def load_data(self, file_path: Union[str, Path], extra_info: Optional[Dict] = None) -> List[Document]:
        """
        Synchronous wrapper for aload_data.
        """
        file_path = Path(file_path)
        self._validate_file(file_path)
        
        try:
            filename = file_path.name
            llama_docs = self._call_llama_parse_sync(str(file_path))
            return self._process_documents(llama_docs, filename, extra_info)
        except Exception as e:
            raise RuntimeError(f"Error parsing file '{file_path}': {str(e)}") from e