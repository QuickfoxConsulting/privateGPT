import os
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from llama_parse import LlamaParse
from llama_index.core.schema import Document
from llama_index.core.readers.base import BaseReader
from private_gpt.users.core.config import settings  # Assuming this is correctly imported

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

    async def aload_data(self, file_path: Union[str, Path], extra_info: Optional[Dict] = None) -> List[Document]:
        """
        Asynchronously load a PDF and return a list of Document objects with metadata.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if file_path.suffix.lower() not in [".pdf", ".docx", ".pptx", ".ppt", ".xlsx", ".xls"]:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        try:
            filename = file_path.name
            
            # Pass file path directly to LlamaParse
            llama_docs = await self.parser.aload_data(str(file_path))
            
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
        
        except Exception as e:
            raise RuntimeError(f"Error parsing file '{file_path}': {str(e)}") from e

    def load_data(self, file_path: Union[str, Path], extra_info: Optional[Dict] = None) -> List[Document]:
        """
        Synchronous wrapper for aload_data (assumes LlamaParse has sync load_data).
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if file_path.suffix.lower() not in [".pdf", ".docx", ".pptx", ".ppt", ".xlsx", ".xls"]:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        try:
            filename = file_path.name
            
            # Pass file path directly to LlamaParse
            llama_docs = self.parser.load_data(str(file_path))
            
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
        
        except Exception as e:
            raise RuntimeError(f"Error parsing file '{file_path}': {str(e)}") from e