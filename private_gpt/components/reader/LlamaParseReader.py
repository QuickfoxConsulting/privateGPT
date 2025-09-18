import os
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import IO, Dict, List, Optional, Union

from llama_parse import LlamaParse
from llama_index.core.schema import Document
from llama_index.core.readers.base import BaseReader

from private_gpt.users.core.config import settings

@dataclass
class TextBlock:
    text: str
    page_num: int
    block_index: int
    metadata: Dict = None


class LlamaParseReader(BaseReader):
    """
    A PDF reader using LlamaParse with improved page reference handling.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        num_workers: int = 2,
        result_type: str = "markdown",
        verbose: bool = True,
    ):
        self.api_key = settings.LLAMA_PARSE
        self.chunk_size = chunk_size
        self.num_workers = num_workers
        self.result_type = result_type
        self.verbose = verbose
        self.parser = LlamaParse(
            api_key=self.api_key,
            num_workers=self.num_workers,
            result_type=self.result_type,
            verbose=self.verbose,
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

            text_blocks.append(
                TextBlock(
                    text=doc.text,
                    page_num=page_num,
                    block_index=idx,
                    metadata=doc.metadata or {},
                )
            )
        return text_blocks
    
    async def load_data(self, file_path: str, extra_info: Optional[Dict] = None) -> List[Document]:
        """
        Load a PDF and return a list of Document objects with enhanced page reference metadata.
        """
        try:
            filename = os.path.basename(file_path)
            with open(file_path, "rb") as f:
                file_bytes = f.read()
                llama_docs = await self.parser.aload_data(file_bytes, extra_info={"file_name": filename})

            text_blocks = self._extract_text_blocks(llama_docs)
            document_id = str(uuid.uuid4())
            result_docs = []
            
            # Combine content from consecutive pages when appropriate
            combined_docs = self._combine_cross_page_content(llama_docs, text_blocks)
            
            for idx, doc in enumerate(combined_docs):
                if doc.metadata is None:
                    doc.metadata = {}

                # Enhanced metadata with better page tracking
                doc.metadata.update({
                    "filename": filename,
                    "page": doc.metadata.get("page_label", idx + 1),  # Preserve original page label
                    "document_id": document_id,
                    "chunk_id": f"{document_id}_{idx}",
                    "chunk_index": idx,
                    "total_chunks": len(combined_docs),
                    "prev_chunk_id": f"{document_id}_{idx-1}" if idx > 0 else None,
                    "next_chunk_id": f"{document_id}_{idx+1}" if idx < len(combined_docs) - 1 else None,
                })
                
                # Add page range information for multi-page content
                if "page_ranges" in doc.metadata:
                    doc.metadata["page_info"] = f"Pages {doc.metadata['page_ranges']}"
                else:
                    doc.metadata["page_info"] = f"Page {doc.metadata.get('page_label', idx + 1)}"
                
                if extra_info:
                    doc.metadata.update(extra_info)

                result_docs.append(doc)

            return result_docs
        except Exception as e:
            print(f"Exception: {e}")
            raise e

    def _combine_cross_page_content(self, llama_docs: List[Document], text_blocks: List[TextBlock]) -> List[Document]:
        """
        Combine content that spans multiple pages to maintain context.
        """
        if len(llama_docs) <= 1:
            return llama_docs
            
        combined_docs = []
        i = 0
        
        while i < len(llama_docs):
            current_doc = llama_docs[i]
            current_block = text_blocks[i]
            
            # Check if this content might continue on the next page
            # (short content or ending with incomplete sentence)
            if (len(current_doc.text.strip()) < 200 or 
                current_doc.text.strip().endswith((',', '-', ':', ';')) or
                (i < len(llama_docs) - 1 and 
                 text_blocks[i+1].page_num == current_block.page_num + 1)):
                
                # Combine with next page(s) if they seem related
                combined_text = current_doc.text
                page_ranges = [current_block.page_num]
                j = i + 1
                
                # Look ahead to combine related content
                while (j < len(llama_docs) and 
                       text_blocks[j].page_num <= current_block.page_num + 2 and
                       (len(llama_docs[j].text.strip()) < 300 or
                        current_doc.text.strip().endswith((',', '-', ':', ';')) or
                        len(combined_text) < 1000)):
                    
                    combined_text += "\n\n" + llama_docs[j].text
                    page_ranges.append(text_blocks[j].page_num)
                    j += 1
                
                # Create combined document
                combined_doc = Document(
                    text=combined_text,
                    metadata=current_doc.metadata.copy() if current_doc.metadata else {}
                )
                
                # Add page range information
                if len(page_ranges) > 1:
                    combined_doc.metadata["page_ranges"] = f"{page_ranges[0]}-{page_ranges[-1]}"
                
                combined_docs.append(combined_doc)
                i = j
            else:
                combined_docs.append(current_doc)
                i += 1
                
        return combined_docs