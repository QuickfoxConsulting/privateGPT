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
    doc_id: uuid.UUID
    text: str
    page_num: int
    block_index: int
    metadata: Dict = None

class LlamaParseReader(BaseReader):
    """
    A Document reader using LlamaParse
    """
    def __init__(
        self,
        num_workers: int = 2,
        result_type: str = "markdown",
        verbose: bool = True,
    ):
        self.api_key = settings.LLAMA_PARSE
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
                    doc_id=doc.id_,
                    text=doc.text,
                    page_num=page_num,
                    block_index=idx,
                    metadata=doc.metadata or {},
                )
            )
        return text_blocks
    
    async def load_data(self, file_path: str, extra_info: Optional[Dict] = None) -> List[Document]:
        """
        Load a PDF and return a list of Document objects, with proper prev/next page IDs.
        """
        try:
            filename = os.path.basename(file_path)
            with open(file_path, "rb") as f:
                file_bytes = f.read()
                llama_docs = await self.parser.aload_data(file_bytes, extra_info={"file_name": filename})

            text_blocks = self._extract_text_blocks(llama_docs)
            result_docs = []

            for idx, doc in enumerate(llama_docs):
                if doc.metadata is None:
                    doc.metadata = {}

                block = text_blocks[idx]
                current_doc_id = str(block.doc_id if getattr(block, "doc_id", None) else uuid.uuid4())
                prev_doc_id = str(text_blocks[idx - 1].doc_id) if idx > 0 else None
                next_doc_id = str(text_blocks[idx + 1].doc_id) if idx < len(text_blocks) - 1 else None

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
            print(f"Exception: {e}")