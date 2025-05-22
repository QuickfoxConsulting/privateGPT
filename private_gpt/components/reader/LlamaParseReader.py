import os
from pathlib import Path
import uuid
from typing import IO, Dict, List, Optional, Union
from dataclasses import dataclass

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_parse import LlamaParse

@dataclass
class TextBlock:
    text: str
    page_num: int
    block_index: int
    metadata: Dict = None


class LlamaParseReader(BaseReader):
    """
    A PDF reader using LlamaParse with optional semantic chunking via Chonkie.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        num_workers: int = 4,
        result_type: str = "markdown",
        verbose: bool = True,
    ):
        self.api_key = "llx-jph9ZiAlC2KXd7L7fqPQfQ66Z0aROZq7KU8ToqVjwL0DvBEP"
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
        Load a PDF and return a list of Document objects, optionally using semantic chunking.
        """
        try:
            filename = os.path.basename(file_path)
            with open(file_path, "rb") as f:
                file_bytes = f.read()
                llama_docs = await self.parser.aload_data(file_bytes, extra_info={"file_name": filename})

            text_blocks = self._extract_text_blocks(llama_docs)
            document_id = str(uuid.uuid4())
            result_docs = []
            for idx, doc in enumerate(llama_docs):
                if doc.metadata is None:
                    doc.metadata = {}

                doc.metadata.update({
                    "filename": filename,
                    "page": text_blocks[idx].page_num,
                    "document_id": document_id,
                    "chunk_id": f"{document_id}_{idx}",
                    "chunk_index": idx,
                    "total_chunks": len(llama_docs),
                    "prev_chunk_id": f"{document_id}_{idx-1}" if idx > 0 else None,
                    "next_chunk_id": f"{document_id}_{idx+1}" if idx < len(llama_docs) - 1 else None,
                })
                if extra_info:
                    doc.metadata.update(extra_info)

                result_docs.append(doc)

            return result_docs
        except Exception as e:
                print(f"Exception: {e}")