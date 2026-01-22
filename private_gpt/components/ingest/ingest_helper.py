import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
from llama_index.core.readers import StringIterableReader
from llama_index.core.readers.base import BaseReader
from llama_index.core.readers.json import JSONReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


# Inspired by the `llama_index.core.readers.file.base` module
def _try_loading_included_file_formats() -> dict[str, type[BaseReader]]:
    try:
        from llama_index.readers.file.docs import (  # type: ignore
            DocxReader,
            HWPReader,
            PDFReader,
        )
        from pymupdf4llm import LlamaMarkdownReader  # type: ignore
        from llama_index.readers.file.epub import EpubReader  # type: ignore
        from llama_index.readers.file.image import ImageReader  # type: ignore
        from llama_index.readers.file.ipynb import IPYNBReader  # type: ignore
        from llama_index.readers.file.markdown import MarkdownReader  # type: ignore
        from llama_index.readers.file.mbox import MboxReader  # type: ignore
        from llama_index.readers.file.slides import PptxReader  # type: ignore
        from llama_index.readers.file.tabular import PandasCSVReader  # type: ignore
        from llama_index.readers.file.video_audio import (  # type: ignore
            VideoAudioReader,
        )
        from llama_index.readers.file.tabular import PandasExcelReader
        from private_gpt.components.reader.LlamaParseReader import LlamaParseReader
    except ImportError as e:
        raise ImportError("`llama-index-readers-file` package not found") from e

    default_file_reader_cls: dict[str, type[BaseReader]] = {
        ".hwp": HWPReader,
        ".pdf": LlamaParseReader,
        ".docx": LlamaParseReader,
        ".pptx": LlamaParseReader,
        ".ppt": LlamaParseReader,
        ".pptm": PptxReader,
        ".jpg": ImageReader,
        ".png": ImageReader,
        ".jpeg": ImageReader,
        ".mp3": VideoAudioReader,
        ".mp4": VideoAudioReader,
        ".csv": PandasCSVReader,
        ".epub": EpubReader,
        ".md": MarkdownReader,
        ".mbox": MboxReader,
        ".ipynb": IPYNBReader,
        ".xlsx": LlamaParseReader,
        ".xls": LlamaParseReader,
    }
    return default_file_reader_cls


# Patching the default file reader to support other file types
FILE_READER_CLS = _try_loading_included_file_formats()
FILE_READER_CLS.update(
    {
        ".json": JSONReader,
    }
)


class IngestionHelper:
    """Helper class to transform a file into a list of documents.

    This class should be used to transform a file into a list of documents.
    These methods are thread-safe (and multiprocessing-safe).
    """

    @staticmethod
    async def transform_file_into_documents(
        file_name: str, file_data: Path, file_metadata: dict[str, Any] | None = None
    ) -> list[Document]:
        documents = await IngestionHelper._load_file_to_documents(file_name, file_data)
        for document in documents:
            # Ensure document has a valid doc_id
            if not hasattr(document, 'doc_id') or document.doc_id is None:
                # Generate a new doc_id if one doesn't exist
                import uuid
                document.doc_id = str(uuid.uuid4())
            
            # Sync LlamaIndex document id_ with our doc_id
            document.id_ = document.doc_id
            
            # Make sure the doc_id is in the document's metadata
            document.metadata["doc_id"] = document.doc_id
            document.metadata["document_id"] = document.doc_id
            
            # Update with provided metadata
            document.metadata.update(file_metadata or {})
            document.metadata["file_name"] = file_name
            document.metadata["ingestion_time"] = datetime.now(timezone.utc).isoformat()
            
            # Ensure excluded metadata keys are set properly
            if not hasattr(document, 'excluded_embed_metadata_keys'):
                document.excluded_embed_metadata_keys = []
            if not hasattr(document, 'excluded_llm_metadata_keys'):
                document.excluded_llm_metadata_keys = []
                
            # Add doc_id to excluded metadata to prevent duplication
            if "doc_id" not in document.excluded_embed_metadata_keys:
                document.excluded_embed_metadata_keys.append("doc_id")
            if "document_id" not in document.excluded_llm_metadata_keys:
                document.excluded_llm_metadata_keys.append("document_id")
                
        IngestionHelper._exclude_metadata(documents)
        return documents

    @staticmethod
    async def _load_file_to_documents(file_name: str, file_data: Path) -> list[Document]:
        logger.debug("Transforming file_name=%s into documents", file_name)
        extension = Path(file_name).suffix
        reader_cls = FILE_READER_CLS.get(extension)
        if reader_cls is None:
            logger.debug(
                "No reader found for extension=%s, using default string reader",
                extension,
            )
            # Read as a plain text
            try:
                string_reader = StringIterableReader()
                return string_reader.load_data([file_data.read_text()])
            except:
                return file_data
        logger.debug("Specific reader found for extension=%s", extension)
        
        # Instantiate the reader
        reader = reader_cls()
        
        # Check if the reader has async support
        if hasattr(reader, 'aload_data'):
            return await reader.aload_data(file_data)
        else:
            # Fallback to synchronous load_data
            return reader.load_data(file_data)

    @staticmethod
    def _exclude_metadata(documents: list[Document]) -> None:
        logger.debug("Excluding metadata from count=%s documents", len(documents))
        for document in documents:
            # Ensure doc_id is always available in metadata for tracking
            if hasattr(document, 'doc_id') and document.doc_id:
                document.metadata["doc_id"] = document.doc_id
                document.metadata["document_id"] = document.doc_id
            # We don't want the Embeddings search to receive this metadata
            excluded_embed_keys = ["doc_id", "document_id"]
            for key in excluded_embed_keys:
                if key not in document.excluded_embed_metadata_keys:
                    document.excluded_embed_metadata_keys.append(key)
            # We don't want the LLM to receive these metadata in the context
            excluded_llm_keys = ["doc_id", "document_id", "file_path", "filename", "chunk_info"]
            for key in excluded_llm_keys:
                if key not in document.excluded_llm_metadata_keys:
                    document.excluded_llm_metadata_keys.append(key)
