"""Ingestion service with dynamic chunking support."""

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, AnyStr, BinaryIO, Sequence, Any, List, Optional
from enum import Enum

from injector import inject, singleton
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter, SentenceWindowNodeParser
from llama_index.core.storage import StorageContext
from llama_index.core.schema import BaseNode , ObjectType , TextNode 

from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.ingest.ingest_component import get_ingestion_component
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.nodeparser.SentenceChunkNodeParser import SentenceChunkWindowNodeParser
from private_gpt.components.nodeparser.PassThroughNodeParser import PassthroughNodeParser
from private_gpt.components.nodeparser.ChonkieContextWindowNodeParser import ChonkieContextWindowNodeParser
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.server.ingest.model import IngestedDoc
from private_gpt.settings.settings import settings
from llama_index.core.extractors import SummaryExtractor
if TYPE_CHECKING:
    from llama_index.core.storage.docstore.types import RefDocInfo

logger = logging.getLogger(__name__)


DEFAULT_CHUNK_SIZE = 512
SENTENCE_CHUNK_OVERLAP = 100
DEFAULT_WINDOW_SIZE = 3

class ChunkingStrategy(str, Enum):
    LATE_CHUNKING = "late_chunking"
    SENTENCE_WINDOW = "sentence_window"
    SENTENCE_CHUNK = "sentence_chunk"
    PASS_THROUGH = "pass_through"

@singleton
class IngestService:
    @inject
    def __init__(
        self,
        llm_component: LLMComponent,
        vector_store_component: VectorStoreComponent,
        embedding_component: EmbeddingComponent,
        node_store_component: NodeStoreComponent,
    ) -> None:
        self.llm_service = llm_component
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store_component.vector_store,
            docstore=node_store_component.doc_store,
            index_store=node_store_component.index_store,
        )      
        # node_parser = ChonkieContextWindowNodeParser.from_defaults(
        #     chunk_size=DEFAULT_CHUNK_SIZE,
        #     context_window=DEFAULT_WINDOW_SIZE
        # )
        node_parser = SentenceWindowNodeParser.from_defaults(
                window_size=10,  # Approximate window size
                window_metadata_key="window",
                original_text_metadata_key="original_text",
                include_metadata=True,
                include_prev_next_rel=True
            )
        self.ingest_component = get_ingestion_component(
            self.storage_context,
            embed_model=embedding_component.embedding_model,
            transformations=[
                node_parser,
                embedding_component.embedding_model,
            ],
            settings=settings(),
        )

    def _get_node_parser(
        self, 
        chunk_size: int = DEFAULT_CHUNK_SIZE, 
        chunk_overlap: int = SENTENCE_CHUNK_OVERLAP,
        window_size: int = DEFAULT_WINDOW_SIZE,
        strategy: ChunkingStrategy = ChunkingStrategy.LATE_CHUNKING
    ):
        """Create node parser based on dynamic parameters."""
        if strategy == ChunkingStrategy.LATE_CHUNKING:
            return ChonkieContextWindowNodeParser.from_defaults(
                chunk_size=chunk_size,
                context_window=window_size,
            )
        elif strategy == ChunkingStrategy.SENTENCE_WINDOW:
            parser = SentenceWindowNodeParser.from_defaults(
                window_size=chunk_size // 50,  # Approximate window size
                window_metadata_key="window",
                original_text_metadata_key="original_text",
                include_metadata=True,
                include_prev_next_rel=True
            )
            return parser
        elif strategy == ChunkingStrategy.SENTENCE_CHUNK:
            return SentenceChunkWindowNodeParser.from_defaults(
                chunk_size=chunk_size // 50,  # Approximate sentences per chunk
                window_size=window_size,
                window_metadata_key="window",
                original_text_metadata_key="original_text",
            )
        else:
            # Default to sentence window node parser
            parser = SentenceWindowNodeParser.from_defaults(
                window_size=chunk_size // 50,  # Approximate window size
                window_metadata_key="window",
                original_text_metadata_key="original_text",
                include_metadata=True,
                include_prev_next_rel=True
            )
            return parser

    def _get_ingest_component_with_parser(self, node_parser, embedding_component):
        """Create ingestion component with specific node parser."""
        # Get the settings from the existing ingest component
        existing_settings = None
        if hasattr(self.ingest_component, 'settings'):
            existing_settings = self.ingest_component.settings
        else:
            # Fallback to getting settings from global settings
            existing_settings = settings()
            
        return get_ingestion_component(
            self.storage_context,
            embed_model=embedding_component,
            transformations=[
                node_parser,
                embedding_component,
            ],
            settings=existing_settings,
        )

    async def _ingest_data(
        self,
        file_name: str,
        file_data: AnyStr,
        file_metadata: dict[str, str] | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = SENTENCE_CHUNK_OVERLAP,
        strategy: ChunkingStrategy = ChunkingStrategy.LATE_CHUNKING,
        window_size: int = DEFAULT_WINDOW_SIZE,
    ) -> list[IngestedDoc]:
        logger.debug("Got file data of size=%s to ingest", len(file_data))
        # Create node parser with dynamic parameters
        node_parser = self._get_node_parser(chunk_size, chunk_overlap, window_size, strategy)
        ingest_component = self._get_ingest_component_with_parser(node_parser, self.ingest_component.transformations[1])  # embedding component
        
        # llama-index mainly supports reading from files, so
        # we have to create a tmp file to read for it to work
        # delete=False to avoid a Windows 11 permission error.
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            try:
                path_to_tmp = Path(tmp.name)
                if isinstance(file_data, bytes):
                    path_to_tmp.write_bytes(file_data)
                else:
                    path_to_tmp.write_text(str(file_data))
                return await self.ingest_file(file_name, path_to_tmp, file_metadata, chunk_size, chunk_overlap, strategy, window_size)
            finally:
                tmp.close()
                path_to_tmp.unlink()

    async def ingest_file(
        self,
        file_name: str,
        file_data: Path,
        file_metadata: dict[str, str] | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = SENTENCE_CHUNK_OVERLAP,
        strategy: ChunkingStrategy = ChunkingStrategy.LATE_CHUNKING,
        window_size: int = DEFAULT_WINDOW_SIZE,
    ) -> list[IngestedDoc]:
        logger.info("Ingesting file_name=%s", file_name)
        # Create node parser with dynamic parameters
        node_parser = self._get_node_parser(chunk_size, chunk_overlap, window_size, strategy)
        ingest_component = self._get_ingest_component_with_parser(node_parser, self.ingest_component.transformations[1])  # embedding component
        documents = await ingest_component.ingest(file_name, file_data, file_metadata)
        logger.info("Finished ingestion file_name=%s", file_name)
        return [IngestedDoc.from_document(document) for document in documents]

    async def ingest_text(
        self, 
        file_name: str, 
        text: str, 
        metadata: dict[str, str] | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = SENTENCE_CHUNK_OVERLAP,
        strategy: ChunkingStrategy = ChunkingStrategy.LATE_CHUNKING,
        window_size: int = DEFAULT_WINDOW_SIZE,
    ) -> list[IngestedDoc]:
        logger.debug("Ingesting text data with file_name=%s", file_name)
        return await self._ingest_data(file_name, text, metadata, chunk_size, chunk_overlap, strategy, window_size)

    async def ingest_bin_data(
        self,
        file_name: str,
        raw_file_data: BinaryIO,
        file_metadata: dict[str, str] | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = SENTENCE_CHUNK_OVERLAP,
        strategy: ChunkingStrategy = ChunkingStrategy.LATE_CHUNKING,
        window_size: int = DEFAULT_WINDOW_SIZE,
    ) -> list[IngestedDoc]:
        logger.debug("Ingesting binary data with file_name=%s", file_name)
        file_data = raw_file_data.read()
        return await self._ingest_data(file_name, file_data, file_metadata, chunk_size, chunk_overlap, strategy, window_size)
    
    async def ingest_url(
        self, 
        url: str, 
        documents,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = SENTENCE_CHUNK_OVERLAP,
        strategy: ChunkingStrategy = ChunkingStrategy.LATE_CHUNKING,
        window_size: int = DEFAULT_WINDOW_SIZE,
    ) -> list[IngestedDoc]:
        logger.debug("Ingesting url=%s", url)
        # Create node parser with dynamic parameters
        node_parser = self._get_node_parser(chunk_size, chunk_overlap, window_size, strategy)
        ingest_component = self._get_ingest_component_with_parser(node_parser, self.ingest_component.transformations[1])  # embedding component
        documents = await ingest_component.ingest(url, documents)
        return [IngestedDoc.from_document(document) for document in documents]

    async def bulk_ingest(
        self, 
        files: list[tuple[str, Path]],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = SENTENCE_CHUNK_OVERLAP,
        strategy: ChunkingStrategy = ChunkingStrategy.LATE_CHUNKING,
        window_size: int = DEFAULT_WINDOW_SIZE,
    ) -> list[IngestedDoc]:
        logger.info("Ingesting file_names=%s", [f[0] for f in files])
        # Create node parser with dynamic parameters
        node_parser = self._get_node_parser(chunk_size, chunk_overlap, window_size, strategy)
        ingest_component = self._get_ingest_component_with_parser(node_parser, self.ingest_component.transformations[1])  # embedding component
        documents = await ingest_component.bulk_ingest(files)
        logger.info("Finished ingestion file_name=%s", [f[0] for f in files])
        return [IngestedDoc.from_document(document) for document in documents]

    def list_ingested(self) -> list[IngestedDoc]:
        ingested_docs: list[IngestedDoc] = []
        try:
            docstore = self.storage_context.docstore
            ref_docs: dict[str, RefDocInfo] | None = docstore.get_all_ref_doc_info()

            if not ref_docs:
                return ingested_docs

            for doc_id, ref_doc_info in ref_docs.items():
                doc_metadata = None
                if ref_doc_info is not None and ref_doc_info.metadata is not None:
                    doc_metadata = IngestedDoc.curate_metadata(ref_doc_info.metadata)
                ingested_docs.append(
                    IngestedDoc(
                        object="ingest.document",
                        doc_id=doc_id,
                        doc_metadata=doc_metadata,
                    )
                )
        except ValueError:
            logger.warning("Got an exception when getting list of docs", exc_info=True)
            pass
        logger.debug("Found count=%s ingested documents", len(ingested_docs))
        return ingested_docs

    async def delete(self, doc_id: str) -> None:
        """Delete an ingested document.

        :raises ValueError: if the document does not exist
        """
        logger.info(
            "Deleting the ingested document=%s in the doc and index store", doc_id
        )
        await self.ingest_component.delete(doc_id)

    def get_doc_ids_by_filename(self, filename: str) -> list[str]:
        doc_ids: set[str] = set()
        try:
            docstore = self.storage_context.docstore
            for node in docstore.docs.values():
                if node.metadata is not None and node.metadata.get("file_name") == filename:
                    doc_ids.add(node.ref_doc_id)

        except ValueError:
            logger.warning("Got an exception when getting doc_ids by filename", exc_info=True)
            pass

        logger.debug("Found count=%s doc_ids for filename '%s'",
                     len(doc_ids), filename)
        return doc_ids

    def get_doc_ids_by_filename_pattern(self, pattern: str) -> list[str]:
        """
        Get document IDs matching a filename pattern.
        
        Args:
            pattern (str): Pattern to match against filenames
            
        Returns:
            list[str]: List of document IDs matching the pattern
        """
        doc_ids: set[str] = set()
        
        try:
            docstore = self.storage_context.docstore
            for node in docstore.docs.values():
                if (node.metadata is not None and 
                    node.metadata.get("file_name") is not None and 
                    pattern in node.metadata["file_name"]):
                    doc_ids.add(node.ref_doc_id)
        except ValueError:
            logger.warning(
                "Got an exception when getting doc_ids by filename pattern",
                exc_info=True
            )
            pass
        
        logger.debug("Found count=%s doc_ids for filename pattern '%s'",
                     len(doc_ids), pattern)
        
        return list(doc_ids)