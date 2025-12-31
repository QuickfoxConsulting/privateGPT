"""Ingestion service with dynamic chunking support."""

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, AnyStr, BinaryIO, Sequence, Any, List, Optional
from enum import Enum

from injector import inject, singleton
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser, 
    SentenceSplitter, 
    SentenceWindowNodeParser,
    HierarchicalNodeParser,
    get_leaf_nodes,
    get_root_nodes,
)
from llama_index.core.storage import StorageContext
from llama_index.core.schema import BaseNode , ObjectType , TextNode 

from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.ingest.ingest_component import get_ingestion_component
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.nodeparser.SentenceChunkNodeParser import SentenceChunkWindowNodeParser
from private_gpt.components.nodeparser.PageByPageNodeParser import PageByPageNodeParser
from private_gpt.components.nodeparser.LateChunkNodeParser import LateChunkNodeParser
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.server.ingest.model import IngestedDoc
from private_gpt.constants import UPLOAD_DIR
from private_gpt.settings.settings import settings
from llama_index.core.extractors import SummaryExtractor
if TYPE_CHECKING:
    from llama_index.core.storage.docstore.types import RefDocInfo

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 512
SENTENCE_CHUNK_OVERLAP = 100
DEFAULT_WINDOW_SIZE = 3
DEFAULT_CHUNK_WINDOW = 10

class ChunkingStrategy(str, Enum):
    LATE_CHUNKING = "late_chunking"
    SENTENCE_WINDOW = "sentence_window"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"
    # PAGE_BY_PAGE = "page_by_page"

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
        self.embedding_component = embedding_component
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store_component.vector_store,
            docstore=node_store_component.doc_store,
            index_store=node_store_component.index_store,
        )      
        self.embedding_model = embedding_component.embedding_model
        self.settings = settings()
        
    def _get_node_parser(
        self, 
        strategy: ChunkingStrategy = ChunkingStrategy.HIERARCHICAL
    ):
        """Create node parser based on dynamic parameters."""
        # if strategy == ChunkingStrategy.LATE_CHUNKING:
        #     return LateChunkNodeParser.from_defaults(
        #         chunk_size=DEFAULT_CHUNK_SIZE,
        #         window_size=DEFAULT_WINDOW_SIZE,
        #     )
        # elif strategy == ChunkingStrategy.SENTENCE_WINDOW:
        #     return SentenceChunkWindowNodeParser.from_defaults(
        #         chunk_size=DEFAULT_CHUNK_WINDOW,
        #         window_size=DEFAULT_WINDOW_SIZE,  
        #         window_metadata_key="window",
        #         original_text_metadata_key="original_text",
        #         include_metadata=True,
        #         include_prev_next_rel=True
        #     )
        # elif strategy == ChunkingStrategy.SEMANTIC:
        #     return SemanticSplitterNodeParser.from_defaults(
        #         buffer_size=2, # Contextual buffer (number of sentences) around split points
        #         breakpoint_percentile_threshold=75, # Sensitivity to semantic shifts
        #         embed_model=self.embedding_model 
        #     )
        # elif strategy == ChunkingStrategy.HIERARCHICAL:
        #     # Create hierarchical chunks: Large (2048) -> Medium (512) -> Small (128)
        #     # This creates parent-child relationships for better context retrieval
        #     return HierarchicalNodeParser.from_defaults(
        #         chunk_sizes=[2048, 512, 128],  # 3 levels of granularity
        #         chunk_overlap=20,               # Overlap between chunks
        #     )
        # else:
        # return SentenceWindowNodeParser.from_defaults(
        #     window_size=10, 
        #     window_metadata_key="window",
        #     original_text_metadata_key="original_text",
        #     include_metadata=True,
        #     include_prev_next_rel=True
        # )
        return HierarchicalNodeParser.from_defaults(
            chunk_sizes=[1024, 512],  # 2 levels of granularity
            chunk_overlap=50,               # Overlap between chunks
        )

    def _get_ingest_component_with_parser(self, node_parser):
        """Create ingestion component with specific node parser."""
        return get_ingestion_component(
            self.storage_context,
            embed_model=self.embedding_model,
            transformations=[
                node_parser,
                self.embedding_model,
            ],
            settings=self.settings,
        )

    async def ingest_text(
        self, 
        file_name: str, 
        text: str, 
        metadata: dict[str, str] | None = None,
        strategy: ChunkingStrategy = ChunkingStrategy.HIERARCHICAL,
    ) -> list[IngestedDoc]:
        logger.debug("Ingesting text data with file_name=%s", file_name)
        try:
            return await self._ingest_data(file_name, text, metadata, strategy)
        except Exception as e:
            logger.error(f"Error during text ingestion for {file_name}: {str(e)}")
            raise

    async def ingest_bin_data(
        self,
        file_name: str,
        raw_file_data: BinaryIO,
        file_metadata: dict[str, str] | None = None,
        strategy: ChunkingStrategy = ChunkingStrategy.HIERARCHICAL,
    ) -> list[IngestedDoc]:
        logger.debug("Ingesting binary data with file_name=%s", file_name)
        try:
            file_data = raw_file_data.read()
            return await self._ingest_data(file_name, file_data, file_metadata, strategy)
        except Exception as e:
            logger.error(f"Error during binary data ingestion for {file_name}: {str(e)}")
            raise
    
    async def _ingest_data(
        self,
        file_name: str,
        file_data: AnyStr,
        file_metadata: dict[str, str] | None = None,
        strategy: ChunkingStrategy = ChunkingStrategy.HIERARCHICAL,
    ) -> list[IngestedDoc]:
        logger.debug("Got file data of size=%s to ingest", len(file_data))
        try:
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
                    return await self.ingest_file(file_name, path_to_tmp, file_metadata, strategy)
                finally:
                    tmp.close()
                    path_to_tmp.unlink()
        except Exception as e:
            logger.error(f"Error during data ingestion for {file_name}: {str(e)}")
            raise

    async def ingest_file(
        self,
        file_name: str,
        file_data: Path,
        file_metadata: dict[str, str] | None = None,
        strategy: ChunkingStrategy = ChunkingStrategy.HIERARCHICAL,
    ) -> list[IngestedDoc]:
        logger.info("Ingesting file_name=%s with strategy=%s", 
                   file_name, strategy.value)
        try:
            node_parser = self._get_node_parser(strategy)
            ingest_component = self._get_ingest_component_with_parser(node_parser)
            
            # Ensure file_path and document_path are in metadata
            file_metadata = file_metadata or {}
            abs_path = str(file_data.resolve())
            if "file_path" not in file_metadata:
                file_metadata["file_path"] = abs_path
            
            # Normalize document_path to be relative to 'documents' folder
            try:
                documents_dir = (Path(UPLOAD_DIR) / "documents").resolve()
                rel_path = file_data.resolve().relative_to(documents_dir)
                file_metadata["document_path"] = str(rel_path)
            except ValueError:
                # Fallback to absolute path or existing value if not in documents folder
                if "document_path" not in file_metadata:
                    file_metadata["document_path"] = abs_path

            documents = await ingest_component.ingest(file_name, file_data, file_metadata)
            logger.info("Finished ingestion file_name=%s", file_name)
            return [IngestedDoc.from_document(document) for document in documents]
        except Exception as e:
            logger.error(f"Error during file ingestion for {file_name}: {str(e)}")
            raise

    async def ingest_url(
        self, 
        url: str, 
        documents,
        strategy: ChunkingStrategy = ChunkingStrategy.HIERARCHICAL,
    ) -> list[IngestedDoc]:
        logger.debug("Ingesting url=%s with strategy=%s", url, strategy.value)
        
        # Create node parser with dynamic parameters
        node_parser = self._get_node_parser(strategy)
        ingest_component = self._get_ingest_component_with_parser(node_parser)
        
        documents = await ingest_component.ingest_url(url, documents)
        return [IngestedDoc.from_document(document) for document in documents]

    async def bulk_ingest(
        self, 
        files: list[tuple[str, Path]],
        strategy: ChunkingStrategy = ChunkingStrategy.HIERARCHICAL,
    ) -> list[IngestedDoc]:
        logger.info("Ingesting file_names=%s with strategy=%s", [f[0] for f in files], strategy.value)
        
        # Create node parser with dynamic parameters
        node_parser = self._get_node_parser(strategy)
        ingest_component = self._get_ingest_component_with_parser(node_parser)
        
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

    async def delete(self, doc_id: str, strategy: ChunkingStrategy = ChunkingStrategy.HIERARCHICAL ) -> None:
        """Delete an ingested document.

        :raises ValueError: if the document does not exist
        """
        logger.info(
            "Deleting the ingested document=%s in the doc and index store", doc_id
        )
        node_parser = self._get_node_parser(strategy)
        ingest_component = self._get_ingest_component_with_parser(node_parser)
        ingest_component.delete(doc_id) 

    async def delete_docs(
        self, 
        doc_ids: [str], 
        strategy: ChunkingStrategy = ChunkingStrategy.HIERARCHICAL,
    ) -> None:
        logger.info(
            "Deleting the ingested document(s) in the doc and index store with doc_ids=%s", doc_ids
        )
        node_parser = self._get_node_parser(strategy)
        ingest_component = self._get_ingest_component_with_parser(node_parser)
        await ingest_component.delete_doc_ids(doc_ids) 
        logger.info("Deleted count=%s documents", len(doc_ids))

    async def delete_by_metadata(
        self, 
        key: str, 
        value: Any, 
        strategy: ChunkingStrategy = ChunkingStrategy.HIERARCHICAL
    ) -> None:
        logger.info(
            "Deleting the ingested document(s) by metadata %s=%s", key, value
        )
        node_parser = self._get_node_parser(strategy)
        ingest_component = self._get_ingest_component_with_parser(node_parser)
        await ingest_component.delete_by_metadata(key, value)

    def get_doc_ids_by_filename(self, filename: str) -> list[str]:
        doc_ids: set[str] = set()
        try:
            docstore = self.storage_context.docstore
            for node in docstore.docs.values():
                if node.metadata is not None and node.metadata.get("file_name") == filename:
                    # Check both ref_doc_id (for nodes) and id_ (for document objects)
                    id_to_add = getattr(node, "ref_doc_id", None) or getattr(node, "id_", None)
                    if id_to_add:
                        doc_ids.add(id_to_add)

        except ValueError:
            logger.warning("Got an exception when getting doc_ids by filename", exc_info=True)
            pass

        logger.debug("Found count=%s doc_ids for filename '%s'",
                     len(doc_ids), filename)
        return list(doc_ids)

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
                    
                    # Check both ref_doc_id (for nodes) and id_ (for document objects)
                    id_to_add = getattr(node, "ref_doc_id", None) or getattr(node, "id_", None)
                    if id_to_add:
                        doc_ids.add(id_to_add)
        except ValueError:
            logger.warning(
                "Got an exception when getting doc_ids by filename pattern",
                exc_info=True
            )
            pass
        
        logger.debug("Found count=%s doc_ids for filename pattern '%s'",
                     len(doc_ids), pattern)
        
        return list(doc_ids)

    def get_doc_ids_by_metadata(self, key: str, value: Any) -> list[str]:
        """
        Get document IDs matching a specific metadata key-value pair.
        
        Args:
            key (str): Metadata key to match
            value (Any): Metadata value to match (exact match)
            
        Returns:
            list[str]: List of document IDs
        """
        doc_ids: set[str] = set()
        try:
            docstore = self.storage_context.docstore
            for node in docstore.docs.values():
                if node.metadata is not None and node.metadata.get(key) == value:
                    # Check both ref_doc_id (for nodes) and id_ (for document objects)
                    id_to_add = getattr(node, "ref_doc_id", None) or getattr(node, "id_", None)
                    if id_to_add:
                        doc_ids.add(id_to_add)

        except ValueError:
            logger.warning("Got an exception when getting doc_ids by metadata", exc_info=True)
            pass

        logger.debug("Found count=%s doc_ids for metadata %s=%s", len(doc_ids), key, value)
        return list(doc_ids)