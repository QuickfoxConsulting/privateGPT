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
from private_gpt.components.nodeparser.PageMetadataBackfiller import PageMetadataBackfiller
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.components.entity.entity_component import EntityComponent
from private_gpt.components.ocr_components.document_reconstructor import DocumentReconstructor

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
    BASIC = "basic"
    UNIFIED_CONTEXT = "unified_context"

@singleton
class IngestService:
    @inject
    def __init__(
        self,
        llm_component: LLMComponent,
        vector_store_component: VectorStoreComponent,
        embedding_component: EmbeddingComponent,
        node_store_component: NodeStoreComponent,
        entity_component: EntityComponent,
        reconstructor: DocumentReconstructor,
    ) -> None:
        self.llm_service = llm_component
        self.embedding_model = embedding_model = embedding_component.embedding_model
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store_component.vector_store,
            docstore=node_store_component.doc_store,
            index_store=node_store_component.index_store,
        )      
        self.embedding_model = embedding_component.embedding_model
        self.settings = settings()
        self.entity_component = entity_component
        self.reconstructor = reconstructor

        
    def _validate_file_name(self, file_name: str) -> None:
        """Validate file name for security and correctness."""
        if not file_name or not file_name.strip():
            raise ValueError("file_name cannot be empty")
        if len(file_name) > 255:
            raise ValueError("file_name too long (max 255 chars)")
        # Check for path traversal attempts
        if any(c in file_name for c in ['/', '\\', '\0']):
            raise ValueError("file_name contains invalid characters")
        if file_name.startswith('..'):
            raise ValueError("file_name cannot start with '..'")
    
    async def health_check(self) -> dict[str, Any]:
        """Check health of ingestion system components."""
        health = {
            "status": "healthy",
            "checks": {}
        }
        
        try:
            # Check vector store
            health["checks"]["vector_store"] = {
                "status": "ok" if self.storage_context.vector_store else "error",
                "type": type(self.storage_context.vector_store).__name__
            }
        except Exception as e:
            health["checks"]["vector_store"] = {"status": "error", "error": str(e)}
            health["status"] = "degraded"
        
        try:
            # Check docstore
            doc_count = len(self.storage_context.docstore.docs)
            health["checks"]["docstore"] = {
                "status": "ok",
                "document_count": doc_count
            }
        except Exception as e:
            health["checks"]["docstore"] = {"status": "error", "error": str(e)}
            health["status"] = "degraded"
        
        try:
            # Check embedding model
            health["checks"]["embedding_model"] = {
                "status": "ok" if self.embedding_model else "error",
                "model": type(self.embedding_model).__name__
            }
        except Exception as e:
            health["checks"]["embedding_model"] = {"status": "error", "error": str(e)}
            health["status"] = "degraded"
        
        return health
        
    def _get_node_parser(
        self, 
        strategy: ChunkingStrategy = None
    ):
        """Create node parser based on dynamic parameters."""
        # Use strategy from settings if not provided
        if strategy is None:
            strategy = ChunkingStrategy(self.settings.chunking.strategy)

        if strategy == ChunkingStrategy.LATE_CHUNKING:
            return LateChunkNodeParser.from_defaults(
                chunk_size=self.settings.chunking.chunk_size,
                window_size=DEFAULT_WINDOW_SIZE,
            )
        elif strategy == ChunkingStrategy.SENTENCE_WINDOW:
            return SentenceChunkWindowNodeParser.from_defaults(
                chunk_size=self.settings.chunking.chunk_size,
                window_size=DEFAULT_WINDOW_SIZE,  
                window_metadata_key="window",
                original_text_metadata_key="original_text",
                include_metadata=True,
                include_prev_next_rel=True
            )
        elif strategy == ChunkingStrategy.SEMANTIC:
            return SemanticSplitterNodeParser.from_defaults(
                buffer_size=2, # Contextual buffer (number of sentences) around split points
                breakpoint_percentile_threshold=75, # Sensitivity to semantic shifts
                embed_model=self.embedding_model 
            )
        elif strategy == ChunkingStrategy.HIERARCHICAL:
            # Create hierarchical chunks based on settings
            return HierarchicalNodeParser.from_defaults(
                chunk_sizes=self.settings.chunking.hierarchical_chunk_sizes,
                chunk_overlap=self.settings.chunking.hierarchical_chunk_overlap,
                include_metadata=True,
                include_prev_next_rel=True
            )
        elif strategy == ChunkingStrategy.BASIC:
            # --- ADVANCED BASIC STRATEGY ---
            # This is an upgraded version of RecursiveCharacterTextSplitter.
            # It intelligently splits on paragraph/sentence boundaries to avoid cutting thoughts.
            return SentenceSplitter(
                chunk_size=self.settings.chunking.chunk_size,
                chunk_overlap=self.settings.chunking.chunk_overlap,
                # Intelligently split on these markers in order (Advanced)
                separator=" ",
                paragraph_separator="\n\n\n",
                secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?",
                include_metadata=True,
                include_prev_next_rel=True
            )
        elif strategy == ChunkingStrategy.UNIFIED_CONTEXT:
            from private_gpt.components.nodeparser.UnifiedContextNodeParser import UnifiedContextNodeParser
            return UnifiedContextNodeParser.from_defaults(
                chunk_sizes=self.settings.chunking.hierarchical_chunk_sizes,
                chunk_overlap=self.settings.chunking.hierarchical_chunk_overlap,
            )
        else:
            # Fallback to SentenceSplitter
            return SentenceSplitter(
                chunk_size=self.settings.chunking.chunk_size,
                chunk_overlap=self.settings.chunking.chunk_overlap,
            )

    def _get_ingest_component_with_parser(self, node_parser):
        """Create ingestion component with specific node parser."""
        # We add PageMetadataBackfiller to ensure every chunk knows its Page/File context.
        return get_ingestion_component(
            self.storage_context,
            embed_model=self.embedding_model,
            transformations=[
                node_parser,
                PageMetadataBackfiller(), # <--- This makes 'Basic' better by adding context back in
                self.entity_component,
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
        idempotent: bool = True,
    ) -> list[IngestedDoc]:
        """Ingest a file with optional idempotent behavior."""
        # Validate file name
        self._validate_file_name(file_name)
        
        logger.info("Ingesting file_name=%s with strategy=%s, idempotent=%s", 
                   file_name, strategy.value, idempotent)
        
        # Check if file already exists (idempotent mode)
        if idempotent:
            existing_docs = self.get_doc_ids_by_filename(file_name)
            if existing_docs:
                logger.info(f"File {file_name} already ingested, skipping (found {len(existing_docs)} docs)")
                
                # Try to populate metadata from the first found doc
                doc_metadata = None
                try:
                    docstore = self.storage_context.docstore
                    first_doc_id = existing_docs[0]
                    if first_doc_id in docstore.docs:
                        node = docstore.docs[first_doc_id]
                        if node.metadata:
                            doc_metadata = IngestedDoc.curate_metadata(node.metadata)
                except Exception:
                    logger.warning("Failed to retrieve metadata for existing document", exc_info=True)

                return [
                    IngestedDoc(
                        object="ingest.document", 
                        doc_id=doc_id,
                        doc_metadata=doc_metadata # May be None, but model allows it now
                    ) for doc_id in existing_docs
                ]
        
        try:
            node_parser = self._get_node_parser(strategy)
            ingest_component = self._get_ingest_component_with_parser(node_parser)
            file_metadata = file_metadata or {}
            abs_path = str(file_data.resolve())            
            file_metadata["file_path"] = abs_path

            documents = await ingest_component.ingest(file_name, file_data, file_metadata)
            
            # --- Vision Pipeline: PDF Reconstruction (The Stamper) ---
            if file_name.lower().endswith(".pdf"):
                try:
                    enhanced_path = self.reconstructor.reconstruct_pdf(file_data, documents)
                    # Add reference to enhanced version in the metadata of the returning docs
                    # This allows the UI to know there is a better version available.
                    for doc in documents:
                        doc.metadata["enhanced_pdf_path"] = str(enhanced_path)
                    logger.info("Vision Reconstruction successful for %s -> %s", file_name, enhanced_path)
                except Exception as rec_e:
                    logger.warning("PDF Reconstruction failed for %s, but ingestion continued: %s", 
                                   file_name, str(rec_e))

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
        doc_ids: list[str], 
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