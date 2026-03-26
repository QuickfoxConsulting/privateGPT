"""Ingestion service with dynamic chunking support."""

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, AnyStr, BinaryIO, Sequence, Any, List, Optional, Dict
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
from llama_index.core.schema import BaseNode , ObjectType , TextNode, Document 

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
from private_gpt.paths import local_data_path

from private_gpt.server.ingest.model import IngestedDoc, IngestPatchInput
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
        self.node_store_component = node_store_component
        
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

    def _get_freshest_docstore(self):
        """Returns the freshest docstore, reloading from disk if using SimpleDocumentStore to sync workers."""
        if self.settings.nodestore.database == "simple":
            self.node_store_component.refresh()
            self.storage_context.docstore = self.node_store_component.doc_store
            self.storage_context.index_store = self.node_store_component.index_store
        return self.storage_context.docstore

    def get_raw_document(self, doc_id: str) -> Optional[Document]:
        """Get the raw (pre-chunked) document by ID."""
        docstore = self._get_freshest_docstore()
        if doc_id in docstore.docs:
            doc = docstore.get_document(doc_id)
            if isinstance(doc, Document):
                return doc
        return None

    def get_raw_page_by_filename_and_page(self, filename: str, page: int) -> Optional[BaseNode]:
        """Get the raw document for a specific file and page number."""
        docstore = self._get_freshest_docstore()
        
        # Priority 1: Find a node with text_locations (indicates high-fidelity page data)
        for node in docstore.docs.values():
            if not node.metadata:
                continue
            
            # Match filename in either variant
            node_filename = node.metadata.get("file_name") or node.metadata.get("filename")
            # In some readers, page might be stored as 'page_label' or be a string
            node_page = node.metadata.get("page") or node.metadata.get("page_label")
            
            if node_filename == filename and str(node_page) == str(page):
                if "text_locations" in node.metadata:
                    return node
                    
        # Priority 2: Return anything that matches filename/page
        for node in docstore.docs.values():
            if not node.metadata:
                continue
            node_filename = node.metadata.get("file_name") or node.metadata.get("filename")
            node_page = node.metadata.get("page") or node.metadata.get("page_label")
            if node_filename == filename and str(node_page) == str(page):
                return node
        return None

    def get_nodes_by_filename_and_page(self, filename: str, page: int) -> list[BaseNode]:
        """Get all nodes (chunks) for a specific file and page number."""
        docstore = self.storage_context.docstore
        found_nodes = []
        for node in docstore.docs.values():
            if not node.metadata:
                continue
            node_filename = node.metadata.get("file_name") or node.metadata.get("filename")
            node_page = node.metadata.get("page") or node.metadata.get("page_label")
            if node_filename == filename and str(node_page) == str(page):
                found_nodes.append(node)
                
        # Sort by their start index if available to maintain reading order
        found_nodes.sort(key=lambda n: getattr(n, "start_char_idx", 0) or 0)
        return found_nodes

    async def patch_raw_page(
        self, 
        filename: str, 
        page: int, 
        text: str, 
        text_locations: Optional[list[dict[str, Any]]] = None
    ) -> dict[str, Any]:
        """Update the raw content of all matching page nodes in the docstore."""
        docstore = self._get_freshest_docstore()
        
        nodes_to_update = []
        updated_ids = []
        for node_id, node in docstore.docs.items():
            if not node.metadata:
                continue
            
            node_filename = node.metadata.get("file_name") or node.metadata.get("filename")
            node_page = node.metadata.get("page") or node.metadata.get("page_label")
            
            if node_filename == filename and str(node_page) == str(page):
                logger.info("Patching node ID: %s for file: %s page: %s", node_id, filename, page)
                
                # Update content
                node.text = text
                if hasattr(node, "set_content"):
                    node.set_content(text)
                
                # Update spatial metadata
                if text_locations is not None:
                    node.metadata["text_locations"] = text_locations
                
                nodes_to_update.append(node)
                updated_ids.append(node_id)

        if nodes_to_update:
            # Save the updated nodes back to the docstore
            docstore.add_documents(nodes_to_update)
            
            # Persist changes to disk (for simple docstore)
            self.storage_context.persist(persist_dir=str(local_data_path))
            
            logger.info("Successfully patched %d raw content nodes for %s page %s", 
                        len(nodes_to_update), filename, page)
            return {"success": True, "count": len(updated_ids), "ids": updated_ids}
            
        logger.warning("Could not find any raw nodes to patch for %s page %s", filename, page)
        return {"success": False, "count": 0, "ids": []}

    def _delete_entities_for_file(self, filename: str) -> int:
        """Delete all NER entity links for nodes belonging to this file."""
        deleted = 0
        try:
            from private_gpt.users.db.session import SessionLocal
            from private_gpt.users.models.entity import NodeEntity
            
            with SessionLocal() as session:
                # Find all node IDs belonging to this file
                docstore = self.storage_context.docstore
                node_ids_for_file = []
                for node_id, node in docstore.docs.items():
                    if not node.metadata:
                        continue
                    node_filename = node.metadata.get("file_name") or node.metadata.get("filename")
                    if node_filename == filename:
                        node_ids_for_file.append(node_id)
                
                if node_ids_for_file:
                    deleted = session.query(NodeEntity).filter(
                        NodeEntity.node_id.in_(node_ids_for_file)
                    ).delete(synchronize_session="fetch")
                    session.commit()
                    logger.info("Deleted %d entity links for file %s", deleted, filename)
        except Exception as e:
            logger.error("Failed to delete entities for file %s: %s", filename, str(e))
        return deleted

    async def reindex_from_raw(
        self,
        filename: str,
        strategy: ChunkingStrategy = ChunkingStrategy.HIERARCHICAL,
    ) -> dict[str, Any]:
        """Re-chunk and re-embed a document from its existing raw page data.
        
        This skips OCR/PDF extraction and uses the (possibly edited) raw pages 
        already in the docstore as the source of truth.
        """
        logger.info("Starting reindex from raw for file=%s strategy=%s", filename, strategy.value)
        
        # Step 1: Collect all raw page nodes from the docstore
        docstore = self._get_freshest_docstore()
        raw_pages = []
        raw_page_ids = set()
        
        from llama_index.core.schema import NodeRelationship
        
        for node_id, node in docstore.docs.items():
            if not node.metadata:
                continue
            
            node_filename = node.metadata.get("file_name") or node.metadata.get("filename")
            if node_filename != filename:
                continue
            
            # STRENGHTENED HEURISTIC FOR RAW PAGES:
            # 1. Must have a page number
            # 2. Must NOT have a PARENT relationship (chunks always point to a parent)
            # 3. Must have 'text_locations' (only the original OCR reader provides this)
            node_page = node.metadata.get("page") or node.metadata.get("page_label")
            has_parent = NodeRelationship.PARENT in node.relationships
            
            # The original page document created by Reader won't have a parent 
            # and will have the OCR text_locations.
            is_raw = node_page is not None and not has_parent and "text_locations" in node.metadata
            
            if is_raw:
                raw_pages.append(node)
                raw_page_ids.add(node_id)
        
        if not raw_pages:
            logger.warning("No raw page nodes found for file=%s", filename)
            return {"success": False, "message": f"No raw pages found for {filename}", "nodes_created": 0}
        
        # Sort by page number
        raw_pages.sort(key=lambda n: int(n.metadata.get("page") or n.metadata.get("page_label") or 0))
        logger.info("Found %d raw page nodes for file=%s", len(raw_pages), filename)
        
        # Step 2: Delete old downstream data (chunks, embeddings, entity links)
        # but KEEP the raw page nodes
        
        # 2a: Delete entity links from Postgres
        self._delete_entities_for_file(filename)
        
        # 2b: Delete embeddings from vector store (Qdrant)
        node_parser = self._get_node_parser(strategy)
        ingest_component = self._get_ingest_component_with_parser(node_parser)
        await ingest_component.delete_by_metadata("file_name", filename)
        await ingest_component.delete_by_metadata("filename", filename)
        logger.info("Deleted old embeddings from vector store for file=%s", filename)
        
        # 2c: Delete non-raw nodes from docstore (chunk nodes)
        # and clean up index references
        nodes_to_delete = []
        ref_doc_ids = set()
        for node_id, node in docstore.docs.items():
            if not node.metadata:
                continue
            node_filename = node.metadata.get("file_name") or node.metadata.get("filename")
            if node_filename != filename:
                continue
            # Keep raw page nodes, delete everything else (chunks)
            if node_id not in raw_page_ids:
                nodes_to_delete.append(node_id)
                ref_id = getattr(node, "ref_doc_id", None)
                if ref_id:
                    ref_doc_ids.add(ref_id)
        
        # Delete chunk nodes from docstore
        for node_id in nodes_to_delete:
            try:
                docstore.delete_document(node_id)
            except Exception:
                pass
        logger.info("Deleted %d chunk nodes from docstore for file=%s", len(nodes_to_delete), filename)
        
        # Clean up index references
        for ref_id in ref_doc_ids:
            try:
                ingest_component._index.delete_ref_doc(ref_id, delete_from_docstore=False)
            except Exception:
                pass
        
        # Step 3: Convert raw pages into LlamaIndex Documents
        documents = []
        # Strict allowlist of metadata keys to avoid 'Metadata length too long' errors
        # during chunking. We must strip 'entity_details', 'text_locations', etc.
        METADATA_ALLOWLIST = {
            "file_name", "filename", "page", "page_label", 
            "file_path", "total_pages", "document_id",
            "document_class", "render_dpi"
        }
        
        for raw_node in raw_pages:
            # Create a fresh Document from the raw page's text and ALLOWED metadata
            chunk_metadata = {
                k: v for k, v in raw_node.metadata.items() 
                if k in METADATA_ALLOWLIST
            }
            doc = Document(
                text=raw_node.get_content(),
                metadata=chunk_metadata,
            )
            documents.append(doc)
        
        logger.info("Created %d Documents from raw pages (metadata cleaned) for re-ingestion", len(documents))
        
        # Step 4: Run the full _save_docs pipeline
        # This does: chunking → PageMetadataBackfiller → NER → embedding → index → persist
        result_docs = await ingest_component._save_docs(documents)
        
        # Step 5: Persist the docstore (raw pages are still there)
        self.storage_context.persist(persist_dir=str(local_data_path))
        
        logger.info("Reindex complete for file=%s. Created %d indexed documents.", filename, len(result_docs))
        return {
            "success": True,
            "message": f"Re-indexed {filename} from {len(raw_pages)} raw pages",
            "raw_pages": len(raw_pages),
            "nodes_created": len(result_docs),
        }

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
            for node_id, node in docstore.docs.items():
                if node.metadata is not None:
                    node_filename = node.metadata.get("file_name") or node.metadata.get("filename")
                    if node_filename == filename:
                        # Capture both the node's individual ID and its reference ID
                        doc_ids.add(node_id)
                        ref_id = getattr(node, "ref_doc_id", None) or getattr(node, "doc_id", None)
                        if ref_id:
                            doc_ids.add(ref_id)

        except ValueError:
            logger.warning("Got an exception when getting doc_ids by filename", exc_info=True)
            pass

        logger.debug("Found count=%s doc_ids for filename '%s'",
                     len(doc_ids), filename)
        return list(doc_ids)

    def get_all_text_locations_by_filename(self, filename: str) -> List[List[Dict[str, Any]]]:
        """
        Collect text_locations from all raw page nodes for the given filename, sorted by page number.
        """
        docstore = self._get_freshest_docstore()
        raw_pages_map: Dict[int, List[Dict[str, Any]]] = {}
        
        from llama_index.core.schema import NodeRelationship
        
        for node in docstore.docs.values():
            if not node.metadata:
                continue
            
            node_filename = node.metadata.get("file_name") or node.metadata.get("filename")
            if node_filename != filename:
                continue
            
            # Identify raw page nodes (no parent, has page number)
            node_page = node.metadata.get("page") or node.metadata.get("page_label")
            has_parent = NodeRelationship.PARENT in node.relationships
            
            if node_page is not None and not has_parent:
                try:
                    page_idx = int(node_page)
                    # We prefer nodes with text_locations as they are the primary source for reconstruction
                    if "text_locations" in node.metadata:
                        raw_pages_map[page_idx] = node.metadata["text_locations"]
                    elif page_idx not in raw_pages_map:
                        # Fallback for pages without spatial metadata (just empty list for that page)
                        raw_pages_map[page_idx] = []
                except (ValueError, TypeError):
                    continue
        
        # Sort by page number and return as a flat list of lists
        sorted_pages = sorted(raw_pages_map.keys())
        return [raw_pages_map[p] for p in sorted_pages]

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
                if node.metadata is not None:
                    node_filename = node.metadata.get("file_name") or node.metadata.get("filename")
                    if node_filename and pattern in node_filename:
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