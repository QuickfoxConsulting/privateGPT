import logging
import uuid
from pathlib import Path
from typing import Any

from llama_index.core import StorageContext
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.ingestion import run_transformations
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import BaseNode, Document, TransformComponent
from llama_index.core.storage.docstore import BaseDocumentStore

from private_gpt.components.ingest.ingest_component import BaseIngestComponentWithIndex
from private_gpt.components.ingest.ingest_helper import IngestionHelper

logger = logging.getLogger(__name__)


class ParentDocumentIngestComponent(BaseIngestComponentWithIndex):
    """
    Ingest component that implements Parent Document Retrieval using HierarchicalNodeParser.
    It creates small chunks (leaf nodes) for vector search and large chunks (parent nodes)
    for providing better context to the LLM.
    """

    def __init__(
        self,
        storage_context: StorageContext,
        embed_model: EmbedType,
        transformations: list[TransformComponent],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(storage_context, embed_model, transformations, *args, **kwargs)
        # We use HierarchicalNodeParser specifically for this component
        # Default chunk sizes if not provided in transformations
        self.node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 512, 128]
        )

    async def ingest(
        self,
        file_name: str,
        file_data: Path,
        file_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        logger.info("Ingesting file_name=%s with Parent-Child strategy", file_name)
        documents = await IngestionHelper.transform_file_into_documents(
            file_name, file_data, file_metadata
        )
        logger.info(
            "Transformed file=%s into count=%s documents", file_name, len(documents)
        )
        return await self._save_docs(documents)

    async def bulk_ingest(self, files: list[tuple[str, Path]]) -> list[Document]:
        saved_documents = []
        for file_name, file_data in files:
            documents = await IngestionHelper.transform_file_into_documents(
                file_name, file_data
            )
            saved_documents.extend(await self._save_docs(documents))
        return saved_documents

    async def _save_docs(self, documents: list[Document]) -> list[Document]:
        logger.debug("Transforming count=%s documents into hierarchical nodes", len(documents))
        
        # 1. Generate hierarchical nodes
        nodes = self.node_parser.get_nodes_from_documents(documents)
        leaf_nodes = get_leaf_nodes(nodes)
        
        # 2. Run other transformations (like embeddings) on LEAF nodes only
        # We want to index leaf nodes in the vector store
        processed_leaf_nodes = run_transformations(
            leaf_nodes,
            self.transformations,
            show_progress=self.show_progress,
        )

        # 3. Store ALL nodes (parents and children) in the docstore
        # This is required so that the retriever can find the parent of a leaf node
        with self._index_thread_lock:
            logger.info("Storing count=%s nodes in docstore", len(nodes))
            self.storage_context.docstore.add_documents(nodes)
            
            logger.info("Inserting count=%s leaf nodes in the index", len(processed_leaf_nodes))
            self._index.insert_nodes(processed_leaf_nodes, show_progress=True)
            
            for document in documents:
                self._index.docstore.set_document_hash(
                    document.get_doc_id(), document.hash
                )
            
            logger.debug("Persisting the index and nodes")
            self._save_index()
            logger.debug("Persisted the index and nodes")
            
        return documents
