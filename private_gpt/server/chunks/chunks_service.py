from typing import TYPE_CHECKING, Literal

from injector import inject, singleton
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.storage import StorageContext
from pydantic import BaseModel, Field

from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.components.entity.entity_component import EntityComponent

from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.server.ingest.model import IngestedDoc

if TYPE_CHECKING:
    from llama_index.core.schema import RelatedNodeInfo


class Chunk(BaseModel):
    object: Literal["context.chunk"]
    score: float = Field(examples=[0.023])
    document: IngestedDoc
    text: str = Field(examples=["Outbound sales increased 20%, driven by new leads."])
    node_id: str | None = Field(default=None, examples=["node_id_123"])
    previous_texts: list[str] | None = Field(
        default=None,
        examples=[["SALES REPORT 2023", "Inbound didn't show major changes."]],
    )
    next_texts: list[str] | None = Field(
        default=None,
        examples=[
            [
                "New leads came from Google Ads campaign.",
                "The campaign was run by the Marketing Department",
            ]
        ],
    )

    @classmethod
    def from_node(cls: type["Chunk"], node: NodeWithScore) -> "Chunk":
        doc_id = node.node.ref_doc_id if node.node.ref_doc_id is not None else "-"
        return cls(
            object="context.chunk",
            score=node.score or 0.0,
            document=IngestedDoc(
                object="ingest.document",
                doc_id=doc_id,
                doc_metadata=node.metadata,
            ),
            text=node.get_content(),
            node_id=node.node.node_id,
        )


@singleton
class ChunksService:
    @inject
    def __init__(
        self,
        llm_component: LLMComponent,
        vector_store_component: VectorStoreComponent,
        embedding_component: EmbeddingComponent,
        node_store_component: NodeStoreComponent,
        entity_component: EntityComponent,
    ) -> None:
        self.vector_store_component = vector_store_component
        self.llm_component = llm_component
        self.embedding_component = embedding_component
        self.entity_component = entity_component

        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store_component.vector_store,
            docstore=node_store_component.doc_store,
            index_store=node_store_component.index_store,
        )

    def _get_sibling_nodes_text(
        self, node_with_score: NodeWithScore, related_number: int, forward: bool = True
    ) -> list[str]:
        import logging
        logger = logging.getLogger(__name__)
        
        explored_nodes_texts = []
        current_node = node_with_score.node
        for _ in range(related_number):
            explored_node_info: RelatedNodeInfo | None = (
                current_node.next_node if forward else current_node.prev_node
            )
            if explored_node_info is None:
                break

            try:
                explored_node = self.storage_context.docstore.get_node(
                    explored_node_info.node_id
                )
                explored_nodes_texts.append(explored_node.get_content())
                current_node = explored_node
            except Exception as e:
                logger.error(f"Async retrieval error: doc_id {explored_node_info.node_id} not found.")
                # Stop traversing siblings if we hit a missing node
                break

        return explored_nodes_texts
    
    def _get_nodes_from_same_document(self, node_with_score: NodeWithScore) -> list[str]:
        """
        Retrieve text from other nodes with the same document ID
        """
        import logging
        logger = logging.getLogger(__name__)
        
        current_node = node_with_score.node
        current_doc_id = current_node.metadata.get('doc_id')
        
        if not current_doc_id:
            return []
        
        # Find nodes with the same doc_id
        same_doc_nodes = []
        for node_id, node_data in self.storage_context.docstore.docs.items():
            if (node_data.metadata.get('doc_id') == current_doc_id and 
                node_id != current_node.node_id):
                try:
                    same_doc_node = self.storage_context.docstore.get_node(node_id)
                    same_doc_nodes.append(same_doc_node.get_content())
                except Exception as e:
                    logger.warning(f"Node {node_id} not found in docstore, skipping")
                    continue
        
        return same_doc_nodes
    
    def _process_retrieved_nodes(self, nodes: list[NodeWithScore], prev_next_chunks: int) -> list[Chunk]:
        """Shared logic to process retrieved nodes into Chunks."""
        import logging
        logger = logging.getLogger(__name__)
        
        nodes.sort(key=lambda n: n.score or 0.0, reverse=True)
        retrieved_nodes = []
        for node in nodes:
            try:
                chunk = Chunk.from_node(node)
                chunk.previous_texts = self._get_sibling_nodes_text(
                    node, prev_next_chunks, False
                )
                chunk.next_texts = self._get_sibling_nodes_text(node, prev_next_chunks)
                retrieved_nodes.append(chunk)
            except Exception as e:
                logger.error(f"Failed to process node {node.node.node_id}: {e}. Skipping this node.")
                continue
        return retrieved_nodes

    def _retrieve_by_entities(self, text: str, limit: int = 5) -> list[NodeWithScore]:
        """Search for nodes linked to entities extracted from the query text."""
        if not self.entity_component.enabled:
            return []
            
        import logging
        logger = logging.getLogger(__name__)
        
        # Extract entities from query
        entities = self.entity_component.extract_entities(text)
        if not entities:
            return []
            
        entity_names = list(set([ent["text"] for ent in entities]))
        logger.info("-> ENTITY RETRIEVAL: Found entities in query: %s", entity_names)
        
        from private_gpt.users.db.session import SessionLocal
        from private_gpt.users.models.entity import Entity, NodeEntity
        
        node_ids = []
        try:
            with SessionLocal() as session:
                # Find matching Entity IDs in DB
                db_entities = session.query(Entity.id).filter(Entity.name.in_(entity_names)).all()
                entity_ids = [e[0] for e in db_entities]
                
                if not entity_ids:
                    logger.info("-> ENTITY RETRIEVAL: No matching entities found in database.")
                    return []
                    
                # Find Node IDs linked to these entities
                # We limit results to avoid overwhelming the context
                links = session.query(NodeEntity.node_id).filter(
                    NodeEntity.entity_id.in_(entity_ids)
                ).limit(limit).all()
                node_ids = list(set([link[0] for link in links]))
        except Exception as e:
            logger.error("Database search for entities failed: %s", str(e))
            return []
            
        # Retrieve actual nodes from docstore
        nodes = []
        for nid in node_ids:
            try:
                node = self.storage_context.docstore.get_node(nid)
                # Score 1.0 to ensure these appear at the top or are highly relevant
                nodes.append(NodeWithScore(node=node, score=1.0))
            except Exception:
                continue
        
        logger.info("-> ENTITY RETRIEVAL: Successfully retrieved %d nodes. Node IDs: %s", len(nodes), node_ids)
        return nodes


    def retrieve_relevant_sync(
        self,
        text: str,
        context_filter: ContextFilter | None = None,
        limit: int = 10,
        prev_next_chunks: int = 0,
    ) -> list[Chunk]:
        index = VectorStoreIndex.from_vector_store(
            self.vector_store_component.vector_store,
            storage_context=self.storage_context,
            llm=self.llm_component.llm,
            embed_model=self.embedding_component.embedding_model,
            show_progress=True,
        )
        vector_index_retriever = self.vector_store_component.get_retriever(
            index=index, context_filter=context_filter, similarity_top_k=limit
        )
        nodes = vector_index_retriever.retrieve(text)
        
        # --- EXPERT RETRIEVAL: Entity Boost ---
        entity_nodes = self._retrieve_by_entities(text, limit=min(5, limit))
        if entity_nodes:
            # Combine and deduplicate
            existing_node_ids = {n.node.node_id for n in nodes}
            for en in entity_nodes:
                if en.node.node_id not in existing_node_ids:
                    nodes.append(en)
        
        return self._process_retrieved_nodes(nodes, prev_next_chunks)


    async def retrieve_relevant(
        self,
        text: str,
        context_filter: ContextFilter | None = None,
        limit: int = 10,
        prev_next_chunks: int = 0,
    ) -> list[Chunk]:
        index = VectorStoreIndex.from_vector_store(
            self.vector_store_component.vector_store,
            storage_context=self.storage_context,
            llm=self.llm_component.llm,
            embed_model=self.embedding_component.embedding_model,
            show_progress=True,
        )
        vector_index_retriever = self.vector_store_component.get_retriever(
            index=index, context_filter=context_filter, similarity_top_k=limit
        )
        nodes = await vector_index_retriever.aretrieve(text)

        # --- EXPERT RETRIEVAL: Entity Boost ---
        # Run DB search in thread pool to avoid blocking async loop
        import asyncio
        entity_nodes = await asyncio.to_thread(self._retrieve_by_entities, text, min(5, limit))
        if entity_nodes:
            # Combine and deduplicate
            existing_node_ids = {n.node.node_id for n in nodes}
            for en in entity_nodes:
                if en.node.node_id not in existing_node_ids:
                    nodes.append(en)

        return self._process_retrieved_nodes(nodes, prev_next_chunks)

    def get_document_page_chunks(
        self, document_id: str, page_num: int
    ) -> list[dict]:
        """Retrieve all chunks belonging to a specific page of a document.

        Handles both Vector Store UUIDs and SQL Database IDs (integers).
        If an integer ID is passed, it looks up the filename in the DB
        and filters chunks by that filename.
        """
        import logging
        logger = logging.getLogger(__name__)

        # --- Bridge: Resolve SQL ID to Filename if needed ---
        target_filename: str | None = None
        if document_id.isdigit():
            try:
                from private_gpt.users.db.session import SessionLocal
                from private_gpt.users.models.document import Document as SQLDocument
                with SessionLocal() as session:
                    sql_doc = session.query(SQLDocument).filter_by(id=int(document_id)).first()
                    if sql_doc:
                        target_filename = sql_doc.filename
                        logger.info("Resolved SQL ID %s to filename '%s'", document_id, target_filename)
            except Exception as e:
                logger.warning("Could not resolve SQL ID (using literal match instead): %s", str(e))

        matching_chunks: list[dict] = []
        try:
            docstore = self.storage_context.docstore
            if not docstore or not hasattr(docstore, "docs") or not docstore.docs:
                logger.warning("Docstore is empty or not available.")
                return []

            for node_id, node in docstore.docs.items():
                node_meta = node.metadata or {}
                # Match on document_id (UUID) OR filename (if we resolved one)
                node_doc_id = node_meta.get("document_id") or node_meta.get("doc_id", "")
                node_filename = node_meta.get("file_name")

                is_match = False
                if node_doc_id == document_id:
                    is_match = True
                elif target_filename and node_filename == target_filename:
                    is_match = True

                node_page = node_meta.get("page")

                if is_match and node_page == page_num:
                    chunk_info = {
                        "node_id": node.node_id,
                        "text": node.get_content(),
                        "page": page_num,
                        "document_id": node_doc_id, # Return the actual UUID from metadata
                        "text_locations": node_meta.get("text_locations", []),
                        "document_class": node_meta.get("document_class", "unknown"),
                        "entities": node_meta.get("entities", []),
                    }
                    matching_chunks.append(chunk_info)

            # Deduplicate: Remove overlapping/redundant chunks (e.g. Hierarchical children)
            matching_chunks = self._deduplicate_chunks(matching_chunks)

            # Sort by node_id for consistent ordering
            matching_chunks.sort(key=lambda c: c["node_id"])
            logger.info(
                "Found %d unique chunks for document_id=%s (filename='%s') page=%d",
                len(matching_chunks), document_id, target_filename or "N/A", page_num,
            )
        except Exception as e:
            logger.error("Failed to retrieve page chunks: %s", str(e))

        return matching_chunks

    def _deduplicate_chunks(self, chunks: list[dict]) -> list[dict]:
        """Filter out redundant chunks that are completely contained within others.
        
        This handles the Hierarchical parser behavior where a page might have 
        a parent chunk and 4 children chunks that contain the exact same text 
        as the parent.
        """
        if not chunks:
            return []
            
        # 1. Sort by text length (descending) so we prefer keeping larger "Parents"
        sorted_chunks = sorted(chunks, key=lambda x: len(x["text"]), reverse=True)
        
        unique_chunks = []
        for candidate in sorted_chunks:
            is_redundant = False
            candidate_text = candidate["text"].strip()
            
            for existing in unique_chunks:
                existing_text = existing["text"].strip()
                # If candidate text is already inside an existing (larger) chunk...
                if candidate_text in existing_text:
                    is_redundant = True
                    break
            
            if not is_redundant:
                unique_chunks.append(candidate)
        
        return unique_chunks

    async def update_chunk(self, node_id: str, new_text: str) -> dict:
        """Surgically update a single chunk's text, re-embed, and re-extract entities.

        Pipeline:
        1. Retrieve the existing node from docstore.
        2. Delete old entity links from Postgres.
        3. Delete old vectors from Vector Store.
        4. Update text, re-generate dense embedding, re-run NER.
        5. Re-insert into Vector Store and Docstore.
        6. Persist storage to disk.
        """
        import logging
        import asyncio
        from llama_index.core.schema import TextNode
        logger = logging.getLogger(__name__)

        # --- Step 1: Retrieve existing node ---
        try:
            node = self.storage_context.docstore.get_node(node_id)
        except Exception as e:
            raise ValueError(f"Node {node_id} not found in docstore: {e}")

        old_text = node.get_content()
        logger.info(
            "CHUNK_EDIT: Updating node %s. Old text length=%d, New text length=%d",
            node_id, len(old_text), len(new_text),
        )

        # --- Step 2: Delete old entity links from Postgres ---
        try:
            from private_gpt.users.db.session import SessionLocal
            from private_gpt.users.models.entity import NodeEntity

            with SessionLocal() as session:
                deleted_count = session.query(NodeEntity).filter_by(
                    node_id=node_id
                ).delete()
                session.commit()
                logger.info(
                    "CHUNK_EDIT: Deleted %d old entity links for node %s",
                    deleted_count, node_id,
                )
        except Exception as e:
            logger.warning(
                "CHUNK_EDIT: Could not delete old entity links (non-fatal): %s", str(e)
            )

        # --- Step 3: Delete old vector from Vector Store ---
        try:
            index = VectorStoreIndex.from_vector_store(
                self.vector_store_component.vector_store,
                storage_context=self.storage_context,
                embed_model=self.embedding_component.embedding_model,
            )
            # Delete the node from vector store by its ID
            self.vector_store_component.vector_store.delete(node_id)
            logger.info("CHUNK_EDIT: Deleted old vector for node %s", node_id)
        except Exception as e:
            logger.warning(
                "CHUNK_EDIT: Could not delete old vector (may not exist): %s", str(e)
            )

        # --- Step 4: Update text and re-embed ---
        if isinstance(node, TextNode):
            node.set_content(new_text)
        else:
            node.text = new_text

        # Generate new dense embedding
        new_embedding = await asyncio.to_thread(
            self.embedding_component.embedding_model.get_text_embedding, new_text
        )
        node.embedding = new_embedding
        logger.info(
            "CHUNK_EDIT: Generated new dense embedding (dim=%d) for node %s",
            len(new_embedding), node_id,
        )

        # --- Step 5: Re-run NER entity extraction ---
        try:
            if self.entity_component.enabled:
                entities = self.entity_component.extract_entities(new_text)
                node.metadata["entities"] = list(set([ent["text"] for ent in entities]))
                node.metadata["entity_details"] = entities
                logger.info(
                    "CHUNK_EDIT: Re-extracted %d entities for node %s",
                    len(entities), node_id,
                )

                # Save new entity links to Postgres
                from private_gpt.components.ingest.ingest_component import BaseIngestComponent
                BaseIngestComponent._save_entities(None, [node])
        except Exception as e:
            logger.warning(
                "CHUNK_EDIT: Entity re-extraction failed (non-fatal): %s", str(e)
            )

        # --- Step 6: Re-insert into Vector Store and Docstore ---
        try:
            # Update docstore (overwrite existing node)
            self.storage_context.docstore.add_documents([node], allow_update=True)
            logger.info("CHUNK_EDIT: Updated node in docstore.")

            # Re-insert into vector store via the index
            index = VectorStoreIndex.from_vector_store(
                self.vector_store_component.vector_store,
                storage_context=self.storage_context,
                embed_model=self.embedding_component.embedding_model,
            )
            index.insert_nodes([node])
            logger.info("CHUNK_EDIT: Re-inserted node into vector store.")
        except Exception as e:
            logger.error("CHUNK_EDIT: Failed to re-insert node: %s", str(e))
            raise

        # --- Step 7: Persist to disk ---
        try:
            from private_gpt.paths import local_data_path
            self.storage_context.persist(persist_dir=local_data_path)
            logger.info("CHUNK_EDIT: Persisted storage context to disk.")
        except Exception as e:
            logger.error("CHUNK_EDIT: Failed to persist storage: %s", str(e))
            raise

        return {
            "node_id": node_id,
            "text": new_text,
            "entities": node.metadata.get("entities", []),
            "status": "updated",
        }

