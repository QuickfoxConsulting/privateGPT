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

