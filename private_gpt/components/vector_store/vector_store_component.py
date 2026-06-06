import logging
import typing

from injector import inject, singleton
from llama_index.core.indices.vector_store import VectorIndexRetriever, VectorStoreIndex
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStore,
    BasePydanticVectorStore
)

from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.paths import local_data_path
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


def _doc_id_metadata_filter(
    context_filter: ContextFilter | None,
) -> MetadataFilters:
    filters = MetadataFilters(filters=[], condition=FilterCondition.OR)

    if context_filter is not None and context_filter.docs_ids is not None:
        for doc_id in context_filter.docs_ids:
            # Use both doc_id and document_id for backward compatibility
            filters.filters.append(MetadataFilter(key="doc_id", value=doc_id))
            filters.filters.append(MetadataFilter(key="document_id", value=doc_id))

    return filters


@singleton
class VectorStoreComponent:
    settings: Settings
    vector_store: VectorStore

    @inject
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        match settings.vectorstore.database:
            case "postgres":
                try:
                    from llama_index.vector_stores.postgres import (  # type: ignore
                        PGVectorStore,
                    )
                except ImportError as e:
                    raise ImportError(
                        "Postgres dependencies not found, install with `poetry install --extras vector-stores-postgres`"
                    ) from e

                if settings.postgres is None:
                    raise ValueError(
                        "Postgres settings not found. Please provide settings."
                    )

                self.vector_store = typing.cast(
                    VectorStore,
                    PGVectorStore.from_params(
                        **settings.postgres.model_dump(exclude_none=True),
                        table_name="embeddings",
                        embed_dim=settings.embedding.embed_dim,
                        hnsw_kwargs={
                            "hnsw_m": 16,
                            "hnsw_ef_construction": 64,
                            "hnsw_ef_search": 40,
                            "hnsw_dist_method": "vector_cosine_ops",
                        },
                        hybrid_search=True,
                    ),
                )

            case "chroma":
                try:
                    import chromadb  # type: ignore
                    from chromadb.config import (  # type: ignore
                        Settings as ChromaSettings,
                    )

                    from private_gpt.components.vector_store.batched_chroma import (
                        BatchedChromaVectorStore,
                    )
                except ImportError as e:
                    raise ImportError(
                        "ChromaDB dependencies not found, install with `poetry install --extras vector-stores-chroma`"
                    ) from e

                chroma_settings = ChromaSettings(anonymized_telemetry=False)
                chroma_client = chromadb.PersistentClient(
                    path=str((local_data_path / "chroma_db").absolute()),
                    settings=chroma_settings,
                )
                chroma_collection = chroma_client.get_or_create_collection(
                    "make_this_parameterizable_per_api_call"
                )  # TODO

                self.vector_store = typing.cast(
                    VectorStore,
                    BatchedChromaVectorStore(
                        chroma_client=chroma_client, chroma_collection=chroma_collection
                    ),
                )

            case "qdrant":
                try:
                    from llama_index.vector_stores.qdrant import (  # type: ignore
                        QdrantVectorStore,
                    )
                    from qdrant_client import QdrantClient, AsyncQdrantClient  # type: ignore
                except ImportError as e:
                    raise ImportError(
                        "Qdrant dependencies not found, install with `poetry install --extras vector-stores-qdrant`"
                    ) from e

                # Qdrant connection configuration with performance optimizations
                qdrant_url = "http://qdrant:6333"
                qdrant_grpc_port = 6334
                qdrant_timeout = 60  # Timeout for operations in seconds
                
                # Connection pool settings
                # Qdrant uses HTTP/gRPC with built-in connection pooling
                # grpc_options can control connection behavior
                grpc_options = {
                    "grpc.max_send_message_length": 100 * 1024 * 1024,  # 100MB
                    "grpc.max_receive_message_length": 100 * 1024 * 1024,  # 100MB
                    "grpc.keepalive_time_ms": 30000,  # Send keepalive ping every 30s
                    "grpc.keepalive_timeout_ms": 10000,  # Wait 10s for keepalive response
                    "grpc.http2.max_pings_without_data": 0,  # No limit on pings
                    "grpc.keepalive_permit_without_calls": 1,  # Allow keepalive without active calls
                }
                
                client = QdrantClient(
                    url=qdrant_url,
                    grpc_port=qdrant_grpc_port,
                    prefer_grpc=True,  # Use gRPC for better performance
                    timeout=qdrant_timeout,
                    grpc_options=grpc_options,
                )
                
                aclient = AsyncQdrantClient(
                    url=qdrant_url,
                    grpc_port=qdrant_grpc_port,
                    prefer_grpc=True,
                    timeout=qdrant_timeout,
                    grpc_options=grpc_options,
                )
                
                self.vector_store = typing.cast(
                    VectorStore,
                    QdrantVectorStore(
                        client=client,
                        aclient=aclient,
                        collection_name="make_this_parameterizable_per_api_call",
                        vector_name="text-dense",
                        sparse_vector_name="text-sparse-new",
                        enable_hybrid=True, 
                        fastembed_sparse_model="prithivida/Splade_PP_en_v1",
                        fastembed_cache_dir="/app/models/cache/fastembed",
                        use_async=True,
                    ),  # TODO
                )
            case "milvus":
                    try:
                        from llama_index.vector_stores.milvus import (  # type: ignore
                            MilvusVectorStore,
                        )
                    except ImportError as e:
                        raise ImportError(
                            "Milvus dependencies not found, install with `poetry install --extras vector-stores-milvus`"
                        ) from e

                    if settings.milvus is None:
                        logger.info(
                            "Milvus config not found. Using default settings.\n"
                            "Trying to connect to Milvus at local_data/private_gpt/milvus/milvus_local.db "
                            "with collection 'make_this_parameterizable_per_api_call'."
                        )

                        self.vector_store = typing.cast(
                            BasePydanticVectorStore,
                            MilvusVectorStore(
                                dim=settings.embedding.embed_dim,
                                collection_name="make_this_parameterizable_per_api_call",
                                overwrite=True,
                                enable_hybrid=True,
                                enable_sparse=True,
                            ),
                        )

                    else:
                        self.vector_store = typing.cast(
                            BasePydanticVectorStore,
                            MilvusVectorStore(
                                dim=settings.embedding.embed_dim,
                                uri=settings.milvus.uri,
                                token=settings.milvus.token,
                                collection_name=settings.milvus.collection_name,
                                overwrite=settings.milvus.overwrite,
                                enable_hybrid=settings.milvus.enable_hybrid,
                                enable_sparse=settings.milvus.enable_sparse,
                            ),
                        )
            case "clickhouse":
                try:
                    from clickhouse_connect import (  # type: ignore
                        get_client,
                    )
                    from llama_index.vector_stores.clickhouse import (  # type: ignore
                        ClickHouseVectorStore,
                    )
                except ImportError as e:
                    raise ImportError(
                        "ClickHouse dependencies not found, install with `poetry install --extras vector-stores-clickhouse`"
                    ) from e

                if settings.clickhouse is None:
                    raise ValueError(
                        "ClickHouse settings not found. Please provide settings."
                    )

                clickhouse_client = get_client(
                    host=settings.clickhouse.host,
                    port=settings.clickhouse.port,
                    username=settings.clickhouse.username,
                    password=settings.clickhouse.password,
                )
                self.vector_store = ClickHouseVectorStore(
                    clickhouse_client=clickhouse_client
                )
            case _:
                # Should be unreachable
                # The settings validator should have caught this
                raise ValueError(
                    f"Vectorstore database {settings.vectorstore.database} not supported"
                )

    def get_retriever(
        self,
        index: VectorStoreIndex,
        context_filter: ContextFilter | None = None,
        similarity_top_k: int = 2,
    ) -> VectorIndexRetriever | AutoMergingRetriever:
        """Get retriever, automatically using AutoMergingRetriever for hierarchical chunks."""
        
        # Check if index contains hierarchical nodes by looking for parent relationships
        has_hierarchical_nodes = False
        try:
            # Sample a few nodes to check for hierarchical structure
            docstore = index.storage_context.docstore
            if docstore and hasattr(docstore, 'docs') and docstore.docs:
                sample_nodes = list(docstore.docs.values())[:5]
                for node in sample_nodes:
                    # Hierarchical nodes have parent_node relationship
                    if hasattr(node, 'relationships') and node.relationships:
                        from llama_index.core.schema import NodeRelationship
                        if NodeRelationship.PARENT in node.relationships:
                            has_hierarchical_nodes = True
                            break
        except Exception as e:
            logger.debug(f"Could not detect hierarchical nodes: {e}")
        
        # Use AutoMergingRetriever for hierarchical chunks
        if has_hierarchical_nodes:
            logger.info("Detected hierarchical chunks, using AutoMergingRetriever")
            return self.get_auto_merging_retriever(
                index=index,
                context_filter=context_filter,
                similarity_top_k=similarity_top_k * 2,  # Retrieve more leaf chunks
                simple_ratio_thresh=0.4,  # Lowered from 0.5 to prefer broader context
            )
        
        # Standard retriever for non-hierarchical chunks
        return VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k,
            doc_ids=context_filter.docs_ids if context_filter else None,
            filters=(
                _doc_id_metadata_filter(context_filter)
                if self.settings.vectorstore.database != "qdrant"
                else None
            ),
            sparse_top_k=20,              # Increased from 12 for better BM25 recall
            vector_store_query_mode="hybrid",
            alpha=0.7,                     # Increased from 0.5 (70% vector, 30% BM25 - better for semantic search)
        )
    
    def file_vector_retriever(
        self, 
        index: VectorStoreIndex, 
        file_name: str,
        similarity_top_k: int = 2,
    ) -> VectorIndexRetriever | AutoMergingRetriever:
        """Get file-specific retriever, automatically using AutoMergingRetriever for hierarchical chunks."""
        
        # Check for hierarchical nodes (same logic as get_retriever)
        has_hierarchical_nodes = False
        try:
            docstore = index.storage_context.docstore
            if docstore and hasattr(docstore, 'docs') and docstore.docs:
                sample_nodes = list(docstore.docs.values())[:5]
                for node in sample_nodes:
                    if hasattr(node, 'relationships') and node.relationships:
                        from llama_index.core.schema import NodeRelationship
                        if NodeRelationship.PARENT in node.relationships:
                            has_hierarchical_nodes = True
                            break
        except Exception as e:
            logger.debug(f"Could not detect hierarchical nodes: {e}")
        
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="file_name", value=f"{file_name}", operator=FilterOperator.EQ),
                MetadataFilter(key="document_id", value=f"{file_name}", operator=FilterOperator.EQ),
            ],
            condition=FilterCondition.OR
        )
        
        if has_hierarchical_nodes:
            logger.info("Detected hierarchical chunks for file retrieval, using AutoMergingRetriever")
            base_retriever = VectorIndexRetriever(
                index=index,
                filters=filters,
                similarity_top_k=similarity_top_k * 2,
                sparse_top_k=20,
                vector_store_query_mode="hybrid",
                alpha=0.7,
            )
            return AutoMergingRetriever(
                base_retriever,
                storage_context=index.storage_context,
                simple_ratio_thresh=0.5,
                verbose=True,
            )
        
        return VectorIndexRetriever(
            index=index,
            filters=filters,
            similarity_top_k=similarity_top_k,
            sparse_top_k=20,              # Increased from 12
            vector_store_query_mode="hybrid",
            alpha=0.7,                     # Increased from 0.5
        )
    
    def get_auto_merging_retriever(
        self,
        index: VectorStoreIndex,
        context_filter: ContextFilter | None = None,
        similarity_top_k: int = 12,
        simple_ratio_thresh: float = 0.4,
    ) -> AutoMergingRetriever:
        """Create AutoMergingRetriever for hierarchical chunk merging.
        
        This retriever intelligently merges child chunks into parent chunks
        when enough children from the same parent are retrieved.
        
        Args:
            index: Vector store index
            context_filter: Optional document filter
            similarity_top_k: Number of chunks to retrieve initially
            simple_ratio_thresh: Ratio of children needed to merge into parent (0.5 = 50%)
        """
        base_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k,
            doc_ids=context_filter.docs_ids if context_filter else None,
            filters=(
                _doc_id_metadata_filter(context_filter)
                if self.settings.vectorstore.database != "qdrant"
                else None
            ),
            sparse_top_k=20,
            vector_store_query_mode="hybrid",
            alpha=0.7,
        )
        
        return AutoMergingRetriever(
            base_retriever,
            storage_context=index.storage_context,
            simple_ratio_thresh=simple_ratio_thresh,
            verbose=True,
        )

    def close(self) -> None:
        if hasattr(self.vector_store.client, "close"):
            self.vector_store.client.close()
