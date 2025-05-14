import logging
import typing

from injector import inject, singleton
from llama_index.core.indices.vector_store import VectorIndexRetriever, VectorStoreIndex
from llama_index.core.vector_stores.types import (
    FilterCondition,
    MetadataFilter,
    MetadataFilters,
    VectorStore,
)

from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.paths import local_data_path
from private_gpt.settings.settings import Settings
from .hybrid_fn import sparse_query_vectors, sparse_doc_vectors, relative_score_fusion_with_threshold

logger = logging.getLogger(__name__)


def _doc_id_metadata_filter(
    context_filter: ContextFilter | None,
) -> MetadataFilters:
    filters = MetadataFilters(filters=[], condition=FilterCondition.OR)

    if context_filter is not None and context_filter.docs_ids is not None:
        for doc_id in context_filter.docs_ids:
            filters.filters.append(MetadataFilter(key="doc_id", value=doc_id))

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

                # if settings.qdrant is None:
                #     logger.info(
                #         "Qdrant config not found. Using default settings."
                #         "Trying to connect to Qdrant at localhost:6333."
                #     )
                client = QdrantClient(url="http://qdrant:6333")
                # aclient = AsyncQdrantClient(url="http://qdrant:6333")
                self.vector_store = typing.cast(
                    VectorStore,
                    QdrantVectorStore(
                        client=client,
                        # aclient=aclient,
                        collection_name="make_this_parameterizable_per_api_call",
                        enable_hybrid=True, 
                        fastembed_sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions",
                        # batch_size=20,
                        # sparse_doc_fn=sparse_doc_vectors,
                        # sparse_query_fn=sparse_query_vectors,
                        use_async=True,
                        # hybrid_fusion_fn=relative_score_fusion,
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
    ) -> VectorIndexRetriever:
        return VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k,
            doc_ids=context_filter.docs_ids if context_filter else None,
            filters=(
                _doc_id_metadata_filter(context_filter)
                if self.settings.vectorstore.database != "qdrant"
                else None
            ),
            sparse_top_k=12, 
            vector_store_query_mode="hybrid",
            alpha=0.5,
            # vector_store_kwargs={"hybrid_fusion_fn": relative_score_fusion_with_threshold}
        )

    def close(self) -> None:
        if hasattr(self.vector_store.client, "close"):
            self.vector_store.client.close()
