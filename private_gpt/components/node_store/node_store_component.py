import logging

from injector import inject, singleton
from llama_index.core.storage.docstore import BaseDocumentStore, SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.index_store.types import BaseIndexStore

from private_gpt.paths import local_data_path
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


@singleton
class NodeStoreComponent:
    index_store: BaseIndexStore
    doc_store: BaseDocumentStore

    @inject
    def __init__(self, settings: Settings) -> None:
        match settings.nodestore.database:
            case "simple":
                try:
                    self.index_store = SimpleIndexStore.from_persist_dir(
                        persist_dir=str(local_data_path)
                    )
                except FileNotFoundError:
                    logger.debug("Local index store not found, creating a new one")
                    self.index_store = SimpleIndexStore()

                try:
                    self.doc_store = SimpleDocumentStore.from_persist_dir(
                        persist_dir=str(local_data_path)
                    )
                except FileNotFoundError:
                    logger.debug("Local document store not found, creating a new one")
                    self.doc_store = SimpleDocumentStore()

            case "postgres":
                try:
                    from llama_index.core.storage.docstore.postgres_docstore import (
                        PostgresDocumentStore,
                    )
                    from llama_index.core.storage.index_store.postgres_index_store import (
                        PostgresIndexStore,
                    )
                    from llama_index.core.storage.kvstore.postgres_kvstore import PostgresKVStore
                    from typing import Any
                except ImportError:
                    raise ImportError(
                        "Postgres dependencies not found, install with `poetry install --extras storage-nodestore-postgres`"
                    ) from None

                if settings.postgres is None:
                    raise ValueError("Postgres index/doc store settings not found.")

                # Define custom KVStore with connection pooling
                class PooledPostgresKVStore(PostgresKVStore):
                    def _connect(self) -> Any:
                        from sqlalchemy import create_engine
                        from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
                        from sqlalchemy.orm import sessionmaker

                        # Add pooling configuration
                        self._engine = create_engine(
                            self.connection_string,
                            echo=self.debug,
                            pool_size=20,     # Increased pool size
                            max_overflow=10    # Allow extra burst connections
                        )
                        self._session = sessionmaker(self._engine)

                        self._async_engine = create_async_engine(
                            self.async_connection_string,
                            pool_size=20,
                            max_overflow=10
                        )
                        self._async_session = sessionmaker(self._async_engine, class_=AsyncSession)

                # Initialize pools with custom KVStore
                # 1. Index Store
                index_kv_store = PooledPostgresKVStore.from_params(
                    **settings.postgres.model_dump(exclude_none=True),
                    table_name="indexstore"
                )
                self.index_store = PostgresIndexStore(index_kv_store)

                # 2. Document Store
                doc_kv_store = PooledPostgresKVStore.from_params(
                    **settings.postgres.model_dump(exclude_none=True),
                    table_name="docstore"
                )
                self.doc_store = PostgresDocumentStore(doc_kv_store)

            case _:
                # Should be unreachable
                # The settings validator should have caught this
                raise ValueError(
                    f"Database {settings.nodestore.database} not supported"
                )
