import abc
import itertools
import logging
import multiprocessing
import multiprocessing.pool
import os
import threading
from pathlib import Path
from queue import Queue
from typing import Any, Sequence

from llama_index.core.data_structs import IndexDict
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.indices import VectorStoreIndex, load_index_from_storage
from llama_index.core.indices.base import BaseIndex
from llama_index.core.ingestion import run_transformations
from llama_index.core.schema import BaseNode, Document, TransformComponent
from llama_index.core.storage import StorageContext

from private_gpt.components.ingest.ingest_helper import IngestionHelper
from private_gpt.paths import local_data_path
from private_gpt.settings.settings import Settings
from private_gpt.utils.eta import eta

class CustomVectorStoreIndex(VectorStoreIndex):
    """
    A subclass of VectorStoreIndex that accommodates asynchronous document ingestion
    during the refresh or update processes of the index. This class enhances the functionality
    of the base class by providing asynchronous operations for inserting and refreshing documents
    within the vector store.

    Methods
    -------
    ainsert(documents, doc_hashes, insert_kwargs)
        Asynchronously inserts a list of documents into the vector store.

    arefresh_ref_docs(documents, **update_kwargs)
        Asynchronously refreshes documents in the index that may have been updated or changed.
    """

    async def ainsert(
        self,
        documents: list[Document],
        doc_hashes: dict[str, str],
        insert_kwargs: Any,
    ) -> None:
        """
        Asynchronously insert documents into the vector store index.

        Parameters
        ----------
        documents : list[Document]
            A list of documents to be inserted into the vector store.
        doc_hashes : dict[str, str]
            A dictionary mapping document IDs to their corresponding hashes.
        insert_kwargs : Any
            Additional keyword arguments to be used during the insertion of nodes.

        Returns
        -------
        None
        """
        with self._callback_manager.as_trace("insert"):
            nodes = await arun_transformations(
                documents,
                transformations=self._transformations,
            )

            for node in nodes:
                if isinstance(node, IndexNode):
                    try:
                        node.dict()
                    except ValueError:
                        self._object_map[node.index_id] = node.obj
                        node.obj = None
            with self._callback_manager.as_trace("insert_nodes"):
                await self._async_add_nodes_to_index(
                    nodes=nodes,
                    index_struct=self.index_struct_cls(),
                    show_progress=True,
                    **insert_kwargs,
                )
                await self.docstore.aset_document_hashes(doc_hashes)
        return None

    async def arefresh_ref_docs(
        self, documents: Sequence[Document], **update_kwargs: Any
    ) -> list[bool]:
        """
        Asynchronously refreshes an index with documents that have changed, ensuring
        the index is up-to-date with the latest document versions.

        Parameters
        ----------
        documents : Sequence[Document]
            A sequence of documents to be refreshed in the index. Each document is checked against
            the existing document in the store to determine if it needs updating.
        **update_kwargs : Any
            Arbitrary keyword arguments. These could include parameters for delete and insert operations
            such as `delete_kwargs` and `insert_kwargs` which are dictionaries of additional parameters.

        Returns
        -------
        list[bool]
            A list of boolean values indicating whether each document was refreshed (True) or not (False).
        """

        with self._callback_manager.as_trace("refresh"):
            refreshed_documents = [False] * len(documents)
            docs_to_insert: list[Document] = []
            doc_hashes: dict[str, str] = {}
            for i, document in enumerate(documents):
                doc_id = document.get_doc_id()
                existing_doc_hash = self._docstore.get_document_hash(doc_id)
                doc_hash = document.hash

                if existing_doc_hash is None:
                    docs_to_insert.append(document)
                    doc_hashes[doc_id] = doc_hash
                    refreshed_documents[i] = True
                elif existing_doc_hash != document.hash:
                    docs_to_insert.append(document)
                    doc_hashes[doc_id] = doc_hash
                    self.delete_ref_doc(
                        doc_id,
                        delete_from_docstore=True,
                        **update_kwargs.pop("delete_kwargs", {}),
                    )
                    refreshed_documents[i] = True

            await self.ainsert(
                documents=docs_to_insert,
                doc_hashes=doc_hashes,
                insert_kwargs=update_kwargs.pop("insert_kwargs", {}),
            )

            return refreshed_documents