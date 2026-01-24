"""Document ID management utilities for the ingest system."""

import uuid
import logging
from typing import Optional
from llama_index.core.schema import Document, TextNode

logger = logging.getLogger(__name__)


def ensure_doc_id(document: Document) -> str:
    """
    Centralized doc_id generation and validation.
    
    Ensures document has a valid doc_id and synchronizes it across
    all relevant fields (doc_id, id_, metadata).
    
    Args:
        document: Document to ensure has a valid doc_id
        
    Returns:
        The document's doc_id (existing or newly generated)
    """
    # Check if document already has a valid doc_id
    if hasattr(document, 'doc_id') and document.doc_id:
        doc_id = document.doc_id
    else:
        # Generate new UUID4
        doc_id = str(uuid.uuid4())
        document.doc_id = doc_id
        logger.debug(f"Generated new doc_id: {doc_id}")
    
    # Sync with id_ field
    document.id_ = doc_id
    
    # Ensure metadata exists and is synchronized
    if not document.metadata:
        document.metadata = {}
    
    document.metadata["doc_id"] = doc_id
    document.metadata["document_id"] = doc_id
    
    return doc_id


def ensure_ref_doc_id(node: TextNode, fallback_doc_id: Optional[str] = None) -> str:
    """
    Ensure a TextNode has a valid ref_doc_id.
    
    Args:
        node: TextNode to ensure has a valid ref_doc_id
        fallback_doc_id: Optional doc_id to use if node doesn't have one
        
    Returns:
        The node's ref_doc_id (existing or newly generated)
    """
    if node.ref_doc_id:
        return node.ref_doc_id
    
    # Try to get from metadata
    if node.metadata and "doc_id" in node.metadata:
        node.ref_doc_id = node.metadata["doc_id"]
        return node.ref_doc_id
    
    # Try to get from source_doc_id attribute
    if hasattr(node, 'source_doc_id') and node.source_doc_id:
        node.ref_doc_id = node.source_doc_id
        return node.ref_doc_id
    
    # Use fallback if provided
    if fallback_doc_id:
        node.ref_doc_id = fallback_doc_id
        return fallback_doc_id
    
    # Generate new UUID4 as last resort
    ref_doc_id = str(uuid.uuid4())
    node.ref_doc_id = ref_doc_id
    logger.warning(f"Generated new ref_doc_id for orphaned node: {ref_doc_id}")
    
    return ref_doc_id


def validate_doc_id_uniqueness(documents: list[Document]) -> bool:
    """
    Validate that all documents have unique doc_ids.
    
    Args:
        documents: List of documents to validate
        
    Returns:
        True if all doc_ids are unique, False otherwise
    """
    doc_ids = set()
    duplicates = []
    
    for doc in documents:
        doc_id = getattr(doc, 'doc_id', None) or getattr(doc, 'id_', None)
        if doc_id:
            if doc_id in doc_ids:
                duplicates.append(doc_id)
            doc_ids.add(doc_id)
    
    if duplicates:
        logger.error(f"Found duplicate doc_ids: {duplicates}")
        return False
    
    return True
