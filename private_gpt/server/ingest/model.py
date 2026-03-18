from typing import Any, Literal
from llama_index.core.schema import Document, NodeWithScore
from pydantic import BaseModel, Field
from private_gpt.constants import UPLOAD_DIR
from pathlib import Path

class IngestedDoc(BaseModel):
    object: Literal["ingest.document"]
    doc_id: str = Field(examples=["c202d5e6-7b69-4869-81cc-dd574ee8ee11"])
    doc_metadata: dict[str, Any] | None = Field(
        default=None,
        examples=[
            {
                "page_label": "2",
                "file_name": "Sales Report Q3 2023.pdf",
            }
        ]
    )

    @staticmethod
    def curate_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Remove unwanted metadata keys."""
        for key in ["doc_id", "window", "original_text"]:
            metadata.pop(key, None)
        return metadata
    
    @staticmethod
    def from_document(document: Document) -> "IngestedDoc":
        # Extract doc_id from document, prioritizing document.doc_id, then metadata
        doc_id = None
        if hasattr(document, 'doc_id') and document.doc_id:
            doc_id = document.doc_id
        elif document.metadata and 'doc_id' in document.metadata and document.metadata['doc_id']:
            doc_id = document.metadata['doc_id']
        elif document.metadata and 'document_id' in document.metadata and document.metadata['document_id']:
            doc_id = document.metadata['document_id']
        
        # Generate a new doc_id if one doesn't exist
        if not doc_id:
            import uuid
            doc_id = str(uuid.uuid4())
        
        # Ensure the document has the doc_id set
        document.doc_id = doc_id
        if not document.metadata:
            document.metadata = {}
        document.metadata["doc_id"] = doc_id
        document.metadata["document_id"] = doc_id
        
        # Note: Document objects don't have ref_doc_id field - that's for TextNode objects
        # The ref_doc_id is handled by the node parser when creating nodes from documents
        
        return IngestedDoc(
            object="ingest.document",
            doc_id=doc_id,
            doc_metadata=IngestedDoc.curate_metadata(document.metadata),
        )


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

