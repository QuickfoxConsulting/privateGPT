from typing import Any, Dict, List, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Security
from pydantic import BaseModel, Field
from private_gpt.users import models
from private_gpt.users.api import deps
from sqlalchemy.orm import Session

from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.server.chunks.chunks_service import Chunk, ChunksService
from private_gpt.server.utils.auth import authenticated

chunks_router = APIRouter(prefix="/v1", dependencies=[Depends(authenticated)])


# ── Existing Schemas ──────────────────────────────────────────────────────────

class ChunksBody(BaseModel):
    text: str = Field(examples=["Q3 2023 sales"])
    context_filter: ContextFilter | None = None
    limit: int = 10
    prev_next_chunks: int = Field(default=0, examples=[2])

class FormattedSource(BaseModel):
    file: str
    page: str
    text: str


class ChunksResponse(BaseModel):
    object: Literal["list"]
    model: Literal["private-gpt"]
    data: List[FormattedSource]


# ── New Schemas for Chunk Editor ──────────────────────────────────────────────

class PageChunkItem(BaseModel):
    """A single chunk belonging to a document page."""
    node_id: str = Field(examples=["c202d5e6-7b69-4869-81cc-dd574ee8ee11"])
    text: str = Field(examples=["Apple was founded in 1976."])
    page: int = Field(examples=[1])
    document_id: str = Field(examples=["doc-uuid-here"])
    text_locations: List[Dict[str, Any]] = Field(default_factory=list)
    document_class: str = Field(default="unknown", examples=["digital", "scanned", "hybrid"])
    entities: List[str] = Field(default_factory=list)


class PageChunksResponse(BaseModel):
    """Response for GET /chunks/document/{document_id}/page/{page_num}."""
    object: Literal["page.chunks"] = "page.chunks"
    document_id: str
    page: int
    total_chunks: int
    chunks: List[PageChunkItem]


class ChunkEditBody(BaseModel):
    """Payload for PUT /chunks/{node_id}."""
    text: str = Field(..., min_length=1, examples=["Apple was founded in 1976."])


class ChunkEditResponse(BaseModel):
    """Response for PUT /chunks/{node_id}."""
    object: Literal["chunk.updated"] = "chunk.updated"
    node_id: str
    text: str
    entities: List[str] = Field(default_factory=list)
    status: str = Field(default="updated")


# ── Existing Endpoint ─────────────────────────────────────────────────────────

@chunks_router.post("/chunks", tags=["Context Chunks"])
async def chunks_retrieval(
    request: Request, 
    body: ChunksBody,
    log_audit: models.Audit = Depends(deps.get_audit_logger),
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
    ),) -> ChunksResponse:
    """Given a `text`, returns the most relevant chunks from the ingested documents.

    The returned information can be used to generate prompts that can be
    passed to `/completions` or `/chat/completions` APIs. Note: it is usually a very
    fast API, because only the Embeddings model is involved, not the LLM. The
    returned information contains the relevant chunk `text` together with the source
    `document` it is coming from. It also contains a score that can be used to
    compare different results.

    The max number of chunks to be returned is set using the `limit` param.

    Previous and next chunks (pieces of text that appear right before or after in the
    document) can be fetched by using the `prev_next_chunks` field.

    The documents being used can be filtered using the `context_filter` and passing
    the document IDs to be used. Ingested documents IDs can be found using
    `/ingest/list` endpoint. If you want all ingested documents to be used,
    remove `context_filter` altogether.
    """
    service = request.state.injector.get(ChunksService)
    results = await service.retrieve_relevant(
        body.text, body.context_filter, body.limit, body.prev_next_chunks
    )
    
    # Create a list of dictionaries with formatted source data
    formatted_sources = []
    for result in results:
        metadata = result.document.doc_metadata or {}
        file_name = str(metadata.get("file_name") or metadata.get("filename") or "Unknown")
        page = str(metadata.get("page_label") or metadata.get("page") or "-")
        formatted_sources.append({
            "file": file_name,
            "page": page,
            "text": result.text,
        })

    log_audit(
        model='Chat', 
        action='Chat',
        details={
            "query": body.text,
            }, 
        user_id=current_user.id
    )
    return ChunksResponse(
        object="list",
        model="private-gpt",
        data=formatted_sources,
    )


# ── New Endpoints for Chunk Editor ────────────────────────────────────────────

@chunks_router.get(
    "/chunks/document/{document_id}/page/{page_num}",
    tags=["Chunk Editor"],
    response_model=PageChunksResponse,
)
async def get_page_chunks(
    request: Request,
    document_id: str,
    page_num: int,
    current_user: models.User = Security(deps.get_current_user),
) -> PageChunksResponse:
    """Retrieve all embedded chunks belonging to a specific page of a document.

    This is designed for the side-by-side chunk editor UI:
    the frontend shows the PDF page on the left and the editable chunks on the right.
    Each chunk includes its text, bounding box coordinates, and extracted entities.
    """
    service = request.state.injector.get(ChunksService)
    chunks = service.get_document_page_chunks(document_id, page_num)

    return PageChunksResponse(
        document_id=document_id,
        page=page_num,
        total_chunks=len(chunks),
        chunks=[PageChunkItem(**c) for c in chunks],
    )


@chunks_router.put(
    "/chunks/{node_id}",
    tags=["Chunk Editor"],
    response_model=ChunkEditResponse,
)
async def update_chunk(
    request: Request,
    node_id: str,
    body: ChunkEditBody,
    log_audit: models.Audit = Depends(deps.get_audit_logger),
    current_user: models.User = Security(deps.get_current_user),
) -> ChunkEditResponse:
    """Surgically edit a single chunk's text.

    This triggers a full re-processing pipeline:
    1. Delete old vector embeddings (dense + sparse).
    2. Delete old entity links from Postgres.
    3. Re-generate dense embedding from the new text.
    4. Re-run NER entity extraction on the new text.
    5. Re-insert into Vector Store, Docstore, and Entity DB.
    6. Persist all changes to disk.
    """
    service = request.state.injector.get(ChunksService)
    try:
        result = await service.update_chunk(node_id, body.text)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chunk update failed: {str(e)}")

    log_audit(
        model="ChunkEditor",
        action="EditChunk",
        details={"node_id": node_id, "new_text_length": len(body.text)},
        user_id=current_user.id,
    )

    return ChunkEditResponse(**result)
