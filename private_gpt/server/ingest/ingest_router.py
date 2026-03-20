import json
import os
import shutil
import logging
import traceback

import aiofiles
from pathlib import Path
from typing import Any, List, Literal, Optional

from private_gpt.users.models.enums import DocumentStatus
from private_gpt.users.models.document import DocumentVersion, MakerCheckerActionType, MakerCheckerStatus
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, status, Security, Body, Form
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

from private_gpt.users import crud, models, schemas
from private_gpt.users.api import deps
from private_gpt.users.constants.role import Role

from private_gpt.server.ingest.ingest_service import IngestService, ChunkingStrategy
from private_gpt.server.ingest.model import IngestedDoc, RawIngestDoc, Chunk, IngestPatchInput
from private_gpt.server.utils.auth import authenticated
from private_gpt.constants import UPLOAD_DIR
from private_gpt.components.ocr_components.reconstruction.engine import ReconstructionEngine

ingest_router = APIRouter(prefix="/v1", dependencies=[Depends(authenticated)])

logger = logging.getLogger(__name__)

class IngestTextBody(BaseModel):
    file_name: str = Field(examples=["Avatar: The Last Airbender"])
    text: str = Field(
        examples=[
            "Avatar is set in an Asian and Arctic-inspired world in which some "
            "people can telekinetically manipulate one of the four elements—water, "
            "earth, fire or air—through practices known as 'bending', inspired by "
            "Chinese martial arts."
        ]
    )
    metadata: Optional[dict[str, Any]] = Field(
        None,
        examples=[
            {
                "title": "Avatar: The Last Airbender",
                "author": "Michael Dante DiMartino, Bryan Konietzko",
                "year": "2005",
                "tags": "#scifi,#avatar",
                "description": "Movie about ....",
            }
        ],
    )
    chunk_size: int = Field(
        512,
        description="The maximum size of each chunk in tokens",
        ge=100,
        le=2048
    )
    chunk_overlap: int = Field(
        100,
        description="The overlap size between consecutive chunks",
        ge=0,
        le=512
    )
    window_size: int = Field(
        3,
        description="The context window size for chunking strategies that support it",
        ge=0,
        le=10
    )
    strategy: ChunkingStrategy = Field(
        ChunkingStrategy.HIERARCHICAL,
        description="The chunking strategy to use"
    )


class IngestFileBody(BaseModel):
    metadata: Optional[dict[str, Any]] = Field(
        None,
        examples=[
            {
                "title": "Avatar: The Last Airbender",
                "author": "Michael Dante DiMartino, Bryan Konietzko",
                "year": "2005",
                "tags": "#scifi,#avatar",
                "description": "Movie about ....",
            }
        ],
    )
    chunk_size: int = Field(
        512,
        description="The maximum size of each chunk in tokens",
        ge=100,
        le=2048
    )
    chunk_overlap: Optional[int] = Field(
        100,
        description="The overlap size between consecutive chunks",
        ge=0,
        le=512
    )
    window_size: Optional[int] = Field(
        3,
        description="The context window size for chunking strategies that support it",
        ge=0,
        le=10
    )
    strategy: ChunkingStrategy = Field(
        ChunkingStrategy.HIERARCHICAL,
        description="The chunking strategy to use"
    )


class IngestResponse(BaseModel):
    object: Literal["list"]
    model: Literal["private-gpt"]
    data: list[IngestedDoc]

class DeleteFilename(BaseModel):
    filename: str
    version_id: Optional[str] = None

@ingest_router.post("/ingest/file", tags=["Ingestion"])
async def ingest_file(
    request: Request, 
    file: UploadFile = File(...), 
    metadata: str = Form(None),
    chunk_size: int = Form(512),
    chunk_overlap: int = Form(100),
    window_size: int = Form(3),
    strategy: ChunkingStrategy = Form(ChunkingStrategy.HIERARCHICAL),
    db: Session = Depends(deps.get_db),
    log_audit: models.Audit = Depends(deps.get_audit_logger),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.ADMIN["name"], Role.SUPER_ADMIN["name"], Role.OPERATOR["name"]],
    )
) -> IngestResponse:
    """Ingests and processes a file with dynamic chunking parameters.

    The context obtained from files is later used in
    `/chat/completions`, `/completions`, and `/chunks` APIs.

    Most common document
    formats are supported, but you may be prompted to install an extra dependency to
    manage a specific file type.

    A file can generate different Documents (for example a PDF generates one Document
    per page). All Documents IDs are returned in the response, together with the
    extracted Metadata (which is later used to improve context retrieval). Those IDs
    can be used to filter the context used to create responses in
    `/chat/completions`, `/completions`, and `/chunks` APIs.
    """
    service = request.state.injector.get(IngestService)
    if file.filename is None:
        raise HTTPException(400, "No file name provided")
    
    # Save directly to the documents subdirectory so it can be served via /media
    upload_path = Path(UPLOAD_DIR) / "documents" / file.filename
    try:
        with open(upload_path, "wb") as f:
            f.write(file.file.read())
        
        # Log the ingestion attempt
        log_audit(
            model='Document',
            action='ingest_attempt',
            details={
                'filename': file.filename,
                'user': current_user.username,
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'window_size': window_size,
                'strategy': strategy.value,
                'file_size': upload_path.stat().st_size if upload_path.exists() else 0
            },
            user_id=current_user.id,
            username=current_user.username,
            severity="INFO"
        )
        
        # Use ingest_file directly since we have the file path
        metadata_dict = None if metadata is None else json.loads(metadata)
        ingested_documents = await service.ingest_file(
            file.filename, 
            upload_path,
            metadata_dict,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            window_size=window_size,
            strategy=strategy
        )
        
        # Log successful ingestion
        log_audit(
            model='Document',
            action='ingest_success',
            details={
                'filename': file.filename,
                'user': current_user.username,
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'window_size': window_size,
                'strategy': strategy.value,
                'document_count': len(ingested_documents),
                'doc_ids': [doc.doc_id for doc in ingested_documents]
            },
            user_id=current_user.id,
            username=current_user.username,
            resource_id=file.filename,
            severity="INFO"
        )
    except Exception as e:
        logger.error(f"Error ingesting file {file.filename}: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Log the ingestion failure
        log_audit(
            model='Document',
            action='ingest_failure',
            details={
                'filename': file.filename,
                'user': current_user.username,
                'error': str(e),
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'window_size': window_size,
                'strategy': strategy.value
            },
            user_id=current_user.id,
            username=current_user.username,
            resource_id=file.filename,
            severity="ERROR"
        )
        
        raise HTTPException(status_code=500, detail=f"There was an error uploading the file(s): {e}")
    finally:
        # We no longer unlink the file here so it can be served via /media
        file.file.close()
    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)

@ingest_router.post("/ingest/text", tags=["Ingestion"])
async def ingest_text(request: Request, body: IngestTextBody) -> IngestResponse:
    """Ingests and processes a text with dynamic chunking parameters.

    The context obtained from files is later used in
    `/chat/completions`, `/completions`, and `/chunks` APIs.

    A Document will be generated with the given text. The Document
    ID is returned in the response, together with the
    extracted Metadata (which is later used to improve context retrieval). That ID
    can be used to filter the context used to create responses in
    `/chat/completions`, `/completions`, and `/chunks` APIs.
    """
    service = request.state.injector.get(IngestService)
    if len(body.file_name) == 0:
        raise HTTPException(400, "No file name provided")
    ingested_documents = await service.ingest_text(
        body.file_name, 
        body.text, 
        body.metadata,
        chunk_size=body.chunk_size,
        chunk_overlap=body.chunk_overlap,
        window_size=body.window_size,
        strategy=body.strategy
    )
    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)

@ingest_router.get("/ingest/list", tags=["Ingestion"])
def list_ingested(request: Request) -> IngestResponse:
    """Lists already ingested Documents including their Document ID and metadata.

    Those IDs can be used to filter the context used to create responses
    in `/chat/completions`, `/completions`, and `/chunks` APIs.
    """
    service = request.state.injector.get(IngestService)
    ingested_documents = service.list_ingested()
    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)


@ingest_router.get("/ingest/{filename}/enhanced", tags=["Ingestion"])
async def download_enhanced_document(request: Request, filename: str):
    """
    Download the enhanced (searchable) version of a document.
    Generated on-the-fly to ensure the latest edited data is used.
    """
    service = request.state.injector.get(IngestService)
    recon_engine = request.state.injector.get(ReconstructionEngine)
    
    # 1. Get the latest spatial metadata from the docstore
    text_locations = service.get_all_text_locations_by_filename(filename)
    
    if not text_locations:
        logger.error(f"No text locations found for {filename}")
        raise HTTPException(status_code=404, detail=f"No enrichment data found for {filename}")

    # 2. Identify the original PDF path
    # Original file is typically in the documents subdirectory
    original_path = Path(UPLOAD_DIR) / "documents" / filename
    if not original_path.exists():
        logger.error(f"Original file not found for {filename} at {original_path}")
        raise HTTPException(status_code=404, detail=f"Original document not found for {filename}")

    # 3. Trigger Reconstruction on-the-fly
    # This will overwrite the existing _enhanced.pdf if it exists, or create it fresh
    stem = original_path.stem
    try:
        recon_result = recon_engine.reconstruct(
            original_images=[], # Empty because we are using original_path (non-destructive)
            text_locations=text_locations,
            output_name=stem,
            original_path=original_path
        )
        
        if not recon_result.success:
            logger.error(f"Reconstruction failed for {filename}: {recon_result.error}")
            raise HTTPException(status_code=500, detail=f"Failed to generate enhanced PDF: {recon_result.error}")
            
        enhanced_path = recon_result.output_path
        return FileResponse(
            path=enhanced_path,
            media_type="application/pdf",
            filename=f"{stem}_enhanced.pdf"
        )
    except Exception as e:
        logger.error(f"Unexpected error during on-the-fly reconstruction for {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error during PDF generation: {str(e)}")



@ingest_router.get("/ingest/{db_id}/raw", tags=["Ingestion"])
def get_raw_ingested(
    request: Request, 
    db_id: int, 
    page: int = 1,
    db: Session = Depends(deps.get_db)
) -> RawIngestDoc:
    """Get the raw content of an ingested page by Database ID and Page Number."""
    service = request.state.injector.get(IngestService)
    
    # 1. Get filename from DB
    db_doc = crud.documents.get(db, id=db_id)
    if not db_doc:
        raise HTTPException(status_code=404, detail=f"Database Document {db_id} not found")
        
    # 2. Get the raw page content from the ingestion service
    document = service.get_raw_page_by_filename_and_page(db_doc.filename, page)
    
    if not document:
        raise HTTPException(status_code=404, detail=f"Raw content for {db_doc.filename} page {page} not found")
        
    return RawIngestDoc.from_document(document)

@ingest_router.post("/ingest/raw/patch", tags=["Ingestion"])
async def patch_raw_ingested(
    request: Request,
    patch_input: IngestPatchInput,
    db: Session = Depends(deps.get_db)
) -> dict:
    """Update the raw content and metadata of an ingested page."""
    service = request.state.injector.get(IngestService)
    
    # 1. Get filename from DB
    db_doc = crud.documents.get(db, id=patch_input.db_id)
    if not db_doc:
        raise HTTPException(status_code=404, detail=f"Database Document {patch_input.db_id} not found")
        
    # 2. Patch the raw content in the docstore
    patch_result = await service.patch_raw_page(
        db_doc.filename, 
        patch_input.page, 
        patch_input.text, 
        patch_input.text_locations
    )
    
    if not patch_result["success"]:
        raise HTTPException(
            status_code=404, 
            detail=f"Raw content for {db_doc.filename} page {patch_input.page} not found in docstore"
        )
        
    return {
        "status": "SUCCESS", 
        "updated_count": patch_result["count"],
        "node_ids": patch_result["ids"],
        "message": f"Raw content updated successfully for {len(patch_result['ids'])} nodes."
    }

@ingest_router.post("/ingest/{db_id}/reindex", tags=["Ingestion"])
async def reindex_document(
    request: Request,
    db_id: int,
    db: Session = Depends(deps.get_db)
) -> dict:
    """Re-chunk and re-embed a document from its existing raw page data.
    
    Use this after editing raw data via /ingest/raw/patch to rebuild
    the chunks, embeddings, and entities without re-running OCR.
    """
    service = request.state.injector.get(IngestService)
    
    # 1. Get filename from DB
    db_doc = crud.documents.get(db, id=db_id)
    if not db_doc:
        raise HTTPException(status_code=404, detail=f"Database Document {db_id} not found")
    
    # 2. Re-index from raw data
    result = await service.reindex_from_raw(db_doc.filename)
    
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["message"])
    
    return {
        "status": "SUCCESS",
        "raw_pages": result["raw_pages"],
        "nodes_created": result["nodes_created"],
        "message": result["message"],
    }

@ingest_router.get("/chunks/document/{db_id}/page/{page_num}", tags=["Ingestion"])
def get_chunks_by_page(
    request: Request, 
    db_id: int, 
    page_num: int,
    db: Session = Depends(deps.get_db)
) -> list[Chunk]:
    """Get all chunks for a specific document and page."""
    service = request.state.injector.get(IngestService)
    db_doc = crud.documents.get(db, id=db_id)
    if not db_doc:
        raise HTTPException(status_code=404, detail=f"Database Document {db_id} not found")
    
    # We find all nodes for this file and page
    nodes = service.get_nodes_by_filename_and_page(db_doc.filename, page_num)
    
    # Wrap nodes in Chunk model
    return [Chunk.from_base_node(node) for node in nodes]

@ingest_router.get("/media/{filename}", tags=["Ingestion"])
async def get_media_file(filename: str):
    """Serve a file from the documents media directory."""
    # Security: Ensure we only serve from the documents subdirectory
    safe_filename = os.path.basename(filename)
    file_path = Path(UPLOAD_DIR) / "documents" / safe_filename
    
    if not file_path.exists():
        logger.error(f"Media file not found: {file_path}")
        raise HTTPException(status_code=404, detail=f"File not found: {safe_filename}")
        
    return FileResponse(
        path=file_path,
        filename=safe_filename
    )

@ingest_router.get("/ingest/{doc_id}", tags=["Ingestion"])
async def delete_ingested(request: Request, doc_id: str) -> None:
    """Delete the specified ingested Document.

    The `doc_id` can be obtained from the `GET /ingest/list` endpoint.
    The document will be effectively deleted from your storage context.
    """
    service = request.state.injector.get(IngestService)
    await service.delete(doc_id)

from pathlib import Path

@ingest_router.post("/ingest/file/delete", tags=["Ingestion"])
async def delete_file(
        request: Request,
        delete_input: DeleteFilename,
        log_audit: models.Audit = Depends(deps.get_audit_logger),
        db: Session = Depends(deps.get_db),
        current_user: models.User = Security(
            deps.get_current_user,
            scopes=[Role.ADMIN["name"], Role.SUPER_ADMIN["name"], Role.OPERATOR["name"]],
        )) -> dict:
    """Delete the specified filename and all related data."""
    filename = delete_input.filename    
    service = request.state.injector.get(IngestService)
    try:
        document = crud.documents.get_by_filename(db, file_name=filename)
        if document:
            document_versions = crud.document_versions.get_by_document_id(db, document_id=document.id)
            chunking_strategy = document.doc_metadata.get(
                'strategy',
                ChunkingStrategy.HIERARCHICAL.value
            )
            for version in document_versions:
                upload_path = version.file_path
                logger.info(f"Deleting file at: {upload_path}")
                filename = os.path.basename(upload_path)
                doc_ids = service.get_doc_ids_by_filename(filename)
                logger.info(f"Deleting doc_ids: {doc_ids} for with: {chunking_strategy}")
                if doc_ids:
                    # for doc_id in doc_ids:
                        # await service.delete(doc_id)
                    # delete everything at once
                    await service.delete_docs(doc_ids, chunking_strategy)
                else:
                    # Fallback: delete by filename directly in vector store if doc_ids not found
                    logger.info(f"No doc_ids found for {filename}, attempting direct metadata deletion")
                    await service.delete_by_metadata("file_name", filename, chunking_strategy)
                
                try:
                    upload_path = Path(upload_path)
                    if upload_path.exists():
                        os.remove(upload_path)
                    
                    # Also delete the enhanced (searchable) version if it exists
                    enhanced_filename = f"{upload_path.stem}_enhanced.pdf"
                    enhanced_path = upload_path.parent / enhanced_filename
                    if enhanced_path.exists():
                        os.remove(enhanced_path)
                        logger.info(f"Deleted enhanced file at: {enhanced_path}")
                        
                except Exception as e:
                    logger.error(f"Error deleting file from static directory: {e}")
                        
            db.execute(
                models.document_department_association.delete().where(
                    models.document_department_association.c.document_id == document.id
                )
            )            
            db.execute(
                models.document_category_association.delete().where(
                    models.document_category_association.c.document_id == document.id
                )
            )            
            crud.documents.remove(db=db, id=document.id)
            db.commit()
            log_audit(
                model='Document', 
                action='delete',
                details={
                    "detail": f"{filename}",
                    'user': current_user.username,
                }, 
                user_id=current_user.id
            )
            
        return {"status": "SUCCESS", "message": f"{filename} deleted successfully."}
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"Error deleting document '{filename}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Internal Server Error"
        )
    
async def create_documents(
    db: Session,
    file_name: str = None,
    current_user: models.User = None,
    documents: schemas.DocumentUpload = None,
    log_audit: models.Audit = None,
):
    """
    Create documents in the `Document` table and update the
    `Document Department Association` table with the department IDs for the documents.
    Using the new metadata JSONB field for storing tags, departments, and categories.
    """
    file_ingested = crud.documents.get_by_filename(db, file_name=file_name)
    if file_ingested:
        raise HTTPException(
            status_code=409,
            detail="File already exists. Choose a different file.",
        )
    
    logger.info(f"{file_name} uploaded by {current_user.id} action {MakerCheckerActionType.INSERT.value} and status {MakerCheckerStatus.PENDING.value}")
    
    # Handle optional metadata - provide defaults when doc_metadata is None
    if documents.doc_metadata is not None:
        metadata_dict = {
            "tags": getattr(documents.doc_metadata, "tags", []),
            "departments": getattr(documents.doc_metadata, "departments", []),
            "category": getattr(documents.doc_metadata, "category", None)
        }
        
        if hasattr(documents.doc_metadata, "custom_fields") and documents.doc_metadata.custom_fields:
            metadata_dict.update(documents.doc_metadata.custom_fields)
    else:
        # Default empty metadata when none provided
        metadata_dict = {
            "tags": [],
            "departments": [],
            "category": None
        }
    
    # Create document with doc_metadata
    docs_in = schemas.DocumentMakerCreate(
        filename=file_name,
        uploaded_by=current_user.id,
        doc_metadata=metadata_dict,
        doc_status=DocumentStatus.INGESTING.value
    )
    
    document = crud.documents.create(db=db, obj_in=docs_in)
    
    # Version 1 is automatically created by crud.documents.create
    
    # Extract department IDs from metadata for backward compatibility
    department_ids = []
    if metadata_dict.get("departments"):
        departments_data = metadata_dict["departments"]
        
        if isinstance(departments_data, str):
            department_names_or_ids = [d.strip() for d in departments_data.split(",") if d.strip()]
        elif isinstance(departments_data, list):
            department_names_or_ids = departments_data
        else:
            department_names_or_ids = []
        
        for dept in department_names_or_ids:
            if isinstance(dept, int) or (isinstance(dept, str) and dept.isdigit()):
                # If it's already an ID, add it directly
                department_ids.append(int(dept))
            elif isinstance(dept, str):
                department = db.query(models.Department).filter(models.Department.name == dept).first()
                if department:
                    department_ids.append(department.id)
                else:
                    logger.warning(f"Department name '{dept}' not found in database")
    
    if not department_ids:
        department_ids = [1]  
    
    # Associate departments (maintain backward compatibility)
    for department_id in department_ids:
        db.execute(
            models.document_department_association.insert().values(
                document_id=document.id,
                department_id=department_id
            )
        )
    
    # Associate category for backward compatibility
    category = metadata_dict.get("category")
    if category:
        category_id = category if isinstance(category, int) else db.query(models.Category).filter(models.Category.name == category).first()
        if isinstance(category_id, models.Category):
            category_id = category_id.id
        
        db.execute(
            models.document_category_association.insert().values(
                document_id=document.id,
                category_id=category_id
            )
        )    
    log_audit(
        model='Document',
        action='create',
        details={
            'filename': file_name,
            'user': current_user.username,
            'metadata': metadata_dict,
        },
        user_id=current_user.id
    )
    return document

async def create_url_documents(
    db: Session, 
    file_name: str = None, 
    category: int = None,
    current_user: models.User = None,
    departments: str = None,
    log_audit: models.Audit = None,
):
    """
    Create documents in the `Document` table and update the
    `Document Department Association` table with the department IDs for the documents.
    """
    department_ids = departments
    file_ingested = crud.documents.get_by_filename(db, file_name=file_name)
    if file_ingested:
        raise HTTPException(
            status_code=409,
            detail="File already exists. Choose a different file.",
        )

    print(f"{file_name} uploaded by {current_user.id} action {MakerCheckerActionType.INSERT.value} and status {MakerCheckerStatus.PENDING.value}")

    docs_in = schemas.UrlMakerChecker(
        filename=file_name, 
        uploaded_by=current_user.id, 
        action_type=MakerCheckerActionType.INSERT,
        status=MakerCheckerStatus.PENDING,
    )
    
    document = crud.documents.create(db=db, obj_in=docs_in)
    department_ids = department_ids if department_ids else "1"
    department_ids = [int(number) for number in department_ids.split(",")]

    for department_id in department_ids:
        db.execute(
            models.document_department_association.insert().values(
                document_id=document.id, 
                department_id=department_id
            )
        )
    if category:  
        db.execute(
            models.document_category_association.insert().values(
                document_id=document.id, 
                category_id=category
            )
        )
    log_audit(
        model='Document', 
        action='create',
        details={
            'filename': f"{file_name}", 
            'user': f"{current_user.username}",
            'departments': f"{department_ids}",
            'categories': f"{category}",
        }, 
        user_id=current_user.id
    )
    return document

from langchain_community.document_loaders import WebBaseLoader
from llama_index.core.schema import Document
async def ingest_url(
    request: Request, 
    url: str,
    chunk_size: int = 512,
    chunk_overlap: int = 100,
    window_size: int = 3,
    strategy: ChunkingStrategy = ChunkingStrategy.HIERARCHICAL
) -> IngestResponse:
    """Ingests and processes a file, storing its chunks to be used as context."""
    service = request.state.injector.get(IngestService)
    try:
        # documents = SimpleWebPageReader(html_to_text=True).load_data(
        #     [url]
        # )
        loader = WebBaseLoader(url)
        langchain_docs = loader.load()        
        llamaindex_docs: List[Document] = [
            Document.from_langchain_format(doc) for doc in langchain_docs
        ]        
        ingested_documents = await service.ingest_url(
            url, 
            llamaindex_docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            window_size=window_size,
            strategy=strategy
        )
    except Exception as e:
        print(traceback.print_exc())
        return {"message": f"There was an error uploading the file(s)\n {e}"}
    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)


# Actual Ingestion Method
async def ingest(
    request: Request, 
    file_path: str, 
    tags: Optional[dict[str, Any]] = None,
    strategy: ChunkingStrategy = ChunkingStrategy.HIERARCHICAL
) -> IngestResponse:
    """Ingests and processes a file, storing its chunks to be used as context."""
    service = request.state.injector.get(IngestService)
    try:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
            
        file_name = file_path_obj.name
        ingested_documents = await service.ingest_file(
            file_name, 
            file_path_obj,
            tags,
            strategy=strategy
        )
    except Exception as e:
        logger.error(f"Error ingesting file {file_path}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"There was an error uploading the file(s): {e}")
    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)


# @ingest_router.post("/ingest/file", response_model=IngestResponse, tags=["Ingestion"])
# async def ingest_file(
#         request: Request,
#         departments: schemas.DocumentUpload = Depends(),
#         file: UploadFile = File(...),
#         log_audit: models.Audit = Depends(deps.get_audit_logger),
#         db: Session = Depends(deps.get_db),
#         current_user: models.User = Security(
#             deps.get_current_user,
#             scopes=[Role.ADMIN["name"], Role.SUPER_ADMIN["name"], Role.OPERATOR["name"]],
# )) -> IngestResponse:
#     """Ingests and processes a file, storing its chunks to be used as context."""
#     service = request.state.injector.get(IngestService)
#     try:
#         original_filename = file.filename
#         print("Original file name is:", original_filename)
#         if original_filename is None:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="No file name provided",
#             )
#         upload_path = Path(f"{UPLOAD_DIR}/{original_filename}")
#         try:
#             contents = await file.read()
#             async with aiofiles.open(upload_path, 'wb') as f:
#                 await f.write(contents)
#         except Exception as e:
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail="Internal Server Error: Unable to ingest file.",
#             )

#         await create_documents(db, original_filename, current_user, departments, log_audit)
#         with open(upload_path, "rb") as f:
#             ingested_documents = service.ingest_bin_data(original_filename, f)

#         logger.info(f"{original_filename} is uploaded by {current_user.username} in {departments.departments_ids}")
#         response = IngestResponse(
#             object="list", model="private-gpt", data=ingested_documents
#         )
#         return response

#     except HTTPException:
#         print(traceback.print_exc())
#         raise

#     except Exception as e:
#         print(traceback.print_exc())
#         logger.error(f"There was an error uploading the file(s): {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Internal Server Error: Unable to ingest file.",
#         )
