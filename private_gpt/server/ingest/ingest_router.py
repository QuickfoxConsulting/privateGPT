import os
import logging
import traceback

import aiofiles
from pathlib import Path
from typing import List, Literal, Optional

from private_gpt.users.models.document import MakerCheckerActionType, MakerCheckerStatus
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, status, Security, Body, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from private_gpt.users import crud, models, schemas
from private_gpt.users.api import deps
from private_gpt.users.constants.role import Role

from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.server.ingest.model import IngestedDoc
from private_gpt.server.utils.auth import authenticated
from private_gpt.constants import UPLOAD_DIR

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


class IngestResponse(BaseModel):
    object: Literal["list"]
    model: Literal["private-gpt"]
    data: list[IngestedDoc]

class DeleteFilename(BaseModel):
    filename: str

# @ingest_router.post("/ingest", tags=["Ingestion"], deprecated=True)
# def ingest(request: Request, file: UploadFile) -> IngestResponse:
#     """Ingests and processes a file.

#     Deprecated. Use ingest/file instead.
#     """
#     return ingest_file(request, file)


@ingest_router.post("/ingest/file1", tags=["Ingestion"])
def ingest_file(request: Request, file: UploadFile = File(...)) -> IngestResponse:
    """Ingests and processes a file, storing its chunks to be used as context.

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
    upload_path = Path(f"{UPLOAD_DIR}/{file.filename}")
    try:
        with open(upload_path, "wb") as f:
            f.write(file.file.read())
        with open(upload_path, "rb") as f:
            ingested_documents = service.ingest_bin_data(file.filename, f)
    except Exception as e:
        return {"message": f"There was an error uploading the file(s)\n {e}"}
    finally:
        file.file.close()
    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)
    

@ingest_router.post("/ingest/text", tags=["Ingestion"])
def ingest_text(request: Request, body: IngestTextBody) -> IngestResponse:
    """Ingests and processes a text, storing its chunks to be used as context.

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
    ingested_documents = service.ingest_text(body.file_name, body.text)
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


@ingest_router.delete("/ingest/{doc_id}", tags=["Ingestion"])
def delete_ingested(request: Request, doc_id: str) -> None:
    """Delete the specified ingested Document.

    The `doc_id` can be obtained from the `GET /ingest/list` endpoint.
    The document will be effectively deleted from your storage context.
    """
    service = request.state.injector.get(IngestService)
    service.delete(doc_id)


@ingest_router.post("/ingest/file/delete", tags=["Ingestion"])
def delete_file(
        request: Request,
        delete_input: DeleteFilename,
        log_audit: models.Audit = Depends(deps.get_audit_logger),
        db: Session = Depends(deps.get_db),
        current_user: models.User = Security(
            deps.get_current_user,
            scopes=[Role.ADMIN["name"], Role.SUPER_ADMIN["name"], Role.OPERATOR["name"]],

        )) -> dict:
    """Delete the specified filename.

    The `filename` can be obtained from the `GET /ingest/list` endpoint.
    The document will be effectively deleted from your storage context.
    """
    filename = delete_input.filename    
    service = request.state.injector.get(IngestService)
    try:
        document = crud.documents.get_by_filename(db,file_name=filename)
        if document:
            doc_ids = service.get_doc_ids_by_filename(filename)
            try:
                if doc_ids:
                    for doc_id in doc_ids:
                        service.delete(doc_id)
                    upload_path = Path(f"{UPLOAD_DIR}/{filename}")
                    os.remove(upload_path)
            except:
                print("Unable to delete file from the static directory")
            log_audit(
                model='Document', 
                action='delete',
                details={
                    "detail": f"{filename}",
                    'user': current_user.username,
                    }, 
                user_id=current_user.id
            )
            crud.documents.remove(db=db, id=document.id)
            db.execute(models.document_department_association.delete().where(
                            models.document_department_association.c.document_id == document.id
                        ))
        return {"status": "SUCCESS", "message": f"{filename}' deleted successfully."}
    except Exception as e:
        print(traceback.print_exc())
        logger.error(
            f"Unexpected error deleting documents with filename '{filename}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal Server Error")


async def create_documents(
    db: Session, 
    file_name: str = None, 
    category: int = None,
    current_user: models.User = None,
    departments: schemas.DocumentDepartmentList = Depends(),
    log_audit: models.Audit = None,
    version_type: str = "NEW",
    previous_document_id: Optional[int] = None,
):
    """
    Create documents in the `Document` table and update the
    `Document Department Association` table with the department IDs for the documents.
    """
    department_ids = departments.departments_ids
    file_ingested = crud.documents.get_by_filename(db, file_name=file_name)
    if file_ingested:
        raise HTTPException(
            status_code=409,
            detail="File already exists. Choose a different file.",
        )

    print(f"{file_name} uploaded by {current_user.id} action {MakerCheckerActionType.INSERT.value} and status {MakerCheckerStatus.PENDING.value}")

    docs_in = schemas.DocumentMakerCreate(
        filename=file_name, 
        uploaded_by=current_user.id, 
        action_type=MakerCheckerActionType.INSERT,
        status=MakerCheckerStatus.PENDING,
        doc_type_id=departments.doc_type_id,
        version_type=version_type,
        previous_document_id=previous_document_id,
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
            'version_type': f"{version_type}",
            'previous_document_id': f"{previous_document_id}" if previous_document_id else "None"
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
async def ingest_url(request: Request, url: str) -> IngestResponse:
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
        ingested_documents = await service.ingest_url(url, llamaindex_docs)
    except Exception as e:
        print(traceback.print_exc())
        return {"message": f"There was an error uploading the file(s)\n {e}"}
    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)


async def ingest(request: Request, file_path: str) -> IngestResponse:
    """Ingests and processes a file, storing its chunks to be used as context."""
    service = request.state.injector.get(IngestService)
    try:
        with open(file_path, 'rb') as file:
            file_name = Path(file_path).name
            upload_path = Path(f"{UPLOAD_DIR}/{file_name}")

            with upload_path.open('wb') as f:
                f.write(file.read())

            with upload_path.open('rb') as f:
                ingested_documents = await service.ingest_bin_data(file_name, f)

    except Exception as e:
        return {"message": f"There was an error uploading the file(s)\n {e}"}

    finally:
        upload_path.unlink(missing_ok=True)
    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)

@ingest_router.post("/ingest/file", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_file(
        request: Request,
        departments: schemas.DocumentDepartmentList = Depends(),
        file: UploadFile = File(...),
        log_audit: models.Audit = Depends(deps.get_audit_logger),
        db: Session = Depends(deps.get_db),
        current_user: models.User = Security(
            deps.get_current_user,
            scopes=[Role.ADMIN["name"], Role.SUPER_ADMIN["name"], Role.OPERATOR["name"]],
)) -> IngestResponse:
    """Ingests and processes a file, storing its chunks to be used as context."""
    service = request.state.injector.get(IngestService)
    try:
        original_filename = file.filename
        print("Original file name is:", original_filename)
        if original_filename is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file name provided",
            )
        upload_path = Path(f"{UPLOAD_DIR}/{original_filename}")
        try:
            contents = await file.read()
            async with aiofiles.open(upload_path, 'wb') as f:
                await f.write(contents)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal Server Error: Unable to ingest file.",
            )

        await create_documents(db, original_filename, current_user, departments, log_audit)
        with open(upload_path, "rb") as f:
            ingested_documents = service.ingest_bin_data(original_filename, f)

        logger.info(f"{original_filename} is uploaded by {current_user.username} in {departments.departments_ids}")
        response = IngestResponse(
            object="list", model="private-gpt", data=ingested_documents
        )
        return response

    except HTTPException:
        print(traceback.print_exc())
        raise

    except Exception as e:
        print(traceback.print_exc())
        logger.error(f"There was an error uploading the file(s): {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error: Unable to ingest file.",
        )
