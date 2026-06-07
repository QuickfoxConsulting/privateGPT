import json
import os
import shutil
import uuid
import logging
import aiofiles
import traceback
from pathlib import Path
from datetime import datetime

from typing import Any, List
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from fastapi_pagination import Page, paginate
from fastapi import File, Form, Query, UploadFile, BackgroundTasks
from fastapi import APIRouter, Depends, HTTPException, status, Security, Request

from private_gpt.users.api import deps
from private_gpt.users.constants.role import Role
from private_gpt.users.core.config import settings
from private_gpt.users import crud, models, schemas
from private_gpt.users.db.session import SessionLocal
from private_gpt.users.models.enums import DocumentStatus
from private_gpt.constants import UNCHECKED_DIR, UPLOAD_DIR
from private_gpt.manager.document_manager import DocumentManager
from private_gpt.server.ingest.ingest_router import create_documents, ingest
from private_gpt.server.ingest.ingest_service import ChunkingStrategy
from private_gpt.users.models.document import MakerCheckerActionType, MakerCheckerStatus

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/documents', tags=['Documents'])


ENABLE_MAKER_CHECKER = settings.ENABLE_MAKER_CHECKER

def get_username(db, id):
    user = crud.user.get_by_id(db=db, id=id)
    return user.username

def get_id(db, username):
    name = crud.user.get_by_name(db=db, name=username)
    return name


def _mention_for_filename(filename: str) -> str:
    if any(char.isspace() for char in filename):
        return f'@"{filename}"'
    return f"@{filename}"


def _accessible_documents_query(db: Session, current_user: models.User):
    role = (
        current_user.user_role.role.name
        if current_user.user_role and current_user.user_role.role
        else None
    )
    if role in ("SUPER_ADMIN", "OPERATOR"):
        return crud.documents.get_multi_documents(db)

    return crud.documents.get_documents_by_departments(
        db,
        department_id=current_user.department_id,
    )


@router.get(
    "/mention-suggestions",
    response_model=List[schemas.DocumentMentionSuggestion],
)
def mention_suggestions(
    query: str = Query("", description="Text typed after @"),
    limit: int = Query(10, ge=1, le=20),
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(deps.get_current_user),
) -> List[schemas.DocumentMentionSuggestion]:
    """
    Return accessible documents for @mention autocomplete.
    """
    typed = (query or "").strip().lstrip("@").lower()
    documents_query = _accessible_documents_query(db, current_user).filter(
        models.Document.is_enabled == True
    )

    if typed:
        filter_safe = typed.replace("%", r"\%").replace("_", r"\_")
        documents_query = documents_query.filter(
            models.Document.filename.ilike(f"%{filter_safe}%")
        )

    documents = documents_query.all()
    if typed:
        documents.sort(
            key=lambda doc: (
                not doc.filename.lower().startswith(typed),
                doc.filename.lower(),
            )
        )
    else:
        documents.sort(key=lambda doc: doc.uploaded_at, reverse=True)

    return [
        schemas.DocumentMentionSuggestion(
            id=doc.id,
            filename=doc.filename,
            mention=_mention_for_filename(doc.filename),
        )
        for doc in documents[:limit]
    ]


@router.get("/{id}", response_model=schemas.DocumentView)
def get_document(
    id: int,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(deps.get_current_user),
) -> schemas.DocumentView:
    """Get document details by its database ID."""
    document = crud.documents.get_by_id(db, id=id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {id} not found",
        )
    
    return schemas.DocumentView(
        id=document.id,
        filename=document.filename,
        doc_status=document.doc_status,
        doc_metadata=document.doc_metadata or {},
        uploaded_by=get_username(db, document.uploaded_by),
        uploaded_at=document.uploaded_at,
        is_enabled=document.is_enabled,
        departments=[
            schemas.DepartmentList(id=dep.id, name=dep.name)
            for dep in document.departments
        ],
        categories=[
            schemas.CategoryList(id=cat.id, name=cat.name)
            for cat in document.categories
        ],
        version=(
            schemas.DocumentVersionOut(
                id=document.current_version.id,
                version_number=document.current_version.version_number,
                status=document.current_version.status,
                action_type=document.current_version.action_type,
                changes=document.current_version.changes,
                file_path=document.current_version.file_path,
                uploaded_at=document.current_version.uploaded_at,
                uploaded_by=document.current_version.uploaded_by,
                reviewed_at=document.current_version.reviewed_at,
                reviewed_by=document.current_version.reviewed_by
            ) if document.current_version else None
        )
    )


@router.get("", response_model=Page[schemas.DocumentView])
def list_files(
    request: Request,
    db: Session = Depends(deps.get_db),
    filter: str = Query(None, description="Filter documents by filename"), 
    current_user: models.User = Security(
        deps.get_current_user, 
    )
) -> Page[schemas.DocumentView]:
    """
    List documents based on user role with pagination and filtering.
    """
    try:
        role = current_user.user_role.role.name if current_user.user_role else None
        if role in ("SUPER_ADMIN", "OPERATOR"):
            base_query = crud.documents.get_multi_documents(db)
        else:
            base_query = crud.documents.get_documents_by_departments(
                db, 
                department_id=current_user.department_id
            )

        if filter:
            filter_safe = filter.replace('%', r'\%').replace('_', r'\_')
            base_query = base_query.filter(
                models.Document.filename.ilike(f"%{filter_safe}%")
            )
        docs_query = base_query.all()
        documents = [
            schemas.DocumentView(
                id=doc.id,
                filename=doc.filename,
                doc_status=doc.doc_status,
                doc_metadata=doc.doc_metadata or {}, 
                uploaded_by=get_username(db, doc.uploaded_by),
                uploaded_at=doc.uploaded_at,
                is_enabled=doc.is_enabled,
                departments=[
                    schemas.DepartmentList(id=dep.id, name=dep.name)
                    for dep in doc.departments
                ],
                categories=[
                    schemas.CategoryList(id=cat.id, name=cat.name)
                    for cat in doc.categories
                ],
                version=( 
                    schemas.DocumentVersionOut(
                        id=doc.current_version.id,
                        version_number=doc.current_version.version_number,
                        status=doc.current_version.status,
                        action_type=doc.current_version.action_type,
                        changes=doc.current_version.changes,
                        file_path=doc.current_version.file_path,
                        uploaded_at=doc.current_version.uploaded_at,
                        uploaded_by=doc.current_version.uploaded_by,
                        reviewed_at=doc.current_version.reviewed_at,
                        reviewed_by=doc.current_version.reviewed_by
                    ) if doc.current_version else None
                )
            )
            for doc in docs_query
        ]

        return paginate(documents)

    except SQLAlchemyError as e:
        logger.error(f"Database error while listing documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred while retrieving documents"
        )
    except Exception as e:
        logger.error(f"Error listing documents: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while retrieving documents"
        )
    

@router.post('/update', response_model=schemas.DocumentEnable)
def update_document(
    request: Request,
    document_in: schemas.DocumentEnable ,
    db: Session = Depends(deps.get_db),
    log_audit: models.Audit = Depends(deps.get_audit_logger),
    current_user: models.User = Security(
        deps.get_current_user
    )
):
    '''
    Function to enable or disable document.
    '''
    try:
        document = crud.documents.get_by_id(db, id=document_in.id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document with this id doesn't exist!",
            )
        docs = crud.documents.update(db=db, db_obj=document, obj_in=document_in)
        log_audit(
            model='Document', 
            action='update',
            details={
                'detail': f'{document.filename} status changed to {document_in.is_enabled} from {document.is_enabled}'
            }, 
            user_id=current_user.id
        )
        return docs
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"There was an error listing the file(s).")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error.",
        )
    
@router.post('/department_update', response_model=schemas.DocumentList)
def update_department(
    request: Request,
    document_in: schemas.DocumentDepartmentUpdate,
    db: Session = Depends(deps.get_db),
    log_audit: models.Audit = Depends(deps.get_audit_logger),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"], Role.OPERATOR["name"]], 
    )
):
    """
    Update the department list for the documents
    """
    try:
        document = crud.documents.get_by_filename(
            db, file_name=document_in.filename)
        old_departments = document.departments
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document with this filename doesn't exist!",
            )
        department_ids = [int(number) for number in document_in.departments]
        for department_id in department_ids:
            db.execute(models.document_department_association.insert().values(document_id=document.id, department_id=department_id))
        log_audit(
            model='Document', 
            action='update',
            details={
                'detail': f'{document_in.filename} assigned to {department_ids} from {old_departments}'
            }, 
            user_id=current_user.id
        )
        return document
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"There was an error listing the file(s).")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error.",
        )


@router.post('/category_update', response_model=schemas.DocumentList)
def update_category(
    request: Request,
    document_in: schemas.DocumentCategoryUpdate,
    db: Session = Depends(deps.get_db),
    log_audit: models.Audit = Depends(deps.get_audit_logger),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"], Role.OPERATOR["name"], Role.ADMIN["name"]],
    )
):
    """
    Update the category list for the document
    """
    try:
        document = crud.documents.get_by_filename(
            db, file_name=document_in.filename)
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document with this filename doesn't exist!",
            )
        
        old_categories = [cat.id for cat in document.categories]
        
        category_ids = [int(number) for number in document_in.categories]
        document.categories = []
        for category_id in category_ids:
            category = db.query(models.Category).get(category_id)
            if category:
                document.categories.append(category)
        
        db.commit()
        
        log_audit(
            model='Document',
            action='update',
            details={
                'detail': f'{document_in.filename} categories updated to {category_ids} from {old_categories}'
            },
            user_id=current_user.id
        )
        return document
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"There was an error updating the categories for the document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error: Unable to update categories.",
        )
    
@router.post('/upload')
async def upload_documents(
    request: Request,
    background_tasks: BackgroundTasks,
    documents: schemas.DocumentUpload = Depends(),
    doc_manager: DocumentManager = Depends(deps.get_document_manager),
    log_audit: models.Audit = Depends(deps.get_audit_logger),
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.ADMIN["name"], Role.SUPER_ADMIN["name"], Role.OPERATOR["name"]],
    )
):
    """Upload a new document version."""
    try:
        temp_path, sanitized_filename = await doc_manager.save_temp_file(documents.file)
        logger.info(f"Temp file:{temp_path} \n filename: {sanitized_filename}")
        
        document = await create_documents(
            db=db,
            file_name=sanitized_filename,
            current_user=current_user,
            documents=documents, 
            log_audit=log_audit,
        )
        if document.current_version:
            version_update = schemas.DocumentVersionUpdate(
                file_path=str(temp_path),
                reviewed_at=datetime.now(),
                reviewed_by=current_user.id,
            )
            crud.document_versions.update(
                db,
                db_obj=document.current_version,
                obj_in=version_update
            )
        if not ENABLE_MAKER_CHECKER:
            # Add auto-approval to background tasks
            background_tasks.add_task(
                verify_document_background,
                document_id=document.id,
                status=MakerCheckerStatus.APPROVED,
                current_user_id=current_user.id,
                doc_manager=doc_manager,
                request=request,
                strategy=documents.strategy
            )
            return {"status": "upload_complete", "message": "Document uploaded and auto-approval started"}
        return document

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload document"
        )

async def verify_document_background(
    document_id: int,
    status: MakerCheckerStatus,
    current_user_id: int,
    doc_manager: DocumentManager,
    request: Request,
    strategy: ChunkingStrategy = ChunkingStrategy.HIERARCHICAL
):
    """Background task to handle document verification with fresh DB session."""
    db = SessionLocal()
    try:
        # Create audit logger for this session
        user_agent = request.headers.get("user-agent", None)
        session_id = request.cookies.get("session_id", None) or request.headers.get("x-session-id", None)
        request_id = request.headers.get("x-request-id", None)
        client_host = request.client.host if request.client else None

        from private_gpt.users.utils.audit import log_audit_entry
        def local_log_audit(model, action, details, user_id=current_user_id):
            log_audit_entry(
                db, model, action, details, user_id=user_id, 
                ip_address=client_host, user_agent=user_agent, 
                session_id=session_id, request_id=request_id
            )

        document = crud.documents.get_by_id(db, id=document_id)
        if not document or not document.current_version:
            logger.error(f"Document or version not found for ID: {document_id}")
            return

        temp_path = Path(document.current_version.file_path)
        if not temp_path.exists():
            logger.error(f"Document file not found at path: {temp_path}")
            return

        if status == MakerCheckerStatus.APPROVED:
            final_path, versioned_filename = await doc_manager.approve_document(
                document_id=document.id,
                original_filename=document.filename,
                temp_path=temp_path,
                version=document.current_version.version_number,
            )
            version_update = schemas.DocumentVersionUpdate(
                status=MakerCheckerStatus.APPROVED,
                action_type=MakerCheckerActionType.UPDATE, 
                reviewed_by=current_user_id,
                reviewed_at=datetime.now(),
                file_path=str(final_path),
            )
            crud.document_versions.update(db, db_obj=document.current_version, obj_in=version_update)
            metadata_dict = {
                "tags": document.doc_metadata.get("tags", []),
                "departments": document.doc_metadata.get("departments", []),
                "category": document.doc_metadata.get("category", None),
                "document_path": str(final_path.relative_to(UPLOAD_DIR)),
                "strategy": strategy.value
            }
            checker = schemas.DocumentCheckerUpdate(
                filename=document.filename, 
                doc_metadata=metadata_dict,
                is_enabled=True,
                verified_at=datetime.now(),
                verified_by=current_user_id,
                verified=True,
            )
            crud.documents.update(db=db, db_obj=document, obj_in=checker)
            db.commit()
            db.refresh(document)

            local_log_audit(
                model='Document',
                action='update',
                details={
                    'filename': document.filename,
                    'approved_by': str(current_user_id)
                }
            )
            
            # Perform ingestion
            await ingest(
                request, 
                final_path, 
                metadata_dict,
                strategy=strategy
            )

            # Update status to READY after ingestion
            # Re-fetch document to ensure we have the latest state
            document = crud.documents.get_by_id(db, id=document_id)
            status_update = schemas.StatusUpdate(
               doc_status=DocumentStatus.READY.value
            )
            document = crud.documents.update(db=db, db_obj=document, obj_in=status_update)
            db.commit()
            db.refresh(document)
            logger.info(f"Document {document.filename} ingestion complete and status set to READY")
            
        elif status == MakerCheckerStatus.REJECTED:
            await doc_manager.reject_document(temp_path)
            
            version_update = schemas.DocumentVersionUpdate(
                action_type=MakerCheckerActionType.DELETE,
                status=MakerCheckerStatus.REJECTED,
                reviewed_by=current_user_id,
                reviewed_at=datetime.now(),
            )
            crud.document_versions.update(db, db_obj=document.current_version, obj_in=version_update)
            metadata_dict = {
                "tags": document.doc_metadata.get("tags", []),
                "departments": document.doc_metadata.get("departments", []),
                "category": document.doc_metadata.get("category", None),
                "document_path": str(temp_path),
                "strategy": strategy.value
            }
            checker = schemas.DocumentCheckerUpdate(
                filename=document.filename,
                doc_metadata=metadata_dict,
                is_enabled=False,
                verified_at=datetime.now(),
                verified_by=current_user_id,
                verified=False,
            )
            crud.documents.update(db=db, db_obj=document, obj_in=checker)
            crud.documents.remove(db, id=document.id)
            db.commit()
            
            local_log_audit(
                model='Document',
                action='update',
                details={
                    'filename': document.filename,
                    'rejected_by': str(current_user_id)
                }
            )
    except Exception as e:
        logger.error(f"Error in background verification: {str(e)}\n{traceback.format_exc()}")
        db.rollback()
    finally:
        db.close()

@router.post('/verify')
async def verify_documents(
    request: Request,
    background_tasks: BackgroundTasks,
    checker_in: schemas.DocumentUpdate,
    doc_manager: DocumentManager = Depends(deps.get_document_manager),
    log_audit: models.Audit = Depends(deps.get_audit_logger),
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"], Role.OPERATOR["name"], Role.ADMIN['name']],
    )
):
    """Verify (approve/reject) a document."""
    try:
        logger.info(f"VERIFYING DOCUMENT::: {checker_in.id}")
        document = crud.documents.get_by_id(db, id=checker_in.id)
        if not document or not document.current_version:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document or version not found"
            )

        if ENABLE_MAKER_CHECKER:
            if document.verified:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Document already verified"
                )
            
            if not current_user.checker:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Not authorized as checker"
                )
            
            if document.uploaded_by == current_user.id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot verify own upload"
                )

        temp_path = Path(document.current_version.file_path)
        if not temp_path.exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document file not found"
            )

        background_tasks.add_task(
            verify_document_background,
            document_id=checker_in.id,
            status=checker_in.status,
            current_user_id=current_user.id,
            doc_manager=doc_manager,
            request=request,
            strategy=ChunkingStrategy.HIERARCHICAL
        )

        return {"status": "verification_started", "message": "Document verification has been started"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify document"
        )




@router.get('/documents/{filename}', response_model=schemas.DocumentFilePath)
async def get_document_by_filename(
    filename: str,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
    )
):
    """
    Get a document by its filename.
    
    Parameters:
    - filename: Name of the file to retrieve
    
    Returns:
    - DocumentFilePath: Object containing filename and relative file path
    """
    try:
        safe_filename = os.path.basename(filename)        
        document = crud.documents.get_by_filename(db, file_name=safe_filename)
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # if not crud.documents.has_access_permission(db, document_id=document.id, user_id=current_user.id):
        #     raise HTTPException(
        #         status_code=status.HTTP_403_FORBIDDEN,
        #         detail="You don't have permission to access this document"
        #     )
        
        # Get file path from current version
        if not document.current_version or not document.current_version.file_path:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document file not found"
            )
            
        file_path = document.current_version.file_path
        
        try:
            full_path = Path(file_path).resolve()
            if not str(full_path).startswith(str(Path(UPLOAD_DIR).resolve())):
                logger.warning(f"Attempted access to file outside upload directory: {file_path}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid file path"
                )
                
            rel_path = full_path.relative_to(UPLOAD_DIR)
            
            if not full_path.exists():
                logger.error(f"File does not exist at path: {full_path}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Document file not found on disk"
                )
                
            return schemas.DocumentFilePath(
                filename=safe_filename,
                file_path=str(rel_path),
            )
            
        except ValueError as e:
            logger.error(f"Path validation error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file path"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document by filename: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get document by filename"
        )

@router.post('/document_update', response_model=schemas.DocumentList)
def update_document_associations(
    request: Request,
    document_in: schemas.DocCatUpdate,
    db: Session = Depends(deps.get_db),
    log_audit: models.Audit = Depends(deps.get_audit_logger),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"], Role.OPERATOR["name"]],
    )
):
    """
    Update the department and category lists for the document.
    Only updates the departments or categories if they are provided.
    """
    try:
        document = crud.documents.get_by_filename(db, file_name=document_in.filename)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document with this filename doesn't exist!",
            )

        old_departments = [dept.id for dept in document.departments]
        old_categories = [cat.id for cat in document.categories]

        # Update departments only if provided
        if document_in.departments is not None:
            document.departments = []
            for department_id in document_in.departments:
                department = db.query(models.Department).get(department_id)
                if department:
                    document.departments.append(department)

        # Update categories only if provided
        if document_in.categories is not None:
            document.categories = []
            for category_id in document_in.categories:
                category = db.query(models.Category).get(category_id)
                if category:
                    document.categories.append(category)

        db.commit()

        log_audit(
            model='Document',
            action='update',
            details={
                'detail': f'{document_in.filename} updated. '
                          f'Departments: {old_departments} -> {document_in.departments or old_departments}, '
                          f'Categories: {old_categories} -> {document_in.categories or old_categories}'
            },
            user_id=current_user.id
        )
        return document

    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"Error updating document associations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error: Unable to update document associations.",
        )

@router.post('/document_selection')
def update_user_document_association(
    request: Request,
    documents: schemas.DocumentSelection,
    db: Session = Depends(deps.get_db),
    log_audit: models.Audit = Depends(deps.get_audit_logger),
    current_user: models.User = Security(
        deps.get_current_user
    )
):
    """
    Update the selected document for the given user for query.
    
    This endpoint allows users to create a personalized document selection for RAG queries.
    Selected documents will be used as the primary source for answering the user's questions.
    
    Args:
        request: FastAPI request object
        documents: Schema containing document IDs to select
        db: Database session dependency
        log_audit: Audit logging dependency
        current_user: Current authenticated user
        
    Returns:
        JSON response with operation status
        
    Raises:
        HTTPException: If document selection operation fails
    """
    try:
        from private_gpt.users.services import DocumentSelectionService
        
        service = DocumentSelectionService(db)
        
        if len(documents.document_ids) == 0:
            service.clear_selection(current_user.id)
            
            log_audit(
                user_id=current_user.id,
                action="document_selection_clear",
                model="Document",
                details="User cleared all selected documents"
            )
            
            return {
                "status": "success",
                "message": "Successfully cleared all document selections",
                "selected_count": 0
            }
        
        current_selections = service.get_selected_documents(user_id=current_user.id)
        current_ids = set(row.id for row in current_selections)
        new_ids = set(documents.document_ids)
        
        to_add = new_ids - current_ids
        to_remove = current_ids - new_ids
        
        if to_remove:
            service.unselect_documents(current_user.id, list(to_remove))
        
        if to_add:
            service.select_documents(current_user.id, list(to_add))
        
        log_audit(
            user_id=current_user.id,
            action="document_selection_update",
            model="Document",
            details=f"User selected {len(documents.document_ids)} documents (added {len(to_add)}, removed {len(to_remove)})"
        )
        
        return {
            "status": "success",
            "message": f"Successfully selected {len(documents.document_ids)} documents",
            "selected_count": len(documents.document_ids),
            "changes": {
                "added": len(to_add),
                "removed": len(to_remove)
            }
        }
        
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Error updating document associations: {str(e)}\n{trace}")
        
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error: Unable to update document associations.",
        )
