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
from fastapi import File, Form, Query, UploadFile
from fastapi_pagination import Page, paginate
from fastapi import APIRouter, Depends, HTTPException, status, Security, Request

from private_gpt.users.api import deps
from private_gpt.constants import UNCHECKED_DIR, UPLOAD_DIR
from private_gpt.users.constants.role import Role
from private_gpt.users.core.config import settings
from private_gpt.users import crud, models, schemas
from private_gpt.server.ingest.ingest_router import create_documents, ingest, create_url_documents, ingest_url
from private_gpt.users.models.document import MakerCheckerActionType, MakerCheckerStatus

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/documents', tags=['Documents'])


ENABLE_MAKER_CHECKER = settings.ENABLE_MAKER_CHECKER
# Constants
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
ALLOWED_EXTENSIONS = {'.pdf', '.doc', '.docx', '.txt'}


def get_username(db, id):
    user = crud.user.get_by_id(db=db, id=id)
    return user.username

def get_id(db, username):
    name = crud.user.get_by_name(db=db, name=username)
    return name


@router.get("", response_model=Page[schemas.DocumentView])
def list_files(
    request: Request,
    db: Session = Depends(deps.get_db),
    filter: str = Query(None, description="Filter documents by filename"), 
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.ADMIN["name"], Role.SUPER_ADMIN["name"], Role.OPERATOR["name"]], 
    )
) -> Page[schemas.DocumentView]:
    """
    List documents based on user role with pagination and filtering.
    """
    try:
        # Get base query based on role
        role = current_user.user_role.role.name if current_user.user_role else None
        if role in ("SUPER_ADMIN", "OPERATOR"):
            base_query = crud.documents.get_multi_documents(db)
        else:
            base_query = crud.documents.get_documents_by_departments(
                db, 
                department_id=current_user.department_id
            )

        # Apply filename filter if provided
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
                tags=doc.tags or "",  # Handle None case
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
        deps.get_current_user,
        scopes=[Role.ADMIN["name"],
                Role.SUPER_ADMIN["name"],
                Role.OPERATOR["name"]]
    )
):
    '''
    Function to enable or disable document.
    '''
    try:
        document = crud.documents.get_by_id(
            db, id=document_in.id)
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
        scopes=[Role.SUPER_ADMIN["name"], Role.OPERATOR["name"]],
    )
):
    """
    Update the category list for the document
    """
    try:
        # Fetch the document by filename
        document = crud.documents.get_by_filename(
            db, file_name=document_in.filename)
        
        # Check if the document exists
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document with this filename doesn't exist!",
            )
        
        # Get the current categories associated with the document
        old_categories = [cat.id for cat in document.categories]
        
        # Update the categories
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

async def validate_file(file: UploadFile, max_size: int = MAX_FILE_SIZE) -> tuple[bool, str]:
    """Validate file size and extension."""
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return False, f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
    
    # Check file size
    first_chunk = await file.read(max_size + 1)
    await file.seek(0)  # Reset file pointer
    
    if len(first_chunk) > max_size:
        return False, f"File too large. Maximum size allowed: {max_size/1024/1024}MB"
    
    return True, ""

async def save_upload_file(file: UploadFile, destination: Path) -> bool:
    """Safely save uploaded file with error handling."""
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(destination, 'wb') as out_file:
            while content := await file.read(1024 * 1024):  # Read in 1MB chunks
                await out_file.write(content)
        return True
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        return False

async def move_file_safely(source: Path, destination: Path) -> bool:
    """Safely move file with error handling."""
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        return True
    except Exception as e:
        logger.error(f"Error moving file: {str(e)}")
        return False


@router.post('/upload', response_model=schemas.Document)
async def upload_documents(
    request: Request,
    departments: schemas.DocumentUpload = Depends(),
    log_audit: models.Audit = Depends(deps.get_audit_logger),
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.ADMIN["name"], Role.SUPER_ADMIN["name"], Role.OPERATOR["name"]],
    )
):
    """Upload the documents."""
    try:
        file = departments.file
        original_filename = file.filename
        
        if not original_filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file name provided",
            )

        # Validate file
        is_valid, error_message = await validate_file(file)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_message,
            )

        # Generate safe filename with original extension
        file_extension = Path(original_filename).suffix
        temp_filename = f"{uuid.uuid4()}{file_extension}"
        upload_path = Path(UNCHECKED_DIR) / temp_filename

        # Save file
        if not await save_upload_file(file, upload_path):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save file",
            )

        try:
            document = await create_documents(
                db=db,
                file_name=original_filename,
                category=departments.category,
                current_user=current_user,
                departments=departments, 
                log_audit=log_audit,
            )

            # Store the temporary path in the document version
            if document.current_version:
                version_update = schemas.DocumentVersionUpdate(
                    file_path=str(upload_path),
                    reviewed_at=datetime.now(),
                    reviewed_by=current_user.id,
                )
                crud.document_versions.update(
                    db, 
                    db_obj=document.current_version, 
                    obj_in=version_update
                )

            logger.info(
                f"{original_filename} uploaded by {current_user.username} "
            )

            # Auto-approve if maker-checker is disabled
            if not ENABLE_MAKER_CHECKER:
                checker_in = schemas.DocumentUpdate(
                    id=document.id,
                    status=MakerCheckerStatus.APPROVED.value
                )
                await verify_documents(
                    request=request,
                    checker_in=checker_in,
                    db=db,
                    log_audit=log_audit,
                    current_user=current_user
                )
            return document

        except Exception:
            # Cleanup on failure
            upload_path.unlink(missing_ok=True)
            raise

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error: Unable to upload file.",
        )

@router.post('/verify')
async def verify_documents(
    request: Request,
    checker_in: schemas.DocumentUpdate,
    log_audit: models.Audit = Depends(deps.get_audit_logger),
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"], Role.OPERATOR["name"]],
    )
):
    """Verify (approve/reject) a document version."""
    try:
        document = crud.documents.get_by_id(db, id=checker_in.id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found",
            )
        
        current_version = document.current_version
        if not current_version:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Document has no versions"
            )
        
        source_path = Path(current_version.file_path)
        if not source_path.exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document file not found",
            )

        if ENABLE_MAKER_CHECKER:
            if document.verified:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Document already verified!",
                )
            
            if not current_user.checker:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="You are not authorized as a checker",
                )
        
            if document.uploaded_by == current_user.id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot verify document you uploaded",
                )

        if checker_in.status == MakerCheckerStatus.APPROVED.value:
            # Prepare new path with version number and original extension
            upload_dir = Path(UPLOAD_DIR) / str(document.id)
            file_extension = Path(document.filename).suffix
            new_path = upload_dir / f"{Path(document.filename).stem}_v{current_version.version_number}{file_extension}"

            # Move file to final location
            if not await move_file_safely(source_path, new_path):
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to move approved file",
                )

            # Update version and document
            version_update = schemas.DocumentVersionUpdate(
                status=MakerCheckerStatus.APPROVED,
                action_type=MakerCheckerActionType.UPDATE,
                reviewed_by=current_user.id,
                reviewed_at=datetime.now(),
                file_path=str(new_path),
            )
            crud.document_versions.update(db, db_obj=current_version, obj_in=version_update)
            
            checker = schemas.DocumentCheckerUpdate(
                is_enabled=True,
                verified_at=datetime.now(),
                verified_by=current_user.id,
                verified=True,
            )
            crud.documents.update(db=db, db_obj=document, obj_in=checker)
            
            log_audit(
                model='Document',
                action='update',
                details={
                    'filename': document.filename,
                    'approved': str(current_user.id)
                },
                user_id=current_user.id
            )
            return await ingest(request, new_path)
            
        elif checker_in.status == MakerCheckerStatus.REJECTED.value:
            # Delete rejected file
            source_path.unlink(missing_ok=True)
            
            # Update version and document
            version_update = schemas.DocumentVersionUpdate(
                action_type=MakerCheckerActionType.DELETE,
                status=MakerCheckerStatus.REJECTED,
                reviewed_by=current_user.id,
                reviewed_at=datetime.now(),
            )
            crud.document_versions.update(db, db_obj=current_version, obj_in=version_update)

            checker = schemas.DocumentCheckerUpdate(
                is_enabled=False,
                verified_at=datetime.now(),
                verified_by=current_user.id,
                verified=False,
            )
            crud.documents.update(db=db, db_obj=document, obj_in=checker)
            crud.documents.remove(db, id=document.id)
            
            log_audit(
                model='Document',
                action='update',
                details={
                    'filename': document.filename,
                    'rejected': str(current_user.id)
                },
                user_id=current_user.id
            )
            return {"status": "rejected"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid status. Cannot change status to PENDING",
            )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error verifying document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error: Unable to verify document",
        )
    


# @router.post('/upload-url', response_model=schemas.Document)
# async def upload_documents(
#     request: Request,
#     url: schemas.UrlUpload,
#     log_audit: models.Audit = Depends(deps.get_audit_logger),
#     db: Session = Depends(deps.get_db),
#     current_user: models.User = Security(
#         deps.get_current_user,
#         scopes=[Role.ADMIN["name"],
#                 Role.SUPER_ADMIN["name"], 
#                 Role.OPERATOR["name"]],
#     )
# ):
#     """Upload the documents."""
#     try:
#         category = url.category
#         if  url.url is None:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="No URL is provided",
#             )

#         document = await create_url_documents(
#             db=db,
#             file_name=url.url,
#             category=category,
#             current_user=current_user,
#             departments=url.departments_ids,
#             log_audit=log_audit,
#         )
#         checker = schemas.DocumentCheckerUpdate(
#             action_type=MakerCheckerActionType.UPDATE,
#             status=MakerCheckerStatus.APPROVED,
#             is_enabled=True,
#             verified_at=datetime.now(),
#             verified_by=current_user.id,
#             verified=True,
#         )
#         crud.documents.update(db=db, db_obj=document, obj_in=checker)
        
#         log_audit(
#             model='Document',
#             action='update',
#             details={
#                 'filename': f'{url.url}',
#                 'approved': f'{current_user.id}'
#             },
#             user_id=current_user.id
#         )
#         return await ingest_url(request, url.url)
#         # return document

#     except HTTPException:
#         print(traceback.print_exc())
#         raise

#     except Exception as e:
#         print(traceback.print_exc())
#         logger.error(f"There was an error uploading the file(s): {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Internal Server Error: Unable to upload file.",
#         )


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
