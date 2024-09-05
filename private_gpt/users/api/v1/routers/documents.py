import os
import logging
import aiofiles
import traceback
from pathlib import Path
from datetime import datetime

from typing import Any, List
from sqlalchemy.orm import Session
from fastapi_pagination import Page, paginate
from fastapi import APIRouter, Depends, HTTPException, status, Security, Request

from private_gpt.users.api import deps
from private_gpt.constants import UNCHECKED_DIR
from private_gpt.users.constants.role import Role
from private_gpt.users.core.config import settings
from private_gpt.users import crud, models, schemas
from private_gpt.server.ingest.ingest_router import create_documents, ingest
from private_gpt.users.models.document import MakerCheckerActionType, MakerCheckerStatus
from private_gpt.components.ocr_components.table_ocr_api import process_ocr

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/documents', tags=['Documents'])

ENABLE_MAKER_CHECKER = settings.ENABLE_MAKER_CHECKER

def get_username(db, id):
    user = crud.user.get_by_id(db=db, id=id)
    return user.username

def get_id(db, username):
    name = crud.user.get_by_name(db=db, name=username)
    return name

@router.get("")
def list_files(
    request: Request,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.ADMIN["name"], Role.SUPER_ADMIN["name"], Role.OPERATOR["name"]], 
    )
)-> Page[schemas.DocumentView]:
    """
    List the documents based on the role. 
    """
    
    try:
        role = current_user.user_role.role.name if current_user.user_role else None
        if (role == "SUPER_ADMIN") or (role == "OPERATOR"):
            docs = crud.documents.get_multi_documents(db)
        else:
            docs = crud.documents.get_documents_by_departments(
                                                            db, 
                                                            department_id=current_user.department_id
                                                        )
        documents = [
            schemas.DocumentView(
                id=doc.id,
                filename=doc.filename,
                uploaded_by=get_username(db, doc.uploaded_by),
                uploaded_at=doc.uploaded_at,
                is_enabled=doc.is_enabled,
                departments=[
                    schemas.DepartmentList(id=dep.id, name=dep.name)
                    for dep in doc.departments
                ],
                action_type=doc.action_type,
                status=doc.status,
                categories=[
                    schemas.CategoryList(id=cat.id, name=cat.name)
                    for cat in doc.categories  
                ]
            )
            for doc in docs
        ]
        return paginate(documents)
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"There was an error listing the file(s).")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error",
        )


@router.get("/pending", response_model=Page[schemas.DocumentVerify])
def list_pending_files(
    request: Request,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"], Role.OPERATOR["name"]], 
    )
):
    """
    List the documents based on the role. 
    """
    def get_username(db, id):
        user = crud.user.get_by_id(db=db, id=id)
        return user.username

    try:
        docs = crud.documents.get_files_to_verify(
            db
        )
        documents = [
            schemas.DocumentVerify(
                id=doc.id,
                filename=doc.filename,
                uploaded_by=get_username(db, doc.uploaded_by),
                uploaded_at=doc.uploaded_at,
                departments=[
                    schemas.DepartmentList(id=dep.id, name=dep.name)
                    for dep in doc.departments
                ],
                status=doc.status,
                categories=[
                    schemas.CategoryList(id=cat.id, name=cat.name)
                    for cat in doc.categories  
                ]
            )
            for doc in docs
        ]
        return paginate(documents)
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"There was an error listing the file(s).")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error",
        )


@router.get('{department_id}', response_model=Page[schemas.DocumentView])
def list_files_by_department(
    request: Request,
    department_id: int,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"]],
    )
):
    '''
    Listing the documents by the department id
    '''
    try:
        docs = crud.documents.get_documents_by_departments(
            db, department_id=department_id)
        documents = [
            schemas.DocumentView(
                id=doc.id,
                filename=doc.filename,
                uploaded_by=get_username(db, doc.uploaded_by),
                uploaded_at=doc.uploaded_at,
                is_enabled=doc.is_enabled,
                departments=[
                    schemas.DepartmentList(id=dep.id, name=dep.name)
                    for dep in doc.departments
                ],
                action_type=doc.action_type,
                status=doc.status,
                categories=[
                    schemas.CategoryList(id=cat.id, name=cat.name)
                    for cat in doc.categories  
                ]
            )
            for doc in docs
        ]
        return paginate(documents)
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"There was an error listing the file(s).")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error.",
        )


@router.get('/files', response_model=Page[schemas.DocumentView])
def list_files_by_department(
    request: Request,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.ADMIN["name"], Role.SUPER_ADMIN["name"], Role.OPERATOR["name"]], 
    )
):
    '''
    Listing the documents by the ADMIN of the Department
    '''
    try:
        department_id = current_user.department_id
        docs = crud.documents.get_documents_by_departments(
            db, department_id=department_id)
        documents = [
            schemas.DocumentView(
                id=doc.id,
                filename=doc.filename,
                uploaded_by=get_username(db, doc.uploaded_by),
                uploaded_at=doc.uploaded_at,
                is_enabled=doc.is_enabled,
                departments=[
                    schemas.DepartmentList(id=dep.id, name=dep.name)
                    for dep in doc.departments
                ],
                action_type=doc.action_type,
                status=doc.status,
                categories=[
                    schemas.CategoryList(id=cat.id, name=cat.name)
                    for cat in doc.categories  
                ]
            )
            for doc in docs
        ]
        return paginate(documents)
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"There was an error listing the file(s).")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error.",
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


@router.post('/upload', response_model=schemas.Document)
async def upload_documents(
    request: Request,
    departments: schemas.DocumentDepartmentList = Depends(),
    log_audit: models.Audit = Depends(deps.get_audit_logger),
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.ADMIN["name"],
                Role.SUPER_ADMIN["name"], 
                Role.OPERATOR["name"]],
    )
):
    """Upload the documents."""
    try:
        file = departments.file
        original_filename = file.filename
        category = departments.category
        version_type = departments.version_type
        previous_document_id = departments.previous_document_id

        if original_filename is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file name provided",
            )

        # Define the upload path for the file
        upload_path = Path(f"{UNCHECKED_DIR}/{original_filename}")

        try:
            # Save the file to the specified upload path
            contents = await file.read()
            async with aiofiles.open(upload_path, 'wb') as f:
                await f.write(contents)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal Server Error: Unable to ingest file.",
            )

        # Create a new document record in the database
        document = await create_documents(
            db=db,
            file_name=original_filename,
            category=category,
            current_user=current_user,
            departments=departments,
            log_audit=log_audit,
            version_type=version_type,
            previous_document_id=previous_document_id
        )

        logger.info(
            f"{original_filename} uploaded by {current_user.username} in {departments.departments_ids} as a {version_type} document."
        )

        # If Maker-Checker is disabled, automatically approve the document
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

    except HTTPException:
        print(traceback.print_exc())
        raise

    except Exception as e:
        print(traceback.print_exc())
        logger.error(f"There was an error uploading the file(s): {str(e)}")
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
        scopes=[
                Role.SUPER_ADMIN["name"],
                Role.OPERATOR["name"]
            ],
    )
):
    """Verify and Ingest the documents."""
    try:
        document = crud.documents.get_by_id(db, id=checker_in.id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document not found!",
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
                    detail="You are not the checker!",
                )
        
            if document.uploaded_by == current_user.id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot verify by same user!",
                )
        
        unchecked_path = Path(f"{UNCHECKED_DIR}/{document.filename}")

        if checker_in.status == MakerCheckerStatus.APPROVED.value:
            checker = schemas.DocumentCheckerUpdate(
                    action_type=MakerCheckerActionType.UPDATE,
                    status=MakerCheckerStatus.APPROVED,
                    is_enabled=True,
                    verified_at=datetime.now(),
                    verified_by=current_user.id,
                    verified=True,
                )
            crud.documents.update(db=db, db_obj= document, obj_in=checker)
            
            log_audit(
                model='Document',
                action='update',
                details={
                    'filename': f'{document.filename}',
                    'approved': f'{current_user.id}'
                },
                user_id=current_user.id
            )

            if document.doc_type_id == 2:
                return await process_ocr(request, unchecked_path) # For Scanned using DocTR
                # return await process_ocr(request, unchecked_path, True) # For Scanned with Marker (GPU Memory full)
            else:
                return await ingest(request, unchecked_path) # For pdf
                # return await process_ocr(request, unchecked_path, False) # For pdf with Marker (GPU Memory full)
            
        elif checker_in.status == MakerCheckerStatus.REJECTED.value:
            checker = schemas.DocumentCheckerUpdate(
                action_type=MakerCheckerActionType.DELETE,
                status=MakerCheckerStatus.REJECTED,
                is_enabled=False,
                verified_at=datetime.now(),
                verified_by=current_user.id,
                verified=True,
            )
            crud.documents.update(db=db, db_obj=document, obj_in=checker)
            os.remove(unchecked_path)
            crud.documents.remove(db, id=document.id)
            log_audit(
                model='Document',
                action='update',
                details={
                    'filename': f'{document.filename}',
                    'rejected': f'{current_user.id}'
                },
                user_id=current_user.id
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot change status to PENDING!",
            )

    except HTTPException:
        print(traceback.print_exc())
        raise

    except Exception as e:
        print(traceback.print_exc())
        logger.error(f"There was an error uploading the file(s): {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error: Unable to upload file.",
        )


@router.get('/filter', response_model=Page[schemas.DocumentView])
async def get_documents(
    document_filter: schemas.DocumentFilter = Depends(),
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[
                Role.SUPER_ADMIN["name"],
                Role.OPERATOR["name"],
                Role.ADMIN["name"],
            ],
    )
)-> Any:
    try:
        uploaded_by = get_id(db, document_filter.uploaded_by)
        id = uploaded_by.id if uploaded_by else None
        docs = crud.documents.filter_query(
            db=db, 
            filename=document_filter.filename, 
            uploaded_by=id, 
            action_type=document_filter.action_type, 
            status=document_filter.status,
            order_by=document_filter.order_by,
            category_id=document_filter.category_id
        )
        documents = [
                schemas.DocumentView(
                    id=doc.id,
                    filename=doc.filename,
                    uploaded_by=get_username(db, doc.uploaded_by),
                    uploaded_at=doc.uploaded_at,
                    is_enabled=doc.is_enabled,
                    departments=[
                        schemas.DepartmentList(id=dep.id, name=dep.name)
                        for dep in doc.departments
                    ],
                    action_type=doc.action_type,
                    status=doc.status,
                    categories=[
                        schemas.CategoryList(id=cat.id, name=cat.name)
                        for cat in doc.categories  
                    ]
                )
                for doc in docs
            ]
        return paginate(documents)
    except Exception as e:
        print(traceback.print_exc())
        raise HTTPException(status_code=500, detail="Internal server error")
    

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
