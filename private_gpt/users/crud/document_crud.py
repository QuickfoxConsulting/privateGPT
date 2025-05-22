import os
from sqlalchemy.sql.expression import desc, asc
from sqlalchemy import or_, and_
from sqlalchemy.orm import Session, joinedload
from typing import Optional, List
from private_gpt.users.schemas.documents import DocumentCreate, DocumentUpdate, DocumentVersionCreate, DocumentVersionUpdate
from private_gpt.users.models.document import Document, DocumentVersion
from private_gpt.users.models.department import Department
from private_gpt.users.models.document_department import document_department_association
from private_gpt.users.models.category import document_category_association, Category
from private_gpt.users.crud.base import CRUDBase
from private_gpt.constants import ALL_DEPARTMENT
from sqlalchemy import and_

def get_versioned_filename_pattern(file_name: str) -> str:
    name, ext = os.path.splitext(file_name)
    return f"{name}_v%{ext}" 


class CRUDDocuments(CRUDBase[Document, DocumentCreate, DocumentUpdate]):

    def get_document(self, db: Session, *, id: int):
        return db.query(Document).options(joinedload(Document.current_version)).filter(Document.id == id).first()
    
    def get_by_id(self, db: Session, *, id: int) -> Optional[Document]:
        """Get a document by its ID."""
        return db.query(self.model).filter(Document.id == id).first()

    def get_by_filename(self, db: Session, *, file_name: str) -> Optional[Document]:
        """Get a document by its filename."""
        return db.query(self.model).filter(Document.filename == file_name).first()
    
    def get_multi_documents(self, db: Session) -> List[Document]:
        """Get all documents with their categories, ordered by upload date (descending)."""
        return (
            db.query(self.model)
            .options(joinedload(Document.categories))
            .order_by(desc(Document.uploaded_at))
            # .all()
        )
    
    def get_documents_by_departments(
            self, db: Session, *, department_id: int
        ) -> List[Document]:
        """Get documents associated with a specific department or the 'ALL_DEPARTMENT' group."""
        all_department_id = ALL_DEPARTMENT
        return (
            db.query(self.model)
            .join(document_department_association)
            .join(Department)
            .options(joinedload(Document.categories))
            .filter(
                or_(
                    document_department_association.c.department_id == department_id,
                    document_department_association.c.department_id == all_department_id,
                )
            )
            .order_by(desc(Document.uploaded_at))
            # .all()
        )
    
    def get_files_to_verify(self, db: Session) -> List[Document]:
        """Get all documents with pending versions that need verification."""
        return (
            db.query(self.model)
            .join(DocumentVersion)
            .filter(DocumentVersion.status == 'PENDING')
            .options(joinedload(Document.categories))
            .all()
        )
    
    def get_enabled_documents_by_departments(
        self,
        db: Session,
        *,
        department_id: int,
        category_ids: Optional[List[int]] = None
    ) -> List[Document]:
        """Get enabled documents for a specific department (or 'ALL_DEPARTMENT') and optional categories."""
        all_department_id = ALL_DEPARTMENT
        query = (
            db.query(self.model)
            .join(document_department_association)
            .join(Department)
            .options(joinedload(Document.categories))
            .filter(
                or_(
                    document_department_association.c.department_id == department_id,
                    document_department_association.c.department_id == all_department_id,
                ),
                Document.is_enabled == True,
            )
        )
        if category_ids:
            query = query.join(document_category_association).filter(
                document_category_association.c.category_id.in_(category_ids)
            )

        return query.all()
        
    def get_documents_by_categories(
            self, db: Session, *, category_id: int
        ) -> List[Document]:
        """Get documents associated with a specific category."""
        return (
            db.query(self.model)
            .join(document_category_association)
            .join(Category)
            .filter(
                document_category_association.c.category_id == category_id
            )
            .order_by(desc(Document.uploaded_at))
            .all()
        )

    def filter_query(
        self, db: Session, *,
        filename: Optional[str] = None,
        uploaded_by: Optional[int] = None,
        status: Optional[str] = None,
        category_id: Optional[int] = None,
        order_by: Optional[str] = None,
    ) -> List[Document]:
        """Filter documents based on various criteria."""
        query = db.query(Document)
        if filename:
            query = query.filter(Document.filename.ilike(f"%{filename}%"))
        if uploaded_by:
            query = query.filter(Document.uploaded_by == uploaded_by)
        if status:
            query = query.join(DocumentVersion).filter(DocumentVersion.status == status)
        if category_id:
            query = query.join(document_category_association).filter(
                document_category_association.c.category_id == category_id
            )
        if order_by == "desc":
            query = query.order_by(desc(Document.uploaded_at))
        else:
            query = query.order_by(asc(Document.uploaded_at))

        return query.all()
    
    def get_by_base_filename(self, db: Session, file_name: str):
        pattern = get_versioned_filename_pattern(file_name)
        return db.query(Document).filter(Document.filename.like(pattern)).first()

class CRUDDocumentVersion(CRUDBase[DocumentVersion, DocumentVersionCreate, DocumentVersionUpdate]):
    def get_by_id(self, db: Session, *, id: int) -> Optional[DocumentVersion]:
        """Get a document version by its ID."""
        return db.query(self.model).filter(DocumentVersion.id == id).first()

    def get_by_document_id(self, db: Session, *, document_id: int) -> List[DocumentVersion]:
        """Get all versions of a specific document, ordered by version number (descending)."""
        return (
            db.query(self.model)
            .filter(DocumentVersion.document_id == document_id)
            .order_by(desc(DocumentVersion.version_number))
            .all()
        )

    def get_pending_versions(self, db: Session) -> List[DocumentVersion]:
        """Get all document versions with status 'PENDING'."""
        return (
            db.query(self.model)
            .filter(DocumentVersion.status == 'PENDING')
            .all()
        )

    def get_latest_version(self, db: Session, *, document_id: int) -> Optional[DocumentVersion]:
        """Get the latest version of a document by its ID."""
        return (
            db.query(self.model)
            .filter(DocumentVersion.document_id == document_id)
            .order_by(desc(DocumentVersion.version_number))
            .first()
        )

    def create_version(self, db: Session, *, obj_in: DocumentVersionCreate) -> DocumentVersion:
        """Create a new document version."""
        # Determine the next version number
        latest_version = self.get_latest_version(db, document_id=obj_in.document_id)
        version_number = latest_version.version_number + 1 if latest_version else 1

        # Create the new version
        db_obj = DocumentVersion(
            document_id=obj_in.document_id,
            version_number=version_number,
            status=obj_in.status,
            action_type=obj_in.action_type,
            file_path=obj_in.file_path,
            changes=obj_in.changes,
            uploaded_by=obj_in.uploaded_by,
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update_version(self, db: Session, *, db_obj: DocumentVersion, obj_in: DocumentVersionUpdate) -> DocumentVersion:
        """Update a document version."""
        if obj_in.status:
            db_obj.status = obj_in.status
        if obj_in.reviewed_by:
            db_obj.reviewed_by = obj_in.reviewed_by
        if obj_in.reviewed_at:
            db_obj.reviewed_at = obj_in.reviewed_at
        if obj_in.changes:
            db_obj.changes = obj_in.changes
        if obj_in.file_path:
            db_obj.file_path = obj_in.file_path

        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def delete_version(self, db: Session, *, id: int) -> DocumentVersion:
        """Delete a document version by its ID."""
        db_obj = db.query(self.model).filter(DocumentVersion.id == id).first()
        if db_obj:
            db.delete(db_obj)
            db.commit()
        return db_obj
    

documents = CRUDDocuments(Document)
document_versions = CRUDDocumentVersion(DocumentVersion)