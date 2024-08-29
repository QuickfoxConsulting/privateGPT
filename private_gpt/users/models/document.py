from fastapi_filter.contrib.sqlalchemy import Filter
from datetime import datetime
from private_gpt.users.models.department import Department
from sqlalchemy.orm import relationship, Session
from sqlalchemy import Boolean, event, select, func, update
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime

from private_gpt.users.db.base_class import Base
from private_gpt.users.models.document_department import document_department_association
from private_gpt.users.models.category import document_category_association
from sqlalchemy import Enum
from enum import Enum as PythonEnum

class MakerCheckerStatus(PythonEnum):
    PENDING = 'PENDING'
    APPROVED = 'APPROVED'
    REJECTED = 'REJECTED'


class MakerCheckerActionType(PythonEnum):
    INSERT = 'INSERT'
    UPDATE = 'UPDATE'
    DELETE = 'DELETE'

class VersionType(PythonEnum):
    NEW = 'NEW'
    AMENDMENT = 'AMENDMENT'

class DocumentType(Base):
    """Models a document table"""
    __tablename__ = "document_type"

    id = Column(Integer, primary_key=True, index=True)
    type = Column(String(225), nullable=False, unique=True)
    documents = relationship("Document", back_populates='doc_type')


class Document(Base):
    """Models a document table"""
    __tablename__ = "document"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(225), nullable=False, unique=True)
    uploaded_by = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False
    )
    uploaded_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
    uploaded_by_user = relationship(
        "User", back_populates="uploaded_documents",
        foreign_keys="[Document.uploaded_by]")
    
    is_enabled = Column(Boolean, default=True)
    verified = Column(Boolean, default=False) 
    
    doc_type_id = Column(Integer, ForeignKey("document_type.id"))
    doc_type = relationship("DocumentType", back_populates='documents')

    action_type = Column(Enum(MakerCheckerActionType), nullable=False,
                         default=MakerCheckerActionType.INSERT)  # 'insert' or 'update' or 'delete'
    status = Column(Enum(MakerCheckerStatus), nullable=False,
                    default=MakerCheckerStatus.PENDING)      # 'pending', 'approved', or 'rejected'

    verified_at = Column(DateTime, nullable=True)
    verified_by = Column(Integer, ForeignKey("users.id"), nullable=True)

    departments = relationship(
        "Department",
        secondary=document_department_association,
        back_populates="documents"
    )
    categories = relationship(
        "Category",
        secondary=document_category_association,
        back_populates="documents"
    )
    version_type = Column(Enum(VersionType), nullable=False, default=VersionType.NEW) # 'new', 'amendment'
    previous_document_id = Column(Integer, ForeignKey("document.id"), nullable=True)
    previous_document = relationship("Document", remote_side=[id], backref="next_documents")


# def get_associated_department_ids(session: Session, document_id: int) -> list:
#     """Get the department IDs associated with a given document."""
#     department_ids = session.query(document_department_association.c.department_id).filter(
#         document_department_association.c.document_id == document_id
#     ).all()
    
#     # Flatten the list of tuples returned by the query
#     return [dept_id for dept_id, in department_ids]

# @event.listens_for(Document, 'after_insert')
# @event.listens_for(Document, 'after_delete')
# def update_total_documents(mapper, connection, target):
#     session = Session(bind=connection)
#     try:
#         # Get the department IDs associated with the target document
#         associated_department_ids = get_associated_department_ids(session, target.id)
        
#         # Update total_documents for each associated department
#         for department_id in associated_department_ids:
#             total_documents = session.query(func.count()).select_from(document_department_association).filter(
#                 document_department_association.c.department_id == department_id
#             ).scalar()
            
#             department = session.query(Department).get(department_id)
#             if department:
#                 department.total_documents = total_documents
        
#         session.commit()
#     except Exception as e:
#         session.rollback()
#         raise
#     finally:
#         session.close()