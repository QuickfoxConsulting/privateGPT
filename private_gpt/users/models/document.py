from datetime import datetime
from sqlalchemy import Enum
from sqlalchemy.orm import relationship, backref
from sqlalchemy import Boolean, event, select, func, update, insert
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, and_
from private_gpt.users.constants.upload_type import UploadType

from private_gpt.users.db.base_class import Base
from private_gpt.users.models.department import Department
from private_gpt.users.models.document_department import document_department_association


class Document(Base):
    """Models a document table"""
    __tablename__ = "document"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(225), nullable=False, unique=True)
    uploaded_by = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False,
    )
    uploaded_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
    uploaded_by_user = relationship(
        "User", back_populates="uploaded_documents")
    is_enabled = Column(Boolean, default=True)
    verified = Column(Boolean, default=False)
    upload_type = Column(Enum(UploadType), nullable=False, default=UploadType.REGULAR)  # 'insert' or 'update'
    departments = relationship(
        "Department",
        secondary=document_department_association,
        back_populates="documents"
    )
    
    # Relationship with MakerChecker
    maker_checker_entry = relationship(
        "MakerChecker",
        backref=backref("document", uselist=False),
        foreign_keys="[MakerChecker.record_id]",
        primaryjoin="and_(MakerChecker.table_name=='document', MakerChecker.record_id==Document.id)",
    )

def update_total_documents(mapper, connection, target):
    document_id = target.id

    # Get the total number of associations for the document
    total_documents = connection.execute(
        select([func.count()])
        .select_from(document_department_association)
        .where(document_department_association.c.document_id == document_id)
    ).scalar()

    # Update total_documents for each associated department
    connection.execute(
        update(Department)
        .values(total_documents=total_documents)
        .where(Department.id.in_(
            select([document_department_association.c.department_id])
            .where(document_department_association.c.document_id == document_id)
        ))
    )

# Register the event listener for Document's after_insert event
event.listen(Document, 'after_insert', update_total_documents)

# Register the event listener for Document's after_delete event
event.listen(Document, 'after_delete', update_total_documents)


# def create_makerchecker_entry(mapper, connection, target):
#     document_id = target.id
#     # Create a MakerChecker entry for the new Document record
#     connection.execute(
#         insert(MakerChecker).values(
#             table_name='document',
#             record_id=document_id,
#             action_type=MakerCheckerActionType.INSERT,
#             status=MakerCheckerStatus.PENDING,
#             verified_at=None,
#             verified_by=None,
#         )
#     )
# event.listen(Document, 'after_insert', create_makerchecker_entry)