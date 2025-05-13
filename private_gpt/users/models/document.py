from datetime import datetime
from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum, Boolean, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from private_gpt.users.db.base_class import Base
from private_gpt.users.models.enums import *

class DocumentVersion(Base):
    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint('document_id', 'version_number', name='uix_document_version'),
    )

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("document.id"))
    version_number = Column(Integer, nullable=False, default=1)
    status = Column(Enum(MakerCheckerStatus), default=MakerCheckerStatus.PENDING)
    action_type = Column(Enum(MakerCheckerActionType))
    
    file_path = Column(String(512))
    changes = Column(String(1024))
    
    uploaded_by = Column(Integer, ForeignKey("users.id"))
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    reviewed_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    
    # Specify the foreign keys for the relationship
    document = relationship(
        "Document",
        foreign_keys=[document_id],
        back_populates="versions",
        primaryjoin="DocumentVersion.document_id == Document.id"
    )

class Document(Base):
    __tablename__ = "document"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(225), nullable=False, unique=True)
    doc_metadata = Column(JSONB, nullable=True)  # JSONB for metadata

    uploaded_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_enabled = Column(Boolean, default=True)
    verified = Column(Boolean, default=False) 
    verified_at = Column(DateTime, nullable=True)
    verified_by = Column(Integer, ForeignKey("users.id"), nullable=True)

    current_version_id = Column(Integer, ForeignKey("document_versions.id"))
    
    current_version = relationship(
        "DocumentVersion",
        foreign_keys=[current_version_id],
        primaryjoin="Document.current_version_id == DocumentVersion.id",
        post_update=True
    )

    versions = relationship(
        "DocumentVersion",
        foreign_keys=[DocumentVersion.document_id],
        primaryjoin="Document.id == DocumentVersion.document_id",
        back_populates="document",
        order_by="desc(DocumentVersion.version_number)",
        cascade="all, delete-orphan"
    )

    departments = relationship(
        "Department",
        secondary="document_department_association",
        back_populates="documents",
    )
    
    categories = relationship(
        "Category",
        secondary="document_category_association",
        back_populates="documents"
    )
    
    uploaded_by_user = relationship(
        "User",
        back_populates="uploaded_documents",
        foreign_keys=[uploaded_by]
    )

    @hybrid_property
    def action_type(self):
        return self.current_version.action_type if self.current_version else None

    @hybrid_property
    def status(self):
        return self.current_version.status if self.current_version else None