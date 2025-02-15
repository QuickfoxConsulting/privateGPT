from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import select, func
from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, String, ForeignKey, Table
from private_gpt.users.db.base_class import Base

class Category(Base):
    """Models a Category table with document associations and dynamic counts."""
    
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, unique=True)
    documents = relationship("Document", secondary="document_category_association", back_populates="categories")

    @hybrid_property
    def document_count(self):
        """Return the total number of documents associated with this category."""
        return len(self.documents)
    
    @document_count.expression
    def document_count(cls):
        """SQL expression for document count."""
        return select([func.count(document_category_association.c.document_id)])\
            .where(document_category_association.c.category_id == cls.id)\
            .label("document_count")
    
    
document_category_association = Table(
    "document_category_association",
    Base.metadata,
    Column("document_id", Integer, ForeignKey("document.id", ondelete="CASCADE", onupdate="CASCADE")),
    Column("category_id", Integer, ForeignKey("categories.id", ondelete="CASCADE", onupdate="CASCADE")),
    extend_existing=True
)

