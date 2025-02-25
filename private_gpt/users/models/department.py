from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import select, func
from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, String, ForeignKey
from private_gpt.users.db.base_class import Base

class Department(Base):
    """Models a Department table with document and user associations"""
    
    __tablename__ = "departments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, unique=True)

    company_id = Column(Integer, ForeignKey('companies.id'))
    company = relationship("Company", back_populates="departments")

    users = relationship("User", back_populates="department")
    documents = relationship("Document", secondary="document_department_association", back_populates="departments")

    @hybrid_property
    def user_count(self):
        """Return the total number of users in this department."""
        return len(self.users)
    
    @user_count.expression
    def user_count(cls):
        """SQL expression for user count."""
        from private_gpt.users.models.user import User
        return select([func.count(User.id)])\
            .where(User.department_id == cls.id)\
            .label("user_count")

    @hybrid_property
    def document_count(self):
        """Return the total number of documents associated with this department."""
        return len(self.documents)
    
    @document_count.expression
    def document_count(cls):
        """SQL expression for document count."""
        from private_gpt.users.models.document_department import document_department_association
        return select([func.count(document_department_association.c.document_id)])\
            .where(document_department_association.c.department_id == cls.id)\
            .label("document_count")