from private_gpt.users.models.document_selection import document_selection_association
from private_gpt.users.models.document_department import document_department_association
from private_gpt.users.models.document import Document
from typing import Optional, List, Dict, Any, Union
from sqlalchemy import or_, and_
from sqlalchemy.orm import Session, joinedload
from private_gpt.users.models.department import Department
from private_gpt.constants import ALL_DEPARTMENT


class DocumentSelectionService:
    def __init__(self, db_session: Session):
        self.db_session = db_session
    
    def select_documents(self, user_id: int, document_ids: List[int]) -> None:
        """Add documents to user's selection"""
        for doc_id in document_ids:
            exists = self.db_session.query(document_selection_association).filter_by(
                user_id=user_id, document_id=doc_id
            ).first()
            
            if not exists:
                self.db_session.execute(
                    document_selection_association.insert().values(
                        user_id=user_id, document_id=doc_id
                    )
                )
        
        self.db_session.commit()
    
    def unselect_documents(self, user_id: int, document_ids: List[int]) -> None:
        """Remove documents from user's selection"""
        for doc_id in document_ids:
            self.db_session.execute(
                document_selection_association.delete().where(
                    document_selection_association.c.user_id == user_id,
                    document_selection_association.c.document_id == doc_id
                )
            )
        
        self.db_session.commit()

    def get_selected_documents(self, user_id: int) -> List[Document]:
        """
        Get all documents selected by user
        """
        query = self.db_session.query(Document).join(
            document_selection_association,
            document_selection_association.c.document_id == Document.id
        ).filter(
            document_selection_association.c.user_id == user_id
        )
        return query.all()

    def get_enabled_documents(
        self,
        user_id: int,
        department_id: int,
    ) -> List[Document]:
        """Get all documents selected by user filtered by department access and enabled status."""
        all_department_id = ALL_DEPARTMENT

        selected_doc_ids = self.db_session.query(
            document_selection_association.c.document_id
        ).filter(
            document_selection_association.c.user_id == user_id
        ).all()

        selected_doc_ids = [doc_id for (doc_id,) in selected_doc_ids]
        base_query = (
            self.db_session.query(Document)
            .join(
                document_department_association,
                document_department_association.c.document_id == Document.id
            )
            .join(Department, document_department_association.c.department_id == Department.id)
            .options(joinedload(Document.categories))
            .filter(
                Document.is_enabled.is_(True),
                or_(
                    document_department_association.c.department_id == department_id,
                    document_department_association.c.department_id == all_department_id
                )
            )
        )

        if selected_doc_ids:
            base_query = base_query.filter(Document.id.in_(selected_doc_ids))

        return base_query.all()


    def clear_selection(self, user_id: int) -> None:
        """Clear all selected documents for a user"""
        self.db_session.execute(
            document_selection_association.delete().where(
                document_selection_association.c.user_id == user_id
            )
        )
        self.db_session.commit()