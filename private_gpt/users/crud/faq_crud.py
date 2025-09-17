import logging
from typing import List, Optional, Dict, Union
from sqlalchemy.orm import Session
from sqlalchemy import func, or_

from private_gpt.users.crud.base import CRUDBase
from private_gpt.users.models.faq import FAQ
from private_gpt.users.schemas.faq import FAQCreate, FAQUpdate

logger = logging.getLogger(__name__)


class CRUDFAQ(CRUDBase[FAQ, FAQCreate, FAQUpdate]):
    """CRUD operations for FAQ model."""
    
    def create_with_user(self, db: Session, *, obj_in: FAQCreate, user_id: int) -> FAQ:
        """Create a new FAQ with user information."""
        db_obj = FAQ(
            question=obj_in.question,
            answer=obj_in.answer,
            category=obj_in.category,
            created_by_id=user_id,
            updated_by_id=user_id,
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update_with_user(
        self, 
        db: Session, 
        *, 
        db_obj: FAQ, 
        obj_in: Union[FAQUpdate, Dict[str, any]], 
        user_id: int
    ) -> FAQ:
        """Update a FAQ with user information."""
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.model_dump(exclude_unset=True)
            
        update_data["updated_by_id"] = user_id
        return super().update(db, db_obj=db_obj, obj_in=update_data)

    def get_active_faqs(self, db: Session, *, skip: int = 0, limit: int = 100) -> List[FAQ]:
        """Get all active FAQs with pagination."""
        return db.query(self.model).filter(FAQ.is_active == True).offset(skip).limit(limit).all()

    def get_frequently_asked(self, db: Session, *, limit: int = 10) -> List[FAQ]:
        """Get the most frequently asked FAQs."""
        return (
            db.query(self.model)
            .filter(FAQ.is_active == True)
            .order_by(FAQ.frequency.desc())
            .limit(limit)
            .all()
        )

    def get_recently_added(self, db: Session, *, limit: int = 10) -> List[FAQ]:
        """Get the most recently added FAQs."""
        return (
            db.query(self.model)
            .filter(FAQ.is_active == True)
            .order_by(FAQ.created_at.desc())
            .limit(limit)
            .all()
        )

    def get_by_category(self, db: Session, *, category: str, skip: int = 0, limit: int = 100) -> List[FAQ]:
        """Get FAQs by category."""
        return (
            db.query(self.model)
            .filter(FAQ.category == category, FAQ.is_active == True)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_categories(self, db: Session) -> List[Dict[str, any]]:
        """Get all FAQ categories with count."""
        return (
            db.query(FAQ.category, func.count(FAQ.id).label("count"))
            .filter(FAQ.category.isnot(None), FAQ.is_active == True)
            .group_by(FAQ.category)
            .all()
        )

    def search(self, db: Session, *, query: str, limit: int = 10) -> List[FAQ]:
        """Search FAQs by query string."""
        logger.debug(f"Searching FAQs in database for query: '{query}' with limit: {limit}")
        
        try:
            result = (
                db.query(self.model)
                .filter(
                    FAQ.is_active == True,
                    or_(
                        FAQ.question.ilike(f"%{query}%"),
                        FAQ.answer.ilike(f"%{query}%"),
                        FAQ.category.ilike(f"%{query}%")
                    )
                )
                .order_by(FAQ.frequency.desc())
                .limit(limit)
                .all()
            )
            
            logger.debug(f"Database search returned {len(result)} results")
            for faq in result:
                logger.debug(f"Found FAQ: {faq.question[:50]}...")
            
            return result
        except Exception as e:
            logger.error(f"Failed to search FAQs in database: {e}", exc_info=True)
            return []

    def increment_frequency(self, db: Session, *, faq_id: int) -> bool:
        """Increment the frequency counter for a FAQ."""
        try:
            db.query(self.model).filter(self.model.id == faq_id).update(
                {FAQ.frequency: FAQ.frequency + 1}
            )
            db.commit()
            return True
        except Exception:
            db.rollback()
            return False

    def get_total_count(self, db: Session) -> int:
        """Get total count of FAQs."""
        return db.query(func.count(self.model.id)).scalar()


faq = CRUDFAQ(FAQ)