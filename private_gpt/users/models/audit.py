from datetime import datetime
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from private_gpt.users.db.base_class import Base


class Audit(Base):
    __tablename__ = "audit"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey(
        "users.id", ondelete="SET NULL"), nullable=True)
    model = Column(String(100), nullable=False)
    action = Column(String(50), nullable=False)
    details = Column(JSONB, nullable=True)
    ip_address = Column(String(45), nullable=True)

    def __repr__(self):
        return f"<Audit(id={self.id}, timestamp={self.timestamp}, user_id={self.user_id}, model={self.model}, action={self.action}, details={self.details})>"