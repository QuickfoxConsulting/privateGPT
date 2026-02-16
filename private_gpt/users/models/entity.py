from sqlalchemy import Column, Integer, String, ForeignKey, Table
from sqlalchemy.orm import relationship
from private_gpt.users.db.base_class import Base

class Entity(Base):
    __tablename__ = "ner_entities"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    type = Column(String, index=True, nullable=False) # e.g., organization, person, location
    
    # Optional: store a hash or standardized name for better deduplication
    
    # Relationships
    nodes = relationship("NodeEntity", back_populates="entity", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Entity(name='{self.name}', type='{self.type}')>"

class NodeEntity(Base):
    """Junction table/model linking entities to specific text nodes (chunks)."""
    __tablename__ = "node_ner_entities"


    id = Column(Integer, primary_key=True, index=True)
    entity_id = Column(Integer, ForeignKey("ner_entities.id", ondelete="CASCADE"), nullable=False)
    node_id = Column(String, index=True, nullable=False) # The LlamaIndex Node ID (UUID string)
    doc_id = Column(String, index=True, nullable=False)  # The LlamaIndex Doc ID for quick aggregation
    
    # Relationships
    entity = relationship("Entity", back_populates="nodes")

    def __repr__(self) -> str:
        return f"<NodeEntity(node_id='{self.node_id}', entity_id={self.entity_id})>"
