from typing import List, Optional, Type
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from private_gpt.server.integrations.base import BaseIntegration, IntegrationMetadata
from private_gpt.users.models.integration import Integration
import logging

logger = logging.getLogger(__name__)

class IntegrationService:
    """Registry and manager for all integrations."""
    
    _registry: List[Type[BaseIntegration]] = []

    def __init__(self, db: Session):
        self.db = db

    @classmethod
    def register(cls, integration_cls: Type[BaseIntegration]):
        """Register a new integration class."""
        cls._registry.append(integration_cls)
        logger.info(f"Registered integration: {integration_cls.__name__}")

    def list_integrations(self, user_id: int) -> List[IntegrationMetadata]:
        """List all integrations with their status for the given user."""
        # Fetch status map from DB
        db_integrations = {
            i.integration_id: i 
            for i in self.db.query(Integration).filter(Integration.user_id == user_id).all()
        }
        
        results = []
        for cls in self._registry:
            try:
                instance = cls(self.db)
                meta = instance.metadata
                
                # Check DB for status
                db_record = db_integrations.get(meta.id)
                meta.is_installed = db_record.is_active if db_record else False
                
                results.append(meta)
            except Exception as e:
                logger.error(f"Error checking status for {cls.__name__}: {e}")
                continue
        return results

    def get_integration(self, integration_id: str) -> Optional[BaseIntegration]:
        """Get an instance of a specific integration."""
        for cls in self._registry:
            instance = cls(self.db)
            if instance.metadata.id == integration_id:
                return instance
        return None

    def install_integration(self, integration_id: str, user_id: int) -> bool:
        """Call internal install AND update generic Integration table."""
        integration = self.get_integration(integration_id)
        if not integration:
            raise ValueError(f"Integration {integration_id} not found")
        
        # 1. Internal setup (e.g. specialized tables)
        success = integration.install(user_id)
        if not success:
            raise RuntimeError(f"Failed to install {integration_id}")

        # 2. Update generic DB record
        record = self.db.query(Integration).filter(
            Integration.user_id == user_id,
            Integration.integration_id == integration_id
        ).first()
        
        if record:
            record.is_active = True
            record.updated_at = datetime.now(timezone.utc)
        else:
            record = Integration(
                user_id=user_id,
                integration_id=integration_id,
                is_active=True,
                settings={} # Default empty settings
            )
            self.db.add(record)
        
        self.db.commit()
        return True

    def uninstall_integration(self, integration_id: str, user_id: int) -> bool:
        """Call internal uninstall AND update generic Integration table."""
        integration = self.get_integration(integration_id)
        if not integration:
            raise ValueError(f"Integration {integration_id} not found")
            
        # 1. Internal teardown
        success = integration.uninstall(user_id)
        
        # 2. Update generic DB record
        record = self.db.query(Integration).filter(
            Integration.user_id == user_id,
            Integration.integration_id == integration_id
        ).first()
        
        if record:
            record.is_active = False
            record.updated_at = datetime.now(timezone.utc)
            self.db.commit()
            
        return success
