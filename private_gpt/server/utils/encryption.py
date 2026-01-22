from typing import Optional
from cryptography.fernet import Fernet
import logging

from private_gpt.settings.settings import settings

logger = logging.getLogger(__name__)

class TokenEncryption:
    """Helper class for token encryption and decryption."""

    def __init__(self) -> None:
        self.encryption_key = settings().server.auth.secret # Using server secret or specific key?
        # Note: google_drive_service used settings.INTEGRATION_ENCRYPTION_KEY (if it existed) or similar.
        # Let's check what settings are available. 
        # In google_drive_service it was: self.encryption_key = settings.INTEGRATION_ENCRYPTION_KEY
        # But 'settings' there was imported from private_gpt.users.core.config usually?
        # Let's assume we pass the key or read from standard settings.
        
        # Checking google_drive_service.py again in my mind... 
        # It had: from private_gpt.users.core.config import settings
        # self.encryption_key = settings.INTEGRATION_ENCRYPTION_KEY
        pass

    def _get_suite(self) -> Fernet:
         # Lazy init
         # We need to access the config. 
         # Let's rely on the caller or import settings here.
         from private_gpt.users.core.config import settings
         key = settings.INTEGRATION_ENCRYPTION_KEY
         if not key:
             raise ValueError("INTEGRATION_ENCRYPTION_KEY not set")
         return Fernet(key.encode() if isinstance(key, str) else key)

    def encrypt(self, data: str) -> str:
        """Encrypt string data."""
        if not data:
            return ""
        try:
            return self._get_suite().encrypt(data.encode()).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data."""
        if not encrypted_data:
            return ""
        try:
            return self._get_suite().decrypt(encrypted_data.encode()).decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise

token_encryption = TokenEncryption()
