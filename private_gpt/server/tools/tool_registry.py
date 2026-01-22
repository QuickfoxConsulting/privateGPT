import logging
import pickle
from typing import List, Optional, Dict, Any, Type
import importlib
from datetime import datetime
import json
import os

# Relax scope requirement for Google OAuth expansion
os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'

from sqlalchemy.orm import Session
from sqlalchemy import select, update, delete
from llama_index.core.tools import BaseTool

from private_gpt.users.models.tool import ToolCategory, ToolDefinition, UserToolInstallation, ToolExecutionLog
from private_gpt.server.tools.tool_interface import BaseMCPTool

logger = logging.getLogger(__name__)

class ToolNotFoundError(Exception):
    pass

class ToolConnectionError(Exception):
    pass

class ToolRegistry:
    """Central registry for tool management"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def list_categories(self) -> List[ToolCategory]:
        """List all tool categories"""
        stmt = select(ToolCategory)
        return self.db.execute(stmt).scalars().all()
    
    def list_available_tools(
        self, 
        category: Optional[str] = None
    ) -> List[ToolDefinition]:
        """List all tools available for installation"""
        stmt = select(ToolDefinition)
        if category:
            stmt = stmt.join(ToolCategory).where(
                ToolCategory.name == category
            )
        return self.db.execute(stmt).scalars().all()
        
    def get_user_tools(self, user_id: int) -> List[ToolDefinition]:
        """List tools installed by a user"""
        stmt = (
            select(ToolDefinition)
            .join(UserToolInstallation)
            .where(UserToolInstallation.user_id == user_id)
            .where(UserToolInstallation.is_enabled == True)
        )
        return self.db.execute(stmt).scalars().all()
    
    def install_tool(
        self, 
        user_id: int, 
        tool_id: int, 
        config: Dict[str, Any]
    ) -> UserToolInstallation:
        """Install a tool for a user"""
        
        # Validate tool exists
        tool_def = self.db.get(ToolDefinition, tool_id)
        if not tool_def:
            raise ToolNotFoundError(f"Tool {tool_id} not found")
        
        # TODO: Validate configuration against schema
        # self._validate_config(tool_def.config_schema, config)
        
        # Check dependencies
        # self._check_dependencies(user_id, tool_def.dependencies)
        
        # Check if already installed
        stmt = select(UserToolInstallation).where(
            UserToolInstallation.user_id == user_id,
            UserToolInstallation.tool_id == tool_id
        )
        existing = self.db.execute(stmt).scalar_one_or_none()
        
        if existing:
            # Update existing installation
            existing.config = config
            existing.is_enabled = True
            existing.installed_at = datetime.utcnow() # Reset install time or update logic?
            self.db.commit()
            self.db.refresh(existing)
            return existing
        
        # Create installation record
        installation = UserToolInstallation(
            user_id=user_id,
            tool_id=tool_id,
            config=config,
            installed_at=datetime.utcnow()
        )
        
        # Inject shared Google Auth credentials if applicable
        if tool_def.name in ["gmail_tool", "calendar_tool", "sheets_tool", "docs_tool"]:
             if "client_id" not in config or "client_secret" not in config:
                 from private_gpt.users.models.integration import GoogleDriveAuth
                 from private_gpt.server.utils.encryption import token_encryption
                 
                 auth = self.db.query(GoogleDriveAuth).filter(
                     GoogleDriveAuth.user_id == user_id
                 ).first()
                 
                 if auth:
                     if "client_id" not in config:
                         config["client_id"] = token_encryption.decrypt(auth.client_id)
                     if "client_secret" not in config:
                         config["client_secret"] = token_encryption.decrypt(auth.client_secret)
        
        self.db.add(installation)
        self.db.commit()
        self.db.refresh(installation)
        
        # Test connection
        try:
            tool_instance = self._load_tool_instance(tool_def, config, user_id)
            # Assuming BaseMCPTool has test_connection. For standard BaseTool we might skip or check attributes.
            if hasattr(tool_instance, 'test_connection'):
                 if not tool_instance.test_connection():
                     raise ToolConnectionError("Tool connection test failed")
        except Exception as e:
            # Rollback if connection fails
            self.db.delete(installation)
            self.db.commit()
            logger.error(f"Failed to install tool {tool_def.name} for user {user_id}. Test connection failed or error during load: {e}", exc_info=True)
            raise ToolConnectionError(f"Tool installation failed: {str(e)}")
        
        return installation

    def uninstall_tool(self, user_id: int, tool_id: int):
        """Uninstall a tool for a user"""
        stmt = select(UserToolInstallation).where(
            UserToolInstallation.user_id == user_id,
            UserToolInstallation.tool_id == tool_id
        )
        installation = self.db.execute(stmt).scalar_one_or_none()
        if installation:
            self.db.delete(installation)
            self.db.commit()

    def load_tools(
        self, 
        user_id: int, 
        tool_names: Optional[List[str]] = None
    ) -> List[BaseTool]:
        """Load user's installed tools"""
        
        # Query database
        query = (
            select(UserToolInstallation, ToolDefinition)
            .join(ToolDefinition)
            .where(UserToolInstallation.user_id == user_id)
            .where(UserToolInstallation.is_enabled == True)
        )
        
        if tool_names:
            query = query.where(ToolDefinition.name.in_(tool_names))
        
        results = self.db.execute(query).all()
        
        # Load tool instances
        tools = []
        for installation, tool_def in results:
            try:
                tool_instance = self._load_tool_instance(
                    tool_def, 
                    installation.config or {},
                    user_id
                )
                
                # Check if the tool provides a list of sub-tools (Toolkits/MCP style)
                if hasattr(tool_instance, 'to_tool_list'):
                    sub_tools = tool_instance.to_tool_list()
                    tools.extend(sub_tools)
                else:
                    tools.append(tool_instance)
                    
            except Exception as e:
                logger.error(f"Failed to load tool {tool_def.name}: {e}")
                
        return tools
    
    def _load_tool_instance(
        self, 
        tool_def: ToolDefinition, 
        config: Dict[str, Any],
        user_id: int
    ) -> BaseTool:
        """Dynamically load a tool class"""
        try:
            # Import module
            module = importlib.import_module(tool_def.module_path)
            tool_class = getattr(module, tool_def.class_name)
            
            # Instantiate with user config
            # We assume tools accept user_id and config, or at least match signature.
            # For standard tools, we might need a wrapper or adapter.
            try:
                tool_instance = tool_class(
                    user_id=user_id,
                    config=config
                )
            except TypeError:
                # Fallback for standard LlamaIndex tools or simpler signature
                tool_instance = tool_class(**config)
                
            return tool_instance
        except (ImportError, AttributeError) as e:
            raise RuntimeError(f"Could not load tool implementation: {e}")

    def get_authorization_url(
        self,
        user_id: int,
        tool_id: int,
        redirect_uri: str,
        state: Optional[str] = None
    ) -> str:
        """Generate OAuth authorization URL for a tool"""
        tool_def = self.db.get(ToolDefinition, tool_id)
        if not tool_def:
            raise ToolNotFoundError(f"Tool {tool_id} not found")
            
        # 1. Fetch Client ID/Secret
        # Try from Google Drive Auth first for Google tools
        client_id = None
        client_secret = None
        
        # Helper to check if it's a Google tool
        is_google_tool = tool_def.name in ["gmail_tool", "calendar_tool", "sheets_tool", "docs_tool"]
        
        if is_google_tool:
            from private_gpt.users.models.integration import GoogleDriveAuth
            from private_gpt.server.utils.encryption import token_encryption
            
            auth = self.db.query(GoogleDriveAuth).filter(
                GoogleDriveAuth.user_id == user_id
            ).first()
            
            if auth:
                client_id = token_encryption.decrypt(auth.client_id)
                client_secret = token_encryption.decrypt(auth.client_secret)
        
        if not client_id or not client_secret:
            # TODO: Check if tool definition has them embedded (not secure but possible for public apps)
            # Or allow user to pass them in request? For now, assume they must exist in Drive Auth
            # or be handled differently. 
            # If missing, we can't generate URL.
            raise ValueError("OAuth credentials (Client ID/Secret) not found. Please configure Google Drive integration first or provide credentials.")

        # 2. Get Scopes
        # Load tool class to get scopes
        module = importlib.import_module(tool_def.module_path)
        tool_class = getattr(module, tool_def.class_name)
        
        # Assume BaseMCPTool interface
        auth_config = tool_class.get_auth_config()
        if not auth_config or auth_config.auth_type != "oauth2":
            raise ValueError(f"Tool {tool_def.name} does not support OAuth2")
            
        scopes = auth_config.oauth_scopes
        
        # 3. Generate URL
        from google_auth_oauthlib.flow import Flow
        
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            },
            scopes=scopes,
            redirect_uri=redirect_uri
        )
        
        auth_url, _ = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent',
            state=state
        )
        
        return auth_url

    def install_with_authorization_code(
        self,
        user_id: int,
        tool_id: int,
        code: str,
        redirect_uri: str
    ) -> UserToolInstallation:
        """Exchange code for tokens and install tool"""
        
        # 1. Re-fetch credentials (same logic as above - should be refactored to helper)
        tool_def = self.db.get(ToolDefinition, tool_id)
        if not tool_def:
            raise ToolNotFoundError(f"Tool {tool_id} not found")
            
        client_id = None
        client_secret = None
        is_google_tool = tool_def.name in ["gmail_tool", "calendar_tool", "sheets_tool", "docs_tool"]
        
        if is_google_tool:
            from private_gpt.users.models.integration import GoogleDriveAuth
            from private_gpt.server.utils.encryption import token_encryption
            auth = self.db.query(GoogleDriveAuth).filter(GoogleDriveAuth.user_id == user_id).first()
            if auth:
                client_id = token_encryption.decrypt(auth.client_id)
                client_secret = token_encryption.decrypt(auth.client_secret)
                
        if not client_id or not client_secret:
             raise ValueError("OAuth credentials not found")
             
        # 2. Get Scopes
        module = importlib.import_module(tool_def.module_path)
        tool_class = getattr(module, tool_def.class_name)
        auth_config = tool_class.get_auth_config()
        scopes = auth_config.oauth_scopes
        
        # 3. Exchange Code
        from google_auth_oauthlib.flow import Flow
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            },
            scopes=scopes,
            redirect_uri=redirect_uri
        )
        
        flow.fetch_token(code=code)
        credentials = flow.credentials
        
        # 4. Prepare Config
        config = {
            "token": credentials.token,
            "refresh_token": credentials.refresh_token,
            "token_uri": credentials.token_uri,
            "client_id": credentials.client_id, 
            "client_secret": credentials.client_secret,
            "scopes": credentials.scopes
        }
        
        # 5. Install
        return self.install_tool(user_id, tool_id, config)

