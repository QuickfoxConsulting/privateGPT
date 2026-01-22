import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from starlette.requests import Request

from private_gpt.users.api.deps import get_db, get_current_user
from private_gpt.users.schemas.user import UserSchema
from private_gpt.users.schemas.tool import (
    ToolCategory, 
    ToolDefinition, 
    UserToolInstallation,
    ToolInstallRequest,
    InstalledToolsResponse,
    ToolAuthUrlRequest,
    ToolAuthUrlResponse,
    ToolAuthExchangeRequest
)
from private_gpt.server.tools.tool_registry import ToolRegistry, ToolNotFoundError, ToolConnectionError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tools", tags=["Tools"])

def get_tool_registry(db: Session = Depends(get_db)) -> ToolRegistry:
    return ToolRegistry(db)

@router.get("", response_model=List[ToolDefinition])
def list_tools(
    category: Optional[str] = None,
    registry: ToolRegistry = Depends(get_tool_registry),
    user: UserSchema = Depends(get_current_user)
):
    """List all available tools"""
    return registry.list_available_tools(category)

@router.get("/categories", response_model=List[ToolCategory])
def list_categories(
    registry: ToolRegistry = Depends(get_tool_registry),
    user: UserSchema = Depends(get_current_user)
):
    """List all tool categories"""
    return registry.list_categories()

@router.get("/installed", response_model=InstalledToolsResponse)
def list_installed_tools(
    registry: ToolRegistry = Depends(get_tool_registry),
    user: UserSchema = Depends(get_current_user)
):
    """List tools installed by the current user"""
    tools = registry.get_user_tools(user.id)
    # Note: get_user_tools returns ToolDefinitions in current implementation, 
    # but for "InstalledToolsResponse" we ideally want the Installation objects to see config/status.
    # The registry methods might need adjustment.
    # Re-checking registry implementation:
    # get_user_tools returns ToolDefinition list.
    # But for UI we need the installation status.
    # I should update registry or add a new method there.
    # For now, let's stick to what registry offers or fix it.
    
    # Fix: Updating logic here to fetch installations directly for better response
    # This logic matches what the schemas expect (UserToolInstallation)
    from private_gpt.users.models.tool import UserToolInstallation as UserToolInstallationModel
    from sqlalchemy import select
    
    stmt = (
        select(UserToolInstallationModel)
        .where(UserToolInstallationModel.user_id == user.id)
        .where(UserToolInstallationModel.is_enabled == True)
    )
    installations = registry.db.execute(stmt).scalars().all()
    
    return InstalledToolsResponse(
        max_count=len(installations),
        tools=installations
    )

@router.post("/{tool_id}/install", response_model=UserToolInstallation)
def install_tool(
    tool_id: int,
    request: ToolInstallRequest,
    registry: ToolRegistry = Depends(get_tool_registry),
    user: UserSchema = Depends(get_current_user)
):
    """Install a tool for the current user"""
    try:
        return registry.install_tool(user.id, tool_id, request.config)
    except ToolNotFoundError:
        raise HTTPException(status_code=404, detail="Tool not found")
    except ToolConnectionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{tool_id}/uninstall")
def uninstall_tool(
    tool_id: int,
    registry: ToolRegistry = Depends(get_tool_registry),
    user: UserSchema = Depends(get_current_user)
):
    """Uninstall a tool for the current user"""
    registry.uninstall_tool(user.id, tool_id)
    return {"message": "Tool uninstalled successfully"}

@router.post(
    "/{tool_id}/auth_url",
    response_model=ToolAuthUrlResponse
)
def get_tool_auth_url(
    tool_id: int,
    request: ToolAuthUrlRequest,
    user: UserSchema = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Generate OAuth URL for tool installation"""
    registry = ToolRegistry(db)
    try:
        url = registry.get_authorization_url(
            user_id=user.id,
            tool_id=tool_id,
            redirect_uri=request.redirect_uri,
            state=request.state
        )
        return ToolAuthUrlResponse(auth_url=url, state=request.state or "")
    except ToolNotFoundError:
        raise HTTPException(status_code=404, detail="Tool not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/{tool_id}/auth_exchange",
    response_model=UserToolInstallation
)
def exchange_tool_auth_code(
    tool_id: int,
    request: ToolAuthExchangeRequest,
    user: UserSchema = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Exchange OAuth code and install tool"""
    registry = ToolRegistry(db)
    try:
        installation = registry.install_with_authorization_code(
            user_id=user.id,
            tool_id=tool_id,
            code=request.code,
            redirect_uri=request.redirect_uri
        )
        return installation
    except ToolNotFoundError:
        raise HTTPException(status_code=404, detail="Tool not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error during tool auth exchange: {e}")
        raise HTTPException(status_code=500, detail=str(e))
