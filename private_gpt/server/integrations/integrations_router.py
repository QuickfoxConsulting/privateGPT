"""API router for Google Drive integrations - FIXED VERSION."""

import logging
import secrets
import json
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query, Security, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.orm import Session

from private_gpt.server.integrations.website_crawl_service import WebsiteCrawlService
from private_gpt.server.integrations.google_drive_service import GoogleDriveService, token_encryption
from private_gpt.server.integrations.integration_service_new import IntegrationService
from private_gpt.server.integrations.base import IntegrationMetadata
from private_gpt.users.models.integration import GoogleDriveAuth, GoogleDriveFile
from private_gpt.users.api import deps
from private_gpt.users.constants.role import Role
from private_gpt.users import models
from private_gpt.users.core.config import settings 

# Register known integrations
IntegrationService.register(GoogleDriveService)
IntegrationService.register(WebsiteCrawlService)

logger = logging.getLogger(__name__)

integrations_router = APIRouter(prefix="/v1/integrations", tags=["Integrations"])

# OAuth state management - Redis with in-memory fallback
_oauth_states_fallback: Dict[str, Tuple[int, datetime]] = {}
_redis_client = None

def _get_redis_client():
    """Get or create Redis client for OAuth state storage."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    
    try:
        import redis
        _redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=int(settings.REDIS_PORT),
            password=settings.REDIS_PASSWORD or None,
            db=int(settings.REDIS_DB),
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        _redis_client.ping()
        logger.info("Redis connected for OAuth state storage")
        return _redis_client
    except Exception as e:
        logger.warning(f"Redis unavailable for OAuth state, using in-memory fallback: {e}")
        return None

def create_oauth_state(user_id: int) -> str:
    """Create a secure state token for OAuth flow."""
    state = secrets.token_urlsafe(32)
    expiry = datetime.now(timezone.utc) + timedelta(minutes=10)
    
    redis_client = _get_redis_client()
    if redis_client:
        try:
            # Store in Redis with 10-minute TTL
            redis_client.setex(
                f"oauth_state:{state}",
                600,  # 10 minutes
                json.dumps({"user_id": user_id, "expiry": expiry.isoformat()})
            )
            return state
        except Exception as e:
            logger.warning(f"Failed to store OAuth state in Redis: {e}")
    
    # Fallback to in-memory
    _oauth_states_fallback[state] = (user_id, expiry)
    return state

def verify_oauth_state(state: str) -> Optional[int]:
    """Verify and consume OAuth state."""
    redis_client = _get_redis_client()
    if redis_client:
        try:
            key = f"oauth_state:{state}"
            data = redis_client.get(key)
            if data:
                redis_client.delete(key)  # Consume the state
                parsed = json.loads(data)
                expiry = datetime.fromisoformat(parsed["expiry"])
                if datetime.now(timezone.utc) > expiry:
                    return None
                return parsed["user_id"]
        except Exception as e:
            logger.warning(f"Failed to verify OAuth state from Redis: {e}")
    
    # Fallback to in-memory
    if state not in _oauth_states_fallback:
        return None
    user_id, expiry = _oauth_states_fallback.pop(state)
    if datetime.now(timezone.utc) > expiry:
        return None
    return user_id


# --- Pydantic Models ---
class WebsiteCrawlConfigResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)        
    id: int
    user_id: int
    name: str
    base_url: str
    max_depth: int
    url_patterns: Optional[dict] = None
    crawl_frequency: Optional[str] = None
    enabled: bool
    last_crawl: Optional[datetime] = None
    crawl_status: str
    pages_crawled: int
    created_at: datetime
    updated_at: datetime

  

class WebsiteCrawlPageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    config_id: int
    url: str
    title: Optional[str] = None
    is_selected: bool
    ingest_status: str
    last_ingested_at: Optional[datetime] = None
    created_at: datetime

class WebsiteCrawlConfigCreate(BaseModel):
    name: str = Field(..., description="Configuration name")
    base_url: str = Field(..., description="Starting URL for crawl")
    max_depth: int = Field(3, description="Maximum crawl depth")
    url_patterns: Optional[dict] = Field(None, description="Include/exclude URL patterns")
    crawl_frequency: Optional[str] = Field(None, description="Cron expression for scheduling") 


class PageSelectionRequest(BaseModel):
    page_ids: List[int]
    is_selected: bool

class GoogleDriveSetupRequest(BaseModel):
    client_id: str = Field(..., description="Google OAuth Client ID")
class GoogleDriveSetupRequest(BaseModel):
    client_id: str = Field(..., description="Google OAuth Client ID")
    client_secret: str = Field(..., description="Google OAuth Client Secret")

class GoogleDriveConfigUpdate(BaseModel):
    sync_frequency: Optional[str] = None # e.g., "0 0 * * *" or "daily"

class GoogleDriveAuthResponse(BaseModel):
    id: int
    user_id: int
    enabled: bool
    last_sync: Optional[datetime] = None
    sync_status: Optional[str] = None
    sync_frequency: Optional[str] = None
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class PaginatedWebsitePagesResponse(BaseModel):
    items: List[WebsiteCrawlPageResponse]
    total: int
    skip: int
    limit: int

class GoogleDriveFileResponse(BaseModel):
    id: int
    auth_id: int
    file_id: str
    name: str
    mime_type: Optional[str] = None
    size: Optional[int] = None
    is_folder: bool = False
    parent_folder_id: Optional[str] = None
    parent_folder_name: Optional[str] = None
    is_selected: bool
    ingest_status: str
    last_ingested_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class PaginatedGoogleDriveFilesResponse(BaseModel):
    items: List[GoogleDriveFileResponse]
    total: int
    skip: int
    limit: int

class GoogleDriveFileSelectionRequest(BaseModel):
    file_ids: List[int]
    is_selected: bool

class GoogleDriveFileDelete(BaseModel):
    file_ids: List[int]

# --- Helpers ---

def get_redirect_uri() -> str:
    """Construct the callback URL based on settings."""
    base = getattr(settings, "API_BASE_URL", "http://localhost:8000")
    return f"{base.rstrip('/')}/v1/integrations/google-drive/callback"

def get_frontend_redirect_url(status_param: str) -> str:
    """Construct the frontend redirect URL."""
    base = getattr(settings, "FRONTEND_BASE_URL", "http://localhost:3000")
    return f"{base.rstrip('/')}/admin/google-drive?{status_param}"

# --- Endpoints ---

admin_security = Security(
    deps.get_current_user,
    scopes=[Role.ADMIN["name"], Role.SUPER_ADMIN["name"]]
)

# --- Generic Integration Endpoints ---

@integrations_router.get("/catalog", response_model=List[IntegrationMetadata])
async def get_integration_catalog(
    current_user: models.User = admin_security,
    db: Session = Depends(deps.get_db)
):
    """List all available integrations and their status."""
    service = IntegrationService(db)
    return service.list_integrations(current_user.id)

@integrations_router.post("/{integration_id}/install")
async def install_integration(
    integration_id: str,
    current_user: models.User = admin_security,
    db: Session = Depends(deps.get_db)
):
    """Install (enable) an integration."""
    service = IntegrationService(db)
    try:
        service.install_integration(integration_id, current_user.id)
        return {"message": f"Integration {integration_id} installed successfully"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Install failed: {e}")
        raise HTTPException(status_code=500, detail="Installation failed")

@integrations_router.post("/{integration_id}/uninstall")
async def uninstall_integration(
    integration_id: str,
    current_user: models.User = admin_security,
    db: Session = Depends(deps.get_db)
):
    """Uninstall (disable/cleanup) an integration."""
    service = IntegrationService(db)
    try:
        service.uninstall_integration(integration_id, current_user.id)
        return {"message": f"Integration {integration_id} uninstalled successfully"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# --- Specific Integration Endpoints ---

@integrations_router.post("/google-drive/setup")
async def setup_google_drive(
    setup_data: GoogleDriveSetupRequest,
    current_user: models.User = admin_security,
    db: Session = Depends(deps.get_db)
):
    """Setup Google Drive integration with user-provided OAuth credentials."""
    service = GoogleDriveService(db)
    
    # 1. Store the Client ID/Secret immediately (Encrypted)
    auth = db.query(GoogleDriveAuth).filter(
        GoogleDriveAuth.user_id == current_user.id
    ).first()
    
    if auth:
        auth.client_id = token_encryption.encrypt(setup_data.client_id)
        auth.client_secret = token_encryption.encrypt(setup_data.client_secret)
        auth.updated_at = datetime.now(timezone.utc)
    else:
        auth = GoogleDriveAuth(
            user_id=current_user.id,
            client_id=token_encryption.encrypt(setup_data.client_id),
            client_secret=token_encryption.encrypt(setup_data.client_secret),
            access_token=token_encryption.encrypt(""), 
            refresh_token=token_encryption.encrypt(""),
            token_expiry=datetime.now(timezone.utc),
            enabled=False 
        )
        db.add(auth)
    
    db.commit()
    db.refresh(auth)
    
    # 2. Generate Auth URL
    # We must pass the RAW strings to the service generator
    auth_url = service.get_authorization_url(
        client_id=setup_data.client_id,
        client_secret=setup_data.client_secret,
        redirect_uri=get_redirect_uri(),
        state=create_oauth_state(current_user.id)
    )
    
    return {
        "message": "OAuth credentials saved. Please complete authorization.",
        "authorization_url": auth_url
    }

@integrations_router.get("/google-drive/auth")
async def initiate_google_drive_auth(
    current_user: models.User = admin_security,
    db: Session = Depends(deps.get_db)
):
    """Get authorization URL for existing OAuth setup."""
    service = GoogleDriveService(db)
    
    auth = db.query(GoogleDriveAuth).filter(
        GoogleDriveAuth.user_id == current_user.id
    ).first()
    
    if not auth or not auth.client_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Please setup OAuth credentials first"
        )
    
    # Decrypt to generate valid Flow
    try:
        decrypted_client_id = token_encryption.decrypt(auth.client_id)
        decrypted_client_secret = token_encryption.decrypt(auth.client_secret)
    except Exception:
        raise HTTPException(status_code=500, detail="Encryption key error")
    
    auth_url = service.get_authorization_url(
        client_id=decrypted_client_id,
        client_secret=decrypted_client_secret,
        redirect_uri=get_redirect_uri(),
        state=create_oauth_state(current_user.id)
    )
    
    return {"authorization_url": auth_url}

@integrations_router.get("/google-drive/callback")
async def google_drive_callback(
    code: str = Query(..., description="Authorization code from Google"),
    state: str = Query(..., description="State parameter"),
    db: Session = Depends(deps.get_db)
):
    """Handle Google Drive OAuth callback."""
    user_id = verify_oauth_state(state)
    if not user_id:
        return RedirectResponse(
            url=get_frontend_redirect_url("error=invalid_state"), 
            status_code=302
        )
    
    service = GoogleDriveService(db)
    
    # Retrieve stored credentials to complete the handshake
    auth = db.query(GoogleDriveAuth).filter(
        GoogleDriveAuth.user_id == user_id
    ).first()
    
    if not auth or not auth.client_id:
        return RedirectResponse(
            url=get_frontend_redirect_url("error=setup_not_found"), 
            status_code=302
        )
    
    try:
        decrypted_client_id = token_encryption.decrypt(auth.client_id)
        decrypted_client_secret = token_encryption.decrypt(auth.client_secret)
        
        service.exchange_code_for_tokens(
            code=code,
            client_id=decrypted_client_id,
            client_secret=decrypted_client_secret,
            redirect_uri=get_redirect_uri(),
            user_id=user_id
        )
        
        return RedirectResponse(
            url=get_frontend_redirect_url("success=true"), 
            status_code=302
        )
    except Exception as e:
        logger.error(f"OAuth callback error: {str(e)}")
        return RedirectResponse(
            url=get_frontend_redirect_url("error=oauth_failed"), 
            status_code=302
        )

@integrations_router.get("/google-drive/status", response_model=Optional[GoogleDriveAuthResponse])
async def get_google_drive_status(
    current_user: models.User = admin_security,
    db: Session = Depends(deps.get_db)
):
    auth = db.query(GoogleDriveAuth).filter(
        GoogleDriveAuth.user_id == current_user.id
    ).first()
    return auth

@integrations_router.post("/google-drive/sync")
async def sync_google_drive(
    folder_id: Optional[str] = Query(None),
    current_user: models.User = admin_security,
    db: Session = Depends(deps.get_db)
):
    """Trigger manual sync."""
    auth = db.query(GoogleDriveAuth).filter(
        GoogleDriveAuth.user_id == current_user.id
    ).first()
    
    if not auth or not auth.enabled:
        raise HTTPException(status_code=404, detail="Google Drive not connected")
    
    service = GoogleDriveService(db)
    try:
        files = await service.sync_files(auth, folder_id=folder_id)
        return {"message": f"Found {len(files)} files", "files_found": len(files)}
    except Exception as e:
        logger.error(f"Sync error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@integrations_router.get("/google-drive/files", response_model=PaginatedGoogleDriveFilesResponse)
async def list_google_drive_files(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    filter_status: Optional[str] = Query(None),
    current_user: models.User = admin_security,
    db: Session = Depends(deps.get_db)
):
    auth = db.query(GoogleDriveAuth).filter(
        GoogleDriveAuth.user_id == current_user.id
    ).first()
    
    if not auth:
        raise HTTPException(status_code=404, detail="Google Drive not connected")
    
    service = GoogleDriveService(db)
    # Service now returns (items, total) tuple
    files, total = service.list_pages(auth.id, current_user.id, skip, limit, filter_status)
    
    return PaginatedGoogleDriveFilesResponse(
        items=files,
        total=total,
        skip=skip,
        limit=limit
    )

@integrations_router.post("/google-drive/files/select")
async def select_google_drive_files(
    selection_data: GoogleDriveFileSelectionRequest,
    current_user: models.User = admin_security,
    db: Session = Depends(deps.get_db)
):
    auth = db.query(GoogleDriveAuth).filter(
        GoogleDriveAuth.user_id == current_user.id
    ).first()
    
    if not auth:
        raise HTTPException(status_code=404, detail="Google Drive not connected")
    
    service = GoogleDriveService(db)
    success = service.update_file_selection(
        auth.id, current_user.id, selection_data.file_ids, selection_data.is_selected
    )
    
    if success:
        return {"message": "Selection updated", "files_updated": len(selection_data.file_ids)}
    raise HTTPException(status_code=404, detail="Update failed")

@integrations_router.post("/google-drive/files/delete")
async def delete_google_drive_files_ingest(
    deletion: GoogleDriveFileDelete,
    current_user: models.User = admin_security,
    db: Session = Depends(deps.get_db)
):
    """Delete ingested documents for specific files."""
    auth = db.query(GoogleDriveAuth).filter(
        GoogleDriveAuth.user_id == current_user.id
    ).first()
    
    if not auth:
        raise HTTPException(status_code=404, detail="Google Drive not connected")
    
    service = GoogleDriveService(db)
    count = await service.delete_ingested_files(auth.id, deletion.file_ids)
    return {"message": f"Deleted {count} ingested documents", "count": count}

@integrations_router.post("/google-drive/ingest")
async def ingest_selected_google_drive_files(
    current_user: models.User = admin_security,
    db: Session = Depends(deps.get_db)
):
    auth = db.query(GoogleDriveAuth).filter(
        GoogleDriveAuth.user_id == current_user.id
    ).first()
    
    if not auth or not auth.enabled:
        raise HTTPException(status_code=404, detail="Google Drive not connected")
    
    service = GoogleDriveService(db)
    try:
        count = await service.ingest_selected_files(auth)
        return {"message": f"Ingested {count} files", "files_ingested": count}
    except Exception as e:
        logger.error(f"Ingestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@integrations_router.delete("/google-drive/disconnect")
async def disconnect_google_drive(
    current_user: models.User = admin_security,
    db: Session = Depends(deps.get_db)
):
    service = GoogleDriveService(db)
    try:
        if await service.disconnect(current_user.id):
            return {"message": "Disconnected successfully"}
        raise HTTPException(status_code=404, detail="Connection not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@integrations_router.post("/google-drive/config")
async def update_google_drive_config(
    config_data: GoogleDriveConfigUpdate,
    current_user: models.User = admin_security,
    db: Session = Depends(deps.get_db)
):
    """Update Google Drive configuration (e.g. sync frequency)."""
    auth = db.query(GoogleDriveAuth).filter(
        GoogleDriveAuth.user_id == current_user.id
    ).first()
    
    if not auth:
        raise HTTPException(status_code=404, detail="Google Drive not connected")
    
    service = GoogleDriveService(db)
    
    # Map friendly names to cron if needed, or assume raw cron
    cron = config_data.sync_frequency
    if cron == "daily":
        cron = "0 0 * * *"
    elif cron == "weekly":
        cron = "0 0 * * 0"
    
    service.update_sync_frequency(auth, cron)
    return {"message": "Configuration updated", "sync_frequency": cron}

# Website Crawl endpoints
@integrations_router.post("/website-crawl", response_model=WebsiteCrawlConfigResponse)
async def create_website_crawl_config(
    config_data: WebsiteCrawlConfigCreate,
    current_user: models.User = admin_security,
    db: Session = Depends(deps.get_db)
):
    """Create a new website crawl configuration."""
    service = WebsiteCrawlService(db)
    
    config = service.create_config(
        user_id=current_user.id,
        name=config_data.name,
        base_url=config_data.base_url,
        max_depth=config_data.max_depth,
        url_patterns=config_data.url_patterns,
        crawl_frequency=config_data.crawl_frequency
    )
    
    return config


@integrations_router.get("/website-crawl", response_model=List[WebsiteCrawlConfigResponse])
async def list_website_crawl_configs(
    current_user: models.User = admin_security,
    db: Session = Depends(deps.get_db)
):
    """List all website crawl configurations."""
    service = WebsiteCrawlService(db)
    return service.list_configs(current_user.id)


@integrations_router.get("/website-crawl/{config_id}", response_model=WebsiteCrawlConfigResponse)
async def get_website_crawl_config(
    config_id: int,
    current_user: models.User = admin_security,
    db: Session = Depends(deps.get_db)
):
    """Get a specific website crawl configuration."""
    service = WebsiteCrawlService(db)
    config = service.get_config(config_id, current_user.id)
    
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Configuration not found"
        )
    
    return config


@integrations_router.put("/website-crawl/{config_id}", response_model=WebsiteCrawlConfigResponse)
async def update_website_crawl_config(
    config_id: int,
    config_data: WebsiteCrawlConfigCreate,
    current_user: models.User = admin_security,
    db: Session = Depends(deps.get_db)
):
    """Update a website crawl configuration."""
    service = WebsiteCrawlService(db)
    
    config = service.update_config(
        config_id=config_id,
        user_id=current_user.id,
        name=config_data.name,
        base_url=config_data.base_url,
        max_depth=config_data.max_depth,
        url_patterns=config_data.url_patterns,
        crawl_frequency=config_data.crawl_frequency
    )
    
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Configuration not found"
        )
    
    return config


@integrations_router.delete("/website-crawl/{config_id}")
async def delete_website_crawl_config(
    config_id: int,
    current_user: models.User = admin_security,
    db: Session = Depends(deps.get_db)
):
    """Delete a website crawl configuration."""
    service = WebsiteCrawlService(db)
    
    if await service.delete_config(config_id, current_user.id):
        return {"message": "Configuration deleted successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Configuration not found"
        )


@integrations_router.post("/website-crawl/{config_id}/scan")
@integrations_router.post("/website-crawl/{config_id}/crawl")
async def scan_website(
    config_id: int,
    max_pages: int = Query(100, description="Maximum pages to scan"),
    current_user: models.User = admin_security,
    db: Session = Depends(deps.get_db)
):
    """Scan a website to discover URLs (no ingestion)."""
    service = WebsiteCrawlService(db)
    config = service.get_config(config_id, current_user.id)
    
    if not config or not config.enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Configuration not found or disabled"
        )
    
    try:
        pages = await service.scan_website(config, max_pages=max_pages)
        return {
            "message": f"Scanned {len(pages)} new pages successfully",
            "pages_found": len(pages)
        }
    except Exception as e:
        logger.error(f"Scan error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scan failed: {str(e)}"
        )


@integrations_router.get("/website-crawl/{config_id}/pages", response_model=PaginatedWebsitePagesResponse)
async def list_website_pages(
    config_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    filter_status: Optional[str] = Query(None),
    search_query: Optional[str] = Query(None),
    current_user: models.User = admin_security,
    db: Session = Depends(deps.get_db)
):
    """List discovered pages for a configuration."""
    service = WebsiteCrawlService(db)
    items, total = service.list_pages(config_id, current_user.id, skip, limit, filter_status, search_query)
    return {
        "items": items,
        "total": total,
        "skip": skip,
        "limit": limit
    }


@integrations_router.post("/website-crawl/{config_id}/pages/select")
async def select_website_pages(
    config_id: int,
    selection_data: PageSelectionRequest,
    current_user: models.User = admin_security,
    db: Session = Depends(deps.get_db)
):
    """Update selection status for pages."""
    service = WebsiteCrawlService(db)
    
    if service.update_page_selection(config_id, current_user.id, selection_data.page_ids, selection_data.is_selected):
        return {"message": "Page selection updated"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Configuration not found"
        )


@integrations_router.post("/website-crawl/{config_id}/ingest")
async def ingest_selected_pages(
    config_id: int,
    current_user: models.User = admin_security,
    db: Session = Depends(deps.get_db)
):
    """Trigger ingestion for selected pages."""
    service = WebsiteCrawlService(db)
    config = service.get_config(config_id, current_user.id)
    
    if not config or not config.enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Configuration not found or disabled"
        )
    
    try:
        count = await service.ingest_selected_pages(config)
        return {
            "message": f"Successfully ingested {count} pages",
            "pages_ingested": count
        }
    except Exception as e:
        logger.error(f"Ingestion error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )
