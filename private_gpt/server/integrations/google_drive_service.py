"""Google Drive OAuth and document ingestion service"""

import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Any, Callable, Dict, Tuple
from cryptography.fernet import Fernet
import os
import time
import asyncio
import functools
import tempfile
from pathlib import Path

# Tenacity imports for robust retries
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
)

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request

from sqlalchemy.orm import Session
from private_gpt.users import crud, models, schemas
from private_gpt.users.models.integration import GoogleDriveAuth, GoogleDriveFile
from private_gpt.users.models.enums import MakerCheckerStatus, MakerCheckerActionType, DocumentStatus
from private_gpt.users.api.deps import create_document_manager
from private_gpt.di import global_injector
from private_gpt.users.core.config import settings
from private_gpt.server.ingest.ingest_router import ingest
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from private_gpt.server.integrations.base import BaseIntegration, IntegrationMetadata

logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler()
if not scheduler.running:
    scheduler.start()
    logger.info("Google Drive Scheduler started")

# Constants
SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.metadata.readonly",
]
OAUTH_AUTH_URI = "https://accounts.google.com/o/oauth2/auth"
OAUTH_TOKEN_URI = "https://oauth2.googleapis.com/token"
DRIVE_V3_VERSION = "v3"
DRIVE_V3_SERVICE = "drive"

DEFAULT_PAGE_SIZE = 100
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1  # seconds
MAX_RETRY_DELAY = 15  # seconds
RATE_LIMIT_DELAY = 0.5  # seconds between API calls

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50MB default limit

# Map Google Workspace types to export MIME types
EXPORT_MIME_MAP = {
    "application/vnd.google-apps.document": "application/pdf",
    "application/vnd.google-apps.spreadsheet": "application/pdf", 
    "application/vnd.google-apps.presentation": "application/pdf",
}

# Supported ingestion types (Native + Google Workspace + Folders)
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
SUPPORTED_MIME_TYPES = [
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    FOLDER_MIME_TYPE,  # Include folders for hierarchy browsing
] + list(EXPORT_MIME_MAP.keys())


def ensure_utc_aware(dt: Optional[datetime]) -> Optional[datetime]:
    """Ensure a datetime is timezone-aware (UTC).
    
    Handles comparison between offset-naive (from DB) and offset-aware datetimes.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def sanitize_error_message(error: Exception) -> str:
    """Sanitize error messages to remove potentially sensitive data."""
    error_str = str(error)
    # Remove common sensitive patterns
    sensitive_patterns = [
        ("token", "[REDACTED]"),
        ("secret", "[REDACTED]"),
        ("key", "[REDACTED]"),
        ("password", "[REDACTED]"),
        ("credential", "[REDACTED]"),
    ]
    sanitized = error_str
    for pattern, replacement in sensitive_patterns:
        if pattern.lower() in sanitized.lower():
            # Only redact if it looks like a value is present (has = or : followed by content)
            import re
            sanitized = re.sub(
                rf'({pattern}["\']?\s*[:=]\s*["\']?)([^"\'\s,}}]+)',
                rf'\1{replacement}',
                sanitized,
                flags=re.IGNORECASE
            )
    return sanitized


def is_retryable_error(e: BaseException) -> bool:
    """Predicate to determine if an exception should trigger a retry."""
    if isinstance(e, HttpError):
        # Retry on:
        # 408: Request Timeout
        # 429: Too Many Requests (Rate Limit)
        # 500: Internal Server Error
        # 502: Bad Gateway
        # 503: Service Unavailable
        # 504: Gateway Timeout
        return e.resp.status in [408, 429, 500, 502, 503, 504]
    
    # Check for common network/rate limit strings in other exceptions
    error_str = str(e).lower()
    retry_keywords = [
        "rate limit", 
        "quota exceeded", 
        "timeout", 
        "connection error",
        "connection reset",
        "broken pipe"
    ]
    return any(k in error_str for k in retry_keywords)


class TokenEncryption:
    """Helper class for token encryption and decryption."""

    def __init__(self) -> None:
        self.encryption_key = settings.INTEGRATION_ENCRYPTION_KEY
        self._cipher_suite: Optional[Fernet] = None

    @property
    def cipher_suite(self) -> Fernet:
        """Get the cipher suite, initializing it if necessary."""
        if self._cipher_suite is None:
            if not self.encryption_key:
                logger.error("INTEGRATION_ENCRYPTION_KEY is not set.")
                raise ValueError(
                    "INTEGRATION_ENCRYPTION_KEY must be set for token encryption/decryption"
                )
            try:
                key = (
                    self.encryption_key.encode()
                    if isinstance(self.encryption_key, str)
                    else self.encryption_key
                )
                self._cipher_suite = Fernet(key)
            except Exception as e:
                logger.error(f"Failed to initialize encryption cipher: {str(e)}")
                raise ValueError(f"Invalid INTEGRATION_ENCRYPTION_KEY: {str(e)}") from e
        return self._cipher_suite

    def encrypt(self, data: str) -> str:
        """Encrypt string data."""
        if not data:
            return ""
        return self.cipher_suite.encrypt(data.encode()).decode()

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data."""
        if not encrypted_data:
            return ""
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()


token_encryption = TokenEncryption()


# -------------------------------------------------------------------------
# Worker Functions (Run in Thread Pool)
# These MUST NOT touch self.db or SQLAlchemy objects directly
# -------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=INITIAL_RETRY_DELAY, max=MAX_RETRY_DELAY),
    retry=retry_if_exception(is_retryable_error),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True
)
def _list_files_worker(
    credentials_dict: dict,
    folder_id: Optional[str],
    page_size: int,
    page_token: Optional[str]
) -> tuple[List[dict], Optional[str]]:
    """Worker function to list files. Runs in a separate thread."""
    
    # Reconstruct credentials from dictionary to avoid pickling issues/thread safety
    creds = Credentials(**credentials_dict)
    
    # cache_discovery=False is CRITICAL to avoid 403 Unregistered Caller errors
    service = build(
        DRIVE_V3_SERVICE, DRIVE_V3_VERSION, credentials=creds, cache_discovery=False
    )

    query = "trashed=false"
    if folder_id:
        query += f" and '{folder_id}' in parents"

    mime_conditions = [f"mimeType='{mt}'" for mt in SUPPORTED_MIME_TYPES]
    if mime_conditions:
        query += f" and ({' or '.join(mime_conditions)})"

    request_params = {
        "q": query,
        "pageSize": page_size,
        "fields": "nextPageToken, files(id, name, mimeType, modifiedTime, size)",
    }
    
    if page_token:
        request_params["pageToken"] = page_token

    results = service.files().list(**request_params).execute()
    time.sleep(RATE_LIMIT_DELAY) # Polite rate limiting inside worker
    
    return results.get("files", []), results.get("nextPageToken")


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=INITIAL_RETRY_DELAY, max=MAX_RETRY_DELAY),
    retry=retry_if_exception(is_retryable_error),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True
)
def _download_file_worker(credentials_dict: dict, file_id: str) -> tuple[str, str]:
    """Worker function to download files. Runs in a separate thread."""
    
    creds = Credentials(**credentials_dict)
    service = build(
        DRIVE_V3_SERVICE, DRIVE_V3_VERSION, credentials=creds, cache_discovery=False
    )

    # Get Metadata
    file_metadata = service.files().get(fileId=file_id, fields="name,mimeType,size").execute()
    file_name = file_metadata.get("name", "unknown_file")
    mime_type = file_metadata.get("mimeType")
    
    # Size Check (if available)
    if "size" in file_metadata:
        file_size = int(file_metadata["size"])
        if file_size > MAX_FILE_SIZE_BYTES:
            raise ValueError(f"File too large: {file_size} bytes")

    # Determine Download Method
    request = None
    if mime_type in EXPORT_MIME_MAP:
        export_mime = EXPORT_MIME_MAP[mime_type]
        request = service.files().export_media(fileId=file_id, mimeType=export_mime)
        if not file_name.lower().endswith(".pdf"):
            file_name = f"{file_name}.pdf"
    else:
        request = service.files().get_media(fileId=file_id)

    # Stream to Temp File
    fd, tmp_path = tempfile.mkstemp(suffix=f"_{file_name}")
    os.close(fd) 
    
    try:
        with open(tmp_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if f.tell() > MAX_FILE_SIZE_BYTES:
                    raise ValueError(f"Download limit exceeded")
                if status:
                    logger.debug(f"Download progress: {int(status.progress() * 100)}%")
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    time.sleep(RATE_LIMIT_DELAY)
    return tmp_path, file_name


class GoogleDriveService(BaseIntegration):
    """Service for Google Drive OAuth and document management."""

    def __init__(self, db: Session) -> None:
        """Initialize the Google Drive service."""
        BaseIntegration.__init__(self)  # Changed: Init from BaseIntegration if it had init, though it's ABC
        self.db = db

    @property
    def metadata(self) -> IntegrationMetadata:
        return IntegrationMetadata(
            id="google_drive",
            name="Google Drive",
            description="Sync and ingest files from your Google Drive.",
            icon="google-drive",  # We'll map this in frontend
            is_configurable=True,
            settings_path="/admin/google-drive"
        )
    
    def is_installed(self, user_id: int) -> bool:
        """Check if integration is active in the registry."""
        from private_gpt.users.models.integration import Integration
        record = self.db.query(Integration).filter(
            Integration.user_id == user_id,
            Integration.integration_id == self.metadata.id,
            Integration.is_active == True
        ).first()
        return bool(record)

    def install(self, user_id: int) -> bool:
        """
        'Installing' creates the placeholder auth record if missing.
        Actual OAuth Setup happens via the /setup endpoint.
        """
        auth = self.db.query(GoogleDriveAuth).filter(
            GoogleDriveAuth.user_id == user_id
        ).first()
        
        if not auth:
            # Create a placeholder record to mark as 'intent to install'
            # Users still need to provide ClientID/Secret via configure
            auth = GoogleDriveAuth(
                user_id=user_id,
                client_id="",
                client_secret="",
                access_token="",
                refresh_token="",
                token_expiry=datetime.now(timezone.utc),
                enabled=False # Will optionally be True after setup
            )
            self.db.add(auth)
            self.db.commit()
        return True

    def uninstall(self, user_id: int) -> bool:
        """Disable and disconnect."""
        return asyncio.run(self.disconnect(user_id))

    def get_config(self, user_id: int) -> Optional[Dict[str, Any]]:
        auth = self.db.query(GoogleDriveAuth).filter(
            GoogleDriveAuth.user_id == user_id
        ).first()
        if not auth:
            return None
        return {
            "sync_status": auth.sync_status,
            "last_sync": auth.last_sync,
            "file_count": self.db.query(GoogleDriveFile).filter(GoogleDriveFile.auth_id == auth.id).count()
        }

    async def _run_in_executor(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Run a blocking function in the default executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

    def create_oauth_flow(
        self, client_id: str, client_secret: str, redirect_uri: str
    ) -> Flow:
        client_config = {
            "web": {
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uris": [redirect_uri],
                "auth_uri": OAUTH_AUTH_URI,
                "token_uri": OAUTH_TOKEN_URI,
            }
        }
        return Flow.from_client_config(
            client_config, scopes=SCOPES, redirect_uri=redirect_uri
        )

    def get_authorization_url(
        self, client_id: str, client_secret: str, redirect_uri: str, state: Optional[str] = None
    ) -> str:
        flow = self.create_oauth_flow(client_id, client_secret, redirect_uri)
        auth_params = {
            "access_type": "offline",
            "include_granted_scopes": "true",
            "prompt": "consent"
        }
        if state:
            auth_params["state"] = state
        auth_url, _ = flow.authorization_url(**auth_params)
        return auth_url

    def exchange_code_for_tokens(
        self, code: str, client_id: str, client_secret: str, redirect_uri: str, user_id: int
    ) -> GoogleDriveAuth:
        flow = self.create_oauth_flow(client_id, client_secret, redirect_uri)
        flow.fetch_token(code=code)
        credentials = flow.credentials

        if not credentials.refresh_token:
            logger.warning("No refresh token received.")

        encrypted_access = token_encryption.encrypt(credentials.token)
        encrypted_refresh = token_encryption.encrypt(credentials.refresh_token) if credentials.refresh_token else None
        
        # Encrypt Client Secrets (BYO-Key requirement)
        encrypted_cid = token_encryption.encrypt(client_id)
        encrypted_csecret = token_encryption.encrypt(client_secret)

        expiry = credentials.expiry
        if expiry and expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)

        existing_auth = (
            self.db.query(GoogleDriveAuth)
            .filter(GoogleDriveAuth.user_id == user_id)
            .first()
        )

        if existing_auth:
            existing_auth.access_token = encrypted_access
            if encrypted_refresh:
                existing_auth.refresh_token = encrypted_refresh
            # Update secrets in case they changed
            existing_auth.client_id = encrypted_cid
            existing_auth.client_secret = encrypted_csecret
            existing_auth.token_expiry = expiry
            existing_auth.enabled = True
            existing_auth.updated_at = datetime.now(timezone.utc)
            self.db.commit()
            self.db.refresh(existing_auth)
            return existing_auth
        else:
            if not encrypted_refresh:
                 raise ValueError("No refresh token received. Revoke access and try again.")

            auth = GoogleDriveAuth(
                user_id=user_id,
                client_id=encrypted_cid,
                client_secret=encrypted_csecret,
                access_token=encrypted_access,
                refresh_token=encrypted_refresh,
                token_expiry=expiry,
                enabled=True,
            )
            self.db.add(auth)
            self.db.commit()
            self.db.refresh(auth)
            return auth

    def _get_fresh_credentials_dict(self, auth_id: int) -> dict:
        """
        Retrieves valid credentials. Refreshes if expired.
        
        CRITICAL: This runs in the MAIN thread (Sync).
        It handles the DB Lock and Refresh to prevent race conditions.
        It returns a plain dict safe to pass to threads.
        """
        # 1. Lock Row
        auth = (
            self.db.query(GoogleDriveAuth)
            .filter(GoogleDriveAuth.id == auth_id)
            .with_for_update() # Locks until commit
            .first()
        )

        if not auth:
            raise ValueError(f"Auth {auth_id} not found")

        # 2. Decrypt
        access_token = token_encryption.decrypt(auth.access_token)
        refresh_token = token_encryption.decrypt(auth.refresh_token)
        client_id = token_encryption.decrypt(auth.client_id)
        client_secret = token_encryption.decrypt(auth.client_secret)

        expiry = auth.token_expiry
        if expiry and expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)

        # Google Auth library typically expects naive UTC datetime for expiry
        # or compares it with naive utcnow(), causing TypeError if aware.
        # So we pass a naive expiry to Credentials, but ensure its value is correct UTC.
        credentials_expiry = expiry
        if credentials_expiry and credentials_expiry.tzinfo is not None:
             credentials_expiry = credentials_expiry.astimezone(timezone.utc).replace(tzinfo=None)

        # 3. Create Object
        credentials = Credentials(
            token=access_token,
            refresh_token=refresh_token,
            token_uri=OAUTH_TOKEN_URI,
            client_id=client_id,
            client_secret=client_secret,
            scopes=SCOPES,
            expiry=credentials_expiry
        )

        # 4. Check & Refresh
        should_refresh = False
        if not credentials.valid:
            should_refresh = True
        elif expiry and datetime.now(timezone.utc) + timedelta(minutes=5) > ensure_utc_aware(expiry):
            should_refresh = True

        if should_refresh and refresh_token:
            try:
                logger.info(f"Refreshing Google Drive token for auth_id {auth.id}")
                # This blocks main thread briefly (~200ms). Acceptable for correctness.
                credentials.refresh(Request())

                # Update DB
                auth.access_token = token_encryption.encrypt(credentials.token)
                auth.token_expiry = credentials.expiry
                auth.updated_at = datetime.now(timezone.utc)
                self.db.commit() # Releases Lock
                
            except Exception as e:
                logger.error(f"Refresh failed: {sanitize_error_message(e)}")
                self.db.rollback()
                raise
        else:
            # Commit to release the lock if no update happened
            self.db.commit()

        # 5. Return serializable dict for threads
        return {
            "token": credentials.token,
            "refresh_token": credentials.refresh_token,
            "token_uri": credentials.token_uri,
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
            "scopes": credentials.scopes,
        }

    async def list_files(
        self,
        auth_id: int, # Pass ID, not object
        folder_id: Optional[str] = None,
        page_size: int = DEFAULT_PAGE_SIZE,
        page_token: Optional[str] = None,
    ) -> tuple[List[dict], Optional[str]]:
        """Async wrapper for listing files."""
        
        # 1. DB Ops (Main Thread)
        creds_dict = self._get_fresh_credentials_dict(auth_id)
        
        # 2. API Ops (Worker Thread)
        return await self._run_in_executor(
            _list_files_worker, creds_dict, folder_id, page_size, page_token
        )

    async def download_file(self, auth_id: int, file_id: str) -> tuple[str, str]:
        """Async wrapper for downloading files."""
        
        # 1. DB Ops (Main Thread)
        creds_dict = self._get_fresh_credentials_dict(auth_id)
        
        # 2. API Ops (Worker Thread)
        return await self._run_in_executor(
            _download_file_worker, creds_dict, file_id
        )

    async def sync_files(
        self, auth: GoogleDriveAuth, folder_id: Optional[str] = None
    ) -> List[GoogleDriveFile]:
        """Sync files from Google Drive to database."""
        discovered_files = []
        page_token = None
        auth_id = auth.id # Capture ID for safe passing
        
        logger.info(f"Starting file sync for auth_id {auth_id}")

        while True:
            try:
                # Call Async Wrapper (handles DB/Thread split)
                files_data, next_page_token = await self.list_files(
                    auth_id, folder_id=folder_id, page_token=page_token
                )
                
                # Process DB updates (Main Thread)
                # We need to re-fetch 'auth' or be careful with the session.
                # Since we are in the main thread async loop, self.db is safe to use 
                # AS LONG AS no other thread is holding it.
                
                for file_info in files_data:
                    modified_time_str = file_info.get("modifiedTime")
                    modified_at = None
                    if modified_time_str:
                        try:
                            modified_at = datetime.fromisoformat(
                                modified_time_str.replace("Z", "+00:00")
                            )
                        except ValueError:
                            pass

                    existing_file = (
                        self.db.query(GoogleDriveFile)
                        .filter(
                            GoogleDriveFile.auth_id == auth_id,
                            GoogleDriveFile.file_id == file_info["id"],
                        )
                        .first()
                    )

                    size_val = int(file_info.get("size", 0))

                    if existing_file:
                        existing_file.name = file_info["name"]
                        existing_file.mime_type = file_info["mimeType"]
                        existing_file.size = size_val
                        existing_file.is_folder = file_info["mimeType"] == FOLDER_MIME_TYPE
                        existing_modified = ensure_utc_aware(existing_file.modified_at)
                        new_modified = ensure_utc_aware(modified_at)
                        if new_modified and (
                            not existing_modified
                            or new_modified > existing_modified
                        ):
                            existing_file.modified_at = modified_at
                            existing_file.ingest_status = "pending"
                        existing_file.updated_at = datetime.now(timezone.utc)
                        discovered_files.append(existing_file)
                    else:
                        new_file = GoogleDriveFile(
                            auth_id=auth_id,
                            file_id=file_info["id"],
                            name=file_info["name"],
                            mime_type=file_info["mimeType"],
                            size=size_val,
                            modified_at=modified_at,
                            is_folder=file_info["mimeType"] == FOLDER_MIME_TYPE,
                            is_selected=False,
                            ingest_status="pending" if file_info["mimeType"] != FOLDER_MIME_TYPE else "n/a",
                        )
                        self.db.add(new_file)
                        discovered_files.append(new_file)
                
                self.db.commit()
                
                if not next_page_token:
                    break
                page_token = next_page_token
                
            except Exception as e:
                logger.error(f"Error during file sync: {sanitize_error_message(e)}")
                # We need to be careful updating auth status if transaction failed
                try:
                    self.db.rollback()
                    err_auth = self.db.get(GoogleDriveAuth, auth_id)
                    if err_auth:
                        err_auth.sync_status = "error"
                        self.db.commit()
                except:
                    pass
                raise

        # Final Update
        final_auth = self.db.get(GoogleDriveAuth, auth_id)
        if final_auth:
            final_auth.last_sync = datetime.now(timezone.utc)
            final_auth.sync_status = "completed"
            self.db.commit()
            
        return discovered_files

    async def ingest_selected_files(self, auth: GoogleDriveAuth) -> int:
        """Ingest content from selected files."""
        auth_id = auth.id
        user_id = auth.user_id
        
        # Initial status update
        auth.sync_status = "ingesting"
        self.db.commit()

        from private_gpt.server.ingest.ingest_service import IngestService
        ingest_service = global_injector.get(IngestService)
        # Get document manager for persistence
        doc_manager = create_document_manager()

        files_to_ingest = (
            self.db.query(GoogleDriveFile)
            .filter(
                GoogleDriveFile.auth_id == auth_id, 
                GoogleDriveFile.is_selected == True
            )
            .all()
        )

        count = 0
        for file_record in files_to_ingest:
            last_ingested = ensure_utc_aware(file_record.last_ingested_at)
            modified = ensure_utc_aware(file_record.modified_at)
            if (
                file_record.ingest_status == "ingested"
                and last_ingested
                and modified
                and last_ingested >= modified
            ):
                continue

            tmp_path = None
            try:
                doc_id = f"google_drive_{file_record.file_id}"
                
                # Clean old data (Async)
                try:
                    await ingest_service.delete_by_metadata("doc_id", doc_id)
                except Exception:
                    pass

                # 1. Download to temporary path
                tmp_path, file_name = await self.download_file(auth_id, file_record.file_id)
                tmp_path_obj = Path(tmp_path)

                # 2. Integrate with main Document system
                # Check if document already exists for this Google Drive file
                document = self.db.query(models.Document).filter(
                    models.Document.doc_metadata["file_id"].astext == file_record.file_id
                ).first()

                if not document:
                    # Create new document record
                    doc_in = schemas.DocumentMakerCreate(
                        filename=file_record.name,
                        uploaded_by=user_id,
                        doc_metadata={
                            "type": "google_drive",
                            "file_id": file_record.file_id,
                            "google_drive_auth_id": str(auth_id),
                        },
                        doc_status=DocumentStatus.INGESTING.value
                    )
                    document = crud.documents.create(self.db, obj_in=doc_in)
                    
                    # Associate with default department (1 = ALL)
                    self.db.execute(
                        models.document_department_association.insert().values(
                            document_id=document.id,
                            department_id=1
                        )
                    )
                    
                    # Create initial version entry
                    version_in = schemas.DocumentVersionCreate(
                        document_id=document.id,
                        version_number=1,
                        status=MakerCheckerStatus.APPROVED,
                        action_type=MakerCheckerActionType.INSERT,
                        file_path=str(tmp_path), # Placeholder path
                        uploaded_by=user_id,
                    )
                    version = crud.document_versions.create(self.db, obj_in=version_in)
                    document.current_version_id = version.id
                    self.db.commit()
                else:
                    document.doc_status = DocumentStatus.INGESTING.value
                    self.db.commit()

                # Refresh to ensure version info is loaded
                self.db.refresh(document)
                if not document.current_version:
                    raise ValueError(f"Document {document.id} missing current version")

                # 3. Persist to media directory
                final_path, _ = await doc_manager.approve_document(
                    document_id=document.id,
                    temp_path=tmp_path_obj,
                    version=document.current_version.version_number,
                    original_filename=file_record.name
                )
                
                # Success: tmp_path is moved to final_path
                tmp_path = None 

                # 4. Update Document and Version records
                document.verified = True
                document.verified_at = datetime.now(timezone.utc)
                document.verified_by = user_id
                
                version_update = schemas.DocumentVersionUpdate(
                    status=MakerCheckerStatus.APPROVED,
                    file_path=str(final_path),
                )
                crud.document_versions.update(self.db, db_obj=document.current_version, obj_in=version_update)

                # 5. Ingest into vector store from permanent path
                metadata = {
                    "type": "google_drive",
                    "file_id": file_record.file_id,
                    "original_name": file_record.name,
                    "google_drive_auth_id": str(auth_id),
                    "doc_id": doc_id,
                    "document_id": str(document.id),
                    "mime_type": file_record.mime_type or "unknown",
                    "modified_at": file_record.modified_at.isoformat() if file_record.modified_at else "unknown",
                    "size_bytes": str(file_record.size or 0),
                    "file_name": file_record.name,
                    "document_path": str(final_path),
                    "strategy": 'hierarchical',
                }

                await ingest_service.ingest_file(
                    file_name=file_name,
                    file_data=Path(final_path),
                    file_metadata=metadata,
                )
                
                file_record.ingest_status = "ingested"
                file_record.last_ingested_at = datetime.now(timezone.utc)
                file_record.error_message = None
                
                document.doc_status = DocumentStatus.READY.value
                
                self.db.commit()
                count += 1

            except Exception as e:
                logger.error(f"Ingestion failed for {file_record.name}: {sanitize_error_message(e)}")
                self.db.rollback()
                # Update status to error
                f = self.db.get(GoogleDriveFile, file_record.id)
                if f:
                    f.ingest_status = "error"
                    f.error_message = str(e)[:500]
                    self.db.commit()
                
                # Also update document if we created it
                try:
                    document = self.db.query(models.Document).filter(
                        models.Document.doc_metadata["file_id"].astext == file_record.file_id
                    ).first()
                    if document:
                        document.doc_status = DocumentStatus.ERROR.value
                        self.db.commit()
                except:
                    pass

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception as e:
                        logger.warning(f"Failed to remove temp file: {str(e)}")

        final_auth = self.db.get(GoogleDriveAuth, auth_id)
        if final_auth:
            final_auth.sync_status = "completed"
            self.db.commit()
            
        return count

    async def delete_ingested_files(self, auth_id: int, file_ids: List[int]) -> int:
        """Delete ingested documents for specific files."""
        if not file_ids:
            return 0

        from private_gpt.server.ingest.ingest_service import IngestService
        ingest_service = global_injector.get(IngestService)

        files = (
            self.db.query(GoogleDriveFile)
            .filter(
                GoogleDriveFile.auth_id == auth_id, 
                GoogleDriveFile.id.in_(file_ids),
                GoogleDriveFile.ingest_status != "n/a"
            )
            .all()
        )

        count = 0
        for file_record in files:
            try:
                # 1. Delete from vector store
                doc_id = f"google_drive_{file_record.file_id}"
                await ingest_service.delete_by_metadata("doc_id", doc_id)
                
                # 2. Delete associated Document and physical file
                document = self.db.query(models.Document).filter(
                    models.Document.doc_metadata["file_id"].astext == file_record.file_id
                ).first()
                if document:
                    # Remove physical file(s)
                    for version in document.versions:
                        if version.file_path and os.path.exists(version.file_path):
                            try:
                                os.remove(version.file_path)
                            except Exception as e:
                                logger.warning(f"Failed to remove file {version.file_path}: {e}")
                    
                    # Remove Document (cascade will handle versions)
                    crud.documents.remove(self.db, id=document.id)

                # 3. Update Google Drive record status
                file_record.ingest_status = "n/a"
                file_record.last_ingested_at = None
                file_record.error_message = None
                self.db.commit()
                count += 1
            except Exception as e:
                logger.error(f"Failed to delete document {file_record.name}: {str(e)}")
        
        return count

    async def disconnect(self, user_id: int) -> bool:
        """Disconnect Google Drive integration."""
        # Main Thread DB Operation
        auth = (
            self.db.query(GoogleDriveAuth)
            .filter(GoogleDriveAuth.user_id == user_id)
            .first()
        )

        if auth:
            from private_gpt.server.ingest.ingest_service import IngestService
            ingest_service = global_injector.get(IngestService)

            try:
                # 1. Clean up vector store
                await ingest_service.delete_by_metadata("google_drive_auth_id", str(auth.id))
            except Exception as e:
                logger.warning(f"Failed to delete by google_drive_auth_id: {str(e)}")

            # 2. Clean up Documents and physical files
            documents = self.db.query(models.Document).filter(
                models.Document.doc_metadata["google_drive_auth_id"].astext == str(auth.id)
            ).all()
            
            for document in documents:
                for version in document.versions:
                    if version.file_path and os.path.exists(version.file_path):
                        try:
                            os.remove(version.file_path)
                        except Exception as e:
                            logger.warning(f"Failed to remove file {version.file_path}: {e}")
                crud.documents.remove(self.db, id=document.id)

            # 3. Final cleanup of Drive records and Auth
            self.db.delete(auth)
            self.db.commit()
            return True
        return False

    def list_pages(
        self, auth_id: int, user_id: int, skip: int = 0, limit: int = 100, ingest_status: Optional[str] = None
    ) ->Tuple[List[GoogleDriveFile], int]:
        """List files for an auth (Synchronous, used by API endpoints)."""
        auth = (
            self.db.query(GoogleDriveAuth)
            .filter(GoogleDriveAuth.id == auth_id, GoogleDriveAuth.user_id == user_id)
            .first()
        )

        if not auth:
            return [], 0
            
        query = self.db.query(GoogleDriveFile).filter(GoogleDriveFile.auth_id == auth_id)
        
        if ingest_status:
             query = query.filter(GoogleDriveFile.ingest_status == ingest_status)

        total = query.count()
        items = query.offset(skip).limit(limit).all()
        
        return items, total

    def update_file_selection(
        self, auth_id: int, user_id: int, file_ids: List[int], is_selected: bool
    ) -> bool:
        """Update selection status for files."""
        auth = (
            self.db.query(GoogleDriveAuth)
            .filter(GoogleDriveAuth.id == auth_id, GoogleDriveAuth.user_id == user_id)
            .first()
        )

        if not auth:
            return False

        if not file_ids:
            return True

        self.db.query(GoogleDriveFile).filter(
            GoogleDriveFile.auth_id == auth_id, GoogleDriveFile.id.in_(file_ids)
        ).update(
            {"is_selected": is_selected, "updated_at": datetime.now(timezone.utc)}, 
            synchronize_session=False
        )

        self.db.commit()
        return True

    def get_ingested_files(self, department_id: int) -> List[GoogleDriveFile]:
        """Get all ingested and selected files for a department."""
        from private_gpt.users.models.user import User

        return (
            self.db.query(GoogleDriveFile)
            .join(GoogleDriveAuth, GoogleDriveFile.auth_id == GoogleDriveAuth.id)
            .join(User, GoogleDriveAuth.user_id == User.id)
            .filter(
                User.department_id == department_id,
                GoogleDriveAuth.enabled == True,
                GoogleDriveFile.is_selected == True,
                GoogleDriveFile.ingest_status == "ingested",
            )
            .all()
            .all()
        )

    def schedule_sync(self, auth_id: int, cron_expression: str):
        """Schedule a sync job."""
        job_id = f"gdrive_sync_{auth_id}"
        
        if scheduler.get_job(job_id):
            scheduler.remove_job(job_id)
        
        try:
            scheduler.add_job(
                self._run_scheduled_sync,
                CronTrigger.from_crontab(cron_expression),
                id=job_id,
                args=[auth_id],
                replace_existing=True
            )
            logger.info(f"Scheduled sync for auth {auth_id} with cron {cron_expression}")
        except Exception as e:
            logger.error(f"Failed to schedule sync for {auth_id}: {str(e)}")

    def remove_schedule(self, auth_id: int):
        """Remove a sync job."""
        job_id = f"gdrive_sync_{auth_id}"
        if scheduler.get_job(job_id):
            scheduler.remove_job(job_id)

    @staticmethod
    def _run_scheduled_sync(auth_id: int):
        """Wrapper to run scheduled sync in a new DB session."""
        import asyncio
        from private_gpt.users.db.session import SessionLocal
        
        db = SessionLocal()
        service = GoogleDriveService(db)
        
        try:
            auth = db.query(GoogleDriveAuth).filter(GoogleDriveAuth.id == auth_id).first()
            if auth and auth.enabled:
                logger.info(f"Starting scheduled sync for auth {auth_id}")
                # 1. Sync file list
                asyncio.run(service.sync_files(auth))
                # 2. Ingest new/updated selected files
                asyncio.run(service.ingest_selected_files(auth))
        except Exception as e:
            logger.error(f"Scheduled GDrive sync failed: {str(e)}")
        finally:
            db.close()

    def update_sync_frequency(self, auth: GoogleDriveAuth, frequency: Optional[str]):
        """Update sync frequency and schedule."""
        auth.sync_frequency = frequency
        self.db.commit()
        
        if frequency:
            self.schedule_sync(auth.id, frequency)
        else:
            self.remove_schedule(auth.id)