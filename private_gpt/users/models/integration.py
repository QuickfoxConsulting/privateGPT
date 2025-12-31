"""Integration models for Google Drive and Website Crawling."""

from datetime import datetime, timezone
from typing import Optional
from private_gpt.users.db.base_class import Base
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    String,
    Text,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship


class GoogleDriveAuth(Base):
    """Store Google OAuth credentials for Drive integration."""

    __tablename__ = "google_drive_auth"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # User-provided OAuth app credentials
    client_id = Column(Text, nullable=False)
    client_secret = Column(Text, nullable=False)
    
    # Encrypted OAuth tokens
    access_token = Column(Text, nullable=False)
    refresh_token = Column(Text, nullable=False)
    token_expiry = Column(DateTime(timezone=True), nullable=False)
    
    # Configuration
    folder_ids = Column(JSONB, nullable=True, default=[])  # List of folder IDs to sync
    enabled = Column(Boolean, default=True, nullable=False)
    
    # Sync tracking
    last_sync = Column(DateTime(timezone=True), nullable=True)
    sync_status = Column(String(50), default="pending")  # pending, syncing, completed, failed
    sync_frequency = Column(String(50), nullable=True) # daily, weekly, etc.
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationship
    user = relationship("User", backref="google_drive_auths")
    files = relationship("GoogleDriveFile", backref="auth", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<GoogleDriveAuth(id={self.id}, user_id={self.user_id}, enabled={self.enabled})>"


class GoogleDriveFile(Base):
    """Store specific files discovered during a Google Drive sync."""
    
    __tablename__ = "google_drive_file"
    
    id = Column(Integer, primary_key=True, index=True)
    auth_id = Column(Integer, ForeignKey("google_drive_auth.id", ondelete="CASCADE"), nullable=False)
    
    file_id = Column(String(512), nullable=False)  # Google Drive file ID
    name = Column(String(1024), nullable=False)
    mime_type = Column(String(255), nullable=True)
    size = Column(Integer, nullable=True)
    
    # Folder hierarchy
    parent_folder_id = Column(String(512), nullable=True, index=True)  # Parent folder's Google Drive ID
    parent_folder_name = Column(String(1024), nullable=True)  # Parent folder name for display
    is_folder = Column(Boolean, default=False, nullable=False)  # True if this is a folder, not a file
    
    # Selection status
    is_selected = Column(Boolean, default=False, nullable=False)
    
    # Ingestion status
    ingest_status = Column(String(50), default="pending") # pending, ingested, error
    last_ingested_at = Column(DateTime(timezone=True), nullable=True)
    modified_at = Column(DateTime(timezone=True), nullable=True) # Last modified time from Google Drive
    error_message = Column(Text, nullable=True)
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    
    def __repr__(self):
        return f"<GoogleDriveFile(id={self.id}, name={self.name}, selected={self.is_selected})>"


class WebsiteCrawlConfig(Base):
    """Store website crawl configurations."""

    __tablename__ = "website_crawl_config"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Configuration
    name = Column(String(255), nullable=False)
    base_url = Column(String(2048), nullable=False)
    max_depth = Column(Integer, default=3, nullable=False)
    
    # URL patterns (include/exclude rules)
    url_patterns = Column(JSONB, nullable=True, default={
        "include": [],
        "exclude": []
    })
    
    # Crawl scheduling
    crawl_frequency = Column(String(100), nullable=True)  # Cron expression (e.g., "0 0 * * *")
    enabled = Column(Boolean, default=True, nullable=False)
    
    # Crawl tracking
    last_crawl = Column(DateTime(timezone=True), nullable=True)
    crawl_status = Column(String(50), default="pending")  # pending, scanning, crawling, completed, failed
    pages_crawled = Column(Integer, default=0)
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationship
    user = relationship("User", backref="website_crawl_configs")
    pages = relationship("WebsiteCrawlPage", backref="config", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<WebsiteCrawlConfig(id={self.id}, name={self.name}, base_url={self.base_url})>"


class WebsiteCrawlPage(Base):
    """Store specific pages discovered during a crawl scan."""
    
    __tablename__ = "website_crawl_page"
    
    id = Column(Integer, primary_key=True, index=True)
    config_id = Column(Integer, ForeignKey("website_crawl_config.id", ondelete="CASCADE"), nullable=False)
    
    url = Column(String(2048), nullable=False)
    title = Column(String(1024), nullable=True)
    
    # Selection status
    is_selected = Column(Boolean, default=False, nullable=False)
    
    # Ingestion status
    ingest_status = Column(String(50), default="pending") # pending, ingested, error
    last_ingested_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    
    def __repr__(self):
        return f"<WebsiteCrawlPage(id={self.id}, url={self.url}, selected={self.is_selected})>"


class Integration(Base):
    """Generic model to track installed integrations and their settings."""

    __tablename__ = "integration"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Integration Identity
    integration_id = Column(String(100), nullable=False) # e.g. "google_drive", "office365"
    
    # Status & Config
    is_active = Column(Boolean, default=True, nullable=False)
    settings = Column(JSONB, nullable=True, default={})
    
    # Audit
    installed_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationship
    user = relationship("User", backref="installed_integrations")

    def __repr__(self):
        return f"<Integration(integration_id={self.integration_id}, user_id={self.user_id}, active={self.is_active})>"
