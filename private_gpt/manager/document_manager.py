import uuid
import shutil
import logging
import aiofiles

from pathlib import Path
from typing import Optional, Tuple
from fastapi import UploadFile, HTTPException, status

logger = logging.getLogger(__name__)

class DocumentManager:
    def __init__(
        self,
        base_upload_dir: Path,
        allowed_extensions: set[str],
        max_file_size: int,
    ):
        self.base_upload_dir = Path(base_upload_dir)
        self.temp_dir = self.base_upload_dir / "temp"
        self.final_dir = self.base_upload_dir / "documents"
        self.allowed_extensions = allowed_extensions
        self.max_file_size = max_file_size
        
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)

    async def _validate_file(self, file: UploadFile) -> Tuple[bool, str]:
        """Validate file extension and size."""
        if not file.filename:
            return False, "No filename provided"
            
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in self.allowed_extensions:
            return False, f"File type not allowed. Allowed types: {', '.join(self.allowed_extensions)}"
        
        # Check file size
        total_size = 0
        while chunk := await file.read(1024 * 1024):  # Read in 1MB chunks
            total_size += len(chunk)
            if total_size > self.max_file_size:
                size_mb = self.max_file_size / (1024 * 1024)
                return False, f"File too large. Maximum size allowed: {size_mb:.1f}MB"
        await file.seek(0)
        
        return True, ""

    def _sanitize_filename(self, filename: str) -> str:
        """Replace spaces with underscores in filenames."""
        return filename.replace(" ", "_")

    def _generate_version_filename(self, original_filename: str, version: int) -> str:
        """Generate versioned filename while preserving original name."""
        sanitized_name = self._sanitize_filename(original_filename)
        stem = Path(sanitized_name).stem
        suffix = Path(sanitized_name).suffix
        return f"{stem}_v{version}{suffix}"

    async def _save_file(self, file: UploadFile, destination: Path) -> bool:
        """Save uploaded file with chunked reading."""
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(destination, 'wb') as out_file:
                while chunk := await file.read(1024 * 1024):  # 1MB chunks
                    await out_file.write(chunk)
            await file.seek(0)  # Reset file pointer after reading
            return True
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            return False

    async def save_temp_file(self, file: UploadFile) -> Tuple[Path, str]:
        """Save file to temporary location."""
        is_valid, error_message = await self._validate_file(file)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_message
            )

        sanitized_filename = self._sanitize_filename(file.filename)
        temp_filename = f"{uuid.uuid4()}_{sanitized_filename}"
        temp_path = self.temp_dir / temp_filename

        if not await self._save_file(file, temp_path):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save temporary file"
            )

        return temp_path, sanitized_filename

    async def approve_document(
        self,
        document_id: int,
        temp_path: Path,
        version: int,
        original_filename: str  # Pass original filename explicitly
    ) -> Path:
        """Move approved document to final location with versioning."""
        try:
            # doc_dir = self.final_dir / str(document_id)
            doc_dir = self.final_dir
            doc_dir.mkdir(parents=True, exist_ok=True)
            versioned_filename = self._generate_version_filename(original_filename, version)
            final_path = doc_dir / versioned_filename

            shutil.move(str(temp_path), str(final_path))
            return final_path, versioned_filename

        except Exception as e:
            logger.error(f"Error approving document: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to approve document"
            )

    async def reject_document(self, temp_path: Path):
        """Remove rejected document."""
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception as e:
            logger.error(f"Error deleting rejected file: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete rejected file"
            )
