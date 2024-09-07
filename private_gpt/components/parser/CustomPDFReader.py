import fitz  # PyMuPDF
import camelot
import pytesseract
from PIL import Image
import io
from pathlib import Path
from typing import List, Dict, Optional
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from tenacity import retry, stop_after_attempt

RETRY_TIMES = 3

class CustomRAGPDFParser(BaseReader):
    """Custom PDF parser with OCR, table extraction, and metadata retrieval using PyMuPDF and Camelot."""

    def __init__(self, return_full_document: Optional[bool] = False) -> None:
        """Initialize CustomRAGPDFParser."""
        super().__init__()
        self.return_full_document = return_full_document

    @retry(stop=stop_after_attempt(RETRY_TIMES))
    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        """Parse PDF file using PyMuPDF."""
        if not isinstance(file, Path):
            file = Path(file)

        docs = []

        with fitz.open(file) as pdf:  
            num_pages = len(pdf)
            
            # Extract metadata
            metadata = pdf.metadata or {}
            metadata["file_name"] = file.name
            metadata["total_pages"] = num_pages
            
            # Merge with any additional information provided
            if extra_info is not None:
                metadata.update(extra_info)

            if self.return_full_document:
                text = self._extract_full_text(pdf)
                tables = self._extract_all_tables(file, num_pages)
                combined_content = f"Text Content:\n{text}\n\nTables:\n{tables}"

                docs.append(Document(text=combined_content, metadata=metadata))

            else:
                for page_num in range(num_pages):
                    page = pdf[page_num]
                    page_text = self._extract_page_text(page)
                    page_tables = self._extract_page_tables(file, page_num + 1)
                    combined_content = f"Text Content:\n{page_text}\n\nTables:\n{page_tables}"

                    page_metadata = {
                        "page_label": page_num + 1,
                    }
                    page_metadata.update(metadata)  # Include general metadata

                    docs.append(Document(text=combined_content, metadata=page_metadata))

        return docs

    def _extract_full_text(self, pdf) -> str:
        """Extract full text from all pages of the PDF."""
        return "\n".join(self._extract_page_text(page) for page in pdf)

    def _extract_page_text(self, page) -> str:
        """Extract text from a single PDF page, fallback to OCR if necessary."""
        text = page.get_text()
        if not text.strip():
            # Fallback to OCR if PyMuPDF fails to extract text
            return self._extract_text_with_tesseract(page)
        return text

    def _extract_text_with_tesseract(self, page) -> str:
        """Extract text from a PDF page using Tesseract OCR."""
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return pytesseract.image_to_string(img)

    def _extract_all_tables(self, file: Path, num_pages: int) -> str:
        """Extract tables from all pages of the PDF."""
        return "\n\n".join(
            self._extract_page_tables(file, page_num + 1) for page_num in range(num_pages)
        )

    def _extract_page_tables(self, file: Path, page_num: int) -> str:
        """Extract tables from a specific page of the PDF using Camelot."""
        try:
            tables = camelot.read_pdf(str(file), pages=str(page_num))
            if tables:
                return self._process_tables_camelot(tables)
            else:
                return "No tables found"
        except Exception as e:
            return f"Error extracting tables: {str(e)}"

    def _process_tables_camelot(self, tables) -> str:
        """Process tables extracted by Camelot into readable text."""
        table_texts = []
        for i, table in enumerate(tables):
            table_text = f"Table {i+1}:\n" + table.df.to_string(index=False)
            table_texts.append(table_text)
        return "\n\n".join(table_texts)
