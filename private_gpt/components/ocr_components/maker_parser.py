import time
import os
import pypdfium2  # Needs to be at the top to avoid warnings
import traceback
import logging  # Add logging for better error handling

from fastapi import HTTPException, status, Request
from private_gpt.constants import OCR_UPLOAD
from private_gpt.server.ingest.ingest_router import ingest

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # For some reason, transformers decided to use .isin for a simple op, which is not supported on MPS
from marker.convert import convert_single_pdf
from marker.logger import configure_logging
from marker.models import load_all_models
from marker.output import save_markdown

configure_logging()

from injector import inject, singleton

@singleton
class PDFConverter:
    @inject
    def __init__(self, output_dir='./output'):
        self.output_dir = output_dir
        self.model_lst = load_all_models()

    async def convert_pdf_to_markdown(self, pdf_path, ocr_all_pages=False):
        """
        Convert a PDF file to Markdown format.
        
        Args:
            ocr_all_pages (bool): Whether to perform OCR on all pages of the PDF.
        
        Returns:
            str: The path to the generated Markdown file.
        """
        start = time.time()
        full_text, images, out_meta = convert_single_pdf(
            pdf_path, self.model_lst, ocr_all_pages=ocr_all_pages
        )
        
        fname = os.path.basename(pdf_path)
        subfolder_path = save_markdown(self.output_dir, fname, full_text, images, out_meta)
        markdown_file_path = os.path.join(subfolder_path, f"{os.path.splitext(fname)[0]}.md")  
        
        logging.info(f"Saved markdown to {markdown_file_path}")
        logging.info(f"Total time: {time.time() - start} seconds")
        return markdown_file_path
    

async def process_pdf_to_markdown(request: Request, pdf_path: str, ocr_all_pages: bool):
    """
    Process PDF to Markdown using the Marker.

    Args:
        request (Request): The FastAPI request object.
        pdf_path (str): The path to the PDF file to be converted.

    Returns:
        str: The path to the generated Markdown file.
    """
    pdf_converter = request.state.injector.get(PDFConverter)
    markdown_path = await pdf_converter.convert_pdf_to_markdown(pdf_path, ocr_all_pages)  # Ensure to await
    return markdown_path

async def process_ocr(request: Request, pdf_path: str, ocr_all_pages: bool):
    """
    Process OCR for the given PDF file.

    Args:
        request (Request): The FastAPI request object.
        pdf_path (str): The path to the PDF file to be processed.

    Returns:
        Any: The ingested documents.
    """
    try:
        markdown_path = await process_pdf_to_markdown(request, pdf_path, ocr_all_pages)
        ingested_documents = await ingest(request=request, file_path=markdown_path)
        return ingested_documents
    except Exception as e:
        logging.error(f"Error processing OCR: {e}")
        logging.error(traceback.format_exc())  # Log the full traceback
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"There was an error processing OCR: {e}"
        )
