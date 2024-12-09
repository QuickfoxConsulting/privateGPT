import os
import time
import logging
import traceback
from typing import Optional, Dict, Any

from fastapi import HTTPException, status, Request
from injector import inject, singleton

from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.logger import configure_logging
from marker.models import create_model_dict
from marker.output import output_exists, save_output

from private_gpt.server.ingest.ingest_router import ingest

# Configure logging
configure_logging()

@singleton
class PDFConverter:
    """
    A singleton class responsible for converting PDF files to Markdown.
    
    This class uses the Marker library for PDF conversion and provides 
    methods for processing PDFs.
    """
    
    DEFAULT_CONFIG = {
        "output_format": "markdown",
    }
    
    @inject
    def __init__(self, output_dir: str = './output'):
        """
        Initialize the PDF converter with default configuration.
        
        Args:
            output_dir (str): Directory to save converted files. Defaults to './output'.
        """
        config_parser = ConfigParser(self.DEFAULT_CONFIG)
        
        self.converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer()
        )
        self.output_dir = output_dir
    
    @staticmethod
    def worker_init(model_dict: Optional[Dict] = None):
        """
        Initialize worker for multiprocessing.
        
        Args:
            model_dict (Optional[Dict]): Dictionary of models to initialize.
        """
        if model_dict is None:
            model_dict = create_model_dict()
        
        global model_refs
        model_refs = model_dict
    
    @staticmethod
    def worker_exit():
        """Clean up worker resources after processing."""
        global model_refs
        del model_refs
    
    @classmethod
    def process_single_pdf(cls, args: tuple):
        """
        Process a single PDF file.
        
        Args:
            args (tuple): A tuple containing file path and CLI options.
        """
        fpath, cli_options = args
        config_parser = ConfigParser(cli_options)
        
        out_folder = config_parser.get_output_folder(fpath)
        base_name = config_parser.get_base_filename(fpath)
        
        if cli_options.get('skip_existing') and output_exists(out_folder, base_name):
            return
        
        try:
            converter = PdfConverter(
                config=config_parser.generate_config_dict(),
                artifact_dict=model_refs,
                processor_list=config_parser.get_processors(),
                renderer=config_parser.get_renderer()
            )
            rendered = converter(fpath)
            save_output(rendered, out_folder, base_name)
        except Exception as e:
            logging.error(f"Error converting {fpath}: {e}")
            logging.error(traceback.format_exc())
    
    async def convert_pdf_to_markdown(self, pdf_path: str) -> str:
        """
        Convert a PDF file to Markdown format.
        
        Args:
            pdf_path (str): Path to the PDF file to convert.
        
        Returns:
            str: Path to the generated Markdown file.
        """
        start_time = time.time()
        
        try:
            rendered = self.converter(pdf_path)
            
            fname = os.path.basename(pdf_path)
            markdown_file_path = os.path.join(
                self.output_dir, 
                f"{os.path.splitext(fname)[0]}.md"
            )
            
            # Save the rendered content (implementation depends on Marker library)
            # You might need to adjust this based on the actual Marker library API
            with open(markdown_file_path, 'w', encoding='utf-8') as f:
                f.write(rendered)
            
            logging.info(f"Saved markdown to {markdown_file_path}")
            logging.info(f"Total conversion time: {time.time() - start_time:.2f} seconds")
            
            return markdown_file_path
        
        except Exception as e:
            logging.error(f"PDF conversion error: {e}")
            logging.error(traceback.format_exc())
            raise


async def process_pdf_to_markdown(
    request: Request, 
    pdf_path: str, 
    ocr_all_pages: bool = False
) -> str:
    """
    Process PDF to Markdown using the Marker converter.
    
    Args:
        request (Request): The FastAPI request object.
        pdf_path (str): Path to the PDF file.
        ocr_all_pages (bool, optional): Whether to perform OCR on all pages.
    
    Returns:
        str: Path to the generated Markdown file.
    """
    pdf_converter = request.state.injector.get(PDFConverter)
    return await pdf_converter.convert_pdf_to_markdown(pdf_path)


async def process_ocr(request: Request, pdf_path: str) -> Any:
    """
    Process OCR for the given PDF file and ingest the results.
    
    Args:
        request (Request): The FastAPI request object.
        pdf_path (str): Path to the PDF file to process.
    
    Returns:
        Any: Ingested documents.
    
    Raises:
        HTTPException: If there's an error during OCR processing.
    """
    try:
        markdown_path = await process_pdf_to_markdown(request, pdf_path)
        return await ingest(request=request, file_path=markdown_path)
    
    except Exception as e:
        logging.error(f"OCR processing error: {e}")
        logging.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing failed: {e}"
        )