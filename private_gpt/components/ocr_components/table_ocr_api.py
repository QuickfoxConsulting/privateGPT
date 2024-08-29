import os
import fitz
import aiofiles
import requests
import traceback
from docx import Document
from fastapi import HTTPException, status, File, UploadFile, APIRouter, Request, Security, Depends

from private_gpt.constants import OCR_UPLOAD
from private_gpt.components.ocr_components.table_ocr import GetOCRText
from private_gpt.components.ocr_components.TextExtraction import ImageToTable
from private_gpt.server.ingest.ingest_router import IngestResponse, ingest
pdf_router = APIRouter(prefix="/v1", tags=["ocr"])


async def save_uploaded_file(file: UploadFile, upload_dir: str):
    file_path = os.path.join(upload_dir, file.filename)
    try:
        contents = await file.read()
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(contents)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"There was an error uploading the file."
        )
    finally:
        await file.close()    
    return file_path


async def process_images_and_generate_doc(request: Request, pdf_path: str, upload_dir: str):
    ocr = request.state.injector.get(GetOCRText)
    # img_tab = request.state.injector.get(ImageToTable)
    pdf_writer = fitz.open()
    pdf_doc = fitz.open(pdf_path)
    
    for page_index in range(len(pdf_doc)):
        page = pdf_doc[page_index]
        image_list = page.get_images()

        if not image_list:
            continue

        for image_index, img in enumerate(image_list, start=1):
            xref = img[0]
            pix = fitz.Pixmap(pdf_doc, xref)

            if pix.n - pix.alpha > 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)(
                    "RGB", [pix.width, pix.height], pix.samples)

            image_path = f"temp_page_{page_index}_image_{image_index}.png"
            pix.save(image_path)

            extracted_text = ocr.extract_text(
                image_file=True, file_path=image_path)
            # Create a new page with the same dimensions as the original page
            pdf_page = pdf_writer.new_page(width=page.rect.width, height=page.rect.height)
            pdf_page.insert_text((10, 10), extracted_text, fontsize=9)
            os.remove(image_path)

    save_path = os.path.join(upload_dir, f"{os.path.splitext(os.path.basename(pdf_path))[0]}.pdf")
    pdf_writer.save(save_path)
    pdf_writer.close()
    return save_path

async def process_ocr(
        request: Request,
        pdf_path: str,
):
    UPLOAD_DIR = OCR_UPLOAD
    try:
        ocr_doc_path = await process_images_and_generate_doc(request, pdf_path, UPLOAD_DIR)
        ingested_documents = await ingest(request=request, file_path=ocr_doc_path)
        return ingested_documents
    except Exception as e:
        print(traceback.print_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"There was an error processing OCR: {e}"
        )

