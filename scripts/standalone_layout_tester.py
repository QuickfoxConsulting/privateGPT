import argparse
import sys
import os
from pathlib import Path

import cv2
import numpy as np
import fitz
from PIL import Image

sys.path.append(os.getcwd())

from private_gpt.components.ocr_components.layout.engine import VisualLayoutEngine

def pdf_page_to_image(pdf_path: str, page_num: int = 0, dpi: int = 300) -> Image.Image:
    """Helper to get a clean image from a PDF page for testing."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img

def main():
    parser = argparse.ArgumentParser(description="Visual Layout Engine Tester")
    parser.add_argument("--file", type=str, required=True, help="Path to PDF or Image")
    parser.add_argument("--page", type=int, default=0, help="Page number (0-indexed) for PDFs")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    print(f"\n[VisualLayoutEngine] 🔬 Analyzing {file_path.name}")
    
    # 1. Load Image
    if file_path.suffix.lower() == ".pdf":
        print(f"Loading page {args.page} from PDF...")
        img = pdf_page_to_image(str(file_path), args.page)
    else:
        print("Loading image...")
        img = Image.open(file_path).convert("RGB")

    # 2. Run Layout Engine
    engine = VisualLayoutEngine() # Using default 25x3 horizontal kernel
    
    import time
    start_time = time.time()
    blocks = engine.extract_layout(img)
    duration = (time.time() - start_time) * 1000
    
    print(f"✅ Extracted {len(blocks)} layout blocks in {duration:.2f} ms")

    # 3. Draw Bounding Boxes
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    for i, block in enumerate(blocks):
        x0, y0, x1, y1 = block.coordinates
        # Draw a bright green rectangle around the discovered block
        cv2.rectangle(img_cv, (x0, y0), (x1, y1), (0, 255, 0), 2)
        
        # Add index label
        cv2.putText(img_cv, str(i), (x0, max(0, y0 - 5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 4. Save Output
    output_dir = Path("layout_output")
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / f"layout_debug_{file_path.stem}_p{args.page}.png"
    
    cv2.imwrite(str(out_path), img_cv)
    print(f"🖼️ Saved debug layout visualization to: {out_path}")
    print("Open this file to verify the 'Strikethrough' contours!")

if __name__ == "__main__":
    main()
