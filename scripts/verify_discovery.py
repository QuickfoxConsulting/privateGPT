import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from private_gpt.components.ocr_components.discovery.engine import DiscoveryEngine
from private_gpt.components.ocr_components.discovery.providers.pdf import PDFDiscoveryProvider
from private_gpt.components.ocr_components.discovery.providers.docx import DOCXDiscoveryProvider
from private_gpt.components.ocr_components.discovery.providers.excel import ExcelDiscoveryProvider
from private_gpt.components.ocr_components.discovery.providers.pptx import PPTXDiscoveryProvider
from private_gpt.components.ocr_components.discovery.providers.rtf import RTFDiscoveryProvider
from private_gpt.components.ocr_components.discovery.providers.md import MDDiscoveryProvider
from private_gpt.components.ocr_components.discovery.base import DiscoveryType

def main():
    parser = argparse.ArgumentParser(description="Discovery Engine Verification Tool")
    parser.add_argument("--file", type=str, required=True, help="Path to any supported file (PDF, DOCX, XLSX, etc.)")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return

    # Initialize Engine with all providers
    engine = DiscoveryEngine([
        PDFDiscoveryProvider(),
        DOCXDiscoveryProvider(),
        ExcelDiscoveryProvider(),
        PPTXDiscoveryProvider(),
        RTFDiscoveryProvider(),
        MDDiscoveryProvider()
    ])

    print(f"\n[DiscoveryEngine] 🚀 Starting Discovery on: {file_path.name}")
    print("-" * 60)

    try:
        result = engine.discover(file_path)
    except Exception as e:
        print(f"Error during discovery: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    print(f"Result: File Type detected as '{result.file_type}'")
    print(f"Total Elements Found: {len(result.elements)}")
    print("-" * 60)

    text_count = 0
    image_count = 0
    
    for i, el in enumerate(result.elements):
        type_str = el.element_type.value.upper()
        if el.element_type == DiscoveryType.TEXT:
            text_count += 1
            # Print first 50 chars of text
            snippet = str(el.content)[:80].replace('\n', ' ')
            print(f"[{i+1:02}] {type_str:7} | P{el.page_number} | {snippet}...")
        else:
            image_count += 1
            is_scan = el.metadata.get("is_full_page_scan", False)
            scan_flag = "[FULL SCAN]" if is_scan else ""
            print(f"[{i+1:02}] {type_str:7} | P{el.page_number} | Visual Object {scan_flag}")
            
            # Save discovered images for inspection
            output_dir = Path("discovery_output")
            output_dir.mkdir(exist_ok=True)
            if hasattr(el.content, 'save'):
                img_name = f"{file_path.stem}_el{i+1}_{type_str}.png"
                el.content.save(output_dir / img_name)
                print(f"      -> Saved image to {output_dir / img_name}")

    print("-" * 60)
    print(f"SUMMARY: {text_count} Text Blocks, {image_count} Visual Elements.")
    print(f"Discovery Success. Open 'discovery_output/' to see extracted visuals.")

if __name__ == "__main__":
    main()
