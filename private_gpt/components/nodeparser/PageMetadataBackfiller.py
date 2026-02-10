from typing import Any, List, Sequence
from llama_index.core.schema import BaseNode, TransformComponent
import re


class PageMetadataBackfiller(TransformComponent):
    """
    A transformation that backfills page metadata for nodes.
    It uses a `page_map` found in the node's metadata (passed down from the Document)
    and the node's `start_char_idx` to determine the correct page number.
    """

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        """Process nodes and backfill page metadata."""
        for node in nodes:
            page_map = node.metadata.get("page_map")
            start_char_idx = getattr(node, "start_char_idx", None)
            end_char_idx = getattr(node, "end_char_idx", None)

            # Strategy 1: Use page_map if available (for joined documents with join_pages=True)
            if page_map and start_char_idx is not None:
                found_pages = set()
                end_idx = end_char_idx if end_char_idx is not None else start_char_idx + len(node.get_content())
                
                for entry in page_map:
                    # Check if the node's range overlaps with this page's range
                    # Node [start, end] and Page [p_start, p_end]
                    # Overlap if: start < p_end AND end > p_start
                    if start_char_idx < entry["end_char_idx"] and end_idx > entry["start_char_idx"]:
                        found_pages.add(entry["page"])
                
                if found_pages:
                    # Sort pages, handling mixed types (int and str)
                    sorted_pages = sorted(list(found_pages), key=lambda x: (isinstance(x, str), x))
                    
                    if len(sorted_pages) > 1:
                        # Node spans multiple pages
                        node.metadata["page"] = sorted_pages
                    else:
                        # Node is on a single page
                        node.metadata["page"] = sorted_pages[0]
                
                # Clean up page_map to avoid bloat
                node.metadata.pop("page_map", None)
            
            # Strategy 2: Page already set correctly (non-joined mode or already processed)
            # DO NOT overwrite existing valid page metadata
            elif "page" in node.metadata and node.metadata["page"] is not None:
                # Page is already set correctly from LlamaParseReader
                # Just clean up page_map if it exists
                node.metadata.pop("page_map", None)
            
            # Strategy 3: ONLY as last resort, extract from text markers
            # This happens when page_map doesn't exist AND page is not set
            else:
                content = node.get_content()
                # Try to find page markers in the content
                # Pattern: START OF PAGE: <page> or END OF PAGE: <page>
                markers = re.findall(r"(?:START|END) OF PAGE:\s*(\S+)", content)
                
                if markers:
                    # Deduplicate while preserving order
                    unique_markers = list(dict.fromkeys(markers))
                    
                    # Try to convert to int for numeric pages
                    found_pages = []
                    for marker in unique_markers:
                        try:
                            found_pages.append(int(marker))
                        except ValueError:
                            # Keep as string for Roman numerals or other formats
                            found_pages.append(marker)
                    
                    if len(found_pages) > 1:
                        node.metadata["page"] = found_pages
                    elif len(found_pages) == 1:
                        node.metadata["page"] = found_pages[0]

            # Clean up duplicate and unnecessary metadata keys
            keys_to_remove = [
                "filename",           # Duplicate of file_name
                "document_path",      # Duplicate of file_path
                "file_path_absolute", # Duplicate of file_path
                "file_path_relative", # Not needed
                "strategy",           # Internal processing detail
                "document_id",        # You might want to keep this - remove from list if needed
            ]
            
            for key in keys_to_remove:
                node.metadata.pop(key, None)

        return nodes