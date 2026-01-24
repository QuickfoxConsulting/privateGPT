from typing import Any, List, Sequence
from llama_index.core.schema import BaseNode, TransformComponent

class PageMetadataBackfiller(TransformComponent):
    """
    A transformation that backfills page metadata for nodes.
    
    It uses a `page_map` found in the node's metadata (passed down from the Document)
    and the node's `start_char_idx` to determine the correct page number.
    """

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        """Process nodes and backfill page metadata."""
        import re
        
        for node in nodes:
            # Check if we have the page_map and character indices
            page_map = node.metadata.get("page_map")
            
            # If this is a child node, it might not have the page_map directly
            # but it has a parent relationship. However, in LlamaIndex transformations,
            # nodes usually keep the original metadata unless stripped.
            
            start_char_idx = getattr(node, "start_char_idx", None)
            end_char_idx = getattr(node, "end_char_idx", None)
            
            if page_map and start_char_idx is not None:
                # Find the range of pages from the map
                # The page_map is a list of dicts: {"page": int, "start_char_idx": int, "end_char_idx": int}
                found_pages = set()
                end_idx = end_char_idx if end_char_idx is not None else start_char_idx + len(node.get_content())
                
                for entry in page_map:
                    # Check if the node's range overlaps with this page's range
                    # Node [start, end] and Page [p_start, p_end]
                    # Overlap if: start < p_end AND end > p_start
                    if start_char_idx < entry["end_char_idx"] and end_idx > entry["start_char_idx"]:
                        found_pages.add(entry["page"])
                
                if found_pages:
                    sorted_pages = sorted(list(found_pages))
                    if len(sorted_pages) > 1:
                        # Store as a list or a range string for the frontend/LLM
                        node.metadata["page"] = sorted_pages
                        # node.metadata["page_label"] = f"{sorted_pages[0]}-{sorted_pages[-1]}"
                    else:
                        node.metadata["page"] = sorted_pages[0]
                
                # Clean up the page_map from metadata to avoid bloat and size issues
                # node.metadata.pop("page_map", None)
            
            # --- Fallback: Extract from text markers if metadata fails ---
            if not node.metadata.get("page") or node.metadata.get("page") == 1:
                markers = re.findall(r"(?:START|END) OF PAGE: (\d+)", node.get_content())
                if markers:
                    found_pages = sorted(list(set(map(int, markers))))
                    if len(found_pages) > 1:
                        node.metadata["page"] = found_pages
                    else:
                        node.metadata["page"] = found_pages[0]
                    
        return nodes
