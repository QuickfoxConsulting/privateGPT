import re
import logging
from pathlib import Path
from typing import List, Optional, Set, Any, Tuple, Dict

from llama_index.core.schema import NodeWithScore, MetadataMode
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from private_gpt.constants import UPLOAD_DIR

logger = logging.getLogger(__name__)

def normalize_path(path: str) -> str:
    """Normalize a path to be relative to the documents folder if it's an absolute path within it."""
    try:
        if not path:
            return path
        p = Path(path)
        documents_dir = (Path(UPLOAD_DIR) / "documents").resolve()
        if p.is_absolute():
            resolved_p = p.resolve()
            if str(resolved_p).startswith(str(documents_dir)):
                return str(resolved_p.relative_to(documents_dir))
    except Exception:
        pass
    return path

def get_cited_nodes(response: str, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
    """
    Extracts citations from the response (e.g., [1], [2]) and returns the corresponding nodes.
    Also validates that the citations exist.
    """
    cited_indices = set()
    # Match [1], [1 (p. 5)], etc.
    matches = re.finditer(r'\[(\d+)(?: \(p\. .*?\))?\]', response)
    for match in matches:
        try:
            # Convert to 0-based index
            idx = int(match.group(1)) - 1
            cited_indices.add(idx)
        except ValueError:
            continue
            
    # Filter nodes
    cited_nodes = []
    seen_node_ids = set()
    
    # We need to map the 1-based index back to the node in the `nodes` list
    # The `nodes` list order corresponds to [1], [2], [3]...
    for idx in sorted(cited_indices):
        if 0 <= idx < len(nodes):
            node = nodes[idx]
            if node.node.node_id not in seen_node_ids:
                cited_nodes.append(node)
                seen_node_ids.add(node.node.node_id)
        else:
            logger.warning(f"Response cited source [{idx+1}] which is out of range (max {len(nodes)}).")

    return cited_nodes

def smart_citation_replacement(response: str, nodes: List[NodeWithScore]) -> str:
    """
    Replaces [1], [2] with [Page 5](document.pdf) or [Title](url) based on metadata.
    Consumes any existing markdown links produced by the LLM to avoid duplicates.
    """
    def replace_match(match):
        try:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(nodes):
                node = nodes[idx]
                metadata = node.node.metadata or {}
                
                # Check for file metadata
                file_name = metadata.get("file_name") or metadata.get("filename")
                document_path = metadata.get("document_path") or metadata.get("file_path")
                page_label = metadata.get("page_label")
                
                # Normalize document_path
                document_path = normalize_path(document_path)
                
                # Check for web metadata
                url = metadata.get("url")
                title = metadata.get("title") or "Web Page"
                
                # Construct tooltip text
                tooltip_parts = []
                if file_name:
                    tooltip_parts.append(f"Source: {file_name}")
                elif title:
                    tooltip_parts.append(f"Source: {title}")
                
                if page_label:
                    tooltip_parts.append(f"Page {page_label}")
                
                tooltip_text = " | ".join(tooltip_parts)
                tooltip_text = tooltip_text.replace('"', "'")

                citation_label = str(idx + 1)
                formatted_citation = f"[{citation_label}]"

                # Prioritize URL for web, then document_path for files, then file_name
                target = url or document_path or file_name
                
                if target:
                    return f'{formatted_citation}({target} "{tooltip_text}")'
                else:
                    return f'{formatted_citation}' # Fallback to plain text if no source
            else:
                return "" # Remove invalid citation
        except (ValueError, IndexError):
            return match.group(0)

    # regex updated to capture [N] and any following (link) to avoid duplication
    citation_regex = r'\[(\d+)(?: \(p\. .*?\))?\](?:\s*\([^\)]+\))?'
    new_response = re.sub(citation_regex, replace_match, response)
    return new_response

class CitationHelper:
    @staticmethod
    def get_cited_nodes(response: str, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        return get_cited_nodes(response, nodes)
    
    @staticmethod
    def smart_citation_replacement(response: str, nodes: List[NodeWithScore]) -> str:
        return smart_citation_replacement(response, nodes)
