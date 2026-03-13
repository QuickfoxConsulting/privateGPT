from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image

@dataclass
class ReconstructionResult:
    """The result of a reconstruction process."""
    success: bool
    output_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

class BaseReconstructionProvider:
    """Abstract base class for document reconstruction."""
    
    def reconstruct(
        self, 
        original_images: List[Image.Image], 
        text_locations: List[List[Dict[str, Any]]],
        output_name: str,
        original_path: Optional[Path] = None
    ) -> ReconstructionResult:
        """
        Reconstructs a document (e.g., Searchable PDF) from images and text locations.
        
        Args:
            original_images: List of PIL images for each page.
            text_locations: List of (list of text segments with bboxes) for each page.
            output_name: The desired output filename (without extension).
            original_path: Optional path to the original document for non-destructive augmentation.
            
        Returns:
            A ReconstructionResult containing the path to the reconstructed file.
        """
        raise NotImplementedError("Subclasses must implement reconstruct()")
