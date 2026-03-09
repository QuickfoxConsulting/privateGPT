from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image


@dataclass
class PreprocessingTask:
    """
    Holds the image and its transformation history.
    
    The transformation_matrix tracks changes (rotation, scaling, translation)
    relative to the original rendered page image.
    """
    image: np.ndarray  # OpenCV format (BGR or Grayscale)
    original_size: tuple  # (width, height)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Cumulative transformation matrix (3x3 for affine transforms)
    # Starts as Identity matrix.
    transformation_matrix: np.ndarray = field(default_factory=lambda: np.eye(3))

    def update_image(self, new_image: np.ndarray, matrix: Optional[np.ndarray] = None):
        """Updates the current image and adjusts the transformation matrix."""
        self.image = new_image
        if matrix is not None:
            # Chain the matrices: Current = New * Current
            self.transformation_matrix = matrix @ self.transformation_matrix


class BasePreprocessingStep(ABC):
    """Abstract base class for all preprocessing transformations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the step (e.g., 'deskew')."""
        pass

    @abstractmethod
    def run(self, task: PreprocessingTask, options: Dict[str, Any]) -> PreprocessingTask:
        """Executes the transformation and returns the updated task."""
        pass
