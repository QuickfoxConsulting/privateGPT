import logging
import torch
from llama_index.core.schema import BaseNode, TransformComponent
from injector import inject, singleton
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)
logger.info("ENTITY_COMPONENT_FILE_LOADED")

# Device selection for GLiNER
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from pydantic import Field, PrivateAttr
from typing import Any

@singleton
class EntityComponent(TransformComponent):
    """Component for Named Entity Recognition using GLiNER."""
    
    settings: Any = Field(default=None, exclude=True)
    enabled: bool = Field(default=False)
    labels: list[str] = Field(default_factory=list)
    _model: Any = PrivateAttr(default=None)

    @inject
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        self.enabled = settings.ner.enabled
        self.labels = settings.ner.labels
        
        if self.enabled:
            logger.info("-> NER: Initializing EntityComponent with GLiNER model: %s", settings.ner.model_name)
            try:
                from gliner import GLiNER
                # Load model to device and use FP16 for memory efficiency if on CUDA
                model = GLiNER.from_pretrained(settings.ner.model_name)
                model.to(device)
                if device.type == "cuda":
                    model.half() # Use half precision to save VRAM
                
                # Use object.__setattr__ to bypass Pydantic's check for "extra" fields
                # This is necessary because TransformComponent likely has extra="forbid"
                object.__setattr__(self, "_model", model)
                logger.info("-> NER: GLiNER model loaded successfully on %s", device)
            except ImportError:


                logger.error("gliner package not found. Entity extraction will be disabled.")
                logger.error("Please install it with: pip install gliner")
                self.enabled = False
            except Exception as e:
                logger.error("Failed to load GLiNER model: %s. Entity extraction will be disabled.", str(e))
                self.enabled = False

    def __call__(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        """Transform nodes by extracting and adding entities to metadata."""
        logger.info("-> NER: __call__ invoked with %d nodes. enabled=%s, model_loaded=%s", len(nodes), self.enabled, self._model is not None)
        if not self.enabled or not self._model:
            logger.warning("-> NER: EntityComponent is disabled or model is not loaded. Skipping.")
            return nodes
        
        logger.info("-> NER: Starting entity extraction from %d text chunks...", len(nodes))
        for i, node in enumerate(nodes):
            try:
                if (i + 1) % 10 == 0 or i == 0:
                    logger.info("-> NER: Processing chunk %d/%d...", i + 1, len(nodes))
                content = node.get_content(metadata_mode="llm")
                logger.debug("-> NER: Extracting from content (len=%d)", len(content))
                entities = self.extract_entities(content)
                if entities:
                    logger.info("-> NER: Found %d entities in chunk %d", len(entities), i + 1)
                    # Add to metadata for downstream use (like expert retrieval)
                    # We store just the text list for simple filtering/matching
                    node.metadata["entities"] = list(set([ent["text"] for ent in entities]))
                    # We store full details if needed for advanced logic
                    node.metadata["entity_details"] = entities
                else:
                    logger.debug("-> NER: No entities found in chunk %d", i + 1)
            except Exception as e:
                logger.warning("-> NER: Failed to extract entities for node %s: %s", node.node_id, str(e))
                
        logger.info("-> NER: Finished extraction for all chunks.")
        return nodes


    def extract_entities(self, text: str) -> list[dict]:

        """Extract entities from the given text using the configured labels.
        
        Each entity in the returned list is a dictionary with:
        - 'start': start character index
        - 'end': end character index
        - 'label': the entity label (e.g., 'person')
        - 'text': the extracted text
        - 'score': confidence score
        """
        if not self.enabled or not self._model:
            return []
        
        try:
            # GLiNER expect a list of labels
            entities = self._model.predict_entities(text, self.labels, threshold=0.5)
            # Standardize output for consistency
            return [
                {
                    "text": ent["text"],
                    "label": ent["label"],
                    "score": float(ent["score"]),
                    "start": ent["start"],
                    "end": ent["end"]
                }
                for ent in entities
            ]
        except Exception as e:
            logger.warning("Error during entity extraction: %s", str(e))
            return []

    def get_unique_entities(self, text: str) -> list[tuple[str, str]]:
        """Utility to get a list of unique (text, label) pairs."""
        entities = self.extract_entities(text)
        seen = set()
        unique = []
        for ent in entities:
            pair = (ent["text"], ent["label"])
            if pair not in seen:
                unique.append(pair)
                seen.add(pair)
        return unique
