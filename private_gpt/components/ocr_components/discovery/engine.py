import logging
from pathlib import Path
from typing import Dict, List, Optional, Type

from private_gpt.components.ocr_components.discovery.base import (
    BaseDiscoveryProvider,
    DiscoveryResult,
)

logger = logging.getLogger(__name__)

class DiscoveryEngine:
    """
    The Orchestrator for Document Discovery.
    
    Detects file types and routes them to the appropriate DiscoveryProvider
    to separate digital text from visual elements.
    """

    def __init__(self, providers: Optional[List[BaseDiscoveryProvider]] = None):
        self._providers: Dict[str, BaseDiscoveryProvider] = {}
        if providers:
            for provider in providers:
                self.register_provider(provider)

    def register_provider(self, provider: BaseDiscoveryProvider) -> None:
        """Registers a provider for its supported extensions."""
        for ext in provider.supported_extensions:
            ext_lower = ext.lower().lstrip('.')
            self._providers[ext_lower] = provider
            logger.debug("Registered %s for extension: .%s", provider.__class__.__name__, ext_lower)

    def discover(self, file_path: Path) -> DiscoveryResult:
        """
        Discovers content in the given file by routing to the correct provider.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = file_path.suffix.lower().lstrip('.')
        provider = self._providers.get(extension)

        if not provider:
            logger.warning("No discovery provider found for extension: .%s. Using generic fallback?", extension)
            raise ValueError(f"Unsupported file format for discovery: .{extension}")

        logger.info("Routing %s to %s", file_path.name, provider.__class__.__name__)
        return provider.discover(file_path)

    def is_supported(self, file_path: Path) -> bool:
        """Checks if a file format is supported by the engine."""
        extension = file_path.suffix.lower().lstrip('.')
        return extension in self._providers
