from injector import Injector

from private_gpt.settings.settings import Settings, unsafe_typed_settings


from private_gpt.components.ocr_components.visual_engine import VisualDocumentEngine, VisualProcessingOptions
from private_gpt.components.ocr_components.vlm_client import NuMarkdownClient
from private_gpt.components.ocr_components.document_reconstructor import DocumentReconstructor

def create_application_injector() -> Injector:
    _injector = Injector(auto_bind=True)
    _injector.binder.bind(Settings, to=unsafe_typed_settings)
    _injector.binder.bind(VisualDocumentEngine, to=VisualDocumentEngine(VisualProcessingOptions(dpi=300)))
    _injector.binder.bind(NuMarkdownClient, to=NuMarkdownClient(unsafe_typed_settings))
    _injector.binder.bind(DocumentReconstructor, to=DocumentReconstructor(unsafe_typed_settings))
    return _injector


"""
Global injector for the application.

Avoid using this reference, it will make your code harder to test.

Instead, use the `request.state.injector` reference, which is bound to every request
"""
global_injector: Injector = create_application_injector()

