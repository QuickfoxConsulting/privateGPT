from injector import Injector

from private_gpt.settings.settings import Settings, unsafe_typed_settings


from private_gpt.components.ocr_components.visual_engine import VisualDocumentEngine, VisualProcessingOptions
from private_gpt.components.ocr_components.detection.engine import DetectionEngine
from private_gpt.components.ocr_components.detection.providers.onnx_detector import OnnxDetectorProvider
from private_gpt.components.ocr_components.detection.downloader import ensure_detection_model
from private_gpt.components.ocr_components.reconstruction.engine import ReconstructionEngine
from private_gpt.components.ocr_components.reconstruction.providers.fitz_reconstructor import FitzReconstructionProvider

def create_application_injector(settings: Settings) -> Injector:
    _injector = Injector(auto_bind=True)
    _injector.binder.bind(Settings, to=settings)
    
    # Text Detection
    model_path = ensure_detection_model()
    onnx_detector = OnnxDetectorProvider(model_path=model_path)
    detection_engine = DetectionEngine([onnx_detector])

    # Document Reconstruction
    fitz_reconstructor = FitzReconstructionProvider()
    reconstruction_engine = ReconstructionEngine([fitz_reconstructor])

    _injector.binder.bind(DetectionEngine, to=detection_engine)
    _injector.binder.bind(ReconstructionEngine, to=reconstruction_engine)
    _injector.binder.bind(VisualDocumentEngine, to=VisualDocumentEngine(
        options=VisualProcessingOptions(dpi=300),
        detection_engine=detection_engine
    ))
    return _injector


"""
Global injector for the application.

Avoid using this reference, it will make your code harder to test.

Instead, use the `request.state.injector` reference, which is bound to every request
"""
global_injector: Injector = create_application_injector(unsafe_typed_settings)

