from bridge.display.vision.detection import DetectionPanelEngine
from bridge.display.vision.image_pairs import ImagePairsPanelEngine

# Back-compat alias. The old generic name is preserved so existing notebooks,
# tests, and provider defaults keep working until feature/notebooks-rewrite
# updates everything to the new names.
Panel = DetectionPanelEngine

__all__ = ["DetectionPanelEngine", "ImagePairsPanelEngine", "Panel"]
