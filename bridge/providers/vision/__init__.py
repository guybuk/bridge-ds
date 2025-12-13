from bridge.providers.vision.classification import ImageFolder, TorchvisionCIFAR10
from bridge.providers.vision.detection import Coco2017Detection
from bridge.providers.vision.paired import AlignedImageFolders, Pix2PixFolder

__all__ = [
    "ImageFolder",
    "TorchvisionCIFAR10",
    "Coco2017Detection",
    "Pix2PixFolder",
    "AlignedImageFolders",
]
