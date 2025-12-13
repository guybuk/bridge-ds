from bridge.providers.dataset_provider import DatasetProvider

# Multimodal providers
from bridge.providers.multimodal import (
    CocoCaptions,
)

# Text providers
from bridge.providers.text import (
    LargeMovieReviewDataset,
    ParallelCorpus,
)

# Vision providers
from bridge.providers.vision import (
    AlignedImageFolders,
    Coco2017Detection,
    ImageFolder,
    Pix2PixFolder,
    TorchvisionCIFAR10,
)

__all__ = [
    # Base
    "DatasetProvider",
    # Vision
    "ImageFolder",
    "TorchvisionCIFAR10",
    "Coco2017Detection",
    "Pix2PixFolder",
    "AlignedImageFolders",
    # Text
    "LargeMovieReviewDataset",
    "ParallelCorpus",
    # Multimodal
    "CocoCaptions",
]
