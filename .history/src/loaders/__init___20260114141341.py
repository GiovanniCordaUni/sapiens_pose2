"""
Model loaders.

produce direttamente 17 keypoints in formato COCO,
nessun mapping aggiuntivo richiesto.    
"""

from .yolo_loader import YoloPersonDetector
from .sapiens_loader import (
    load_sapiens,
    get_device,
    SapiensPoseEstimator,
)

__all__ = [
    "YoloPersonDetector",
    "load_sapiens",
    "get_device",
    "SapiensPoseEstimator",
]
