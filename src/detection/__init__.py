"""
Package di rilevamento delle persone.

Contiene funzioni e classi per il rilevamento delle persone nelle immagini e nei video.
"""

from .person_detector import (
    iou_xyxy,
    clamp_box_xyxy,
    add_padding_xyxy,
    pick_person_box,
)

__all__ = [
    "iou_xyxy",
    "clamp_box_xyxy",
    "add_padding_xyxy",
    "pick_person_box",
]
