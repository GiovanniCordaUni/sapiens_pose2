"""
Visualization utilities.

Contiene funzioni di disegno per scheletri e pose.
"""

from .draw import (
    COCO17_EDGES,
    draw_coco17,
    draw_bbox,
    draw_pose_overlay,
)

__all__ = [
    "COCO17_EDGES",
    "draw_coco17",
    "draw_bbox",
    "draw_pose_overlay",
]
