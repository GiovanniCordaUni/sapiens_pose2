"""
Visualization utilities for pose estimation.

Funzioni per disegnare "scheletri" e keypoints sulle immagini.
"""

import cv2
import numpy as np


# COCO 17-keypoint scheletro edges (coppie di indici di keypoint)
COCO17_EDGES = [
    # Braccia
    (5, 7), (7, 9),     # braccio sinistro
    (6, 8), (8, 10),    # braccio destro
    # Torso
    (5, 6),             # spalle
    (5, 11), (6, 12),   # spalle a anche
    (11, 12),           # anche
    # Gambe
    (11, 13), (13, 15), # gamba sinistra
    (12, 14), (14, 16), # gamba destra
    # Faccia
    (0, 1), (0, 2),     # naso a occhi
    (1, 3), (2, 4),     # occhi a orecchie
]


def draw_coco17(
    frame_bgr: np.ndarray,
    keypoints_xy: np.ndarray,
    scores: np.ndarray,
    keypoint_threshold: float = 0.35,
    keypoint_radius: int = 5,
    skeleton_thickness: int = 2,
    keypoint_color: tuple[int, int, int] = (0, 255, 0),
    skeleton_color: tuple[int, int, int] = (0, 255, 0),
    edges: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """
    Disegna lo scheletro COCO a 17 keypoint su un'immagine.
    
    Args:
        frame_bgr: Immagine BGR di input
        keypoints_xy: Coordinate dei keypoint di shape (17, 2)
        scores: Punteggi di confidenza dei keypoint di shape (17,)
        keypoint_threshold: Confidenza minima per disegnare un keypoint
        keypoint_radius: Raggio dei cerchi dei keypoint
        skeleton_thickness: Spessore delle linee dello scheletro
        keypoint_color: Colore BGR per i keypoint
        skeleton_color: Colore BGR per gli edges dello scheletro
        edges: Edges personalizzati dello scheletro (default: COCO17_EDGES)
        
    Returns:
        Immagine con scheletro disegnato (copia dell'input)
    """
    out = frame_bgr.copy()
    edges = edges or COCO17_EDGES
    
    # Disegna i keypoint
    for i in range(17):
        if float(scores[i]) < keypoint_threshold:
            continue
        x, y = int(keypoints_xy[i, 0]), int(keypoints_xy[i, 1])
        cv2.circle(out, (x, y), keypoint_radius, keypoint_color, -1)
    
    # Draw skeleton edges
    for a, b in edges:
        if float(scores[a]) < keypoint_threshold or float(scores[b]) < keypoint_threshold:
            continue
        ax, ay = int(keypoints_xy[a, 0]), int(keypoints_xy[a, 1])
        bx, by = int(keypoints_xy[b, 0]), int(keypoints_xy[b, 1])
        cv2.line(out, (ax, ay), (bx, by), skeleton_color, skeleton_thickness)
    
    return out


def draw_bbox(
    frame_bgr: np.ndarray,
    box_xyxy: list[int],
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw a bounding box on an image.
    
    Args:
        frame_bgr: Input BGR image
        box_xyxy: Box coordinates as [x1, y1, x2, y2]
        color: BGR color for the box
        thickness: Line thickness
        
    Returns:
        Image with drawn box (copy of input)
    """
    out = frame_bgr.copy()
    x1, y1, x2, y2 = box_xyxy
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


def draw_pose_overlay(
    frame_bgr: np.ndarray,
    keypoints_xy: np.ndarray,
    scores: np.ndarray,
    box_xyxy: list[int] | None = None,
    keypoint_threshold: float = 0.35,
    keypoint_radius: int = 5,
    skeleton_thickness: int = 2,
    keypoint_color: tuple[int, int, int] = (0, 255, 0),
    skeleton_color: tuple[int, int, int] = (0, 255, 0),
    bbox_color: tuple[int, int, int] = (255, 0, 0),
    edges: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """
    Draw full pose overlay including skeleton and optional bounding box.
    
    Args:
        frame_bgr: Input BGR image
        keypoints_xy: Keypoint coordinates of shape (17, 2)
        scores: Keypoint confidence scores of shape (17,)
        box_xyxy: Optional bounding box as [x1, y1, x2, y2]
        keypoint_threshold: Minimum confidence to draw a keypoint
        keypoint_radius: Radius of keypoint circles
        skeleton_thickness: Thickness of skeleton lines
        keypoint_color: BGR color for keypoints
        skeleton_color: BGR color for skeleton edges
        bbox_color: BGR color for bounding box
        edges: Custom skeleton edges
        
    Returns:
        Image with full pose overlay (copy of input)
    """
    out = frame_bgr.copy()
    
    # Draw bounding box if provided
    if box_xyxy is not None:
        x1, y1, x2, y2 = box_xyxy
        cv2.rectangle(out, (x1, y1), (x2, y2), bbox_color, 2)
    
    # Draw skeleton
    out = draw_coco17(
        out, keypoints_xy, scores,
        keypoint_threshold=keypoint_threshold,
        keypoint_radius=keypoint_radius,
        skeleton_thickness=skeleton_thickness,
        keypoint_color=keypoint_color,
        skeleton_color=skeleton_color,
        edges=edges,
    )
    
    return out
