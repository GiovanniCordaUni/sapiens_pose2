"""
Person detection utilities.

Produce funzioni e classi per il rilevamento delle persone nelle immagini e nei video.
"""

import numpy as np
from typing import Any


def iou_xyxy(box_a: list | np.ndarray, box_b: list | np.ndarray) -> float:
    """
    Calcola Intersection over Union (IoU) per due box in formato xyxy.
    
    Args:
        box_a: Primo box come [x1, y1, x2, y2]
        box_b: Secondo box come [x1, y1, x2, y2]
        
    Returns:
        Valore di IoU tra 0 e 1
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    
    # Intersection
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    
    # Union
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    
    return float(inter / denom) if denom > 0 else 0.0


def clamp_box_xyxy(
    box: list | np.ndarray,
    frame_width: int,
    frame_height: int,
) -> list[int]:
    """
    Limita una bounding box per rimanere all'interno dei confini del frame.
    
    Garantisce che le coordinate della box siano valide (x1 < x2, y1 < y2) e
    all'interno delle dimensioni del frame.
    
    Args:
        box: Bounding box come [x1, y1, x2, y2]
        frame_width: Larghezza del frame
        frame_height: Altezza del frame
        
    Returns:
        box [x1, y1, x2, y2] con coordinate intere
    """
    x1, y1, x2, y2 = box
    
    # Limita ai confini del frame
    x1 = int(max(0, min(frame_width - 1, x1)))
    y1 = int(max(0, min(frame_height - 1, y1)))
    x2 = int(max(0, min(frame_width - 1, x2)))
    y2 = int(max(0, min(frame_height - 1, y2)))
    
    # Garantisce box valido (x1 < x2, y1 < y2)
    if x2 <= x1:
        x2 = min(frame_width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(frame_height - 1, y1 + 1)
        
    return [x1, y1, x2, y2]


def add_padding_xyxy(
    box: list | np.ndarray,
    frame_width: int,
    frame_height: int,
    padding_ratio: float = 0.20,
) -> list[int]:
    """
    Aggiunge padding a una bounding box e limita ai confini del frame.
    
    Args:
        box: Bounding box come [x1, y1, x2, y2]
        frame_width: Larghezza del frame
        frame_height: Altezza del frame
        padding_ratio: Padding come frazione delle dimensioni della box (default: 0.20)
        
    Returns:
        Box con padding e limitata come [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = box
    box_w, box_h = (x2 - x1), (y2 - y1)
    
    # Calcola padding in pixel
    pad_x = int(box_w * padding_ratio)
    pad_y = int(box_h * padding_ratio)
    
    # Applica padding e limita
    return clamp_box_xyxy(
        [x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y],
        frame_width,
        frame_height
    )


def pick_person_box(
    detection_result: Any,
    prev_box: list | None = None,
    iou_bias: float = 0.7,
    person_class_id: int = 0,
) -> tuple[list | None, float | None]:
    """
    Seleziona la migliore rilevazione di persona dai risultati YOLO.
    
    Usa una combinazione di confidenza e IoU con la box precedente (se disponibile)
    per la coerenza temporale nel tracciamento.
    
    Args:
        detection_result: Oggetto risultato del rilevamento YOLO con attributo .boxes
        prev_box: Box del frame precedente per continuità nel tracciamento (opzionale)
        iou_bias: Peso per IoU rispetto alla confidenza (0 = solo confidenza, 1 = solo IoU)
        person_class_id: ID classe per persona nel modello (default: 0 per COCO)
        
    Returns:
        Tuple of (box_xyxy, confidence) or (None, None) if no person found
    """
    boxes = detection_result.boxes
    if boxes is None or len(boxes) == 0:
        return None, None
    
    # Extract numpy arrays
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    
    # Filter for person class
    candidates = [
        (xyxy[i], float(conf[i])) 
        for i in range(len(cls)) 
        if cls[i] == person_class_id
    ]
    
    if not candidates:
        return None, None
    
    # If no previous box, return highest confidence
    if prev_box is None:
        best = max(candidates, key=lambda t: t[1])
        return best[0].tolist(), best[1]
    
    # Score combining confidence and IoU with previous box
    def score_fn(item):
        box, c = item
        return (1 - iou_bias) * c + iou_bias * iou_xyxy(box, prev_box)
    
    best = max(candidates, key=score_fn)
    return best[0].tolist(), best[1]
