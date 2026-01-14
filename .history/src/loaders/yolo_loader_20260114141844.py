"""
YOLO model loader and detector wrapper.

Incapsula il caricamento del modello YOLO e il rilevamento delle persone.
"""

from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from ..detection.person_detector import pick_person_box, clamp_box_xyxy, add_padding_xyxy


class YoloPersonDetector:
    """
    Rilevatore di persone basato su YOLO con supporto al tracciamento.
    
    Incapsula il modello YOLO per il rilevamento delle persone e mantiene la
    coerenza temporale attraverso i frame video.
    
    Args:
        weights_path: Percorso al file dei pesi YOLO
        confidence: Minimum detection confidence
        imgsz: Input image size for YOLO
        person_class_id: Class ID for person in the model (default: 0)
        iou_bias: Weight for IoU vs confidence in tracking (0-1)
        
    Example:
        detector = YoloPersonDetector("yolov8n.pt")
        box, conf = detector.detect(frame)
        # For tracking:
        box, conf = detector.detect(frame, prev_box=previous_box)
    """
    
    def __init__(
        self,
        weights_path: str | Path,
        confidence: float = 0.30,
        imgsz: int = 640,
        person_class_id: int = 0,
        iou_bias: float = 0.7,
    ):
        self.model = YOLO(str(weights_path))
        self.confidence = confidence
        self.imgsz = imgsz
        self.person_class_id = person_class_id
        self.iou_bias = iou_bias
        
    def predict(
        self,
        frame: np.ndarray,
        confidence: float | None = None,
        imgsz: int | None = None,
    ) -> Any:
        """
        Run YOLO prediction on a frame.
        
        Args:
            frame: BGR image as numpy array
            confidence: Override default confidence threshold
            imgsz: Override default image size
            
        Returns:
            YOLO Results object
        """
        conf = confidence if confidence is not None else self.confidence
        size = imgsz if imgsz is not None else self.imgsz
        
        results = self.model.predict(
            frame,
            conf=conf,
            imgsz=size,
            verbose=False,
        )
        return results[0]
    
    def detect(
        self,
        frame: np.ndarray,
        prev_box: list | None = None,
        confidence: float | None = None,
    ) -> tuple[list | None, float | None]:
        """
        Detect the best person in a frame.
        
        Args:
            frame: BGR image as numpy array
            prev_box: Previous frame's box for tracking (optional)
            confidence: Override default confidence threshold
            
        Returns:
            Tuple of (box_xyxy, confidence) or (None, None) if no person found
        """
        result = self.predict(frame, confidence=confidence)
        return pick_person_box(
            result,
            prev_box=prev_box,
            iou_bias=self.iou_bias,
            person_class_id=self.person_class_id,
        )
    
    def detect_with_padding(
        self,
        frame: np.ndarray,
        prev_box: list | None = None,
        padding_ratio: float = 0.20,
        confidence: float | None = None,
    ) -> tuple[list | None, float | None]:
        """
        Detect person and return padded, clamped bounding box.
        
        Args:
            frame: BGR image as numpy array
            prev_box: Previous frame's box for tracking (optional)
            padding_ratio: Padding as fraction of box dimensions
            confidence: Override default confidence threshold
            
        Returns:
            Tuple of (padded_box_xyxy, confidence) or (None, None)
        """
        H, W = frame.shape[:2]
        box, conf = self.detect(frame, prev_box=prev_box, confidence=confidence)
        
        if box is None:
            return None, None
        
        # Clamp and add padding
        box = clamp_box_xyxy(box, W, H)
        box = add_padding_xyxy(box, W, H, padding_ratio=padding_ratio)
        
        return box, conf
