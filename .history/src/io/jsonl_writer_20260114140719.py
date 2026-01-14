"""
JSONL output utilities.

Funzioni e classi per la lettura e scrittura di file JSONL.
"""

import json
from pathlib import Path
from typing import Any


class JsonlWriter:
    """
    Context manager per la scrittura di file JSONL.
    
    Example:
        with JsonlWriter("output.jsonl") as writer:
            writer.write({"frame": 0, "keypoints": [...]})
    """
    
    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        
    def __enter__(self):
        self._file = self.filepath.open("w", encoding="utf-8")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()
        return False
    
    def write(self, record: dict[str, Any]) -> None:
        """Write a single record as a JSON line."""
        if self._file is None:
            raise RuntimeError("JsonlWriter must be used as context manager")
        self._file.write(json.dumps(record) + "\n")


def create_pose_record(
    frame_idx: int,
    fps: float,
    box_xyxy: list[int] | None,
    box_conf: float | None,
    keypoints: list[list[float]] | None = None,
    keypoint_indices: list[int] | None = None,
    num_keypoints_total: int | None = None,
    status: str = "ok",
) -> dict[str, Any]:
    """
    Create a standardized pose record for JSONL output.
    
    Con il checkpoint COCO 17 nativo, keypoints contiene sempre 17 elementi
    e keypoint_indices è identity [0..16].
    
    Args:
        frame_idx: Frame index in the video
        fps: Video frame rate
        box_xyxy: Person bounding box or None
        box_conf: Detection confidence or None
        keypoints: List of [x, y, score] for 17 COCO keypoints
        keypoint_indices: Identity indices [0..16]
        num_keypoints_total: Always 17 for COCO native checkpoint
        status: Status string ("ok", "no_person", "empty_crop")
        
    Returns:
        Dictionary ready for JSON serialization
    """
    record = {
        "frame": frame_idx,
        "time_sec": float(frame_idx / fps),
        "person_box_xyxy": box_xyxy,
        "person_box_conf": box_conf,
        "status": status,
    }
    
    if keypoints is not None:
        record["keypoints17"] = keypoints
        
    if keypoint_indices is not None:
        record["kp17_indices"] = keypoint_indices
        
    if num_keypoints_total is not None:
        record["num_keypoints_total"] = num_keypoints_total
        
    return record


def keypoints_to_list(
    keypoints: Any,  # np.ndarray
    scores: Any,     # np.ndarray
) -> list[list[float]]:
    """
    Convert keypoints and scores arrays to JSON-serializable list.
    
    Args:
        keypoints: Array of shape (K, 2) with x, y coordinates
        scores: Array of shape (K,) with confidence scores
        
    Returns:
        List of [x, y, score] for each keypoint
    """
    return [
        [float(keypoints[i, 0]), float(keypoints[i, 1]), float(scores[i])]
        for i in range(len(scores))
    ]
