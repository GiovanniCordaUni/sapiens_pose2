"""
Video reading utilities.

Classi e funzioni per la lettura di file video e l'iterazione sui frame.
"""

from pathlib import Path
from typing import Iterator, Any

import cv2
import numpy as np


class VideoReader:
    """
    Lettore di file video con accesso ai metadati e iterazione sui frame.
    
    Args:
        video_path: Percorso al file video
        
    Esempio:
        with VideoReader("video.mp4") as reader:
            print(f"Video: {reader.width}x{reader.height}, {reader.fps} fps")
            for frame_idx, frame in reader.iter_frames(stride=10):
                process(frame)
    """
    
    def __init__(self, video_path: str | Path):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        self._cap = None
        
    def __enter__(self):
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cap:
            self._cap.release()
        return False
    
    @property
    def fps(self) -> float:
        """Video frame rate."""
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    @property
    def width(self) -> int:
        """Video frame width."""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    @property
    def height(self) -> int:
        """Video frame height."""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    @property
    def frame_count(self) -> int:
        """Numero di frame totali."""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    @property
    def duration_sec(self) -> float:
        """Durata del video in secondi."""
        return self.frame_count / self.fps
    
    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        """Legge un singolo frame."""
        return self._cap.read()
    
    def iter_frames(
        self,
        stride: int = 1,
        start_frame: int = 0,
        end_frame: int | None = None,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """
        Itera sui frame del video con stride opzionale.
        
        Args:
            stride: Process every N-th frame (default: 1 = all frames)
            start_frame: Start from this frame index
            end_frame: Stop at this frame index (exclusive)
            
        Yields:
            Tuple of (frame_index, frame_bgr)
        """
        if end_frame is None:
            end_frame = self.frame_count
            
        # Seek to start frame if needed
        if start_frame > 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
        frame_idx = start_frame
        while frame_idx < end_frame:
            ret, frame = self._cap.read()
            if not ret:
                break
                
            if (frame_idx - start_frame) % stride == 0:
                yield frame_idx, frame
                
            frame_idx += 1
    
    def iter_all_with_skip(
        self,
        stride: int = 1,
    ) -> Iterator[tuple[int, np.ndarray | None]]:
        """
        Iterate over all frames, yielding None for skipped frames.
        
        Useful when you need to track frame indices but only process
        some frames.
        
        Args:
            stride: Process every N-th frame
            
        Yields:
            Tuple of (frame_index, frame_bgr or None for skipped)
        """
        frame_idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
                
            if frame_idx % stride == 0:
                yield frame_idx, frame
            else:
                yield frame_idx, None
                
            frame_idx += 1
