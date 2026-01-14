"""
Video writing utilities.

Classi e funzioni per la scrittura di file video.
"""

from pathlib import Path

import cv2
import numpy as np


class VideoWriter:
    """
    Scrittore di file video con supporto per context manager.
    
    Args:
        output_path: Percorso per il file video di output
        width: Larghezza del frame
        height: Altezza del frame
        fps: Frame rate
        fourcc: Video codec (default: "mp4v")
        
    Example:
        with VideoWriter("output.mp4", 1920, 1080, 30.0) as writer:
            writer.write(frame)
    """
    
    def __init__(
        self,
        output_path: str | Path,
        width: int,
        height: int,
        fps: float,
        fourcc: str = "mp4v",
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self._writer = None
        self._frame_count = 0
        
    def __enter__(self):
        fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
        self._writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (self.width, self.height),
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot create video writer: {self.output_path}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._writer:
            self._writer.release()
        return False
    
    @property # @property serve per fare in modo che frame_count sia accessibile come attributo
    def frame_count(self) -> int:
        """Numero di frame scritti."""
        return self._frame_count
    
    def write(self, frame: np.ndarray) -> None:
        """Scrive un singolo frame."""
        if self._writer is None:
            raise RuntimeError("VideoWriter deve essere usato come context manager")
        self._writer.write(frame)
        self._frame_count += 1


def create_overlay_writer(
    output_path: str | Path,
    source_width: int,
    source_height: int,
    source_fps: float,
    frame_stride: int = 1,
) -> VideoWriter:
    """
    Crea un VideoWriter configurato per video con overlay.
    
    Regola il FPS in base al frame stride per mantenere la corretta velocità di riproduzione.
    
    Args:
        output_path: Percorso per il video di output
        source_width: Larghezza del video originale
        source_height: Altezza del video originale
        source_fps: FPS del video originale
        frame_stride: Passo di elaborazione dei frame
        
    Returns:
        Configured VideoWriter instance
    """
    output_fps = source_fps / frame_stride
    return VideoWriter(
        output_path,
        source_width,
        source_height,
        output_fps,
    )
