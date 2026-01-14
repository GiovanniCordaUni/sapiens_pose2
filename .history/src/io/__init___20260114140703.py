"""
I/O utilities.

Funzioni e classi per la lettura e scrittura di video e file JSONL.
"""

from .video_reader import VideoReader
from .video_writer import VideoWriter, create_overlay_writer
from .jsonl_writer import (
    JsonlWriter,
    create_pose_record,
    keypoints_to_list,
)

__all__ = [
    "VideoReader",
    "VideoWriter",
    "create_overlay_writer",
    "JsonlWriter",
    "create_pose_record",
    "keypoints_to_list",
]
