"""
Pose estimation utilities.

Esegue preprocessing, decoding, e operazioni correlate alla stima della posa.
"""

from .preprocess import (
    preprocess_crop_bgr_to_tensor,
    crop_person_from_frame,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

from .decode import (
    heatmaps_to_keypoints,
    select_keypoints,
    scale_keypoints_to_frame,
)

__all__ = [
    # Preprocessing
    "preprocess_crop_bgr_to_tensor",
    "crop_person_from_frame",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    # Decoding
    "heatmaps_to_keypoints",
    "select_keypoints",
    "scale_keypoints_to_frame",
]
