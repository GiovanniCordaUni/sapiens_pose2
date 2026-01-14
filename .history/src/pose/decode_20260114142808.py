"""
Decoding utilities per l'estimazione della posa umana dagli heatmaps.

Pure, stateless functions for converting model outputs to keypoints.
"""

import numpy as np


def heatmaps_to_keypoints(
    heatmaps: np.ndarray,
    output_width: int,
    output_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode heatmaps to keypoint coordinates and confidence scores.
    
    Takes heatmaps from a pose model and finds the argmax location for each
    keypoint, then scales the coordinates to the desired output size.
    
    Args:
        heatmaps: Heatmaps of shape (K, Hm, Wm) where K is number of keypoints,
                  Hm and Wm are the heatmap height and width
        output_width: Desired output width for coordinate scaling
        output_height: Desired output height for coordinate scaling
        
    Returns:
        keypoints: Array of shape (K, 2) with (x, y) coordinates
        scores: Array of shape (K,) with confidence scores (heatmap max values)
        
    Note:
        The scaling assumes the heatmap covers the full output area.
        Coordinates are scaled as: x = hm_x * (output_width / (Wm - 1))
    """
    K, Hm, Wm = heatmaps.shape
    
    # Flatten spatial dimensions to find argmax
    flat = heatmaps.reshape(K, -1)
    idx = np.argmax(flat, axis=1)
    scores = flat[np.arange(K), idx]
    
    # Convert flat indices to 2D coordinates
    ys = (idx // Wm).astype(np.float32)
    xs = (idx % Wm).astype(np.float32)
    
    # Scale to output dimensions
    # Avoid division by zero for single-pixel heatmaps
    xs = xs * (output_width / max(Wm - 1, 1))
    ys = ys * (output_height / max(Hm - 1, 1))
    
    keypoints = np.stack([xs, ys], axis=1)
    return keypoints, scores


def select_keypoints(
    keypoints: np.ndarray,
    scores: np.ndarray,
    indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select a subset of keypoints by indices.
    
    Con il checkpoint COCO 17 nativo, questa funzione usa identity mapping
    (indices = [0..16]) e restituisce tutti i keypoints senza trasformazioni.
    
    Args:
        keypoints: Full keypoint array of shape (K, 2)
        scores: Full scores array of shape (K,)
        indices: List of indices to select (identity: list(range(17)))
        
    Returns:
        selected_keypoints: Array of shape (len(indices), 2)
        selected_scores: Array of shape (len(indices),)
    """
    return keypoints[indices].copy(), scores[indices].copy()


def scale_keypoints_to_frame(
    keypoints: np.ndarray,
    crop_box_xyxy: list[int],
    model_input_width: int,
    model_input_height: int,
) -> np.ndarray:
    """
    Transform keypoints from model input space to original frame space.
    
    Args:
        keypoints: Keypoints in model input coordinates, shape (K, 2)
        crop_box_xyxy: The bounding box [x1, y1, x2, y2] used for cropping
        model_input_width: Width of the model input
        model_input_height: Height of the model input
        
    Returns:
        Keypoints transformed to frame coordinates, shape (K, 2)
    """
    x1, y1, x2, y2 = crop_box_xyxy
    crop_width = x2 - x1
    crop_height = y2 - y1
    
    # Scale and translate
    kpts = keypoints.copy()
    kpts[:, 0] = kpts[:, 0] * (crop_width / model_input_width) + x1
    kpts[:, 1] = kpts[:, 1] * (crop_height / model_input_height) + y1
    
    return kpts
