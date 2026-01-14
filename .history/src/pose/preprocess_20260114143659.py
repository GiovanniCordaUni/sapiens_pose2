"""
Preprocessing utilities for pose estimation.

Funzioni stateless per preparare le immagini per i modelli di posa umana.
"""

import cv2
import numpy as np
import torch


# ImageNet costanti di normalizzazione
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_crop_bgr_to_tensor(
    frame_bgr: np.ndarray,
    target_width: int,
    target_height: int,
    device: torch.device | str = "cpu",
    mean: np.ndarray = IMAGENET_MEAN,
    std: np.ndarray = IMAGENET_STD,
) -> torch.Tensor:
    """
    Preprocess a BGR image crop for pose estimation.
    
    Performs:
    1. Resize to target dimensions
    2. BGR -> RGB conversion
    3. Normalize to [0, 1]
    4. Apply mean/std normalization (ImageNet by default)
    5. Convert to CHW tensor format
    6. Move to specified device
    
    Args:
        frame_bgr: Input BGR image as numpy array (H, W, 3)
        target_width: Target width for resizing
        target_height: Target height for resizing
        device: PyTorch device to place tensor on ("cpu", "cuda", torch.device)
        mean: Per-channel mean for normalization (default: ImageNet)
        std: Per-channel std for normalization (default: ImageNet)
        
    Returns:
        Preprocessed tensor of shape (1, 3, H, W) on specified device
    """
    # Resize to target dimensions
    resized = cv2.resize(
        frame_bgr, 
        (target_width, target_height), 
        interpolation=cv2.INTER_LINEAR
    )
    
    # BGR -> RGB and normalize to [0, 1]
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # Apply mean/std normalization
    rgb = (rgb - mean) / std
    
    # HWC -> CHW and add batch dimension
    chw = np.transpose(rgb, (2, 0, 1))
    tensor = torch.from_numpy(chw).unsqueeze(0).float()
    
    # Move to device
    return tensor.to(device)


def crop_person_from_frame(
    frame_bgr: np.ndarray,
    box_xyxy: list[int],
) -> np.ndarray:
    """
    Extract a person crop from a frame given bounding box.
    
    Args:
        frame_bgr: Full frame as BGR numpy array
        box_xyxy: Bounding box as [x1, y1, x2, y2]
        
    Returns:
        Cropped BGR image
    """
    x1, y1, x2, y2 = box_xyxy
    return frame_bgr[y1:y2, x1:x2].copy()
