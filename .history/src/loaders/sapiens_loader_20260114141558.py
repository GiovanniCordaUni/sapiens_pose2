"""
Sapiens model loader.

Functions for loading and managing Sapiens pose estimation models.
"""

from pathlib import Path

import torch
import numpy as np

from ..pose.preprocess import preprocess_crop_bgr_to_tensor
from ..pose.decode import heatmaps_to_keypoints, select_keypoints, scale_keypoints_to_frame


def get_device(device: str = "auto") -> torch.device:
    """
    Get the appropriate torch device.
    
    Args:
        device: Device specification - "auto", "cuda", "cpu", or "mps"
        
    Returns:
        torch.device object
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device)


def load_sapiens(
    checkpoint_path: str | Path,
    device: str | torch.device = "auto",
) -> tuple[torch.jit.ScriptModule, torch.device]:
    """
    Load a Sapiens TorchScript model.
    
    Args:
        checkpoint_path: Path to the .pt2 checkpoint file
        device: Device to load model on ("auto", "cuda", "cpu", or torch.device)
        
    Returns:
        Tuple of (model, device) where model is loaded and in eval mode
        
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Sapiens checkpoint not found: {ckpt_path}")
    
    # Resolve device
    if isinstance(device, str):
        device = get_device(device)
    
    # Load model to device
    model = torch.jit.load(str(ckpt_path), map_location=device)
    model.eval()
    model.to(device)
    
    return model, device


class SapiensPoseEstimator:
    """
    Sapiens pose estimation wrapper.
    
    Encapsulates model loading, preprocessing, inference, and decoding.
    Il checkpoint COCO 17 produce direttamente 17 keypoints in formato COCO,
    nessun mapping aggiuntivo richiesto.
    
    Args:
        checkpoint_path: Path to Sapiens checkpoint (COCO 17 native)
        device: Device to run on ("auto", "cuda", "cpu")
        input_width: Model input width (default: 768)
        input_height: Model input height (default: 1024)
        
    Example:
        estimator = SapiensPoseEstimator("sapiens_0.3b_coco.pt2")
        keypoints, scores = estimator.predict_crop(crop_bgr)  # Returns 17 keypoints
    """
    
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device = "auto",
        input_width: int = 768,
        input_height: int = 1024,
    ):
        self.model, self.device = load_sapiens(checkpoint_path, device)
        self.input_width = input_width
        self.input_height = input_height
        
    @property
    def num_keypoints(self) -> int | None:
        """Number of keypoints in model output (determined after first inference)."""
        return getattr(self, "_num_keypoints", None)
        
    def predict_heatmaps(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        Run inference and return raw heatmaps.
        
        Args:
            crop_bgr: BGR image crop as numpy array
            
        Returns:
            Heatmaps of shape (K, Hm, Wm)
        """
        # Preprocess
        x = preprocess_crop_bgr_to_tensor(
            crop_bgr,
            self.input_width,
            self.input_height,
            device=self.device,
        )
        
        # Inference
        with torch.no_grad():
            y = self.model(x)
        
        if not torch.is_tensor(y) or y.ndim != 4:
            raise RuntimeError(
                f"Unexpected Sapiens output: type={type(y)}, shape={getattr(y, 'shape', None)}"
            )
        
        heatmaps = y[0].detach().cpu().numpy()
        self._num_keypoints = heatmaps.shape[0]
        
        return heatmaps
    
    def predict_crop(
        self,
        crop_bgr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict keypoints from a crop in model input space.
        
        Args:
            crop_bgr: BGR image crop as numpy array
            
        Returns:
            keypoints: Array of shape (K, 2) in model input coordinates
            scores: Array of shape (K,) with confidence scores
        """
        heatmaps = self.predict_heatmaps(crop_bgr)
        return heatmaps_to_keypoints(
            heatmaps,
            self.input_width,
            self.input_height,
        )
    
    def predict_frame(
        self,
        crop_bgr: np.ndarray,
        crop_box_xyxy: list[int],
        keypoint_indices: list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predice i keypoints da un crop e li trasforma nelle coordinate del frame originale.
        
        Con il checkpoint COCO 17 nativo, il modello restituisce direttamente
        17 keypoints. keypoint_indices usa identity mapping [0..16].
        
        Args:
            crop_bgr: Crop dell'immagine in formato BGR come array numpy
            crop_box_xyxy: Bounding box [x1, y1, x2, y2] usato per il crop
            keypoint_indices: Indici di identità [0..16] per COCO 17 nativo
            
        Returns:
            keypoints: Array di forma (17, 2) nelle coordinate del frame
            scores: Array di forma (17,) con i punteggi di confidenza
        """
        keypoints, scores = self.predict_crop(crop_bgr)
        
        # Seleziona subset di keypoints se richiesto
        if keypoint_indices is not None:
            keypoints, scores = select_keypoints(keypoints, scores, keypoint_indices)
        
        # Transform to frame coordinates
        keypoints = scale_keypoints_to_frame(
            keypoints,
            crop_box_xyxy,
            self.input_width,
            self.input_height,
        )
        
        return keypoints, scores
