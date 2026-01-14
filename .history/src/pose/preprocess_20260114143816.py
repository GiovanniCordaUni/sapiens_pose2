"""
Preprocessing utilities for pose estimation.

Funzioni stateless per preparare le immagini per i modelli di posa umana.
"""

import cv2
import numpy as np
import torch


# ImageNet costanti di normalizzazione predefinite con i valori standard
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
    Preprocessa un crop di immagine BGR per la stima della posa.
    
    Esegue:
    1. Ridimensionamento alle dimensioni target
    2. Conversione BGR -> RGB
    3. Normalizzazione a [0, 1]
    4. Applicazione della normalizzazione mean/std (ImageNet di default)
    5. Conversione al formato tensor CHW
    6. Spostamento sul device specificato
    
    Args:
        frame_bgr: Immagine BGR di input come array numpy (H, W, 3)
        target_width: Larghezza target per il ridimensionamento
        target_height: Altezza target per il ridimensionamento
        device: Device PyTorch su cui posizionare il tensor ("cpu", "cuda", torch.device)
        mean: Media per canale per la normalizzazione (default: ImageNet)
        std: Deviazione standard per canale per la normalizzazione (default: ImageNet)
        
    Returns:
        Tensor preprocessato di shape (1, 3, H, W) sul device specificato
    """
    # Ridimensiona alle dimensioni target
    resized = cv2.resize(
        frame_bgr, 
        (target_width, target_height), 
        interpolation=cv2.INTER_LINEAR
    )
    
    # BGR -> RGB e normalizza a [0, 1]
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # Applica la normalizzazione mean/std
    rgb = (rgb - mean) / std
    
    # HWC -> CHW e aggiungi dimensione batch
    chw = np.transpose(rgb, (2, 0, 1))
    tensor = torch.from_numpy(chw).unsqueeze(0).float()
    
    # Sposta sul device
    return tensor.to(device)


def crop_person_from_frame(
    frame_bgr: np.ndarray,
    box_xyxy: list[int],
) -> np.ndarray:
    """
    Estrae un crop della persona da un frame dato un bounding box.
    
    Args:
        frame_bgr: FFrame completo come array numpy BGR
        box_xyxy: Bounding box come [x1, y1, x2, y2]
        
    Returns:
        Immagine BGR ritagliata
    """
    x1, y1, x2, y2 = box_xyxy
    return frame_bgr[y1:y2, x1:x2].copy()
