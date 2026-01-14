"""
Decoding utilities per la stima della posa umana dalle heatmaps .

funzione stateless per convertire le uscite del modello in keypoints.
"""

import numpy as np


def heatmaps_to_keypoints(
    heatmaps: np.ndarray,
    output_width: int,
    output_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decodifica le heatmaps in coordinate dei keypoints e punteggi di confidenza.
    
    Prende le heatmaps da un modello di posa e trova la posizione dell'argmax per ogni
    keypoint, quindi scala le coordinate alla dimensione di output desiderata.
    
    Args:
        heatmaps: Heatmaps con shape (K, Hm, Wm) dove K è il numero di keypoints,
                  Hm e Wm sono l'altezza e la larghezza della heatmap
        output_width: Larghezza di output desiderata per la scalatura delle coordinate
        output_height: Altezza di output desiderata per la scalatura delle coordinate
        
    Returns:
        keypoints: Array di shape (K, 2) con coordinate (x, y)
        scores: Array di shape (K,) con punteggi di confidenza (valori massimi delle heatmap)
        
    Note:
        Lo scaling assume che la heatmap copra l'intera area di output.
        Le coordinate sono scalate come: x = hm_x * (output_width / (Wm - 1))
    """
    K, Hm, Wm = heatmaps.shape
    
    # Appiattisci le dimensioni spaziali per trovare l'argmax
    flat = heatmaps.reshape(K, -1)
    idx = np.argmax(flat, axis=1)
    scores = flat[np.arange(K), idx]
    
    # Converti gli indici piatti in coordinate 2D
    ys = (idx // Wm).astype(np.float32)
    xs = (idx % Wm).astype(np.float32)
    
    # Scala alle dimensioni di output
    # Evita la divisione per zero per heatmap a singolo pixel
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
    Seleziona un sottoinsieme di keypoints per indici.
    
    Con il checkpoint COCO 17 nativo, questa funzione usa identity mapping
    (indices = [0..16]) e restituisce tutti i keypoints senza trasformazioni.
    
    Args:
        keypoints: Array completo di keypoints di shape (K, 2)
        scores: Array completo di punteggi di shape (K,)
        indices: Lista di indici da selezionare (identity: list(range(17)))
        
    Returns:
        selected_keypoints: Array di shape (len(indices), 2)
        selected_scores: Array di shape (len(indices),)
    """
    return keypoints[indices].copy(), scores[indices].copy()


def scale_keypoints_to_frame(
    keypoints: np.ndarray,
    crop_box_xyxy: list[int],
    model_input_width: int,
    model_input_height: int,
) -> np.ndarray:
    """
    Trasforma i keypoints dallo spazio di input del modello allo spazio del frame originale.
    
    Args:
        keypoints: Keypoints nelle coordinate di input del modello, shape (K, 2)
        crop_box_xyxy: Il bounding box [x1, y1, x2, y2] usato per il ritaglio
        model_input_width: Larghezza dell'input del modello
        model_input_height: Altezza dell'input del modello
        
    Returns:
        Keypoints trasformati alle coordinate del frame, shape (K, 2)
    """
    x1, y1, x2, y2 = crop_box_xyxy
    crop_width = x2 - x1
    crop_height = y2 - y1
    
    # Scala e trasla i keypoints per adattarli al frame originale
    kpts = keypoints.copy()
    kpts[:, 0] = kpts[:, 0] * (crop_width / model_input_width) + x1
    kpts[:, 1] = kpts[:, 1] * (crop_height / model_input_height) + y1
    
    return kpts
