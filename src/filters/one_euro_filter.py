"""
One-Euro Filter per la stabilizzazione temporale dei keypoints.

Riduce il jitter (vibrazione) mantenendo la reattività ai movimenti veloci.
Reference: https://cristal.univ-lille.fr/~casiez/1euro/
"""

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LowPassFilter:
    """Filtro passa-basso di primo ordine."""
    alpha: float = 1.0
    y: float | None = None
    s: float | None = None
    
    def filter(self, value: float) -> float:
        if self.y is None:
            self.s = value
        else:
            self.s = self.alpha * value + (1.0 - self.alpha) * self.s
        self.y = value
        return self.s
    
    def reset(self):
        self.y = None
        self.s = None


@dataclass 
class OneEuroFilter:
    """
    One-Euro Filter per segnali 1D.
    
    Args:
        min_cutoff: Frequenza di cutoff minima (Hz). Valori bassi = più smoothing.
        beta: Coefficiente di velocità. Valori alti = più reattivo ai movimenti rapidi.
        d_cutoff: Frequenza di cutoff per la derivata.
    """
    min_cutoff: float = 1.0
    beta: float = 0.007
    d_cutoff: float = 1.0
    
    x_filter: LowPassFilter = field(default_factory=LowPassFilter)
    dx_filter: LowPassFilter = field(default_factory=LowPassFilter)
    last_time: float | None = None
    
    def __post_init__(self):
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
    
    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
    
    def filter(self, x: float, t: float) -> float:
        """
        Filtra un valore.
        
        Args:
            x: Valore da filtrare
            t: Timestamp in secondi
            
        Returns:
            Valore filtrato
        """
        if self.last_time is None:
            self.last_time = t
            self.x_filter.s = x
            self.dx_filter.s = 0.0
            return x
        
        dt = t - self.last_time
        if dt <= 0:
            dt = 1e-6  # Evita divisione per zero
        
        self.last_time = t
        
        # Stima della derivata
        dx = (x - (self.x_filter.s or x)) / dt
        self.dx_filter.alpha = self._alpha(self.d_cutoff, dt)
        edx = self.dx_filter.filter(dx)
        
        # Cutoff adattivo basato sulla velocità
        cutoff = self.min_cutoff + self.beta * abs(edx)
        self.x_filter.alpha = self._alpha(cutoff, dt)
        
        return self.x_filter.filter(x)
    
    def reset(self):
        """Resetta lo stato del filtro."""
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None


class KeypointStabilizer:
    """
    Stabilizzatore per keypoints 2D usando One-Euro Filter.
    
    Mantiene un filtro separato per ogni keypoint (x, y).
    """
    
    def __init__(
        self,
        num_keypoints: int = 17,
        min_cutoff: float = 1.5,
        beta: float = 0.01,
        d_cutoff: float = 1.0,
    ):
        """
        Args:
            num_keypoints: Numero di keypoints da tracciare
            min_cutoff: Cutoff minimo (più basso = più smooth)
            beta: Reattività ai movimenti rapidi (più alto = più reattivo)
            d_cutoff: Cutoff per la derivata
        """
        self.num_keypoints = num_keypoints
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        # Filtri per x e y di ogni keypoint
        self.filters_x: list[OneEuroFilter] = []
        self.filters_y: list[OneEuroFilter] = []
        self._init_filters()
    
    def _init_filters(self):
        """Inizializza i filtri per tutti i keypoints."""
        self.filters_x = [
            OneEuroFilter(self.min_cutoff, self.beta, self.d_cutoff)
            for _ in range(self.num_keypoints)
        ]
        self.filters_y = [
            OneEuroFilter(self.min_cutoff, self.beta, self.d_cutoff)
            for _ in range(self.num_keypoints)
        ]
    
    def filter(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        timestamp: float,
        confidence_threshold: float = 0.3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Filtra i keypoints per ridurre il jitter.
        
        Args:
            keypoints: Array (N, 2) di coordinate (x, y) per ogni keypoint
            scores: Array (N,) di confidenze
            timestamp: Timestamp del frame in secondi
            confidence_threshold: Soglia minima per applicare il filtro
            
        Returns:
            Tuple di (keypoints filtrati, scores)
        """
        filtered_keypoints = keypoints.copy()
        num_kpts = min(len(keypoints), self.num_keypoints)
        
        for i in range(num_kpts):
            x, y = keypoints[i]
            conf = scores[i]
            
            # Se la confidenza è troppo bassa, resetta il filtro
            if conf < confidence_threshold:
                self.filters_x[i].reset()
                self.filters_y[i].reset()
                continue
            
            # Applica il filtro
            filtered_keypoints[i, 0] = self.filters_x[i].filter(x, timestamp)
            filtered_keypoints[i, 1] = self.filters_y[i].filter(y, timestamp)
        
        return filtered_keypoints, scores
    
    def reset(self):
        """Resetta tutti i filtri (es. per nuova persona/video)."""
        self._init_filters()
