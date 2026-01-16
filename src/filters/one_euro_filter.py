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
    Stabilizzatore per keypoints 2D usando One-Euro Filter + Hold.
    
    Combina due tecniche:
    1. Hold: se conf < threshold, mantiene la posizione precedente
    2. One-Euro: smooth sui keypoints validi per ridurre jitter
    
    Mantiene un filtro separato per ogni keypoint (x, y).
    """
    
    def __init__(
        self,
        num_keypoints: int = 17,
        min_cutoff: float = 1.5,
        beta: float = 0.01,
        d_cutoff: float = 1.0,
        use_one_euro: bool = True,
        use_hold: bool = True,
        hold_decay: float = 0.95,
    ):
        """
        Args:
            num_keypoints: Numero di keypoints da tracciare
            min_cutoff: Cutoff minimo (più basso = più smooth)
            beta: Reattività ai movimenti rapidi (più alto = più reattivo)
            d_cutoff: Cutoff per la derivata
            use_one_euro: Abilita filtro One-Euro per smoothing
            use_hold: Abilita hold per keypoints con bassa confidenza
            hold_decay: Decay della confidenza quando si usa hold (0.9-0.99)
        """
        self.num_keypoints = num_keypoints
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.use_one_euro = use_one_euro
        self.use_hold = use_hold
        self.hold_decay = hold_decay
        
        # Filtri One-Euro per x e y di ogni keypoint
        self.filters_x: list[OneEuroFilter] = []
        self.filters_y: list[OneEuroFilter] = []
        
        # Buffer per hold
        self.prev_keypoints: np.ndarray | None = None
        self.prev_scores: np.ndarray | None = None
        
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
        Filtra i keypoints per ridurre il jitter e mantenere stabilità.
        
        Logica:
        1. Se conf >= threshold: aggiorna posizione (+ smooth se One-Euro attivo)
        2. Se conf < threshold e hold attivo: usa posizione precedente
        
        Args:
            keypoints: Array (N, 2) delle coordinate x, y
            scores: Array (N,) delle confidenze
            timestamp: Timestamp del frame in secondi
            confidence_threshold: Soglia minima per considerare il keypoint valido
            
        Returns:
            Tuple (keypoints_filtrati, scores_filtrati)
        """
        # Prima chiamata: inizializza buffer
        if self.prev_keypoints is None:
            self.prev_keypoints = keypoints.copy()
            self.prev_scores = scores.copy()
            # Non applico One-Euro al primo frame
            return keypoints.copy(), scores.copy()
        
        filtered_kpts = keypoints.copy()
        filtered_scores = scores.copy()
        num_kpts = min(len(scores), self.num_keypoints)
        
        for i in range(num_kpts):
            if scores[i] >= confidence_threshold:
                # Confidenza alta → aggiorna (con o senza smooth)
                if self.use_one_euro:
                    filtered_kpts[i, 0] = self.filters_x[i].filter(keypoints[i, 0], timestamp)
                    filtered_kpts[i, 1] = self.filters_y[i].filter(keypoints[i, 1], timestamp)
                # else: mantieni keypoints originali (già in filtered_kpts)
                
                # Aggiorna buffer
                self.prev_keypoints[i] = filtered_kpts[i]
                self.prev_scores[i] = scores[i]
                
            elif self.use_hold and self.prev_scores[i] >= confidence_threshold:
                # Confidenza bassa ma hold attivo e precedente valido
                filtered_kpts[i] = self.prev_keypoints[i]
                filtered_scores[i] = self.prev_scores[i] * self.hold_decay
                
                # Aggiorna buffer con score decayed
                self.prev_scores[i] = filtered_scores[i]
                
                # Resetta filtro One-Euro per questo keypoint
                # (evita salti quando torna visibile)
                if self.use_one_euro:
                    self.filters_x[i].reset()
                    self.filters_y[i].reset()
            else:
                # Confidenza bassa e nessun hold valido
                # Resetta filtro
                if self.use_one_euro:
                    self.filters_x[i].reset()
                    self.filters_y[i].reset()
        
        return filtered_kpts, filtered_scores
    
    def reset(self):
        """Resetta tutti i filtri e buffer (es. per nuova persona/video)."""
        self._init_filters()
        self.prev_keypoints = None
        self.prev_scores = None
