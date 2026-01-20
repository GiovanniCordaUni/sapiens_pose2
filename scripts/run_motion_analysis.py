#!/usr/bin/env python3
"""
Analisi del movimento dai keypoints JSONL.
Rileva intervalli di movimento e calcola tempi di esecuzione esercizi.

Usage:
    python run_motion_analysis.py --input file.jsonl
    python run_motion_analysis.py --input cartella/ --batch
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from sklearn.cluster import KMeans

import numpy as np
import matplotlib.pyplot as plt

# Indici COCO-17
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Pesi per keypoint (parti piu mobili hanno peso maggiore)
# Arti inferiori e superiori pesano di piu del torso/testa
KEYPOINT_WEIGHTS = {
    0: 0.0,   # nose
    1: 0.0,   # left_eye
    2: 0.0,   # right_eye
    3: 0.0,   # left_ear
    4: 0.0,   # right_ear
    5: 1.0,   # left_shoulder
    6: 1.0,   # right_shoulder
    7: 0.8,   # left_elbow
    8: 0.8,   # right_elbow
    9: 0.7,   # left_wrist
    10: 0.7,  # right_wrist
    11: 1.0,  # left_hip
    12: 1.0,  # right_hip
    13: 0.8,  # left_knee
    14: 0.8,  # right_knee
    15: 0.7,  # left_ankle
    16: 0.7,  # right_ankle
}


def carica_jsonl(path: Path) -> list[dict]:
    """Carica records da file JSONL o JSON."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    
    if text.startswith("["):
        return json.loads(text)
    
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def calcola_scala_corpo(kpts: list, conf_th: float) -> float:
    """
    Calcola scala del corpo per normalizzazione.
    Usa la diagonale del bounding box dei keypoints validi.
    """
    valid_pts = []
    for i, kpt in enumerate(kpts):
        if len(kpt) >= 3 and kpt[2] >= conf_th:
            valid_pts.append([kpt[0], kpt[1]])
    
    if len(valid_pts) < 2:
        return 1.0
    
    pts = np.array(valid_pts)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    diag = np.linalg.norm(max_xy - min_xy)
    
    return max(diag, 1.0)


def calcola_segnale_movimento(
    records: list[dict],
    conf_th: float = 0.25,
    smooth_window: int = 5,
    use_weights: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcola segnale di movimento normalizzato usando TUTTI i keypoints.
    
    Logica:
    1. Per ogni frame, calcola lo spostamento di ogni keypoint valido
    2. Aggrega con media pesata (parti mobili pesano di piu)
    3. Normalizza per dimensione corpo (diagonale bbox keypoints)
    4. Applica smoothing per ridurre rumore di movimenti casuali
    
    Returns:
        (timestamps, movimento_smooth)
    """
    # Filtra records validi
    usable = [
        r for r in records
        if r.get("status", "ok") == "ok"
        and isinstance(r.get("keypoints17"), list)
        and "time_sec" in r
    ]
    
    if len(usable) < 2:
        raise ValueError(f"Servono almeno 2 frame validi, trovati {len(usable)}")
    
    usable.sort(key=lambda r: r.get("frame_idx", r.get("frame", 0)))
    
    t = np.array([r["time_sec"] for r in usable])
    m = np.zeros(len(usable))
    
    prev_kpts = None
    prev_confs = None
    
    for i, r in enumerate(usable):
        kpts = r["keypoints17"]
        num_kpts = len(kpts)
        
        # Estrai coordinate e confidenze
        curr_kpts = np.zeros((num_kpts, 2))
        curr_confs = np.zeros(num_kpts)
        
        for j, kpt in enumerate(kpts):
            # se kpt ha almeno x,y,conf
            if len(kpt) >= 3:
                curr_kpts[j] = [kpt[0], kpt[1]]
                curr_confs[j] = kpt[2]
        
        # Primo frame -> inizializza prev
        if i == 0:
            prev_kpts = curr_kpts.copy()
            prev_confs = curr_confs.copy()
            continue
        
        # Calcola spostamenti pesati per ogni keypoint
        displacements = []
        weights = []
        
        for j in range(num_kpts):
            # Entrambi i frame devono avere confidenza sufficiente
            if curr_confs[j] >= conf_th and prev_confs[j] >= conf_th:
                dist = np.linalg.norm(curr_kpts[j] - prev_kpts[j])
                w = KEYPOINT_WEIGHTS.get(j, 0.5) if use_weights else 1.0
                displacements.append(dist)
                weights.append(w)
        
        # Aggiorna prev
        prev_kpts = curr_kpts.copy()
        prev_confs = curr_confs.copy()
        
        if not displacements: # Nessun keypoint valido
            m[i] = m[i-1] if i > 0 else 0
            continue
        
        # Media pesata degli spostamenti
        displacements = np.array(displacements)
        weights = np.array(weights)
        disp = np.sum(displacements * weights) / np.sum(weights)
        
        # Normalizza per scala corpo
        scale = calcola_scala_corpo(kpts, conf_th)
        m[i] = disp / scale
    
    # Smooth con moving average
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        m = np.convolve(m, kernel, mode="same")
    
    return t, m

def rileva_intervalli(
    t: np.ndarray,
    m: np.ndarray,
    soglia_on: float = 0.02,
    soglia_off: float = 0.008,
    min_frames: int = 3,
    merge_gap: float = 1.0,   # parametro per unire intervalli vicini
    min_duration: float = 1.0 # Durata minima del blocco UNITO finale
) -> list[tuple[float, float]]:
    """
    Rileva, unisce intervalli vicini e filtra quelli troppo brevi.
    Gestisce sia ripetizioni veloci che rumore isolato.
    """
    n = len(m)
    raw_intervals = []
    
    # rilevamento intervalli base (grezzi)
    in_movimento = False
    above_count = below_count = 0
    start_idx = None
    
    for i in range(n):
        if not in_movimento:
            above_count = above_count + 1 if m[i] > soglia_on else 0
            if above_count >= min_frames:
                in_movimento = True
                start_idx = i - min_frames + 1
                below_count = 0
        else:
            below_count = below_count + 1 if m[i] < soglia_off else 0
            if below_count >= min_frames:
                in_movimento = False
                end_idx = i - min_frames + 1
                if start_idx is not None and end_idx > start_idx:
                    raw_intervals.append((t[start_idx], t[end_idx]))
                start_idx = None
                above_count = 0
    
    if in_movimento and start_idx is not None:
        raw_intervals.append((t[start_idx], t[-1]))

    if not raw_intervals:
        return []

    
    # Unisce intervalli se la pausa tra loro è < merge_gap
    merged_intervals = [raw_intervals[0]]
    
    for curr_start, curr_end in raw_intervals[1:]:
        last_start, last_end = merged_intervals[-1]
        
        # Calcola il buco temporale tra la fine del precedente e l'inizio del corrente
        gap = curr_start - last_end
        
        if gap <= merge_gap:
            # caso di unione fra intervalli
            merged_intervals[-1] = (last_start, curr_end)
        else:
            # se Il buco è troppo grande, è un altro evento (o rumore)
            merged_intervals.append((curr_start, curr_end))

    
    # pulisce intervalli troppo brevi dopo l'unione, causati da movimenti isolati non facenti parte
    # di un esercizio (es. un movimento casuale alla fine)
    final_intervals = []
    for start, end in merged_intervals:
        duration = end - start
        if duration >= min_duration:
            final_intervals.append((start, end))
            
    return final_intervals

"""
    Sono stati testati diversi metodi per il calcolo automatico delle soglie
    basati sulla distribuzione del segnale di movimento.
    I metodi includono percentili, Otsu e K-Means.

    a scelta si possono decommentare le funzioni corrispondenti e usarle commentando
    la funzione attualmente in uso.

    NOTA: k-means sembra dare i risultati più stabili e affidabili considerando la variabilità degli esercizi
"""
# def calcola_soglie_automatiche(m: np.ndarray) -> tuple[float, float]:
#     """
#     Calcola soglie automatiche basate sulla distribuzione del segnale.
    
#     Usa percentili per adattarsi al tipo di esercizio.
#     """
#     # Rimuovi valori nulli/iniziali
#     m_valid = m[m > 0]
#     if len(m_valid) < 10:
#         return 0.02, 0.008
    
#     # Soglia ON: percentile 60-70 (sopra la mediana)
#     soglia_on = np.percentile(m_valid, 60)
    
#     # Soglia OFF: percentile 20-30 (sotto la mediana)
#     soglia_off = np.percentile(m_valid, 35)
    
#     # Assicura che ON > OFF con margine
#     if soglia_on <= soglia_off:
#         soglia_on = soglia_off * 1.5
    
#     return float(soglia_on), float(soglia_off)

# def calcola_soglie_automatiche(m: np.ndarray) -> tuple[float, float]:
#     """
#     Calcola soglie automatiche con metodo Otsu.
#     Trova la soglia che meglio separa movimento da fermo.
#     """
#     m_valid = m[m > 0]
#     if len(m_valid) < 10:
#         return 0.02, 0.008
    
#     # Normalizza a 0-255 per Otsu
#     m_min, m_max = m_valid.min(), m_valid.max()
#     if m_max - m_min < 1e-6:
#         return 0.02, 0.008
    
#     m_norm = ((m_valid - m_min) / (m_max - m_min) * 255).astype(np.uint8)
    
#     # Istogramma
#     hist, _ = np.histogram(m_norm, bins=256, range=(0, 256))
#     hist = hist.astype(float) / hist.sum()
    
#     # Trova soglia ottimale Otsu
#     best_thresh, best_var = 0, 0
#     for t in range(1, 255):
#         w0, w1 = hist[:t].sum(), hist[t:].sum()
#         if w0 == 0 or w1 == 0:
#             continue
#         m0 = (np.arange(t) * hist[:t]).sum() / w0
#         m1 = (np.arange(t, 256) * hist[t:]).sum() / w1
#         var = w0 * w1 * (m0 - m1) ** 2
#         if var > best_var:
#             best_var, best_thresh = var, t
    
#     # Converti a scala originale
#     soglia_otsu = m_min + (best_thresh / 255) * (m_max - m_min)
#     soglia_on = soglia_otsu * 0.8
#     soglia_off = soglia_on * 0.4
    
#     return float(soglia_on), float(soglia_off)


def calcola_soglie_automatiche(m: np.ndarray) -> tuple[float, float]:
    """
    Metodo K-Means: Divide i valori di movimento in 2 cluster
    (Fermo vs Mosso) e prende il punto medio tra i centroidi.
    """
    m_valid = m[m > 0].reshape(-1, 1)
    if len(m_valid) < 10: return 0.02, 0.008

    # Clustering in 2 gruppi
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(m_valid)
    
    # Centroidi dei due gruppi
    centers = sorted(kmeans.cluster_centers_.flatten())
    low_center = centers[0]  # Centro del gruppo "fermo"
    high_center = centers[1] # Centro del gruppo "mosso"
    
    # Spesso K-means mette la soglia troppo alta se il movimento è esplosivo.
    # Prendo un punto in percentuale della distanza tra Low e High
    soglia_on = low_center + 0.4 * (high_center - low_center)
    
    soglia_off = soglia_on * 0.5
    
    return float(soglia_on), float(soglia_off)


def genera_grafici(
    t: np.ndarray,
    m: np.ndarray,
    intervalli: list[tuple[float, float]],
    out_path: Path,
    soglia_on: float,
    soglia_off: float,
) -> None:
    """Genera grafico combinato movimento + intervalli."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    
    # Segnale movimento
    ax1.plot(t, m, 'b-', linewidth=1)
    ax1.axhline(soglia_on, color='g', linestyle='--', alpha=0.7, label=f'ON={soglia_on:.4f}')
    ax1.axhline(soglia_off, color='r', linestyle='--', alpha=0.7, label=f'OFF={soglia_off:.4f}')
    ax1.set_ylabel("Movimento (norm.)")
    ax1.legend(loc='upper right')
    ax1.set_title("Analisi Movimento (tutti i keypoints)")
    
    # Stato binario
    state = np.zeros(len(t))
    for start, end in intervalli:
        mask = (t >= start) & (t <= end)
        state[mask] = 1
    
    ax2.fill_between(t, state, step="post", alpha=0.5, color='orange')
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_xlabel("Tempo (s)")
    ax2.set_ylabel("Stato")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Fermo", "In movimento"])
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def processa_file(
    input_path: Path,
    output_dir: Path,
    conf_th: float,
    smooth_window: int,
    soglia_on: float | None,
    soglia_off: float | None,
    min_frames: int,
    salva_grafici: bool,
    use_weights: bool,
    auto_threshold: bool,
) -> dict | None:
    """Processa un singolo file JSONL."""
    print(f"\n[FILE] Elaborazione: {input_path.name}")
    
    try:
        records = carica_jsonl(input_path)
        if not records:
            print("  [WARN] Nessun record trovato")
            return None
        
        t, m = calcola_segnale_movimento(records, conf_th, smooth_window, use_weights)
        
        # Soglie automatiche se richiesto
        if auto_threshold or soglia_on is None or soglia_off is None:
            soglia_on_auto, soglia_off_auto = calcola_soglie_automatiche(m)
            if auto_threshold:
                soglia_on = soglia_on_auto
                soglia_off = soglia_off_auto
                print(f"  [INFO] Soglie automatiche: ON={soglia_on:.4f}, OFF={soglia_off:.4f}")
        
    
        intervalli = rileva_intervalli(
            t, m, 
            soglia_on, soglia_off, 
            min_frames,
            merge_gap=1.5,      # Unisce ripetizioni se pausa < 1.5s (gestisce bene 5xSTS)
            min_duration=2.0    # Ignora blocchi isolati < 2.0s (cancella il rumore finale)
        )

    except Exception as e:
        print(f"  [ERROR] {e}")
        return None
    
    # Calcola statistiche
    durata_totale = t[-1] - t[0]
    tempo_movimento = sum(end - start for start, end in intervalli)
    
    risultato = {
        "file": input_path.name,
        "durata_totale_sec": round(durata_totale, 2),
        "tempo_movimento_sec": round(tempo_movimento, 2),
        "percentuale_movimento": round(tempo_movimento / durata_totale * 100, 1) if durata_totale > 0 else 0,
        "num_intervalli": len(intervalli),
        "soglia_on": round(soglia_on, 4),
        "soglia_off": round(soglia_off, 4),
        "intervalli": [{"inizio": round(a, 2), "fine": round(b, 2), "durata": round(b-a, 2)} 
                       for a, b in intervalli],
    }
    
    # Output
    file_out_dir = output_dir / input_path.stem
    file_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Salva JSON
    with open(file_out_dir / "risultato.json", 'w', encoding='utf-8') as f:
        json.dump(risultato, f, indent=2, ensure_ascii=False)
    
    # Salva grafici
    if salva_grafici:
        genera_grafici(t, m, intervalli, file_out_dir / "grafico.png", soglia_on, soglia_off)
    
    # Stampa risultati
    print(f"  [INFO] Durata totale: {durata_totale:.1f}s")
    print(f"  [INFO] Tempo in movimento: {tempo_movimento:.1f}s ({risultato['percentuale_movimento']}%)")
    print(f"  [INFO] Intervalli rilevati: {len(intervalli)}")
    
    if intervalli:
        for i, (start, end) in enumerate(intervalli, 1):
            print(f"      {i}) {start:.2f}s -> {end:.2f}s (durata: {end-start:.2f}s)")
    
    return risultato


def main():
    parser = argparse.ArgumentParser(description="Analisi movimento da keypoints")
    parser.add_argument("--input", "-i", required=True, help="File JSONL o cartella")
    parser.add_argument("--output", "-o", default=None, help="Cartella output")
    parser.add_argument("--batch", action="store_true", help="Processa tutti i .jsonl")
    parser.add_argument("--conf_th", type=float, default=0.25, help="Soglia confidenza keypoints")
    parser.add_argument("--smooth", type=int, default=5, help="Finestra smoothing")
    parser.add_argument("--ton", type=float, default=None, help="Soglia inizio movimento (auto se non specificata)")
    parser.add_argument("--toff", type=float, default=None, help="Soglia fine movimento (auto se non specificata)")
    parser.add_argument("--auto_threshold", action="store_true", help="Calcola soglie automaticamente")
    parser.add_argument("--min_frames", type=int, default=3, help="Frame consecutivi per cambio stato")
    parser.add_argument("--no_weights", action="store_true", help="Non usare pesi per keypoint")
    parser.add_argument("--no_grafici", action="store_true", help="Non generare grafici")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    # Output directory di default
    if args.output:
        output_dir = Path(args.output)
    else:
        # Default: data/output/test_tempi nella root del progetto
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        output_dir = project_root / "data" / "output" / "test_tempi"
    
    # Trova file da processare
    if args.batch or input_path.is_dir():
        files = sorted(input_path.glob("*.jsonl"))
    else:
        files = [input_path]
    
    if not files:
        print(f"[ERROR] Nessun file .jsonl trovato in {input_path}")
        return
    
    # Se non specificate soglie, usa auto
    auto_threshold = args.auto_threshold or (args.ton is None and args.toff is None)
    soglia_on = args.ton if args.ton is not None else 0.02
    soglia_off = args.toff if args.toff is not None else 0.008
    
    print("=" * 60) 
    print("ANALISI MOVIMENTO")
    print("=" * 60)
    print(f"[CONFIG] File da elaborare: {len(files)}")
    print(f"[CONFIG] Output: {output_dir}")
    print(f"[CONFIG] Soglie: {'AUTO' if auto_threshold else f'ON={soglia_on}, OFF={soglia_off}'}")
    print(f"[CONFIG] Pesi keypoint: {'NO' if args.no_weights else 'SI'}")
    
    risultati = []
    for file_path in files:
        r = processa_file(
            file_path, output_dir,
            args.conf_th, args.smooth,
            soglia_on, soglia_off, args.min_frames,
            not args.no_grafici,
            not args.no_weights,
            auto_threshold,
        )
        if r:
            risultati.append(r)
    
    # Salva riepilogo globale
    if len(risultati) > 1:
        with open(output_dir / "riepilogo.json", 'w', encoding='utf-8') as f:
            json.dump(risultati, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print(f"[DONE] Elaborati {len(risultati)}/{len(files)} file")
    print(f"[DONE] Risultati salvati in: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()