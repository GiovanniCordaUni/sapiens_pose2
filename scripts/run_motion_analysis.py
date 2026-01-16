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

import numpy as np
import matplotlib.pyplot as plt

# Indici COCO-17 per il torso
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12
TORSO_INDICES = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]


def carica_jsonl(path: Path) -> list[dict]:
    """Carica records da file JSONL o JSON."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    
    if text.startswith("["):
        return json.loads(text)
    
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def calcola_segnale_movimento(
    records: list[dict],
    conf_th: float = 0.4,
    smooth_window: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcola segnale di movimento normalizzato.
    
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
    
    prev_pts = {}
    
    for i, r in enumerate(usable):
        kpts = r["keypoints17"]
        
        # Estrai keypoints torso
        curr = {}
        for idx in TORSO_INDICES:
            if idx < len(kpts) and len(kpts[idx]) >= 3:
                curr[idx] = kpts[idx]
        
        if i == 0:
            prev_pts = curr
            continue
        
        # Calcola spostamenti
        dists = []
        for idx, (x, y, c) in curr.items():
            if idx in prev_pts:
                px, py, pc = prev_pts[idx]
                if c >= conf_th and pc >= conf_th:
                    dists.append(np.sqrt((x - px)**2 + (y - py)**2))
        
        prev_pts = curr
        
        if not dists:
            m[i] = m[i-1] if i > 0 else 0
            continue
        
        disp = np.median(dists)
        
        # Normalizza per lunghezza torso
        scale = 1.0
        if all(idx in curr for idx in TORSO_INDICES):
            ls, rs = curr[LEFT_SHOULDER], curr[RIGHT_SHOULDER]
            lh, rh = curr[LEFT_HIP], curr[RIGHT_HIP]
            if all(p[2] >= conf_th for p in [ls, rs, lh, rh]):
                mid_sh = np.array([(ls[0]+rs[0])/2, (ls[1]+rs[1])/2])
                mid_hp = np.array([(lh[0]+rh[0])/2, (lh[1]+rh[1])/2])
                torso = np.linalg.norm(mid_sh - mid_hp)
                if torso > 1:
                    scale = torso
        
        m[i] = disp / scale
    
    # Smooth con moving average
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        m = np.convolve(m, kernel, mode="same")
    
    return t, m


def rileva_intervalli(
    t: np.ndarray,
    m: np.ndarray,
    soglia_on: float = 0.07,
    soglia_off: float = 0.015,
    min_frames: int = 3,
) -> list[tuple[float, float]]:
    """
    Rileva intervalli di movimento con isteresi.
    
    Returns:
        Lista di tuple (inizio, fine) in secondi
    """
    n = len(m)
    in_movimento = False
    above_count = below_count = 0
    start_idx = None
    intervalli = []
    
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
                    intervalli.append((t[start_idx], t[end_idx]))
                start_idx = None
                above_count = 0
    
    # Chiudi intervallo se ancora in movimento
    if in_movimento and start_idx is not None:
        intervalli.append((t[start_idx], t[-1]))
    
    return intervalli


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
    ax1.axhline(soglia_on, color='g', linestyle='--', alpha=0.7, label=f'ON={soglia_on}')
    ax1.axhline(soglia_off, color='r', linestyle='--', alpha=0.7, label=f'OFF={soglia_off}')
    ax1.set_ylabel("Movimento (norm.)")
    ax1.legend(loc='upper right')
    ax1.set_title("Analisi Movimento")
    
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
    soglia_on: float,
    soglia_off: float,
    min_frames: int,
    salva_grafici: bool,
) -> dict | None:
    """Processa un singolo file JSONL."""
    print(f"\n[FILE] Elaborazione: {input_path.name}")
    
    try:
        records = carica_jsonl(input_path)
        if not records:
            print("  [WARN] Nessun record trovato")
            return None
        
        t, m = calcola_segnale_movimento(records, conf_th, smooth_window)
        intervalli = rileva_intervalli(t, m, soglia_on, soglia_off, min_frames)
        
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
    parser.add_argument("--conf_th", type=float, default=0.4, help="Soglia confidenza keypoints")
    parser.add_argument("--smooth", type=int, default=5, help="Finestra smoothing")
    parser.add_argument("--ton", type=float, default=1, help="Soglia inizio movimento")
    parser.add_argument("--toff", type=float, default=0.015, help="Soglia fine movimento")
    parser.add_argument("--min_frames", type=int, default=3, help="Frame consecutivi per cambio stato")
    parser.add_argument("--no_grafici", action="store_true", help="Non generare grafici")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else (
        input_path.parent / "analisi_movimento" if input_path.is_file() 
        else input_path / "analisi_movimento"
    )
    
    # Trova file da processare
    if args.batch or input_path.is_dir():
        files = sorted(input_path.glob("*.jsonl"))
    else:
        files = [input_path]
    
    if not files:
        print(f"[ERROR] Nessun file .jsonl trovato in {input_path}")
        return
    
    print("=" * 60)
    print("ANALISI MOVIMENTO")
    print("=" * 60)
    print(f"[CONFIG] File da elaborare: {len(files)}")
    print(f"[CONFIG] Output: {output_dir}")
    print(f"[CONFIG] Soglie: ON={args.ton}, OFF={args.toff}")
    
    risultati = []
    for file_path in files:
        r = processa_file(
            file_path, output_dir,
            args.conf_th, args.smooth,
            args.ton, args.toff, args.min_frames,
            not args.no_grafici,
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
