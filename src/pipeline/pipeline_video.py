"""
Video pose estimation pipeline.

Orchestra video reading, detection, pose estimation, e output writing.
"""

from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from ..loaders.yolo_loader import YoloPersonDetector
from ..loaders.sapiens_loader import SapiensPoseEstimator
from ..io.video_reader import VideoReader
from ..io.video_writer import VideoWriter, create_overlay_writer
from ..io.jsonl_writer import JsonlWriter, create_pose_record, keypoints_to_list
from ..viz.draw import draw_pose_overlay
from ..detection.person_detector import clamp_box_xyxy, add_padding_xyxy
from ..filters.one_euro_filter import KeypointStabilizer

# # NOTA: La funzione EMA potrebbe causare ritardi indesiderati nei bounding box sulla persona.
# # Exponential moving average per stabilizzare i bounding box sulla persona
# def ema_box(prev: list[float] | None, curr: list[float], alpha: float = 0.2) -> list[float]:
#     if prev is None:
#         return [float(x) for x in curr]
#     return [(1 - alpha) * float(prev[i]) + alpha * float(curr[i]) for i in range(4)]

def run_pose_pipeline(
    video_path: str | Path,
    output_jsonl: str | Path,
    output_video: str | Path,
    yolo_detector: YoloPersonDetector,
    pose_estimator: SapiensPoseEstimator,
    keypoint_indices: list[int],
    frame_stride: int = 1,
    box_padding: float = 0.20,
    keypoint_threshold: float = 0.35,
    keypoint_radius: int = 5,
    skeleton_thickness: int = 2,
    keypoint_color: tuple[int, int, int] = (0, 255, 0),
    skeleton_color: tuple[int, int, int] = (0, 255, 0),
    bbox_color: tuple[int, int, int] = (255, 0, 0),
    show_progress: bool = True,
    stabilizer_conf_threshold: float = 0.15,
    stabilize_keypoints: bool = True,
    stabilizer_min_cutoff: float = 1.5,
    stabilizer_beta: float = 0.01,
    stabilizer_use_one_euro: bool = True,
    stabilizer_use_hold: bool = True,
    stabilizer_hold_decay: float = 0.95,
) -> dict[str, Any]:
    """
    Esegue la pipeline completa di stima della posa su un video.
    
    Args:
        video_path: Percorso del file video di input
        output_jsonl: Percorso del file JSONL di output per i keypoints
        output_video: Percorso del file video di output con overlay
        yolo_detector: Rilevatore di persone YOLO configurato
        pose_estimator: Stimatore di posa Sapiens configurato
        keypoint_indices: Indici identità [0..16] per COCO 17 nativo
        frame_stride: Processa ogni N-esimo frame
        box_padding: Rapporto di padding per i bounding box
        keypoint_threshold: Soglia di confidenza per disegnare i keypoints
        keypoint_radius: Raggio per i cerchi dei keypoints
        skeleton_thickness: Spessore per le linee dello scheletro
        keypoint_color: Colore BGR per i keypoints
        skeleton_color: Colore BGR per lo scheletro
        bbox_color: Colore BGR per il bounding box
        show_progress: Mostra la barra di progresso
        stabilize_keypoints: Abilita filtro One-Euro per stabilizzare i keypoints
        stabilizer_min_cutoff: Cutoff minimo del filtro (più basso = più smooth)
        stabilizer_beta: Reattività ai movimenti rapidi (più alto = più reattivo)
        
    Returns:
        Dizionario con le statistiche della pipeline
    """
    video_path = Path(video_path)
    output_jsonl = Path(output_jsonl)
    output_video = Path(output_video)
    
    # Crea le directory di output
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "frames_total": 0,
        "frames_processed": 0,
        "frames_with_person": 0,
        "frames_no_person": 0,
        "frames_empty_crop": 0,
    }
    
    prev_box = None
    
    # Inizializza stabilizzatore keypoints se abilitato
    stabilizer = None
    if stabilize_keypoints:
        stabilizer = KeypointStabilizer(
            num_keypoints=len(keypoint_indices),
            min_cutoff=stabilizer_min_cutoff,
            beta=stabilizer_beta,
            use_one_euro=stabilizer_use_one_euro,
            use_hold=stabilizer_use_hold,
            hold_decay=stabilizer_hold_decay,
        )
    
    with VideoReader(video_path) as reader:
        W, H = reader.width, reader.height
        fps = reader.fps
        total_frames = reader.frame_count
        
        # Crea il writer video di output con FPS regolato
        output_fps = fps / frame_stride
        
        with VideoWriter(output_video, W, H, output_fps) as writer:
            with JsonlWriter(output_jsonl) as jsonl:
                
                # Barra di progresso per visualizzare lo stato
                pbar = tqdm(
                    total=total_frames,
                    desc="Pose estimation",
                    disable=not show_progress,
                )
                
                frame_idx = -1
                while True:
                    ret, frame = reader.read_frame()
                    if not ret or frame is None:
                        break
                    frame_idx += 1
                    pbar.update(1)
                    stats["frames_total"] += 1
                    
                    # Salta i frame in base allo stride
                    if frame_idx % frame_stride != 0:
                        continue
                    
                    stats["frames_processed"] += 1
                    
                    # Rileva la persona
                    box_xyxy, box_conf = yolo_detector.detect(frame, prev_box=prev_box)
                    
                    if box_xyxy is None:
                        # Nessuna persona rilevata
                        record = create_pose_record(
                            frame_idx=frame_idx,
                            fps=fps,
                            box_xyxy=None,
                            box_conf=None,
                            status="no_person",
                        )
                        jsonl.write(record)
                        writer.write(frame)
                        stats["frames_no_person"] += 1
                        continue

                    # smooth_box = ema_box(prev_box, box_xyxy, alpha=0.2)

                    # prev_box = smooth_box
                    prev_box = box_xyxy

                    #box_xyxy = clamp_box_xyxy(smooth_box, W, H)
                    box_xyxy = clamp_box_xyxy(box_xyxy, W, H)
                    # box_xyxy = add_padding_xyxy(box_xyxy, W, H, padding_ratio=box_padding) # opzionale se si vuole padding sul box persona
                    x1, y1, x2, y2 = box_xyxy
                    
                    # Ritaglia la persona
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        record = create_pose_record(
                            frame_idx=frame_idx,
                            fps=fps,
                            box_xyxy=[int(x1), int(y1), int(x2), int(y2)],
                            box_conf=box_conf,
                            status="empty_crop",
                        )
                        jsonl.write(record)
                        writer.write(frame)
                        stats["frames_empty_crop"] += 1
                        continue
                    
                    # Esegui la stima della posa
                    keypoints, scores = pose_estimator.predict_frame(
                        crop,
                        box_xyxy,
                        keypoint_indices=keypoint_indices,
                    )

                    
                    
                    # Applica stabilizzazione temporale se abilitata (in coordinate normalizzate al bbox)
                    if stabilizer is not None:
                        timestamp = frame_idx / fps

                        # bbox dims (evita divisioni per zero)
                        bw = float(max(1, x2 - x1))
                        bh = float(max(1, y2 - y1))

                    # keypoints in [0..1] rispetto al bbox: elimina jitter indotto da bbox variabili
                        kpts_rel = keypoints.copy()
                        kpts_rel[:, 0] = (kpts_rel[:, 0] - float(x1)) / bw
                        kpts_rel[:, 1] = (kpts_rel[:, 1] - float(y1)) / bh

                        # Filtra su coordinate normalizzate con soglia dedicata
                        kpts_rel_f, scores_f = stabilizer.filter(
                            kpts_rel, scores, timestamp, stabilizer_conf_threshold
                        )

                        # Riporta in coordinate assolute frame
                        keypoints = kpts_rel_f.copy()
                        keypoints[:, 0] = keypoints[:, 0] * bw + float(x1)
                        keypoints[:, 1] = keypoints[:, 1] * bh + float(y1)

                        scores = scores_f

                    
                    # Disegna l'overlay
                    vis = draw_pose_overlay(
                        frame,
                        keypoints,
                        scores,
                        box_xyxy=box_xyxy,
                        keypoint_threshold=keypoint_threshold,
                        keypoint_radius=keypoint_radius,
                        skeleton_thickness=skeleton_thickness,
                        keypoint_color=keypoint_color,
                        skeleton_color=skeleton_color,
                        bbox_color=bbox_color,
                    )
                    writer.write(vis)
                    
                    # Scrivi il record
                    record = create_pose_record(
                        frame_idx=frame_idx,
                        fps=fps,
                        box_xyxy=[int(x1), int(y1), int(x2), int(y2)],
                        box_conf=box_conf,
                        keypoints=keypoints_to_list(keypoints, scores),
                        keypoint_indices=keypoint_indices,
                        num_keypoints_total=pose_estimator.num_keypoints,
                        status="ok",
                    )
                    jsonl.write(record)
                    stats["frames_with_person"] += 1
                
                pbar.close()
    
    return stats
