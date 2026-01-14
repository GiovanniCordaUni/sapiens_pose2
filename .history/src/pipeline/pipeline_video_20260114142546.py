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
        
    Returns:
        Dizionario con le statistiche della pipeline
    """
    video_path = Path(video_path)
    output_jsonl = Path(output_jsonl)
    output_video = Path(output_video)
    
    # Create output directories
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
    
    with VideoReader(video_path) as reader:
        W, H = reader.width, reader.height
        fps = reader.fps
        total_frames = reader.frame_count
        
        # Create output video writer with adjusted FPS
        output_fps = fps / frame_stride
        
        with VideoWriter(output_video, W, H, output_fps) as writer:
            with JsonlWriter(output_jsonl) as jsonl:
                
                # Progress bar
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
                    
                    # Skip frames based on stride
                    if frame_idx % frame_stride != 0:
                        continue
                    
                    stats["frames_processed"] += 1
                    
                    # Detect person
                    box_xyxy, box_conf = yolo_detector.detect(frame, prev_box=prev_box)
                    
                    if box_xyxy is None:
                        # No person detected
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
                    
                    # Update tracking box and apply padding
                    prev_box = box_xyxy
                    box_xyxy = clamp_box_xyxy(box_xyxy, W, H)
                    box_xyxy = add_padding_xyxy(box_xyxy, W, H, padding_ratio=box_padding)
                    x1, y1, x2, y2 = box_xyxy
                    
                    # Crop person
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
                    
                    # Run pose estimation
                    keypoints, scores = pose_estimator.predict_frame(
                        crop,
                        box_xyxy,
                        keypoint_indices=keypoint_indices,
                    )
                    
                    # Draw overlay
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
                    
                    # Write record
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
