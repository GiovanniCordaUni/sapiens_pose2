#!/usr/bin/env python3
"""
Sapiens Pose Estimation Pipeline - YOLO + Sapiens

Main script per l'esecuzione della stima della posa su video
utilizzando YOLO per il rilevamento delle persone
e Sapiens per la stima dei keypoint.

Usage:
    # Singolo video (specifica soggetto)
    python run_pose_estimation.py --video <path> --subject soggetto001

    # Intero dataset
    python run_pose_estimation.py --dataset data/input/videos

    # Con config personalizzato
    python run_pose_estimation.py --dataset <path> --config <config.yaml>
"""

import argparse
import sys
from pathlib import Path
from typing import Generator

import yaml

# funzioni di utilità per aggiungere src al path per le importazioni
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.loaders.yolo_loader import YoloPersonDetector
from src.loaders.sapiens_loader import SapiensPoseEstimator, get_device
from src.pipeline.pipeline_video import run_pose_pipeline


def load_config(config_path: Path) -> dict:
    """carica file di configurazione YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_keypoint_config(config_path: Path) -> dict:
    """carica configurazione di mapping dei keypoint."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(config: dict, project_root: Path) -> dict:
    """Risolve i percorsi relativi nella configurazione in percorsi assoluti."""
    paths = config.get("paths", {})
    
    resolved = {}
    for key, value in paths.items():
        if value and not Path(value).is_absolute():
            resolved[key] = project_root / value
        else:
            resolved[key] = Path(value) if value else None
    
    return resolved


def discover_dataset_videos(dataset_root: Path) -> Generator[tuple[str, Path], None, None]:
    """
    trova video nella struttura del dataset: dataset_root/soggettoNNN/*.mp4
    
    Yields:
        Tuples of (subject_name, video_path)
    """
    if not dataset_root.exists():
        return
    
    # Itera su tutte le cartelle soggetto
    for subject_dir in sorted(dataset_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        
        subject_name = subject_dir.name
        
        # Trova tutti i video nella cartella soggetto
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        for video_path in sorted(subject_dir.iterdir()):
            if video_path.suffix.lower() in video_extensions:
                yield subject_name, video_path


def process_single_video(
    video_path: Path,
    subject_name: str,
    output_dir: Path,
    yolo_detector: YoloPersonDetector,
    pose_estimator: SapiensPoseEstimator,
    keypoint_indices: list[int],
    frame_stride: int,
    box_padding: float,
    keypoint_threshold: float,
    keypoint_radius: int,
    skeleton_thickness: int,
    keypoint_color: tuple[int, int, int],
    skeleton_color: tuple[int, int, int],
    bbox_color: tuple[int, int, int],
    skip_video_output: bool,
    quiet: bool,
) -> dict:
    """
    Processa un singolo video e salva gli output nella cartella del soggetto.
    
    Output structure:
        output_dir/
            soggettoNNN/
                videos/
                    nome_video.mp4
                keypoints_jsonl/
                    nome_video.jsonl
    """
    # Crea percorsi output: output_dir/soggettoNNN/nome_video.*
    subject_output_dir = output_dir / subject_name
    subject_output_dir.mkdir(parents=True, exist_ok=True)
    
    video_stem = video_path.stem
    output_jsonl = subject_output_dir / f"{video_stem}.jsonl"
    output_video = subject_output_dir / f"{video_stem}.mp4"
    
    if not quiet:
        print(f"[INFO] Processing: {subject_name}/{video_path.name}")
        print(f"[INFO] Output JSONL: {output_jsonl}")
        if not skip_video_output:
            print(f"[INFO] Output Video: {output_video}")
    
    # Run pipeline
    stats = run_pose_pipeline(
        video_path=video_path,
        output_jsonl=output_jsonl,
        output_video=output_video if not skip_video_output else Path("/dev/null"),
        yolo_detector=yolo_detector,
        pose_estimator=pose_estimator,
        keypoint_indices=keypoint_indices,
        frame_stride=frame_stride,
        box_padding=box_padding,
        keypoint_threshold=keypoint_threshold,
        keypoint_radius=keypoint_radius,
        skeleton_thickness=skeleton_thickness,
        keypoint_color=keypoint_color,
        skeleton_color=skeleton_color,
        bbox_color=bbox_color,
        show_progress=not quiet,
    )
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Run Sapiens pose estimation on video(s)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Input modes (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--video", "-v",
        type=Path,
        help="Single video file path (requires --subject)",
    )
    input_group.add_argument(
        "--dataset",
        type=Path,
        help="Dataset root with structure: dataset/soggettoNNN/*.mp4",
    )
    
    # Subject name (required for single video)
    parser.add_argument(
        "--subject",
        type=str,
        help="Subject folder name (e.g., soggetto001). Required with --video",
    )
    
    # Output
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        help="Output root directory (default: data/output)",
    )
    
    # Config
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=PROJECT_ROOT / "config" / "default.yaml",
        help="Configuration file",
    )
    parser.add_argument(
        "--keypoints-config",
        type=Path,
        default=PROJECT_ROOT / "config" / "keypoints.yaml",
        help="Keypoints mapping configuration",
    )
    
    # Model paths (override config)
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        help="YOLO weights file",
    )
    parser.add_argument(
        "--sapiens-checkpoint",
        type=Path,
        help="Sapiens checkpoint file",
    )
    
    # Processing options (override config)
    parser.add_argument(
        "--stride", "-s",
        type=int,
        help="Process every N-th frame",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device for inference",
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        help="YOLO confidence threshold",
    )
    parser.add_argument(
        "--box-padding",
        type=float,
        help="Bounding box padding ratio",
    )
    parser.add_argument(
        "--keypoint-threshold",
        type=float,
        help="Keypoint confidence threshold for visualization",
    )
    
    # Flags
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip overlay video output",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos that already have JSONL output",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.video and not args.subject:
        parser.error("--subject is required when using --video")
    
    # Load configurations
    config = load_config(args.config)
    kp_config = load_keypoint_config(args.keypoints_config)
    
    # Resolve paths
    paths = resolve_paths(config, PROJECT_ROOT)
    
    # Output directory
    output_dir = args.output_dir or paths.get("output_root", PROJECT_ROOT / "data" / "output")
    
    # Model paths
    sapiens_host = paths.get("sapiens_host", PROJECT_ROOT / "sapiens_host")
    
    yolo_weights = args.yolo_weights or (
        PROJECT_ROOT / config["models"]["yolo"]["weights"]
    )
    sapiens_checkpoint = args.sapiens_checkpoint or (
        sapiens_host / config["models"]["sapiens"]["checkpoint"]
    )
    
    if not sapiens_checkpoint.exists():
        print(f"[ERROR] Sapiens checkpoint not found: {sapiens_checkpoint}", file=sys.stderr)
        sys.exit(1)
    
    # Processing parameters (CLI overrides config)
    device = args.device or config["processing"]["device"]
    frame_stride = args.stride or config["processing"]["frame_stride"]
    yolo_conf = args.yolo_conf or config["models"]["yolo"]["confidence"]
    box_padding = args.box_padding or config["detection"]["box_padding"]
    iou_bias = config["detection"]["iou_bias"]
    person_class_id = config["models"]["yolo"]["person_class_id"]
    
    # Model input dimensions
    input_width = config["models"]["sapiens"]["input_width"]
    input_height = config["models"]["sapiens"]["input_height"]
    
    # Visualization parameters
    viz_config = config["visualization"]
    keypoint_threshold = args.keypoint_threshold or viz_config["keypoint_threshold"]
    keypoint_radius = viz_config["keypoint_radius"]
    skeleton_thickness = viz_config["skeleton_thickness"]
    keypoint_color = tuple(viz_config["keypoint_color"])
    skeleton_color = tuple(viz_config["skeleton_color"])
    bbox_color = tuple(viz_config["bbox_color"])
    
    # Keypoint indices (identity mapping per COCO 17 nativo)
    keypoint_indices = kp_config["coco17"]["indices"]
    
    # Resolve device
    resolved_device = get_device(device)
    
    # Build list of videos to process
    videos_to_process: list[tuple[str, Path]] = []
    
    if args.video:
        # Singolo video
        if not args.video.exists():
            print(f"[ERROR] Video not found: {args.video}", file=sys.stderr)
            sys.exit(1)
        videos_to_process.append((args.subject, args.video))
    else:
        # Dataset mode
        if not args.dataset.exists():
            print(f"[ERROR] Dataset directory not found: {args.dataset}", file=sys.stderr)
            sys.exit(1)
        videos_to_process = list(discover_dataset_videos(args.dataset))
        
        if not videos_to_process:
            print(f"[ERROR] No videos found in dataset: {args.dataset}", file=sys.stderr)
            sys.exit(1)
    
    # Filter existing if requested
    if args.skip_existing:
        filtered = []
        for subject_name, video_path in videos_to_process:
            output_jsonl = output_dir / subject_name / f"{video_path.stem}.jsonl"
            if output_jsonl.exists():
                if not args.quiet:
                    print(f"[SKIP] Already exists: {output_jsonl}")
            else:
                filtered.append((subject_name, video_path))
        videos_to_process = filtered
    
    if not videos_to_process:
        print("[INFO] No videos to process (all skipped or none found)")
        sys.exit(0)
    
    # Print configuration
    if not args.quiet:
        print("=" * 60)
        print("Sapiens Pose Estimation Pipeline")
        print("=" * 60)
        print(f"[CONFIG] Mode: {'Single video' if args.video else 'Dataset'}")
        print(f"[CONFIG] Videos to process: {len(videos_to_process)}")
        print(f"[CONFIG] Output directory: {output_dir}")
        print(f"[CONFIG] Device: {resolved_device}")
        print(f"[CONFIG] Frame stride: {frame_stride}")
        print(f"[CONFIG] YOLO confidence: {yolo_conf}")
        print(f"[CONFIG] Box padding: {box_padding}")
        print(f"[CONFIG] Keypoint threshold: {keypoint_threshold}")
        print("-" * 60)
    
    # Initialize models (una sola volta)
    if not args.quiet:
        print(f"[INFO] Loading YOLO: {yolo_weights}")
    
    yolo_detector = YoloPersonDetector(
        weights_path=yolo_weights,
        confidence=yolo_conf,
        person_class_id=person_class_id,
        iou_bias=iou_bias,
    )
    
    if not args.quiet:
        print(f"[INFO] Loading Sapiens: {sapiens_checkpoint}")
    
    pose_estimator = SapiensPoseEstimator(
        checkpoint_path=sapiens_checkpoint,
        device=device,
        input_width=input_width,
        input_height=input_height,
    )
    
    # Process all videos
    total_stats = {
        "videos_processed": 0,
        "videos_failed": 0,
        "frames_total": 0,
        "frames_with_person": 0,
    }
    
    for idx, (subject_name, video_path) in enumerate(videos_to_process, 1):
        if not args.quiet:
            print("=" * 60)
            print(f"[PROGRESS] Video {idx}/{len(videos_to_process)}")
        
        try:
            stats = process_single_video(
                video_path=video_path,
                subject_name=subject_name,
                output_dir=output_dir,
                yolo_detector=yolo_detector,
                pose_estimator=pose_estimator,
                keypoint_indices=keypoint_indices,
                frame_stride=frame_stride,
                box_padding=box_padding,
                keypoint_threshold=keypoint_threshold,
                keypoint_radius=keypoint_radius,
                skeleton_thickness=skeleton_thickness,
                keypoint_color=keypoint_color,
                skeleton_color=skeleton_color,
                bbox_color=bbox_color,
                skip_video_output=args.no_video,
                quiet=args.quiet,
            )
            
            total_stats["videos_processed"] += 1
            total_stats["frames_total"] += stats["frames_total"]
            total_stats["frames_with_person"] += stats["frames_with_person"]
            
            if not args.quiet:
                print(f"[DONE] {subject_name}/{video_path.name}: {stats['frames_with_person']}/{stats['frames_processed']} frames with person")
                
        except Exception as e:
            total_stats["videos_failed"] += 1
            print(f"[ERROR] Failed to process {video_path}: {e}", file=sys.stderr)
            if not args.quiet:
                import traceback
                traceback.print_exc()
    
    # Final summary
    if not args.quiet:
        print("=" * 60)
        print("[SUMMARY] Pipeline completed")
        print(f"  Videos processed: {total_stats['videos_processed']}")
        print(f"  Videos failed: {total_stats['videos_failed']}")
        print(f"  Total frames: {total_stats['frames_total']}")
        print(f"  Frames with person: {total_stats['frames_with_person']}")
        print(f"  Output directory: {output_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()