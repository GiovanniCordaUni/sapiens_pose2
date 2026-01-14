#!/usr/bin/env python3
"""
Sapiens Pose Estimation Pipeline - YOLO + Sapiens

Run pose estimation on video files using YOLO for person detection
and Sapiens for keypoint estimation.

Usage:
    python run_sapiens_pose17_jsonl_yolo.py --video <path> [options]
    python run_sapiens_pose17_jsonl_yolo.py --config <config.yaml>
"""

import argparse
import sys
from pathlib import Path

import yaml

# Add src to path for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.loaders.yolo_loader import YoloPersonDetector
from src.loaders.sapiens_loader import SapiensPoseEstimator, get_device
from src.pipeline.pipeline_video import run_pose_pipeline


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_keypoint_config(config_path: Path) -> dict:
    """Load keypoint mapping configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(config: dict, project_root: Path) -> dict:
    """Resolve relative paths in config to absolute paths."""
    paths = config.get("paths", {})
    
    # Resolve paths relative to project root
    resolved = {}
    for key, value in paths.items():
        if value and not Path(value).is_absolute():
            resolved[key] = project_root / value
        else:
            resolved[key] = Path(value) if value else None
    
    return resolved


def main():
    parser = argparse.ArgumentParser(
        description="Run Sapiens pose estimation on video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Input/Output
    parser.add_argument(
        "--video", "-v",
        type=Path,
        help="Input video file path",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        help="Output directory (default: data/output)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        help="Output filename prefix (default: video filename)",
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
    
    args = parser.parse_args()
    
    # Load configurations
    config = load_config(args.config)
    kp_config = load_keypoint_config(args.keypoints_config)
    
    # Resolve paths
    paths = resolve_paths(config, PROJECT_ROOT)
    
    # Determine video path
    if args.video:
        video_path = args.video
    else:
        parser.error("--video is required")
    
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)
    
    # Determine output paths
    output_dir = args.output_dir or paths.get("output_root", PROJECT_ROOT / "data" / "output")
    output_name = args.output_name or video_path.stem
    
    output_jsonl = output_dir / "keypoints_jsonl" / f"{output_name}.jsonl"
    output_video = output_dir / "overlay_videos" / f"{output_name}_overlay.mp4"
    
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
    
    # Keypoint indices from config
    keypoint_indices = kp_config["coco17"]["indices"]
    
    # Resolve device
    resolved_device = get_device(device)
    
    # Print configuration
    if not args.quiet:
        print("=" * 60)
        print("Sapiens Pose Estimation Pipeline")
        print("=" * 60)
        print(f"[CONFIG] Video: {video_path}")
        print(f"[CONFIG] Output JSONL: {output_jsonl}")
        print(f"[CONFIG] Output Video: {output_video}")
        print(f"[CONFIG] Device: {resolved_device}")
        print(f"[CONFIG] Frame stride: {frame_stride}")
        print(f"[CONFIG] YOLO confidence: {yolo_conf}")
        print(f"[CONFIG] Box padding: {box_padding}")
        print(f"[CONFIG] Keypoint threshold: {keypoint_threshold}")
        print("-" * 60)
    
    # Initialize models
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
    
    # Run pipeline
    if not args.quiet:
        print("-" * 60)
    
    stats = run_pose_pipeline(
        video_path=video_path,
        output_jsonl=output_jsonl,
        output_video=output_video if not args.no_video else "/dev/null",
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
        show_progress=not args.quiet,
    )
    
    # Print results
    if not args.quiet:
        print("=" * 60)
        print("[DONE] Pipeline completed")
        print(f"  Total frames: {stats['frames_total']}")
        print(f"  Processed frames: {stats['frames_processed']}")
        print(f"  Frames with person: {stats['frames_with_person']}")
        print(f"  Frames no person: {stats['frames_no_person']}")
        print(f"  Frames empty crop: {stats['frames_empty_crop']}")
        print(f"[DONE] JSONL: {output_jsonl}")
        if not args.no_video:
            print(f"[DONE] Overlay video: {output_video}")
        print("=" * 60)


if __name__ == "__main__":
    main()

