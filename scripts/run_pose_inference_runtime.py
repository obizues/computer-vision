from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".seq"}


def load_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text(encoding="utf-8"))


def pick_video(cfg: dict) -> Path:
    runtime_cfg = cfg.get("pose_inference_runtime", {})
    configured_video = str(runtime_cfg.get("video_file", "")).strip()
    if configured_video:
        path = Path(configured_video)
        if not path.exists():
            raise SystemExit(f"Configured pose_inference_runtime.video_file not found: {path}")
        return path

    video_dir = Path(cfg.get("ingestion", {}).get("local_video_dir", "data/raw_videos"))
    videos = sorted(path for path in video_dir.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise SystemExit(
            f"No video files found in {video_dir}. Add a real video and/or set pose_inference_runtime.video_file"
        )
    return videos[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run external continuous-video pose inference command")
    parser.add_argument("--config", type=Path, default=Path("configs/mvp_config.json"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    pose_mode = str(cfg.get("pose_stage", {}).get("mode", ""))
    if pose_mode != "continuous_video_external_inference":
        raise SystemExit("pose_stage.mode is not continuous_video_external_inference")

    runtime_cfg = cfg.get("pose_inference_runtime", {})
    command_template = str(runtime_cfg.get("command", "")).strip()
    if not command_template:
        raise SystemExit(
            "pose_inference_runtime.command is required. "
            "Example: python tools/run_dlc.py --video \"{video}\" --output \"{output}\""
        )

    video_path = pick_video(cfg)
    output_path = Path(cfg.get("pose_stage", {}).get("external_pose_file", ""))
    if not str(output_path):
        raise SystemExit("pose_stage.external_pose_file is required for continuous_video_external_inference")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = command_template.format(
        video=str(video_path).replace("\\", "/"),
        output=str(output_path).replace("\\", "/"),
        config=str(args.config).replace("\\", "/"),
    )

    print(f"Running external pose inference command:\n{command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        raise SystemExit(f"Pose inference command failed with exit code {result.returncode}")

    if not output_path.exists():
        raise SystemExit(f"Expected pose output not found after command: {output_path}")

    print(f"Pose inference completed. Output: {output_path}")


if __name__ == "__main__":
    main()
