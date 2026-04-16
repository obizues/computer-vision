from __future__ import annotations

import argparse
import json
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path


def run_script(path: str, argv: list[str]) -> None:
    sys.argv = argv
    runpy.run_path(path, run_name="__main__")


def try_run_script(path: str, argv: list[str]) -> None:
    try:
        run_script(path, argv)
    except SystemExit as exc:
        print(f"Skipped optional step {path}: {exc}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def append_registry_line(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full MVP pipeline")
    parser.add_argument("--config", type=Path, default=Path("configs/mvp_config.json"))
    args = parser.parse_args()

    cfg = load_json(args.config)
    out_dir = Path(cfg["outputs"]["artifacts_dir"])
    pose_mode = cfg.get("pose_stage", {}).get("mode")

    if pose_mode == "continuous_video_external_inference":
        run_script(
            "scripts/run_pose_inference_runtime.py",
            ["run_pose_inference_runtime.py", "--config", str(args.config)],
        )

    run_script("scripts/video_to_pose.py", ["video_to_pose.py", "--config", str(args.config)])
    run_script("scripts/build_features.py", ["build_features.py", "--config", str(args.config)])
    run_script("scripts/train_eval.py", ["train_eval.py", "--config", str(args.config)])
    run_script("scripts/predict_batch.py", ["predict_batch.py", "--config", str(args.config)])
    try_run_script("scripts/render_pose_overlay.py", ["render_pose_overlay.py", "--config", str(args.config)])

    model_summary = load_json(out_dir / "baseline_model_summary.json")
    quality_report = load_json(out_dir / "feature_quality_report.json")

    registry_entry = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "mouse-vision-mvp",
        "config": str(args.config).replace('\\', '/'),
        "pose_mode": cfg.get("pose_stage", {}).get("mode"),
        "dataset_name": cfg["dataset"]["name"],
        "dataset_version": cfg["dataset"]["version"],
        "model_type": model_summary.get("model_type"),
        "auc": model_summary.get("auc"),
        "train_size": model_summary.get("train_size"),
        "test_size": model_summary.get("test_size"),
        "quality": {
            "rows": quality_report.get("rows"),
            "null_cells": quality_report.get("null_cells"),
            "out_of_bounds_x": quality_report.get("out_of_bounds_x"),
            "out_of_bounds_y": quality_report.get("out_of_bounds_y"),
        },
        "artifacts": {
            "pose_input": str(Path(cfg["pose_stage"]["output_top_keypoints_file"]).as_posix()),
            "features": str((out_dir / "features_top_view.csv").as_posix()),
            "model": str((out_dir / "baseline_model.joblib").as_posix()),
            "summary": str((out_dir / "baseline_model_summary.json").as_posix()),
            "predictions": str((out_dir / "batch_predictions.csv").as_posix()),
            "pose_overlay_preview": str((out_dir / "pose_overlay_preview.mp4").as_posix()),
        },
    }

    registry_path = Path(cfg["outputs"]["run_registry"])
    append_registry_line(registry_path, registry_entry)
    print(f"Appended run registry: {registry_path}")


if __name__ == "__main__":
    main()
