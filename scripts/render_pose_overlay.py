from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import cv2


def load_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text(encoding="utf-8"))


def parse_frame_index(filename: str) -> int | None:
    match = re.search(r"_frame_(\d+)\.", filename)
    if not match:
        return None
    return int(match.group(1))


def draw_mouse_points(frame, xs, ys, color, radius: int = 5, thickness: int = -1):
    for x, y in zip(xs, ys):
        if x is None or y is None:
            continue
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        if not math.isfinite(float(x)) or not math.isfinite(float(y)):
            continue
        cv2.circle(frame, (int(x), int(y)), radius, color, thickness)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render pose overlay preview video")
    parser.add_argument("--config", type=Path, default=Path("configs/mvp_config.json"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["outputs"]["artifacts_dir"])
    pose_file = Path(cfg["pose_stage"]["output_top_keypoints_file"])

    if not pose_file.exists():
        raise SystemExit(f"Missing pose file: {pose_file}")

    records = json.loads(pose_file.read_text(encoding="utf-8"))
    if not records:
        raise SystemExit("No pose records found")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "pose_overlay_preview.mp4"
    fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
    fourcc_raw: Any = fourcc_fn(*"mp4v") if callable(fourcc_fn) else 0
    fourcc = int(fourcc_raw)

    first_video = records[0].get("source_video")
    if first_video:
        video_path = Path(first_video)
        if not video_path.exists():
            raise SystemExit(f"Source video not found: {video_path}")

        index_map = {}
        for rec in records:
            idx = rec.get("source_frame_idx")
            if idx is None:
                idx = parse_frame_index(rec.get("filename", ""))
            if idx is not None:
                index_map[int(idx)] = rec

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise SystemExit(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        frame_idx = 0
        written = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rec = index_map.get(frame_idx)
            if rec is not None:
                black = rec["coords"]["black"]
                white = rec["coords"]["white"]
                draw_mouse_points(frame, black["x"], black["y"], (0, 255, 0), radius=5, thickness=-1)    # lime green filled
                draw_mouse_points(frame, white["x"], white["y"], (0, 165, 255), radius=8, thickness=2)   # orange ring
                cv2.putText(frame, f"frame {frame_idx}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
                writer.write(frame)
                written += 1

            frame_idx += 1

        cap.release()
        writer.release()
        print(f"Wrote overlay preview: {out_path} ({written} frames)")
        return

    raw_top_dir = Path(cfg.get("dataset", {}).get("raw_images_top_dir", "data/raw_images_top"))
    if not raw_top_dir.exists():
        raise SystemExit("Pose records do not include source_video and raw_images_top_dir was not found")

    first_filename = records[0].get("filename", "")
    direct_dir = raw_top_dir
    nested_dir = raw_top_dir / "raw_images_top"
    image_root = direct_dir if (direct_dir / first_filename).exists() else nested_dir
    first_image_path = image_root / first_filename
    first_frame = cv2.imread(str(first_image_path))
    if first_frame is None:
        raise SystemExit(f"Could not open first raw image frame: {first_image_path}")

    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(str(out_path), fourcc, 6.0, (width, height))

    written = 0
    for row_idx, rec in enumerate(records):
        filename = str(rec.get("filename", ""))
        image_path = image_root / filename
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        black = rec["coords"]["black"]
        white = rec["coords"]["white"]
        draw_mouse_points(frame, black["x"], black["y"], (0, 0, 255))
        draw_mouse_points(frame, white["x"], white["y"], (0, 255, 255))
        cv2.putText(frame, f"frame {row_idx}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
        writer.write(frame)
        written += 1

    writer.release()
    print(f"Wrote overlay preview: {out_path} ({written} frames)")


if __name__ == "__main__":
    main()
