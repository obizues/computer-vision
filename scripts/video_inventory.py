from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".seq"}


def inspect_video(path: Path) -> dict:
    result = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
    }

    if cv2 is None:
        result["error"] = "opencv-python is not installed"
        return result

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        result["error"] = "could not open video"
        return result

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = frame_count / fps if fps else None
    capture.release()

    result.update(
        {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_sec": duration_sec,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory video files in a folder")
    parser.add_argument("input_dir", type=Path, help="Directory containing videos")
    parser.add_argument("--output", type=Path, default=Path("video_inventory.json"))
    args = parser.parse_args()

    files = sorted(
        path for path in args.input_dir.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS
    )

    report = [inspect_video(path) for path in files]
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {len(report)} records to {args.output}")


if __name__ == "__main__":
    main()
