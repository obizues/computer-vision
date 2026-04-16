from __future__ import annotations

import argparse
from pathlib import Path

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def sample_frames(video_path: Path, output_dir: Path, every_n_frames: int) -> int:
    if cv2 is None:
        raise RuntimeError("opencv-python is not installed")

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    saved = 0
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index % every_n_frames == 0:
            target = output_dir / f"{video_path.stem}_frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(target), frame)
            saved += 1

        frame_index += 1

    capture.release()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample frames from a video")
    parser.add_argument("video_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--every-n-frames", type=int, default=300)
    args = parser.parse_args()

    if args.video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise SystemExit("Unsupported video extension")

    saved = sample_frames(args.video_path, args.output_dir, args.every_n_frames)
    print(f"Saved {saved} frames")


if __name__ == "__main__":
    main()
