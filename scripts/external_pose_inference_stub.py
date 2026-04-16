from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pose_adapters import _contour_centroids, _fallback_centers, _synth_keypoints, CANONICAL_LABELS
from pose_adapters import _assign_black_white_centers


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="External-command pose inference stub for continuous video mode")
    parser.add_argument("--config", type=Path, default=Path("configs/mvp_config.json"))
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    max_frames = cfg.get("pose_stage", {}).get("max_frames")
    max_frames = int(max_frames) if isinstance(max_frames, int) else None

    video_path = args.video.resolve()
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    records: list[dict] = []
    frame_index = 0
    prev_black: tuple[float, float] | None = None
    prev_white: tuple[float, float] | None = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        height, width = frame.shape[:2]
        centers = _contour_centroids(frame)
        if len(centers) < 2:
            if len(centers) == 1 and prev_black is not None and prev_white is not None:
                detected = centers[0]
                db = (detected[0] - prev_black[0]) ** 2 + (detected[1] - prev_black[1]) ** 2
                dw = (detected[0] - prev_white[0]) ** 2 + (detected[1] - prev_white[1]) ** 2
                if db <= dw:
                    centers = [detected, prev_white]
                else:
                    centers = [prev_black, detected]
            elif len(centers) == 0 and prev_black is not None and prev_white is not None:
                centers = [prev_black, prev_white]

        if len(centers) < 2:
            black_fb, white_fb = _fallback_centers(frame_index, width, height)
            if len(centers) == 1:
                detected = centers[0]
                if detected[0] < width * 0.5:
                    centers = [detected, white_fb]
                else:
                    centers = [black_fb, detected]
            else:
                centers = [black_fb, white_fb]

        assigned = _assign_black_white_centers(frame, centers, prev_black=prev_black, prev_white=prev_white)
        if assigned is None:
            black_center, white_center = _fallback_centers(frame_index, width, height)
        else:
            black_center, white_center = assigned

        prev_black, prev_white = black_center, white_center

        black_x, black_y = _synth_keypoints(black_center[0], black_center[1], width, height)
        white_x, white_y = _synth_keypoints(white_center[0], white_center[1], width, height)

        records.append(
            {
                "filename": f"{video_path.stem}_frame_{frame_index:06d}.jpg",
                "width": int(width),
                "height": int(height),
                "labels": CANONICAL_LABELS,
                "coords": {
                    "black": {"x": black_x, "y": black_y},
                    "white": {"x": white_x, "y": white_y},
                },
                "source_video": str(video_path).replace("\\", "/"),
                "source_frame_idx": frame_index,
            }
        )

        frame_index += 1
        if max_frames is not None and len(records) >= max_frames:
            break

    cap.release()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records), encoding="utf-8")
    print(f"Wrote external pose predictions: {args.output} ({len(records)} frames)")


if __name__ == "__main__":
    main()