from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
CANONICAL_LABELS = [
    "nose tip",
    "right ear",
    "left ear",
    "neck",
    "right side body",
    "left side body",
    "tail base",
]


def _limit_records(records: list[dict], max_frames: int | None) -> list[dict]:
    if isinstance(max_frames, int) and max_frames > 0:
        return records[:max_frames]
    return records


def _validate_canonical(records: list[dict]) -> None:
    required = {"filename", "width", "height", "labels", "coords"}
    for index, rec in enumerate(records[:20]):
        missing = required - rec.keys()
        if missing:
            raise SystemExit(f"Canonical pose record {index} missing keys: {sorted(missing)}")
        coords = rec["coords"]
        for mouse in ("black", "white"):
            if mouse not in coords:
                raise SystemExit(f"Canonical pose record {index} missing mouse '{mouse}'")
            for axis in ("x", "y"):
                if axis not in coords[mouse]:
                    raise SystemExit(f"Canonical pose record {index} missing axis '{axis}' for {mouse}")


def adapter_dataset_json_passthrough(cfg: dict) -> list[dict]:
    source = Path(cfg["dataset"]["top_keypoints_file"])
    records = json.loads(source.read_text(encoding="utf-8"))
    return _limit_records(records, cfg.get("pose_stage", {}).get("max_frames"))


def _contour_centroids(frame: np.ndarray) -> list[tuple[float, float]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    frame_area = float(frame.shape[0] * frame.shape[1])
    min_area = 150.0
    max_area = frame_area * 0.2

    # Collect candidate centers from several masks:
    # - dark blobs (inverted Otsu)
    # - bright blobs (non-inverted Otsu)
    # - near-black and near-white hard thresholds for synthetic footage
    all_centers: list[tuple[float, float, float]] = []  # (cx, cy, area)

    dark_mask = cv2.inRange(blur, np.array([0], dtype=np.uint8), np.array([40], dtype=np.uint8))
    bright_mask = cv2.inRange(blur, np.array([245], dtype=np.uint8), np.array([255], dtype=np.uint8))

    for mask in (255 - thresh, thresh, dark_mask, bright_mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            m = cv2.moments(cnt)
            if m["m00"] == 0:
                continue
            cx = float(m["m10"] / m["m00"])
            cy = float(m["m01"] / m["m00"])
            all_centers.append((cx, cy, area))

    # Deduplicate: merge any two candidate centers that are within 20 px of each other
    merged: list[tuple[float, float, float]] = []
    for cx, cy, area in sorted(all_centers, key=lambda t: t[2], reverse=True):
        if any(((cx - mx) ** 2 + (cy - my) ** 2) < 400 for mx, my, _ in merged):
            continue
        merged.append((cx, cy, area))

    centers: list[tuple[float, float]] = [(cx, cy) for cx, cy, _ in merged[:2]]
    return centers


def _synth_keypoints(cx: float, cy: float, width: int, height: int) -> tuple[list[float], list[float]]:
    offsets = [
        (0, -12),
        (8, -7),
        (-8, -7),
        (0, 0),
        (10, 8),
        (-10, 8),
        (0, 13),
    ]
    xs: list[float] = []
    ys: list[float] = []
    for ox, oy in offsets:
        x = max(0.0, min(float(width - 1), cx + ox))
        y = max(0.0, min(float(height - 1), cy + oy))
        xs.append(round(x, 2))
        ys.append(round(y, 2))
    return xs, ys


def _fallback_centers(frame_index: int, width: int, height: int) -> tuple[tuple[float, float], tuple[float, float]]:
    black = (
        width * 0.30 + 0.14 * width * np.sin(frame_index / 18.0),
        height * 0.55 + 0.10 * height * np.cos(frame_index / 15.0),
    )
    white = (
        width * 0.68 + 0.12 * width * np.cos(frame_index / 20.0),
        height * 0.50 + 0.12 * height * np.sin(frame_index / 17.0),
    )
    return black, white


def _center_intensity(gray: np.ndarray, center: tuple[float, float], radius: int = 10) -> float:
    cx, cy = int(center[0]), int(center[1])
    y0 = max(0, cy - radius)
    y1 = min(gray.shape[0], cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(gray.shape[1], cx + radius + 1)
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return 127.0
    return float(np.mean(patch))


def _assign_black_white_centers(
    frame: np.ndarray,
    centers: list[tuple[float, float]],
    prev_black: tuple[float, float] | None = None,
    prev_white: tuple[float, float] | None = None,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if len(centers) < 2:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    c0, c1 = centers[0], centers[1]
    i0 = _center_intensity(gray, c0)
    i1 = _center_intensity(gray, c1)

    if prev_black is not None and prev_white is not None:
        d_keep = (c0[0] - prev_black[0]) ** 2 + (c0[1] - prev_black[1]) ** 2 + (c1[0] - prev_white[0]) ** 2 + (c1[1] - prev_white[1]) ** 2
        d_swap = (c1[0] - prev_black[0]) ** 2 + (c1[1] - prev_black[1]) ** 2 + (c0[0] - prev_white[0]) ** 2 + (c0[1] - prev_white[1]) ** 2
        if d_keep <= d_swap:
            return c0, c1
        return c1, c0

    intensity_margin = 6.0
    if abs(i0 - i1) >= intensity_margin:
        # In these videos, black mouse is darker and white mouse is brighter.
        if i0 <= i1:
            return c0, c1
        return c1, c0

    # Final fallback: stable deterministic ordering.
    ordered = sorted((c0, c1), key=lambda p: p[0])
    return ordered[0], ordered[1]


def adapter_video_stub(cfg: dict) -> list[dict]:
    video_dir = Path(cfg["ingestion"]["local_video_dir"])
    sample_every_n = int(cfg.get("pose_stage", {}).get("sample_every_n_frames", 30))
    max_frames = cfg.get("pose_stage", {}).get("max_frames")
    max_frames = int(max_frames) if isinstance(max_frames, int) else None

    videos = sorted(path for path in video_dir.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)
    records: list[dict] = []
    prev_black: tuple[float, float] | None = None
    prev_white: tuple[float, float] | None = None

    for video in videos:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            continue

        frame_index = 0
        kept = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % sample_every_n != 0:
                frame_index += 1
                continue

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
                    "filename": f"{video.stem}_frame_{frame_index:06d}.jpg",
                    "width": int(width),
                    "height": int(height),
                    "labels": CANONICAL_LABELS,
                    "coords": {
                        "black": {"x": black_x, "y": black_y},
                        "white": {"x": white_x, "y": white_y},
                    },
                    "source_video": str(video).replace("\\", "/"),
                    "source_frame_idx": frame_index,
                }
            )

            kept += 1
            frame_index += 1
            if max_frames is not None and kept >= max_frames:
                break

        cap.release()

    if not records:
        raise SystemExit(
            f"No frames produced from video directory: {video_dir}. "
            f"Put videos there or switch pose_stage.mode."
        )

    return records


def _adapter_external_canonical_json(cfg: dict) -> list[dict]:
    external_path = cfg.get("pose_stage", {}).get("external_pose_file")
    if not external_path:
        raise SystemExit("pose_stage.external_pose_file is required for external canonical adapters")

    path = Path(external_path)
    if not path.exists():
        raise SystemExit(f"External pose file not found: {path}")

    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise SystemExit("External canonical pose file must be a JSON list of records")

    _validate_canonical(records)
    return _limit_records(records, cfg.get("pose_stage", {}).get("max_frames"))


def _validate_continuous_video_external(records: list[dict]) -> None:
    if not records:
        raise SystemExit("External canonical pose file is empty")

    source_videos = {str(rec.get("source_video", "")) for rec in records}
    if "" in source_videos or len(source_videos) != 1:
        raise SystemExit(
            "continuous_video_external_inference requires exactly one non-empty source_video across all records"
        )

    frame_indices: list[int] = []
    for index, rec in enumerate(records):
        idx = rec.get("source_frame_idx")
        if idx is None:
            raise SystemExit(
                f"continuous_video_external_inference requires source_frame_idx on every record (missing at row {index})"
            )
        frame_indices.append(int(idx))

    sorted_indices = sorted(frame_indices)
    if len(set(sorted_indices)) != len(sorted_indices):
        raise SystemExit("continuous_video_external_inference found duplicate source_frame_idx values")

    expected = list(range(sorted_indices[0], sorted_indices[-1] + 1))
    if sorted_indices != expected:
        raise SystemExit(
            "continuous_video_external_inference expects contiguous frame coverage with no gaps in source_frame_idx"
        )


def adapter_deeplabcut_canonical_json(cfg: dict) -> list[dict]:
    return _adapter_external_canonical_json(cfg)


def adapter_sleap_canonical_json(cfg: dict) -> list[dict]:
    return _adapter_external_canonical_json(cfg)


def adapter_continuous_video_external_inference(cfg: dict) -> list[dict]:
    records = _adapter_external_canonical_json(cfg)
    _validate_continuous_video_external(records)
    return records


def get_adapter(mode: str):
    registry = {
        "dataset_json_passthrough": adapter_dataset_json_passthrough,
        "video_stub": adapter_video_stub,
        "deeplabcut_canonical_json": adapter_deeplabcut_canonical_json,
        "sleap_canonical_json": adapter_sleap_canonical_json,
        "continuous_video_external_inference": adapter_continuous_video_external_inference,
    }
    if mode not in registry:
        supported = ", ".join(sorted(registry.keys()))
        raise SystemExit(f"Unsupported pose_stage.mode '{mode}'. Supported: {supported}")
    return registry[mode]
