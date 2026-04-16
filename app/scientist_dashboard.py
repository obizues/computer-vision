from __future__ import annotations

import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError:
    cv2 = None

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Mouse Vision Scientist Dashboard", layout="wide")

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data" / "eda_outputs"
RUN_REGISTRY = BASE / "data" / "run_registry.jsonl"
POSE_PATH = BASE / "data" / "processed" / "pose_top_keypoints.json"
CONFIG_PATH = BASE / "configs" / "mvp_config.json"

if CONFIG_PATH.exists():
    CFG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
else:
    CFG = {}

RAW_TOP_DIR = BASE / CFG.get("dataset", {}).get("raw_images_top_dir", "data/raw_images_top")


def resolve_raw_image_path(filename: str) -> Path | None:
    if not filename:
        return None
    candidates = [
        RAW_TOP_DIR / filename,
        RAW_TOP_DIR / "raw_images_top" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def parse_frame_idx_from_name(name: str) -> int | None:
    match = re.search(r"_frame_(\d+)\.", str(name))
    if match is None:
        return None
    return int(match.group(1))


@st.cache_data(show_spinner=False)
def load_pose_index() -> tuple[dict[int, dict], str | None]:
    if not POSE_PATH.exists():
        return {}, None

    raw_text = POSE_PATH.read_text(encoding="utf-8").strip()
    if not raw_text:
        return {}, None

    try:
        records = json.loads(raw_text)
    except json.JSONDecodeError:
        return {}, None

    if not isinstance(records, list):
        return {}, None

    index: dict[int, dict] = {}
    source_video: str | None = None
    for fallback_idx, rec in enumerate(records):
        idx = rec.get("source_frame_idx")
        if idx is None:
            idx = parse_frame_idx_from_name(rec.get("filename", ""))
        if idx is None:
            idx = fallback_idx
        index[int(idx)] = rec
        if source_video is None:
            source_video = rec.get("source_video")
    return index, source_video


def _valid_pt(x, y) -> tuple[int, int] | None:
    """Return (int(x), int(y)) if both are finite non-zero, else None."""
    if x is None or y is None:
        return None
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None
    if not math.isfinite(float(x)) or not math.isfinite(float(y)):
        return None
    return (int(x), int(y))


def draw_pose_overlay(frame, rec: dict) -> None:
    """Draw keypoints for both mice.  Black mouse = filled lime-green circle.
    White mouse = open orange ring slightly larger, so both show when overlapping."""
    if cv2 is None:
        return
    black = rec["coords"]["black"]
    white = rec["coords"]["white"]
    # Black mouse: filled lime-green dot (radius 5)
    for x, y in zip(black["x"], black["y"]):
        pt = _valid_pt(x, y)
        if pt is None:
            continue
        cv2.circle(frame, pt, 5, (0, 255, 0), -1)   # lime green filled
    # White mouse: open orange ring (radius 8, thickness 2) — sits around black dots when co-located
    for x, y in zip(white["x"], white["y"]):
        pt = _valid_pt(x, y)
        if pt is None:
            continue
        cv2.circle(frame, pt, 8, (0, 165, 255), 2)  # orange ring


def _count_valid_mouse_points(rec: dict, mouse: str) -> int:
    try:
        coords = rec.get("coords", {}).get(mouse, {})
        xs = coords.get("x", [])
        ys = coords.get("y", [])
    except Exception:
        return 0

    count = 0
    for x, y in zip(xs, ys):
        if x is None or y is None:
            continue
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        if not math.isfinite(float(x)) or not math.isfinite(float(y)):
            continue
        if float(x) <= 0.0 or float(y) <= 0.0:
            continue
        count += 1
    return count


def segment_has_both_mice(
    pose_index: dict[int, dict],
    start_frame: int,
    end_frame: int,
    min_points_per_mouse: int = 3,
    min_ratio: float = 0.9,
) -> bool:
    total = 0
    both = 0
    for frame_idx in range(int(start_frame), int(end_frame) + 1):
        rec = pose_index.get(frame_idx)
        if rec is None:
            continue
        total += 1
        black_pts = _count_valid_mouse_points(rec, "black")
        white_pts = _count_valid_mouse_points(rec, "white")
        if black_pts >= min_points_per_mouse and white_pts >= min_points_per_mouse:
            both += 1

    if total == 0:
        return False
    return (both / total) >= float(min_ratio)


def segment_has_low_nose_dist(
    features_df: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    distance_quantile: float = 0.25,
) -> bool:
    """Check if segment's average nose_dist is in bottom quantile (close together).
    Requires both mice to have valid (non-zero) coordinates."""
    if features_df.empty or "nose_dist" not in features_df.columns:
        return True  # If no features available, don't exclude
    
    segment_frames = features_df[
        (features_df["frame_idx"] >= int(start_frame)) & (features_df["frame_idx"] <= int(end_frame))
    ]
    
    if segment_frames.empty:
        return False
    
    # Filter to valid measurements: nose_dist not NaN, not 0.0, AND both mice noses have valid xyz coords
    # (both black and white nose x,y coordinates should be non-zero and non-NaN)
    valid_frames = segment_frames[
        (segment_frames["nose_dist"].notna()) 
        & (segment_frames["nose_dist"] > 0.0)
        & (segment_frames["b_nose_x"] > 0.0)
        & (segment_frames["b_nose_y"] > 0.0)
        & (segment_frames["w_nose_x"] > 0.0)
        & (segment_frames["w_nose_y"] > 0.0)
    ]
    
    # If >50% of frames lack valid bilateral pose data, reject the segment
    if len(valid_frames) < len(segment_frames) * 0.5:
        return False
    
    # Compute the distance threshold from all valid frames in full dataset
    valid_all = features_df[
        (features_df["nose_dist"].notna()) 
        & (features_df["nose_dist"] > 0.0)
        & (features_df["b_nose_x"] > 0.0)
        & (features_df["b_nose_y"] > 0.0)
        & (features_df["w_nose_x"] > 0.0)
        & (features_df["w_nose_y"] > 0.0)
    ]
    if valid_all.empty:
        return True  # No valid data anywhere, don't filter
    
    nose_dist_threshold = valid_all["nose_dist"].quantile(distance_quantile)
    avg_nose_dist = valid_frames["nose_dist"].mean()
    
    return float(avg_nose_dist) <= float(nose_dist_threshold)


def _both_mice_visible(rec: dict, min_points_per_mouse: int = 3) -> bool:
    black_pts = _count_valid_mouse_points(rec, "black")
    white_pts = _count_valid_mouse_points(rec, "white")
    return black_pts >= min_points_per_mouse and white_pts >= min_points_per_mouse


def find_both_mice_span(
    pose_index: dict[int, dict],
    start_frame: int,
    end_frame: int,
    min_points_per_mouse: int = 3,
) -> tuple[int, int] | None:
    visible_frames: list[int] = []
    for frame_idx in range(int(start_frame), int(end_frame) + 1):
        rec = pose_index.get(frame_idx)
        if rec is None:
            continue
        if _both_mice_visible(rec, min_points_per_mouse=min_points_per_mouse):
            visible_frames.append(frame_idx)

    if not visible_frames:
        return None
    return int(min(visible_frames)), int(max(visible_frames))


@st.cache_data(max_entries=300, show_spinner=False)
def _cached_load_frame_rgb(source_video: str, frame_idx: int):
    """Load a single frame from disk; cached by (source_video, frame_idx)."""
    if cv2 is None:
        return None
    source_video_path = Path(source_video)
    if source_video_path.exists():
        cap = cv2.VideoCapture(str(source_video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        cap.release()
        if ok:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def load_frame_rgb(rec: dict, frame_idx: int, source_video: str | None):
    if cv2 is None:
        return None
    if source_video:
        result = _cached_load_frame_rgb(source_video, frame_idx)
        if result is not None:
            return result

    filename = str(rec.get("filename", ""))
    if filename:
        image_path = resolve_raw_image_path(filename)
        if image_path is not None:
            image = cv2.imread(str(image_path))
            if image is not None:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return None


def load_overlay_pair(rec: dict, frame_idx: int, source_video: str | None):
    raw_rgb = load_frame_rgb(rec, frame_idx, source_video)
    if raw_rgb is None:
        return None, None

    overlay_rgb = raw_rgb.copy()
    draw_pose_overlay(overlay_rgb, rec)
    return raw_rgb, overlay_rgb


def build_event_segments(pred_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    high = pred_df[pred_df["y_proba_close"] >= threshold].copy()
    if high.empty:
        return pd.DataFrame(columns=["segment_id", "start_frame", "end_frame", "num_frames", "duration_frames", "peak_proba", "mean_proba"])

    frame_diffs = pred_df["frame_idx"].diff().dropna()
    positive_diffs = frame_diffs[frame_diffs > 0]
    expected_step = int(positive_diffs.median()) if not positive_diffs.empty else 1
    if expected_step <= 0:
        expected_step = 1

    high = high.sort_values("frame_idx").reset_index(drop=True)
    high["new_segment"] = high["frame_idx"].diff().fillna(expected_step + 1) > expected_step
    # cumsum starts at 1 for the first row (which is always True), so IDs are 1-based with no +1 offset
    high["segment_id"] = high["new_segment"].cumsum().astype(int)

    segments = (
        high.groupby("segment_id", as_index=False)
        .agg(
            start_frame=("frame_idx", "min"),
            end_frame=("frame_idx", "max"),
            num_frames=("frame_idx", "count"),
            peak_proba=("y_proba_close", "max"),
            mean_proba=("y_proba_close", "mean"),
        )
        .sort_values("start_frame")
        .reset_index(drop=True)
    )
    segments["duration_frames"] = segments["end_frame"] - segments["start_frame"] + expected_step
    
    # Post-process: merge segments separated by <= 2 frames (user's close interactions)
    segments = _merge_nearby_segments(segments, max_gap_frames=2)
    
    return segments


def build_distance_segments(
    features_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    distance_quantile: float,
) -> pd.DataFrame:
    required_cols = {"frame_idx", "nose_dist"}
    if features_df.empty or not required_cols.issubset(features_df.columns):
        return pd.DataFrame(columns=["segment_id", "start_frame", "end_frame", "num_frames", "duration_frames", "peak_proba", "mean_proba"])

    valid = features_df[
        features_df["nose_dist"].notna() & (features_df["nose_dist"] > 0)
    ][["frame_idx", "nose_dist"]].copy()
    if valid.empty:
        return pd.DataFrame(columns=["segment_id", "start_frame", "end_frame", "num_frames", "duration_frames", "peak_proba", "mean_proba"])

    threshold = float(valid["nose_dist"].quantile(float(distance_quantile)))
    low = valid[valid["nose_dist"] <= threshold][["frame_idx"]].drop_duplicates().sort_values("frame_idx").reset_index(drop=True)
    if low.empty:
        return pd.DataFrame(columns=["segment_id", "start_frame", "end_frame", "num_frames", "duration_frames", "peak_proba", "mean_proba"])

    frame_diffs = valid["frame_idx"].diff().dropna()
    positive_diffs = frame_diffs[frame_diffs > 0]
    expected_step = int(positive_diffs.median()) if not positive_diffs.empty else 1
    if expected_step <= 0:
        expected_step = 1

    proba_by_frame: dict[int, float] = {}
    if "frame_idx" in pred_df.columns and "y_proba_close" in pred_df.columns:
        proba_by_frame = (
            pred_df[["frame_idx", "y_proba_close"]]
            .drop_duplicates(subset=["frame_idx"], keep="last")
            .set_index("frame_idx")["y_proba_close"]
            .to_dict()
        )

    low["y_proba_close"] = low["frame_idx"].map(proba_by_frame).fillna(0.0)
    low["new_segment"] = low["frame_idx"].diff().fillna(expected_step + 1) > expected_step
    low["segment_id"] = low["new_segment"].cumsum().astype(int)

    segments = (
        low.groupby("segment_id", as_index=False)
        .agg(
            start_frame=("frame_idx", "min"),
            end_frame=("frame_idx", "max"),
            num_frames=("frame_idx", "count"),
            peak_proba=("y_proba_close", "max"),
            mean_proba=("y_proba_close", "mean"),
        )
        .sort_values("start_frame")
        .reset_index(drop=True)
    )
    segments["duration_frames"] = segments["end_frame"] - segments["start_frame"] + expected_step
    segments = _merge_nearby_segments(segments, max_gap_frames=2)
    return segments


def _merge_nearby_segments(segments_df: pd.DataFrame, max_gap_frames: int = 2) -> pd.DataFrame:
    """Merge segments separated by <= max_gap_frames into continuous events."""
    if segments_df.empty or len(segments_df) < 2:
        return segments_df.reset_index(drop=True)
    
    segments_df = segments_df.sort_values("start_frame").reset_index(drop=True)
    merged = []
    current_start = int(segments_df.iloc[0]["start_frame"])
    current_end = int(segments_df.iloc[0]["end_frame"])
    current_peak = float(segments_df.iloc[0]["peak_proba"])
    total_frames_in_merge = int(segments_df.iloc[0]["num_frames"])
    total_proba_sum = float(segments_df.iloc[0]["mean_proba"]) * total_frames_in_merge
    
    for idx in range(1, len(segments_df)):
        next_start = int(segments_df.iloc[idx]["start_frame"])
        next_end = int(segments_df.iloc[idx]["end_frame"])
        gap = next_start - current_end - 1
        
        if gap <= max_gap_frames:
            # Merge: extend end and update aggregates
            current_end = next_end
            current_peak = max(current_peak, float(segments_df.iloc[idx]["peak_proba"]))
            next_frames = int(segments_df.iloc[idx]["num_frames"])
            total_proba_sum += float(segments_df.iloc[idx]["mean_proba"]) * next_frames
            total_frames_in_merge += next_frames
        else:
            # Gap too large: save current segment and start a new one
            merged.append({
                "start_frame": current_start,
                "end_frame": current_end,
                "num_frames": total_frames_in_merge,
                "peak_proba": current_peak,
                "mean_proba": total_proba_sum / total_frames_in_merge,
                "duration_frames": current_end - current_start + 1,
            })
            current_start = next_start
            current_end = next_end
            current_peak = float(segments_df.iloc[idx]["peak_proba"])
            total_frames_in_merge = int(segments_df.iloc[idx]["num_frames"])
            total_proba_sum = float(segments_df.iloc[idx]["mean_proba"]) * total_frames_in_merge
    
    # Add final segment
    merged.append({
        "start_frame": current_start,
        "end_frame": current_end,
        "num_frames": total_frames_in_merge,
        "peak_proba": current_peak,
        "mean_proba": total_proba_sum / total_frames_in_merge,
        "duration_frames": current_end - current_start + 1,
    })
    
    merged_df = pd.DataFrame(merged)
    merged_df["segment_id"] = range(1, len(merged_df) + 1)
    return merged_df[["segment_id", "start_frame", "end_frame", "num_frames", "duration_frames", "peak_proba", "mean_proba"]]


def find_segment_for_frame(segments_df: pd.DataFrame, frame_idx: int) -> dict | None:
    if segments_df.empty:
        return None

    matches = segments_df[
        (segments_df["start_frame"] <= frame_idx) & (segments_df["end_frame"] >= frame_idx)
    ]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def render_probability_timeline(
    pred_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    selected_frame: int,
    threshold: float,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(pred_df["frame_idx"], pred_df["y_proba_close"], color="#2563EB", linewidth=1.5)
    ax.axhline(threshold, color="#DC2626", linestyle="--", linewidth=1, label=f"threshold={threshold:.2f}")
    ax.axvline(selected_frame, color="#111827", linestyle=":", linewidth=2, label=f"selected={selected_frame}")

    for _, segment in segments_df.iterrows():
        ax.axvspan(
            float(segment["start_frame"]),
            float(segment["end_frame"]),
            color="#FDE68A",
            alpha=0.35,
        )

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Close-event probability")
    ax.set_title("Frame-by-frame event score timeline")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    st.pyplot(fig)


def resolve_source_video_path(source_video: str | None) -> Path | None:
    candidates: list[str] = []
    if source_video:
        candidates.append(str(source_video))

    configured = str(CFG.get("pose_inference_runtime", {}).get("video_file", "")).strip()
    if configured:
        candidates.append(configured)

    for candidate in candidates:
        path = Path(candidate)
        if not path.is_absolute():
            path = BASE / path
        if path.exists():
            return path
    return None


def json_file_has_records(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        raw_text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    if not raw_text:
        return False
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, list) and len(payload) > 0


def build_pipeline_status(pred_exists: bool, segments_count: int, source_video_path: Path | None) -> list[tuple[str, bool]]:
    external_pose_path = Path(CFG.get("pose_stage", {}).get("external_pose_file", "data/processed/external_pose_predictions.json"))
    if not external_pose_path.is_absolute():
        external_pose_path = BASE / external_pose_path

    statuses = [
        ("Video loaded", source_video_path is not None and source_video_path.exists()),
        ("Pose extracted", json_file_has_records(external_pose_path)),
        ("Pose normalized", json_file_has_records(POSE_PATH)),
        ("Frames scored", pred_exists),
        ("Segments detected", segments_count > 0),
    ]
    return statuses


def _resolve_path_from_cfg(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = BASE / candidate
    return candidate


def _format_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 ** 2:
        return f"{num_bytes / 1024:.1f} KB"
    if num_bytes < 1024 ** 3:
        return f"{num_bytes / (1024 ** 2):.1f} MB"
    return f"{num_bytes / (1024 ** 3):.2f} GB"


def _format_eta(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds <= 0:
        return "estimating…"
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours}h {mins}m {secs}s"
    if mins > 0:
        return f"{mins}m {secs}s"
    return f"{secs}s"


def _estimate_eta_seconds(progress_pct: float) -> float | None:
    samples: list[tuple[float, float]] = st.session_state.setdefault("pipeline_progress_samples", [])
    now = time.time()
    samples.append((now, float(progress_pct)))
    if len(samples) > 30:
        del samples[:-30]

    if len(samples) < 4:
        return None

    recent = samples[-10:]
    t0, p0 = recent[0]
    t1, p1 = recent[-1]
    dt = t1 - t0
    dp = p1 - p0
    if dt <= 0 or dp <= 0.5:
        return None

    rate = dp / dt
    if rate <= 0:
        return None
    return max(0.0, (100.0 - progress_pct) / rate)


def build_pipeline_progress_snapshot(video_path_value: str | None) -> dict[str, Any]:
    runtime_cfg = CFG.get("pose_inference_runtime", {})
    dlc_cfg = runtime_cfg.get("dlc", {})

    input_video = _resolve_path_from_cfg(video_path_value) or _resolve_path_from_cfg(str(runtime_cfg.get("video_file", "")).strip())
    stem = input_video.stem if input_video is not None else "current_video"
    input_video_bytes = input_video.stat().st_size if input_video is not None and input_video.exists() else 0

    dlc_out_dir = _resolve_path_from_cfg(str(dlc_cfg.get("output_dir", "data/processed/dlc_outputs")).strip())
    if dlc_out_dir is None:
        dlc_out_dir = BASE / "data" / "processed" / "dlc_outputs"

    black_masked = dlc_out_dir / f"{stem}_black_masked.mp4"
    white_masked = dlc_out_dir / f"{stem}_white_masked.mp4"
    black_h5 = dlc_out_dir / f"{stem}_black_maskedDLC_snapshot-200000.h5"
    white_h5 = dlc_out_dir / f"{stem}_white_maskedDLC_snapshot-200000.h5"
    black_csv = dlc_out_dir / f"{stem}_black_maskedDLC_snapshot-200000.csv"
    white_csv = dlc_out_dir / f"{stem}_white_maskedDLC_snapshot-200000.csv"

    masked_total_bytes = sum(p.stat().st_size for p in [black_masked, white_masked] if p.exists())
    masked_target_bytes = max(1, int(input_video_bytes * 2.0)) if input_video_bytes > 0 else max(masked_total_bytes, 1)
    masked_progress = min(1.0, masked_total_bytes / masked_target_bytes)

    h5_count = int(black_h5.exists()) + int(white_h5.exists())
    csv_count = int(black_csv.exists()) + int(white_csv.exists())
    inference_progress = 0.4 * (h5_count / 2.0) + 0.6 * (csv_count / 2.0)

    external_pose_path = _resolve_path_from_cfg(str(CFG.get("pose_stage", {}).get("external_pose_file", "data/processed/external_pose_predictions.json")))
    external_pose_done = json_file_has_records(external_pose_path) if external_pose_path is not None else False
    normalized_pose_done = json_file_has_records(POSE_PATH)
    scored_done = (OUT / "batch_predictions.csv").exists()

    progress_pct = 100.0 * (
        0.35 * masked_progress
        + 0.35 * inference_progress
        + 0.15 * (1.0 if external_pose_done else 0.0)
        + 0.10 * (1.0 if normalized_pose_done else 0.0)
        + 0.05 * (1.0 if scored_done else 0.0)
    )

    return {
        "progress_pct": max(0.0, min(100.0, progress_pct)),
        "h5_count": h5_count,
        "csv_count": csv_count,
        "masked_total_bytes": masked_total_bytes,
        "masked_target_bytes": masked_target_bytes,
    }


def is_demo_mode() -> bool:
    env_value = str(os.environ.get("MOUSE_VISION_DEMO_MODE", "")).strip().lower()
    if env_value in {"1", "true", "yes", "on"}:
        return True
    if env_value in {"0", "false", "no", "off"}:
        return False
    hostname = str(os.environ.get("HOSTNAME", "")).lower()
    if "streamlit" in hostname:
        return True
    return False


def find_precomputed_clip(
    segment_id: int,
    mode_tag: str,
    clip_start: int,
    clip_end: int,
    context_frames: int = 0,
) -> Path | None:
    """Look for precomputed segment clip before trying to generate it.
    Clips should be in data/eda_outputs/segments/ from pipeline run."""
    segments_dir = OUT / "segments"
    if not segments_dir.exists():
        return None
    
    # Try exact expected filenames first.
    for suffix in [".webm", ".mp4"]:
        clip_path = segments_dir / f"segment_{segment_id}_{mode_tag}_ctx{context_frames}_{clip_start}_{clip_end}{suffix}"
        if clip_path.exists() and clip_path.stat().st_size > 0:
            return clip_path

    # Legacy exact format support: segment_<id>_<start>_<end>_<mode>.<ext>
    for suffix in [".webm", ".mp4"]:
        legacy_path = segments_dir / f"segment_{segment_id}_{clip_start}_{clip_end}_{mode_tag}{suffix}"
        if legacy_path.exists() and legacy_path.stat().st_size > 0:
            return legacy_path

    # Fallback: choose the precomputed clip whose frame span is closest to the
    # requested span, but keep context strict and reject oversized alternatives.
    target_len = max(1, int(clip_end) - int(clip_start) + 1)
    min_reasonable_len = 8
    max_allowed_len = max(12, target_len * 3)

    parsed_candidates: list[tuple[Path, int, int, int]] = []
    pattern = re.compile(
        rf"^segment_{segment_id}_{mode_tag}_ctx(\d+)_(\d+)_(\d+)\.(?:webm|mp4)$"
    )

    for path in list(segments_dir.glob(f"segment_{segment_id}_{mode_tag}_*.webm")) + list(
        segments_dir.glob(f"segment_{segment_id}_{mode_tag}_*.mp4")
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            continue
        match = pattern.match(path.name)
        if match is None:
            continue
        ctx_val = int(match.group(1))
        start_val = int(match.group(2))
        end_val = int(match.group(3))
        if ctx_val != int(context_frames):
            continue
        clip_len = max(1, end_val - start_val + 1)
        if clip_len > max_allowed_len:
            continue
        parsed_candidates.append((path, ctx_val, start_val, end_val))

    # Legacy fallback: segment_<id>_<start>_<end>_<mode>.<ext>
    legacy_pattern = re.compile(
        rf"^segment_{segment_id}_(\d+)_(\d+)_{mode_tag}\.(?:webm|mp4)$"
    )
    for path in list(segments_dir.glob(f"segment_{segment_id}_*_{mode_tag}.webm")) + list(
        segments_dir.glob(f"segment_{segment_id}_*_{mode_tag}.mp4")
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            continue
        match = legacy_pattern.match(path.name)
        if match is None:
            continue
        start_val = int(match.group(1))
        end_val = int(match.group(2))
        clip_len = max(1, end_val - start_val + 1)
        if clip_len > max_allowed_len:
            continue
        parsed_candidates.append((path, int(context_frames), start_val, end_val))

    if not parsed_candidates:
        return None

    long_enough = [
        item for item in parsed_candidates if (item[3] - item[2] + 1) >= min_reasonable_len
    ]
    pool = long_enough if long_enough else parsed_candidates

    def candidate_score(item: tuple[Path, int, int, int]) -> tuple[int, int, int]:
        _path, ctx_val, start_val, end_val = item
        clip_len = max(1, end_val - start_val + 1)
        return (abs(clip_len - target_len), abs(ctx_val - context_frames), ctx_val)

    best = min(pool, key=candidate_score)
    if best:
        return best[0]
    return None


def build_segment_clip(
    source_video_path: Path,
    pose_index: dict[int, dict],
    segment_id: int,
    start_frame: int,
    end_frame: int,
    with_pose_overlay: bool,
    context_frames: int = 30,
) -> Path | None:
    if cv2 is None:
        return None
    segments_dir = OUT / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    mode_tag = "overlay" if with_pose_overlay else "raw"
    codec_candidates: list[tuple[str, str]] = [
        (".webm", "VP80"),
        (".mp4", "mp4v"),
    ]

    context_frames = max(0, int(context_frames))
    clip_start = max(0, int(start_frame) - context_frames)
    clip_end = int(end_frame) + context_frames

    for suffix, _ in codec_candidates:
        existing_clip = segments_dir / f"segment_{segment_id}_{mode_tag}_ctx{context_frames}_{clip_start}_{clip_end}{suffix}"
        if existing_clip.exists() and existing_clip.stat().st_size > 0:
            return existing_clip

    cap = cv2.VideoCapture(str(source_video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames > 0:
        clip_end = min(clip_end, total_frames - 1)

    clip_path: Path | None = None
    writer = None
    fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
    for suffix, codec in codec_candidates:
        candidate_clip = segments_dir / f"segment_{segment_id}_{mode_tag}_ctx{context_frames}_{clip_start}_{clip_end}{suffix}"
        fourcc_raw: Any = fourcc_fn(*codec) if callable(fourcc_fn) else 0
        fourcc = int(fourcc_raw)
        candidate_writer = cv2.VideoWriter(str(candidate_clip), fourcc, float(fps), (width, height))
        if candidate_writer.isOpened():
            clip_path = candidate_clip
            writer = candidate_writer
            break
        candidate_writer.release()

    if clip_path is None or writer is None:
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(clip_start))
    written = 0
    frame_id = int(clip_start)
    while frame_id <= int(clip_end):
        ok, frame = cap.read()
        if not ok:
            break

        if with_pose_overlay:
            rec = pose_index.get(frame_id)
            if rec is not None:
                draw_pose_overlay(frame, rec)
        writer.write(frame)

        written += 1
        frame_id += 1

    writer.release()
    cap.release()

    if written == 0:
        if clip_path.exists():
            clip_path.unlink()
        return None

    return clip_path


def video_format_for_path(path: Path) -> str:
    return "video/webm" if path.suffix.lower() == ".webm" else "video/mp4"

# ── pipeline runner helpers ──────────────────────────────────────────────────
def _stream_pipeline(proc: subprocess.Popen, log_q: "queue.Queue[str | None]") -> None:
    """Read stdout + stderr from *proc* and push lines onto *log_q*."""
    assert proc.stdout is not None
    for line in iter(proc.stdout.readline, ""):
        log_q.put(line)
    proc.wait()
    log_q.put(None)  # sentinel – pipeline finished


def launch_pipeline() -> None:
    """Start a pipeline subprocess and store it in session state."""
    proc = subprocess.Popen(
        [sys.executable, str(BASE / "scripts" / "run_pipeline.py"),
         "--config", str(BASE / "configs" / "mvp_config.json")],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(BASE),
    )
    log_q: queue.Queue[str | None] = queue.Queue()
    t = threading.Thread(target=_stream_pipeline, args=(proc, log_q), daemon=True)
    t.start()
    st.session_state["pipeline_proc"] = proc
    st.session_state["pipeline_log_q"] = log_q
    st.session_state["pipeline_log"] = []
    st.session_state["pipeline_done"] = False
    st.session_state["pipeline_started_at"] = time.time()
    st.session_state["pipeline_progress_samples"] = []


def set_pipeline_video_file(video_path: str) -> tuple[bool, str]:
    if not video_path:
        return False, "Video path is empty."

    candidate = Path(video_path)
    if not candidate.is_absolute():
        candidate = BASE / candidate
    if not candidate.exists():
        return False, f"Video file not found: {candidate}"

    cfg = {}
    if CONFIG_PATH.exists():
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    pose_runtime = cfg.setdefault("pose_inference_runtime", {})
    try:
        saved_value = str(candidate.relative_to(BASE).as_posix())
    except ValueError:
        saved_value = str(candidate.resolve().as_posix())
    pose_runtime["video_file"] = saved_value

    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    CFG["pose_inference_runtime"] = pose_runtime
    return True, f"Updated pipeline video: {saved_value}"


def save_uploaded_video(uploaded_file) -> tuple[bool, str, str | None]:
    if uploaded_file is None:
        return False, "No uploaded file provided.", None

    raw_video_dir = BASE / "data" / "raw_videos"
    raw_video_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(str(uploaded_file.name)).name
    if not safe_name:
        return False, "Uploaded file name is invalid.", None

    target = raw_video_dir / safe_name
    try:
        target.write_bytes(uploaded_file.getbuffer())
    except Exception as exc:
        return False, f"Failed to save uploaded video: {exc}", None

    relative_target = str(target.relative_to(BASE).as_posix())
    return True, f"Saved uploaded video to {relative_target}", relative_target


def list_local_videos() -> list[str]:
    raw_video_dir = BASE / "data" / "raw_videos"
    if not raw_video_dir.exists():
        return []

    allowed_ext = {".mp4", ".avi", ".mov", ".mkv"}
    videos: list[str] = []
    for item in sorted(raw_video_dir.iterdir()):
        if item.is_file() and item.suffix.lower() in allowed_ext:
            videos.append(str(item.relative_to(BASE).as_posix()))
    return videos


st.title("Mouse Behavior Analytics Dashboard")

summary_path = OUT / "baseline_model_summary.json"
quality_path = OUT / "feature_quality_report.json"
pred_path = OUT / "batch_predictions.csv"
overlay_path = OUT / "pose_overlay_preview.mp4"

# ── shared data load ──────────────────────────────────────────────────────────
pose_index, source_video = load_pose_index()
source_video_path = resolve_source_video_path(source_video)

@st.cache_data(show_spinner=False)
def _load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).sort_values("frame_idx").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _load_features(artifacts_dir: Path) -> pd.DataFrame:
    """Load features CSV to access nose_dist for distance gating."""
    features_path = artifacts_dir / "features_top_view.csv"
    if not features_path.exists():
        return pd.DataFrame()
    return pd.read_csv(features_path).sort_values("frame_idx").reset_index(drop=True)


pred_df: pd.DataFrame = _load_predictions(pred_path)
features_df: pd.DataFrame = _load_features(OUT)

# ── tabs ──────────────────────────────────────────────────────────────────────
tab_replay, tab_analytics = st.tabs(["Event Replay", "Analytics & Details"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Event Replay
# ═════════════════════════════════════════════════════════════════════════════
with tab_replay:
    # ── minimal run controls: video dropdown + run button ────────────────────
    configured_video = str(CFG.get("pose_inference_runtime", {}).get("video_file", "")).strip()
    if "dashboard_video_path" not in st.session_state:
        st.session_state["dashboard_video_path"] = configured_video

    local_video_options = list_local_videos()
    if local_video_options:
        current_path = str(st.session_state.get("dashboard_video_path", "")).strip()
        if current_path in local_video_options:
            default_idx = local_video_options.index(current_path)
        else:
            default_idx = 0
        selected_local_video = st.selectbox(
            "Choose local video",
            options=local_video_options,
            index=default_idx,
            help="Videos found in data/raw_videos",
        )
        if selected_local_video != current_path:
            st.session_state["dashboard_video_path"] = selected_local_video
            ok, message = set_pipeline_video_file(selected_local_video)
            if not ok:
                st.error(message)
    else:
        st.warning(
            "No local videos found in data/raw_videos. For full replay and pipeline controls, run this project locally and place a video in data/raw_videos (or use launch_mvp.bat)."
        )

    # ── Run Pipeline button ───────────────────────────────────────────────────
    pipeline_running = (
        "pipeline_proc" in st.session_state
        and st.session_state["pipeline_proc"] is not None
        and not st.session_state.get("pipeline_done", True)
    )
    demo_mode = is_demo_mode()

    run_col, status_col = st.columns([2, 5])
    with run_col:
        if pipeline_running:
            st.button("⏳ Pipeline running…", disabled=True, width="stretch")
        elif demo_mode:
            st.button("▶ Run Full Pipeline", disabled=True, width="stretch")
        else:
            if st.button("▶ Run Full Pipeline", type="primary", width="stretch"):
                launch_pipeline()
                st.rerun()
    if demo_mode and not pipeline_running:
        st.info(
            "Demo mode is enabled for hosted viewing, so pipeline execution is disabled in this app instance. "
            "To run ingestion, training, and scoring end-to-end, use the local setup from README and launch the pipeline locally."
        )

    # Drain any pending log lines from the background thread
    if "pipeline_log_q" in st.session_state:
        log_q: queue.Queue[str | None] = st.session_state["pipeline_log_q"]
        while True:
            try:
                line = log_q.get_nowait()
                if line is None:
                    st.session_state["pipeline_done"] = True
                    break
                st.session_state["pipeline_log"].append(line)
            except queue.Empty:
                break

    if pipeline_running:
        snap = build_pipeline_progress_snapshot(str(st.session_state.get("dashboard_video_path", "")).strip())
        progress_pct = float(snap.get("progress_pct", 0.0))
        eta_seconds = _estimate_eta_seconds(progress_pct)
        st.progress(min(1.0, max(0.0, progress_pct / 100.0)))
        st.caption(
            "Pipeline progress: "
            f"{progress_pct:.1f}% · ETA {_format_eta(eta_seconds)} · "
            f"H5 {snap['h5_count']}/2 · CSV {snap['csv_count']}/2 · "
            f"masked bytes {_format_bytes(snap['masked_total_bytes'])}/{_format_bytes(snap['masked_target_bytes'])}"
        )

    if pipeline_running or st.session_state.get("pipeline_log"):
        log_text = "".join(st.session_state.get("pipeline_log", []))
        with st.expander("Pipeline log", expanded=pipeline_running):
            st.code(log_text or "(starting…)", language="");
        if pipeline_running:
            time.sleep(2)
            st.rerun()   # keep refreshing to drain the queue and progress indicators

    with status_col:
        if st.session_state.get("pipeline_done") and st.session_state.get("pipeline_log"):
            rc = st.session_state["pipeline_proc"].returncode
            if rc == 0:
                st.success("Pipeline finished successfully. Reload the page to see updated results.")
            else:
                st.error(f"Pipeline exited with code {rc}. Check the log above for details.")

    st.divider()

    if pred_df.empty:
        st.warning("No predictions found yet. Run the pipeline first.")
    else:
        confidence_threshold = float(st.session_state.get("confidence_threshold", 0.85))
        segments_df = pd.DataFrame()
        if "y_proba_close" in pred_df.columns and "frame_idx" in pred_df.columns:
            segments_df = build_event_segments(pred_df, confidence_threshold)

        if len(segments_df) < 3 and not features_df.empty:
            distance_quantile = float(CFG.get("feature_build", {}).get("close_interaction_quantile", 0.075))
            distance_segments = build_distance_segments(
                features_df=features_df,
                pred_df=pred_df,
                distance_quantile=distance_quantile,
            )
            if not distance_segments.empty:
                if segments_df.empty:
                    segments_df = distance_segments
                else:
                    segments_df = pd.concat([segments_df, distance_segments], ignore_index=True)
                    segments_df = (
                        segments_df
                        .sort_values(["start_frame", "end_frame", "peak_proba"], ascending=[True, True, False])
                        .drop_duplicates(subset=["start_frame", "end_frame"], keep="first")
                        .reset_index(drop=True)
                    )
                    segments_df["segment_id"] = range(1, len(segments_df) + 1)

        if not segments_df.empty:
            sparse_candidates = len(segments_df) < 3

            # Use a strict filter when we have enough candidates. If sparse,
            # require bilateral visibility only so replay still has useful events.
            if sparse_candidates:
                keep_mask = segments_df.apply(
                    lambda row: segment_has_both_mice(
                        pose_index,
                        int(row["start_frame"]),
                        int(row["end_frame"]),
                    ),
                    axis=1,
                )
            else:
                keep_mask = segments_df.apply(
                    lambda row: (
                        segment_has_both_mice(
                            pose_index,
                            int(row["start_frame"]),
                            int(row["end_frame"]),
                        )
                        and segment_has_low_nose_dist(
                            features_df,
                            int(row["start_frame"]),
                            int(row["end_frame"]),
                        )
                    ),
                    axis=1,
                )
            segments_df = segments_df[keep_mask].reset_index(drop=True)

        if segments_df.empty:
            st.info("No segments detected with current predictions.")
        else:
            segment_ids = segments_df["segment_id"].astype(int).tolist()
            if "selected_segment_id" not in st.session_state or int(st.session_state.selected_segment_id) not in segment_ids:
                st.session_state.selected_segment_id = int(segment_ids[0])

            longest_segment_row = segments_df.sort_values(["num_frames", "peak_proba"], ascending=[False, False]).iloc[0]

            selected_segment_id = int(st.session_state.selected_segment_id)
            selected_segment = segments_df[segments_df["segment_id"] == selected_segment_id].iloc[0]
            start_frame = int(selected_segment["start_frame"])
            end_frame = int(selected_segment["end_frame"])
            both_span = find_both_mice_span(pose_index, start_frame, end_frame)
            min_visible_span_frames = 8
            if both_span is not None:
                both_start, both_end = both_span
                both_len = int(both_end) - int(both_start) + 1
                # Avoid collapsing playback to 1-3 frame clips.
                if both_len >= min_visible_span_frames:
                    start_frame, end_frame = both_start, both_end

            # Fixed replay defaults for demo clarity: Overlay + Segment only
            clip_context_frames = 0

            segment_overlay_clip_path = None
            segment_raw_clip_path = None
            # First, try to find precomputed clips (generated during pipeline run).
            # This path is used by hosted demo mode and does not require local source video.
            segment_overlay_clip_path = find_precomputed_clip(
                segment_id=selected_segment_id,
                mode_tag="overlay",
                clip_start=start_frame,
                clip_end=end_frame,
                context_frames=clip_context_frames,
            )
            segment_raw_clip_path = find_precomputed_clip(
                segment_id=selected_segment_id,
                mode_tag="raw",
                clip_start=start_frame,
                clip_end=end_frame,
                context_frames=clip_context_frames,
            )

            # If precomputed clips are missing, try to build them only when local
            # source video and cv2 are available.
            if source_video_path is not None and source_video_path.exists() and pose_index:
                if segment_overlay_clip_path is None and cv2 is not None:
                    segment_overlay_clip_path = build_segment_clip(
                        source_video_path=source_video_path,
                        pose_index=pose_index,
                        segment_id=selected_segment_id,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        with_pose_overlay=True,
                        context_frames=clip_context_frames,
                    )
                if segment_raw_clip_path is None and cv2 is not None:
                    segment_raw_clip_path = build_segment_clip(
                        source_video_path=source_video_path,
                        pose_index=pose_index,
                        segment_id=selected_segment_id,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        with_pose_overlay=False,
                        context_frames=clip_context_frames,
                    )

            # Videos area (first): fixed overlay-only replay
            if segment_overlay_clip_path is not None and segment_overlay_clip_path.exists():
                st.video(
                    str(segment_overlay_clip_path),
                    format=video_format_for_path(segment_overlay_clip_path),
                )
            else:
                st.info("Overlay clip unavailable. Ensure source video and pose records exist locally.")

            # Segment selector (under videos)
            sel_col, nav1, nav2, nav3 = st.columns([3, 1, 1, 1])
            ui_segment_id = int(
                sel_col.selectbox(
                    "Selected segment",
                    options=segment_ids,
                    index=segment_ids.index(selected_segment_id),
                    format_func=lambda sid: (
                        f"Segment {sid}  "
                        f"({int(segments_df[segments_df['segment_id'] == sid]['start_frame'].iloc[0])}"
                        f"–"
                        f"{int(segments_df[segments_df['segment_id'] == sid]['end_frame'].iloc[0])})"
                    ),
                )
            )
            if nav1.button("◀ Prev"):
                current_idx = segment_ids.index(selected_segment_id)
                ui_segment_id = int(segment_ids[max(0, current_idx - 1)])
            if nav2.button("Next ▶"):
                current_idx = segment_ids.index(selected_segment_id)
                ui_segment_id = int(segment_ids[min(len(segment_ids) - 1, current_idx + 1)])
            if nav3.button("Longest segment"):
                ui_segment_id = int(longest_segment_row["segment_id"])

            if ui_segment_id != selected_segment_id:
                st.session_state.selected_segment_id = ui_segment_id
                st.rerun()

            # Probability trace (under selector)
            segment_pred = pred_df[
                (pred_df["frame_idx"] >= start_frame) & (pred_df["frame_idx"] <= end_frame)
            ].copy()
            if not segment_pred.empty and "y_proba_close" in segment_pred.columns:
                fig_seg, ax_seg = plt.subplots(figsize=(10, 3.2))
                ax_seg.plot(segment_pred["frame_idx"], segment_pred["y_proba_close"], linewidth=1.5, color="#2563EB")
                ax_seg.axhline(confidence_threshold, color="#DC2626", linestyle="--", linewidth=1, label=f"threshold={confidence_threshold:.2f}")
                ax_seg.axhspan(confidence_threshold, 1.0, color="#DBEAFE", alpha=0.25, zorder=0)
                ax_seg.fill_between(
                    segment_pred["frame_idx"],
                    segment_pred["y_proba_close"],
                    confidence_threshold,
                    where=(segment_pred["y_proba_close"] >= confidence_threshold).tolist(),
                    alpha=0.35,
                    color="#2563EB",
                    interpolate=True,
                )
                ax_seg.set_title(f"Segment {selected_segment_id} — probability trace (frames {start_frame}–{end_frame})")
                ax_seg.set_xlabel("Frame index")
                ax_seg.set_ylabel("Event probability")
                ax_seg.set_ylim(-0.02, 1.02)
                ax_seg.legend(loc="upper right")
                ax_seg.grid(alpha=0.2)
                st.pyplot(fig_seg)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Analytics & Details
# ═════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    # ── model / quality summary ───────────────────────────────────────────────
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        mcols = st.columns(3)
        mcols[0].metric("Model AUC", f"{summary.get('auc', 'n/a')}")
        mcols[1].metric("Train Size", str(summary.get("train_size", "n/a")))
        mcols[2].metric("Test Size", str(summary.get("test_size", "n/a")))
    else:
        st.warning("Missing model summary. Run pipeline first.")

    if quality_path.exists():
        q = json.loads(quality_path.read_text(encoding="utf-8"))
        st.subheader("Data Quality")
        qcols = st.columns(4)
        qcols[0].metric("Rows", str(q.get("rows", "n/a")))
        qcols[1].metric("Null Cells", str(q.get("null_cells", "n/a")))
        qcols[2].metric("Out-of-bounds X", str(q.get("out_of_bounds_x", "n/a")))
        qcols[3].metric("Out-of-bounds Y", str(q.get("out_of_bounds_y", "n/a")))

    if not pred_df.empty:
        # re-compute segments using latest threshold (defaults to 0.80 if tab_replay not visited)
        _threshold = st.session_state.get("_analytics_threshold", 0.85)
        if "y_proba_close" in pred_df.columns:
            _threshold = st.slider(
                "Threshold (analytics view)",
                min_value=0.50, max_value=0.99, value=_threshold, step=0.01,
                key="analytics_threshold",
            )
            st.session_state["_analytics_threshold"] = _threshold
        _segments_df = build_event_segments(pred_df, _threshold) if "y_proba_close" in pred_df.columns else pd.DataFrame()

        # ── pipeline progress ─────────────────────────────────────────────────
        st.subheader("Pipeline Progress")
        statuses = build_pipeline_status(True, len(_segments_df), source_video_path)
        status_cols = st.columns(len(statuses))
        for i, (label, ok) in enumerate(statuses):
            status_cols[i].metric(label, "✅" if ok else "⏳")

        # ── run snapshot ──────────────────────────────────────────────────────
        st.subheader("Run Snapshot")
        frames_processed = len(pred_df)
        peak_segment_proba = float(_segments_df["peak_proba"].max()) if not _segments_df.empty else 0.0
        summary_cols = st.columns(5)
        summary_cols[0].metric("Source video", source_video_path.name if source_video_path else "n/a")
        summary_cols[1].metric("Frames processed", frames_processed)
        summary_cols[2].metric("Segments detected", len(_segments_df))
        summary_cols[3].metric("Peak segment score", f"{peak_segment_proba:.3f}")
        summary_cols[4].metric("Threshold", f"{_threshold:.2f}")

        # ── full-timeline chart ───────────────────────────────────────────────
        if "y_proba_close" in pred_df.columns:
            st.subheader("Full-run Probability Timeline")
            _full_min = int(pred_df["frame_idx"].min())
            _full_max = int(pred_df["frame_idx"].max())
            _selected_frame_ref = int(st.session_state.get("selected_frame", _full_min))
            render_probability_timeline(pred_df, _segments_df, _selected_frame_ref, _threshold)

        # ── segment frame review ──────────────────────────────────────────────
        if "frame_idx" in pred_df.columns:
            st.subheader("Frame-level Review")
            _full_min_frame = int(pred_df["frame_idx"].min())
            _full_max_frame = int(pred_df["frame_idx"].max())

            _sel_seg_start: int | None = None
            _sel_seg_end: int | None = None
            if not _segments_df.empty and "selected_segment_id" in st.session_state:
                _sid = int(st.session_state.selected_segment_id)
                _seg_match = _segments_df[_segments_df["segment_id"] == _sid]
                if not _seg_match.empty:
                    _sel_seg_start = int(_seg_match.iloc[0]["start_frame"])
                    _sel_seg_end = int(_seg_match.iloc[0]["end_frame"])

            _default_constrain = _sel_seg_start is not None
            constrain_to_segment = st.checkbox(
                "Constrain review to selected segment",
                value=_default_constrain,
            )

            if constrain_to_segment and _sel_seg_start is not None and _sel_seg_end is not None:
                min_frame = _sel_seg_start
                max_frame = _sel_seg_end
            else:
                min_frame = _full_min_frame
                max_frame = _full_max_frame

            if "selected_frame" not in st.session_state:
                st.session_state.selected_frame = min_frame
            if st.session_state.selected_frame < min_frame or st.session_state.selected_frame > max_frame:
                st.session_state.selected_frame = min_frame

            frame_review_segments = pd.DataFrame()
            event_frames: list[int] = []
            if "y_proba_close" in pred_df.columns:
                event_frames = pred_df[pred_df["y_proba_close"] >= _threshold]["frame_idx"].astype(int).tolist()
                frame_review_segments = build_event_segments(pred_df, _threshold)

                st.caption("Frame Controls")
                nav1, nav2, nav3, nav4, nav5, nav6 = st.columns([1, 1, 1, 1, 1, 2])
                if nav1.button("-10 frames"):
                    st.session_state.selected_frame = max(min_frame, int(st.session_state.selected_frame) - 10)
                if nav2.button("◀ Prev", key="fr_prev"):
                    st.session_state.selected_frame = max(min_frame, int(st.session_state.selected_frame) - 1)
                if nav3.button("Next ▶", key="fr_next"):
                    st.session_state.selected_frame = min(max_frame, int(st.session_state.selected_frame) + 1)
                if nav4.button("+10 frames"):
                    st.session_state.selected_frame = min(max_frame, int(st.session_state.selected_frame) + 10)
                if nav5.button("◀ Prev event") and event_frames:
                    prev = [f for f in event_frames if f < int(st.session_state.selected_frame)]
                    st.session_state.selected_frame = prev[-1] if prev else event_frames[-1]
                if nav6.button("Next event ▶") and event_frames:
                    nxt = [f for f in event_frames if f > int(st.session_state.selected_frame)]
                    st.session_state.selected_frame = nxt[0] if nxt else event_frames[0]
                st.caption(f"High-confidence frames detected: {len(event_frames)}")

            slider_col, jump_col = st.columns([4, 1])
            with jump_col:
                if st.button("Jump to segment start", disabled=(_sel_seg_start is None)):
                    if _sel_seg_start is not None:
                        st.session_state.selected_frame = int(_sel_seg_start)
            with slider_col:
                if min_frame < max_frame:
                    st.slider(
                        "Select frame index",
                        min_value=min_frame,
                        max_value=max_frame,
                        step=1,
                        key="selected_frame",
                    )
                else:
                    st.session_state.selected_frame = min_frame
                    st.info(f"Single frame available: {min_frame}")
            selected_frame = int(st.session_state.selected_frame)

            selected_rows = pred_df[pred_df["frame_idx"] == selected_frame]
            if not selected_rows.empty:
                selected_row = selected_rows.iloc[0]
                active_segment = find_segment_for_frame(frame_review_segments, selected_frame)

                info1, info2, info3, info4 = st.columns(4)
                info1.metric("Selected frame", selected_frame)
                info2.metric("Detected event", "Yes" if int(selected_row["y_pred"]) == 1 else "No")
                if "y_proba_close" in selected_row:
                    info3.metric("Event probability", f"{float(selected_row['y_proba_close']):.3f}")
                info4.metric(
                    "Active segment",
                    "None" if active_segment is None else f"#{int(active_segment['segment_id'])}",
                )

                if active_segment is not None:
                    st.success(
                        f"Frame {selected_frame} is inside segment {int(active_segment['segment_id'])} "
                        f"(frames {int(active_segment['start_frame'])}–{int(active_segment['end_frame'])}, "
                        f"peak={float(active_segment['peak_proba']):.3f})."
                    )
                else:
                    st.info("Selected frame is outside any detected high-confidence event segment.")

                st.caption("Prediction rows near selected frame")
                near = pred_df[
                    (pred_df["frame_idx"] >= selected_frame - 5) & (pred_df["frame_idx"] <= selected_frame + 5)
                ].copy()
                near["selected"] = near["frame_idx"] == selected_frame

                def highlight_selected(row):
                    return ["background-color: #FFF3B0" if bool(row["selected"]) else "" for _ in row]

                st.dataframe(near.style.apply(highlight_selected, axis=1), width="stretch")

                pose_rec = pose_index.get(selected_frame)
                if pose_rec is not None:
                    raw_rgb, overlay_rgb = load_overlay_pair(pose_rec, selected_frame, source_video)
                    if raw_rgb is not None and overlay_rgb is not None:
                        sc1, sc2 = st.columns(2)
                        sc1.image(raw_rgb, caption=f"Frame {selected_frame} — raw", width="stretch")
                        sc2.image(overlay_rgb, caption=f"Frame {selected_frame} — pose overlay", width="stretch")

                        if active_segment is not None:
                            st.caption("Event segment filmstrip")
                            fs = int(active_segment["start_frame"])
                            fe = int(active_segment["end_frame"])
                            segment_frame_ids = sorted({fs, fe, selected_frame, (fs + fe) // 2})
                            filmstrip_cols = st.columns(len(segment_frame_ids))
                            for idx, frame_id in enumerate(segment_frame_ids):
                                segment_rec = pose_index.get(frame_id)
                                if segment_rec is None:
                                    continue
                                _, segment_overlay = load_overlay_pair(segment_rec, frame_id, source_video)
                                if segment_overlay is None:
                                    continue
                                proba_match = pred_df[pred_df["frame_idx"] == frame_id]
                                frame_proba = None if proba_match.empty else float(proba_match.iloc[0]["y_proba_close"])
                                cap_text = f"frame {frame_id}" + (f"\np={frame_proba:.3f}" if frame_proba is not None else "")
                                filmstrip_cols[idx].image(segment_overlay, caption=cap_text, width="stretch")
                    else:
                        st.info("Raw frame not found locally. Run download_data with --include-raw-images-top.")

        # ── expanders ─────────────────────────────────────────────────────────
        with st.expander("Event table", expanded=False):
            st.dataframe(_segments_df, width="stretch")

