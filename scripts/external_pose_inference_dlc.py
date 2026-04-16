from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve(path_value: str | None, base: Path) -> Path | None:
    if not path_value:
        return None
    p = Path(path_value)
    if not p.is_absolute():
        p = base / p
    return p


def find_dlc_csv(output_dir: Path, video_stem: str) -> Path | None:
    candidates = sorted(output_dir.glob(f"{video_stem}*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    nested = sorted(output_dir.rglob(f"{video_stem}*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if nested:
        return nested[0]
    return None


def build_single_mouse_mask_videos(
    *,
    video_path: Path,
    output_dir: Path,
    mask_radius: int,
    max_frames: int | None,
) -> tuple[Path, Path]:
    from pose_adapters import _assign_black_white_centers, _contour_centroids, _fallback_centers

    output_dir.mkdir(parents=True, exist_ok=True)
    black_path = output_dir / f"{video_path.stem}_black_masked.mp4"
    white_path = output_dir / f"{video_path.stem}_white_masked.mp4"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        raise SystemExit(f"Could not read video dimensions: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    black_writer = cv2.VideoWriter(str(black_path), fourcc, fps, (width, height))
    white_writer = cv2.VideoWriter(str(white_path), fourcc, fps, (width, height))
    if not black_writer.isOpened() or not white_writer.isOpened():
        raise SystemExit("Could not create masked single-mouse videos for pretrained DLC fallback")

    prev_black: tuple[float, float] | None = None
    prev_white: tuple[float, float] | None = None
    prev_gray: cv2.typing.MatLike | None = None
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            # Compute optical flow for motion-based center refinement
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_mask = None
            if prev_gray is not None and prev_gray.shape == gray.shape:
                try:
                    # Compute optical flow between frames (Farneback algorithm)
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.1, 0
                    )
                    # Compute motion magnitude
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    # Create motion mask: regions with significant motion (likely mice)
                    motion_threshold = np.percentile(mag[mag > 0], 50) if np.any(mag > 0) else 1.0
                    motion_mask = (mag > motion_threshold).astype(np.uint8) * 255
                except Exception:
                    # If optical flow fails, skip motion refinement for this frame
                    motion_mask = None

            prev_gray = gray

            centers = _contour_centroids(frame)
            
            # Refine centers using motion information if available
            if motion_mask is not None and len(centers) >= 1:
                motion_scores = []
                for cx, cy in centers:
                    # Extract region around center
                    x1, y1 = max(0, int(cx) - mask_radius), max(0, int(cy) - mask_radius)
                    x2, y2 = min(width, int(cx) + mask_radius), min(height, int(cy) + mask_radius)
                    region = motion_mask[y1:y2, x1:x2]
                    # Score: fraction of this mouse's region with detected motion
                    score = np.sum(region) / (region.size + 1e-6) if region.size > 0 else 0.0
                    motion_scores.append(score)
                
                # Sort centers by motion score (more motion = more likely to be a mouse)
                centers = [c for _, c in sorted(zip(motion_scores, centers), key=lambda x: -x[0])]
            
            if len(centers) < 2:
                if len(centers) == 1 and prev_black is not None and prev_white is not None:
                    detected = centers[0]
                    db = (detected[0] - prev_black[0]) ** 2 + (detected[1] - prev_black[1]) ** 2
                    dw = (detected[0] - prev_white[0]) ** 2 + (detected[1] - prev_white[1]) ** 2
                    if db <= dw:
                        black_center = detected
                        white_center = prev_white
                    else:
                        black_center = prev_black
                        white_center = detected
                else:
                    fallback = _fallback_centers(width, height)
                    black_center, white_center = fallback[0], fallback[1]
            else:
                black_center, white_center = _assign_black_white_centers(frame, centers, prev_black, prev_white)

            prev_black, prev_white = black_center, white_center

            def _masked(center: tuple[float, float], other_center: tuple[float, float]) -> cv2.typing.MatLike:
                """Create non-overlapping circular mask: this mouse's circle minus the other's circle."""
                mask = np.zeros((height, width), dtype=np.uint8)
                # Draw this mouse's circle
                cv2.circle(
                    img=mask,
                    center=(int(round(center[0])), int(round(center[1]))),
                    radius=mask_radius,
                    color=255,
                    thickness=-1,
                )
                # Subtract the other mouse's circle to avoid overlap
                cv2.circle(
                    img=mask,
                    center=(int(round(other_center[0])), int(round(other_center[1]))),
                    radius=mask_radius,
                    color=0,
                    thickness=-1,
                )
                return cv2.bitwise_and(frame, frame, mask=mask)

            black_writer.write(_masked(black_center, white_center))
            white_writer.write(_masked(white_center, black_center))
            frame_idx += 1
    finally:
        cap.release()
        black_writer.release()
        white_writer.release()

    return black_path, white_path


def h5_to_csv(h5_path: Path) -> Path:
    csv_path = h5_path.with_suffix(".csv")
    df = pd.read_hdf(h5_path)
    df.to_csv(csv_path)
    return csv_path


def infer_total_frames(video_path: Path, max_frames: int | None) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        if max_frames is not None:
            return max_frames
        raise SystemExit(f"Could not open video for frame-count inference: {video_path}")
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if max_frames is not None and max_frames > 0:
        return min(n, max_frames) if n > 0 else max_frames
    return n


def densify_records(records: list[dict], total_frames: int, video_path: Path) -> list[dict]:
    """Fill sparse frames using velocity-based interpolation instead of static repeat.
    
    For frames without real detections, estimates position using velocity from nearby real frames.
    This produces smoother trajectories and eliminates visual jumping on playback.

    TODO: Reduce residual jumpiness by adding bounded interpolation with per-keypoint max
    displacement constraints, and audit missing tail frames (e.g., ending at frame 2376 vs
    expected total) to guarantee full-length continuity.
    """
    if total_frames <= 0:
        return records
    by_idx = {int(r.get("source_frame_idx", -1)): r for r in records if isinstance(r.get("source_frame_idx"), int)}
    if not by_idx:
        return records

    first_idx = min(by_idx)
    first = by_idx[first_idx]
    labels = first.get("labels", [])
    width = int(first.get("width", 1024))
    height = int(first.get("height", 570))

    def _zero_coords() -> dict:
        n = len(labels)
        return {
            "black": {"x": [0.0] * n, "y": [0.0] * n},
            "white": {"x": [0.0] * n, "y": [0.0] * n},
        }

    def _estimate_velocity(coords_a: dict, coords_b: dict, frame_diff: int) -> dict:
        """Compute per-keypoint velocity from two frame samples (frame_diff frames apart)."""
        velocity = {"black": {"x": [], "y": []}, "white": {"x": [], "y": []}}
        for mouse in ["black", "white"]:
            for coord in ["x", "y"]:
                vel_list = []
                for i in range(len(coords_a.get(mouse, {}).get(coord, []))):
                    a_val = coords_a.get(mouse, {}).get(coord, [])[i] if i < len(coords_a.get(mouse, {}).get(coord, [])) else 0.0
                    b_val = coords_b.get(mouse, {}).get(coord, [])[i] if i < len(coords_b.get(mouse, {}).get(coord, [])) else 0.0
                    vel = (b_val - a_val) / frame_diff if frame_diff > 0 else 0.0
                    vel_list.append(vel)
                velocity[mouse][coord] = vel_list
        return velocity

    def _interpolate_position(base_coords: dict, velocity: dict, num_steps: int) -> dict:
        """Forward-extrapolate position using velocity for num_steps frames."""
        result = {"black": {"x": [], "y": []}, "white": {"x": [], "y": []}}
        for mouse in ["black", "white"]:
            for coord in ["x", "y"]:
                pos_list = []
                for i in range(len(base_coords.get(mouse, {}).get(coord, []))):
                    base = base_coords.get(mouse, {}).get(coord, [])[i] if i < len(base_coords.get(mouse, {}).get(coord, [])) else 0.0
                    vel = velocity.get(mouse, {}).get(coord, [])[i] if i < len(velocity.get(mouse, {}).get(coord, [])) else 0.0
                    pos = base + vel * num_steps
                    pos_list.append(pos)
                result[mouse][coord] = pos_list
        return result

    sorted_indices = sorted(by_idx.keys())
    densified: list[dict] = []
    prev_coords = first.get("coords", _zero_coords())
    prev_velocity = {
        "black": {"x": [0.0] * len(labels), "y": [0.0] * len(labels)},
        "white": {"x": [0.0] * len(labels), "y": [0.0] * len(labels)},
    }
    last_real_idx = first_idx
    next_real_ptr = 0

    for frame_idx in range(total_frames):
        while next_real_ptr < len(sorted_indices) and sorted_indices[next_real_ptr] < frame_idx:
            next_real_ptr += 1

        if frame_idx in by_idx:
            rec = by_idx[frame_idx]
            new_coords = rec.get("coords", prev_coords)

            if frame_idx > last_real_idx:
                last_coords = by_idx.get(last_real_idx, {}).get("coords", prev_coords)
                frame_gap = frame_idx - last_real_idx
                prev_velocity = _estimate_velocity(last_coords, new_coords, frame_gap)

            prev_coords = new_coords
            last_real_idx = frame_idx
            densified.append(rec)
            continue

        if frame_idx < first_idx:
            fill_coords = first.get("coords", _zero_coords())
        else:
            steps_since_detection = frame_idx - last_real_idx
            fill_coords = _interpolate_position(prev_coords, prev_velocity, steps_since_detection)

        densified.append(
            {
                "filename": f"frame_{frame_idx:06d}.jpg",
                "width": width,
                "height": height,
                "labels": labels,
                "coords": fill_coords,
                "source_video": str(video_path).replace("\\", "/"),
                "source_frame_idx": frame_idx,
            }
        )

    return densified


def run_pretrained_single_mouse_inference(
    *,
    deeplabcut_module,
    video_path: Path,
    output_dir: Path,
    pretrained_model: str,
    scale_list: list[int],
    batch_size: int,
) -> Path:
    from deeplabcut.modelzoo.api.superanimal_inference import video_inference

    output_dir.mkdir(parents=True, exist_ok=True)
    _, datafiles = video_inference(
        videos=[str(video_path)],
        superanimal_name=pretrained_model,
        scale_list=scale_list,
        videotype=video_path.suffix,
        destfolder=str(output_dir),
        batchsize=batch_size,
    )
    if not datafiles:
        raise SystemExit(f"Pretrained DLC inference did not produce output for {video_path}")
    h5_path = Path(datafiles[0])
    if not h5_path.exists():
        raise SystemExit(f"Expected pretrained DLC output not found: {h5_path}")
    return h5_to_csv(h5_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DeepLabCut inference and export canonical pose JSON")
    parser.add_argument("--config", type=Path, default=Path("configs/mvp_config.json"))
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    base = Path.cwd()

    runtime = cfg.get("pose_inference_runtime", {})
    dlc_cfg = runtime.get("dlc", {})
    pose_stage = cfg.get("pose_stage", {})

    project_config = resolve(str(dlc_cfg.get("project_config", "")).strip(), base)

    dlc_out_dir = resolve(str(dlc_cfg.get("output_dir", "data/processed/dlc_outputs")).strip(), base)
    assert dlc_out_dir is not None
    dlc_out_dir.mkdir(parents=True, exist_ok=True)

    use_pretrained_fallback = bool(dlc_cfg.get("pretrained_fallback", True))
    pretrained_model = str(dlc_cfg.get("pretrained_model", "superanimal_topviewmouse")).strip() or "superanimal_topviewmouse"
    mask_radius = int(dlc_cfg.get("mask_radius", 140))
    pretrained_scale = int(dlc_cfg.get("pretrained_scale", 256))
    pretrained_batch_size = int(dlc_cfg.get("pretrained_batch_size", 8))
    max_frames_value = pose_stage.get("max_frames")
    max_frames = int(max_frames_value) if isinstance(max_frames_value, int) else None

    try:
        import deeplabcut
    except Exception as exc:
        raise SystemExit(
            "DeepLabCut is not installed in this environment. "
            "Install DLC in a supported Python env (typically 3.10/3.11), then retry. "
            f"Import error: {exc}"
        )

    video_path = args.video
    if not video_path.is_absolute():
        video_path = base / video_path
    if not video_path.exists():
        raise SystemExit(f"Video file not found: {video_path}")

    from pose_adapter_dlc import convert_dlc_multianimal, convert_dlc_single_animal_pair

    keypoint_map_file = resolve(str(dlc_cfg.get("keypoint_map_file", "")).strip(), base)
    keypoint_map = None
    if keypoint_map_file is not None and keypoint_map_file.exists():
        keypoint_map = json.loads(keypoint_map_file.read_text(encoding="utf-8"))

    black_id = str(dlc_cfg.get("black_id", "mouse1"))
    white_id = str(dlc_cfg.get("white_id", "mouse2"))
    likelihood_threshold = float(dlc_cfg.get("likelihood_threshold", 0.6))

    records: list[dict]
    project_ready = project_config is not None and project_config.exists()
    try:
        if not project_ready:
            raise FileNotFoundError("DeepLabCut project config not found")

        print(f"Running DLC analyze_videos on: {video_path}")
        deeplabcut.analyze_videos(
            str(project_config),
            [str(video_path)],
            save_as_csv=True,
            destfolder=str(dlc_out_dir),
        )

        csv_path = find_dlc_csv(dlc_out_dir, video_path.stem)
        if csv_path is None:
            raise SystemExit(f"Could not find DLC CSV output in {dlc_out_dir} for stem {video_path.stem}")

        records = convert_dlc_multianimal(
            csv_path=csv_path,
            black_id=black_id,
            white_id=white_id,
            video_path=str(video_path).replace("\\", "/"),
            keypoint_map=keypoint_map,
            likelihood_threshold=likelihood_threshold,
        )
    except Exception as exc:
        if not use_pretrained_fallback:
            raise

        print(f"Falling back to pretrained DLC model '{pretrained_model}' because project inference failed: {exc}")
        black_video, white_video = build_single_mouse_mask_videos(
            video_path=video_path,
            output_dir=dlc_out_dir,
            mask_radius=mask_radius,
            max_frames=max_frames,
        )
        existing_black_csv = find_dlc_csv(dlc_out_dir, black_video.stem)
        existing_white_csv = find_dlc_csv(dlc_out_dir, white_video.stem)

        if existing_black_csv is not None and existing_black_csv.exists():
            print(f"Reusing existing black masked CSV: {existing_black_csv}")
            black_csv = existing_black_csv
        else:
            print(f"Running pretrained inference for black masked video: {black_video}")
            black_csv = run_pretrained_single_mouse_inference(
                deeplabcut_module=deeplabcut,
                video_path=black_video,
                output_dir=dlc_out_dir,
                pretrained_model=pretrained_model,
                scale_list=[pretrained_scale],
                batch_size=pretrained_batch_size,
            )

        if existing_white_csv is not None and existing_white_csv.exists():
            print(f"Reusing existing white masked CSV: {existing_white_csv}")
            white_csv = existing_white_csv
        else:
            print(f"Running pretrained inference for white masked video: {white_video}")
            white_csv = run_pretrained_single_mouse_inference(
                deeplabcut_module=deeplabcut,
                video_path=white_video,
                output_dir=dlc_out_dir,
                pretrained_model=pretrained_model,
                scale_list=[pretrained_scale],
                batch_size=pretrained_batch_size,
            )
        records = convert_dlc_single_animal_pair(
            black_csv=black_csv,
            white_csv=white_csv,
            video_path=str(video_path).replace("\\", "/"),
            keypoint_map=keypoint_map,
            likelihood_threshold=likelihood_threshold,
        )

    total_frames = infer_total_frames(video_path, max_frames)
    records = densify_records(records, total_frames=total_frames, video_path=video_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records), encoding="utf-8")
    print(f"Wrote external pose predictions: {args.output} ({len(records)} frames)")


if __name__ == "__main__":
    main()
