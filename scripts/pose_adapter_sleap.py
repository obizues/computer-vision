"""Convert a SLEAP analysis HDF5 export to the canonical pose JSON format.

SLEAP produces an analysis HDF5 via:
    sleap-convert --format analysis predictions.slp -o analysis.h5
  or from the GUI: File → Export Analysis HDF5

Expected HDF5 structure (SLEAP ≥1.2):
    /tracks          float32  (n_tracks, n_frames, n_nodes, 2)  x,y; NaN = missing
    /track_names     bytes    track labels, e.g. b"mouse1"
    /node_names      bytes    skeleton node labels, e.g. b"nose"
    /point_scores    float32  (n_tracks, n_frames, n_nodes)     per-keypoint confidence

Usage
-----
    python scripts/pose_adapter_sleap.py \\
        --input       path/to/analysis.h5 \\
        --output      data/processed/external_pose_predictions.json \\
        --black-track mouse1 \\
        --white-track mouse2 \\
        --video       data/raw_videos/my_session.mp4 \\
        [--keypoint-map configs/sleap_keypoint_map.json]

Keypoint map (optional JSON)
----------------------------
Maps SLEAP node names → canonical pipeline slots:
    {
        "nose":       "nose tip",
        "right_ear":  "right ear",
        "left_ear":   "left ear",
        "neck":       "neck",
        "body_right": "right side body",
        "body_left":  "left side body",
        "tail_base":  "tail base"
    }
If not provided, the adapter tries a set of common naming conventions
automatically (see _build_default_keypoint_map).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CANONICAL_LABELS = [
    "nose tip",
    "right ear",
    "left ear",
    "neck",
    "right side body",
    "left side body",
    "tail base",
]

# Common synonym groups for each canonical slot.
# First match wins (case-insensitive, spaces/underscores normalised).
_SYNONYMS: dict[str, list[str]] = {
    "nose tip":        ["nose", "snout", "nose_tip", "noseTip"],
    "right ear":       ["right_ear", "rightear", "ear_right", "r_ear", "rear"],
    "left ear":        ["left_ear", "leftear", "ear_left", "l_ear", "lear"],
    "neck":            ["neck", "head_neck", "neck_base"],
    "right side body": ["right_side", "rightside", "body_right", "r_body", "right_body", "mid_right"],
    "left side body":  ["left_side", "leftside", "body_left", "l_body", "left_body", "mid_left"],
    "tail base":       ["tail_base", "tailbase", "tail", "tail_root"],
}


def _normalise(s: str) -> str:
    return s.strip().lower().replace(" ", "_").replace("-", "_")


def _build_default_keypoint_map(node_names: list[str]) -> dict[str, str]:
    """Auto-map SLEAP node names to canonical labels using synonym lists."""
    mapping: dict[str, str] = {}
    norm_nodes = {_normalise(n): n for n in node_names}
    for canonical, synonyms in _SYNONYMS.items():
        for syn in synonyms:
            if _normalise(syn) in norm_nodes:
                mapping[norm_nodes[_normalise(syn)]] = canonical
                break
    return mapping


def _decode_bytes_list(arr) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in arr]


def convert_sleap_h5(
    h5_path: Path,
    black_track: str,
    white_track: str,
    video_path: str,
    keypoint_map: dict[str, str] | None,
) -> list[dict]:
    try:
        import h5py
    except ImportError:
        raise SystemExit("h5py is required: pip install h5py")

    with h5py.File(h5_path, "r") as f:
        tracks = f["tracks"][:]           # (n_tracks, n_frames, n_nodes, 2)
        track_names = _decode_bytes_list(f["track_names"][:])
        node_names  = _decode_bytes_list(f["node_names"][:])
        point_scores = f["point_scores"][:] if "point_scores" in f else None

    if black_track not in track_names:
        raise SystemExit(
            f"--black-track '{black_track}' not found. Available: {track_names}"
        )
    if white_track not in track_names:
        raise SystemExit(
            f"--white-track '{white_track}' not found. Available: {track_names}"
        )

    bi = track_names.index(black_track)
    wi = track_names.index(white_track)

    if keypoint_map is None:
        keypoint_map = _build_default_keypoint_map(node_names)

    # Verify all canonical slots are covered
    covered = set(keypoint_map.values())
    missing = [c for c in CANONICAL_LABELS if c not in covered]
    if missing:
        raise SystemExit(
            f"Keypoint map does not cover canonical slots: {missing}\n"
            f"SLEAP nodes available: {node_names}\n"
            f"Provide --keypoint-map to specify the mapping explicitly."
        )

    # Build ordered index: canonical position → SLEAP node index
    canon_to_node_idx = {}
    for sleap_node, canon in keypoint_map.items():
        if sleap_node in node_names:
            canon_to_node_idx[canon] = node_names.index(sleap_node)

    n_tracks, n_frames, n_nodes, _ = tracks.shape

    # Infer image size from the max valid coordinate in the whole file
    valid = tracks[~np.isnan(tracks)]
    if valid.size == 0:
        raise SystemExit("All keypoints are NaN — check that tracking was run.")
    approx_w = int(np.nanmax(tracks[:, :, :, 0])) + 10
    approx_h = int(np.nanmax(tracks[:, :, :, 1])) + 10

    records: list[dict] = []
    for frame_idx in range(n_frames):
        b_frame = tracks[bi, frame_idx]   # (n_nodes, 2)
        w_frame = tracks[wi, frame_idx]

        # Skip frames where either animal has majority NaN keypoints
        b_valid = int(np.sum(~np.isnan(b_frame[:, 0])))
        w_valid = int(np.sum(~np.isnan(w_frame[:, 0])))
        if b_valid < 4 or w_valid < 4:
            continue

        bx, by, wx, wy = [], [], [], []
        for canon in CANONICAL_LABELS:
            ni = canon_to_node_idx[canon]
            bx.append(float(np.nan_to_num(b_frame[ni, 0])))
            by.append(float(np.nan_to_num(b_frame[ni, 1])))
            wx.append(float(np.nan_to_num(w_frame[ni, 0])))
            wy.append(float(np.nan_to_num(w_frame[ni, 1])))

        records.append({
            "filename": f"frame_{frame_idx:06d}.jpg",
            "width": approx_w,
            "height": approx_h,
            "labels": CANONICAL_LABELS,
            "coords": {
                "black": {"x": bx, "y": by},
                "white": {"x": wx, "y": wy},
            },
            "source_video": str(video_path).replace("\\", "/"),
            "source_frame_idx": frame_idx,
        })

    if not records:
        raise SystemExit("No valid frames found after filtering NaN keypoints.")

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SLEAP analysis HDF5 to canonical pose JSON"
    )
    parser.add_argument("--input", type=Path, required=True, help="SLEAP analysis .h5 file")
    parser.add_argument("--output", type=Path, required=True, help="Output canonical JSON path")
    parser.add_argument("--black-track", default="mouse1", help="SLEAP track name for black mouse")
    parser.add_argument("--white-track", default="mouse2", help="SLEAP track name for white mouse")
    parser.add_argument("--video", default="", help="Source video path (stored in JSON metadata)")
    parser.add_argument(
        "--keypoint-map",
        type=Path,
        default=None,
        help="JSON file mapping SLEAP node names to canonical labels",
    )
    args = parser.parse_args()

    keypoint_map: dict[str, str] | None = None
    if args.keypoint_map is not None:
        keypoint_map = json.loads(args.keypoint_map.read_text(encoding="utf-8"))

    records = convert_sleap_h5(
        h5_path=args.input,
        black_track=args.black_track,
        white_track=args.white_track,
        video_path=args.video,
        keypoint_map=keypoint_map,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} canonical pose records -> {args.output}")


if __name__ == "__main__":
    main()
