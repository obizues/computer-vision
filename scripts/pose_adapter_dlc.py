"""Convert a DeepLabCut (DLC) CSV or HDF5 output to canonical pose JSON.

DLC produces per-video CSV files (and optionally HDF5) with a 3- or 4-level
column header:

    Single-animal DLC (3 levels):
        scorer      | DLC_resnet50_...
        bodyparts   | nose | nose | nose | left_ear | ...
        coords      | x    | y    | likelihood | x | ...

    Multi-animal DLC (4 levels):
        scorer      | DLC_resnet50_...
        individuals | mouse1 | mouse1 | ... | mouse2 | ...
        bodyparts   | nose  | nose  | ...  | nose  | ...
        coords      | x     | y     | ...  | x     | ...

Usage
-----
    # Multi-animal DLC (most common for two-mouse studies)
    python scripts/pose_adapter_dlc.py \\
        --input       path/to/video_DLC.csv \\
        --output      data/processed/external_pose_predictions.json \\
        --black-id    mouse1 \\
        --white-id    mouse2 \\
        --video       data/raw_videos/my_session.mp4 \\
        [--keypoint-map configs/dlc_keypoint_map.json]

    # Single-animal DLC (two separate files, one per mouse)
    python scripts/pose_adapter_dlc.py \\
        --input-black  black_mouse_DLC.csv \\
        --input-white  white_mouse_DLC.csv \\
        --output       data/processed/external_pose_predictions.json \\
        --video        data/raw_videos/my_session.mp4

Keypoint map (optional JSON)
----------------------------
Maps DLC bodypart names → canonical pipeline slots:
    {
        "nose":       "nose tip",
        "right_ear":  "right ear",
        "left_ear":   "left ear",
        "neck":       "neck",
        "right_body": "right side body",
        "left_body":  "left side body",
        "tail_base":  "tail base"
    }
If not provided, automatic synonym matching is attempted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL_LABELS = [
    "nose tip",
    "right ear",
    "left ear",
    "neck",
    "right side body",
    "left side body",
    "tail base",
]

_SYNONYMS: dict[str, list[str]] = {
    "nose tip":        ["nose", "snout", "nose_tip"],
    "right ear":       ["right_ear", "rightear", "ear_right", "r_ear"],
    "left ear":        ["left_ear",  "leftear",  "ear_left",  "l_ear"],
    "neck":            ["neck", "neckbase", "head_neck"],
    "right side body": ["right_side", "body_right", "r_body", "mid_right", "right_midside"],
    "left side body":  ["left_side",  "body_left",  "l_body", "mid_left", "left_midside"],
    "tail base":       ["tail_base", "tailbase", "tail"],
}


def _normalise(s: str) -> str:
    return s.strip().lower().replace(" ", "_").replace("-", "_")


def _build_default_keypoint_map(bodypart_names: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    norm_parts = {_normalise(n): n for n in bodypart_names}
    for canonical, synonyms in _SYNONYMS.items():
        for syn in synonyms:
            if _normalise(syn) in norm_parts:
                mapping[norm_parts[_normalise(syn)]] = canonical
                break
    return mapping


def _read_dlc_csv(path: Path) -> pd.DataFrame:
    """Read a DLC CSV into a flat DataFrame, handling multi-level headers."""
    raw = pd.read_csv(path, header=None)

    # Find how many header rows there are (scorer, [individuals,] bodyparts, coords)
    n_header = 3
    if str(raw.iloc[0, 0]).lower() == "scorer":
        row1_unique = raw.iloc[1].dropna().unique().tolist()
        # If the second row has values that look like identities (individuals),
        # there are 4 header rows
        if "bodyparts" not in [str(v).lower() for v in row1_unique]:
            n_header = 4

    header_rows = raw.iloc[:n_header].values
    data_rows = raw.iloc[n_header:].reset_index(drop=True)
    data_rows.columns = range(len(data_rows.columns))  # type: ignore[assignment]

    return header_rows, data_rows, n_header


def _extract_animal_coords(
    header_rows: np.ndarray,
    data_rows: pd.DataFrame,
    animal_id: str | None,
    n_header: int,
) -> dict[str, pd.DataFrame]:
    """
    Returns dict: {bodypart_name: DataFrame with columns ['x', 'y', 'likelihood']}
    """
    # Col 0 is the frame index column provided by DLC
    has_individual_row = n_header == 4

    # Build column guide from header rows
    # header_rows shape: (n_header, n_cols)
    bodypart_row = header_rows[2 if has_individual_row else 1]
    coord_row    = header_rows[3 if has_individual_row else 2]

    if has_individual_row:
        individual_row = header_rows[1]
    else:
        individual_row = np.array([animal_id] * len(bodypart_row))

    result: dict[str, dict[str, list]] = {}
    for col_idx in range(1, len(bodypart_row)):
        ind   = str(individual_row[col_idx]) if individual_row[col_idx] is not None else ""
        part  = str(bodypart_row[col_idx])
        coord = str(coord_row[col_idx]).lower()

        if animal_id is not None and _normalise(ind) != _normalise(animal_id):
            continue
        if part not in result:
            result[part] = {"x": [], "y": [], "likelihood": []}
        if coord in ("x", "y", "likelihood"):
            # We'll store the column index instead of values here, then vectorise
            result[part][coord] = col_idx

    # Now extract as Series
    extracted: dict[str, pd.DataFrame] = {}
    for part, coord_cols in result.items():
        missing = [k for k in ("x", "y", "likelihood") if not isinstance(coord_cols.get(k), int)]
        if missing:
            continue
        part_df = pd.DataFrame({
            "x":           pd.to_numeric(data_rows[coord_cols["x"]], errors="coerce"),
            "y":           pd.to_numeric(data_rows[coord_cols["y"]], errors="coerce"),
            "likelihood":  pd.to_numeric(data_rows[coord_cols["likelihood"]], errors="coerce"),
        })
        extracted[part] = part_df

    return extracted


def _build_records(
    black_coords: dict[str, pd.DataFrame],
    white_coords: dict[str, pd.DataFrame],
    keypoint_map: dict[str, str],
    video_path: str,
    likelihood_threshold: float,
) -> list[dict]:
    """Convert per-bodypart DataFrames into canonical pose records."""
    # Infer image dimensions from max coordinate
    all_x = [df["x"].max() for df in {**black_coords, **white_coords}.values()]
    all_y = [df["y"].max() for df in {**black_coords, **white_coords}.values()]
    approx_w = int(max(all_x)) + 10
    approx_h = int(max(all_y)) + 10

    n_frames = len(next(iter(black_coords.values())))

    records: list[dict] = []
    for frame_idx in range(n_frames):
        bx, by, wx, wy = [], [], [], []
        b_has_any = False
        w_has_any = False

        for canon in CANONICAL_LABELS:
            # Find DLC bodypart that maps to this canonical slot
            dlc_part = next((k for k, v in keypoint_map.items() if v == canon), None)

            def _get(coords, part, axis):
                if part is None or part not in coords:
                    return 0.0, False
                row = coords[part].iloc[frame_idx]
                if row["likelihood"] < likelihood_threshold:
                    return 0.0, False
                return float(row[axis]), True

            bxi, bv = _get(black_coords, dlc_part, "x")
            byi, _  = _get(black_coords, dlc_part, "y")
            wxi, wv = _get(white_coords, dlc_part, "x")
            wyi, _  = _get(white_coords, dlc_part, "y")

            bx.append(bxi); by.append(byi)
            wx.append(wxi); wy.append(wyi)
            if bv:
                b_has_any = True
            if wv:
                w_has_any = True

        # Keep frame if at least one animal has any confident keypoint.
        # This avoids dropping useful frames when one mouse is partially occluded.
        if not (b_has_any or w_has_any):
            continue

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

    return records


def convert_dlc_multianimal(
    csv_path: Path,
    black_id: str,
    white_id: str,
    video_path: str,
    keypoint_map: dict[str, str] | None,
    likelihood_threshold: float,
) -> list[dict]:
    header_rows, data_rows, n_header = _read_dlc_csv(csv_path)
    black_coords = _extract_animal_coords(header_rows, data_rows, black_id, n_header)
    white_coords = _extract_animal_coords(header_rows, data_rows, white_id, n_header)

    all_parts = list(set(list(black_coords.keys()) + list(white_coords.keys())))
    if keypoint_map is None:
        keypoint_map = _build_default_keypoint_map(all_parts)

    covered = set(keypoint_map.values())
    missing = [c for c in CANONICAL_LABELS if c not in covered]
    if missing:
        raise SystemExit(
            f"Keypoint map missing canonical slots: {missing}\n"
            f"DLC bodyparts found: {all_parts}\n"
            f"Provide --keypoint-map to specify the mapping."
        )

    return _build_records(black_coords, white_coords, keypoint_map, video_path, likelihood_threshold)


def convert_dlc_single_animal_pair(
    black_csv: Path,
    white_csv: Path,
    video_path: str,
    keypoint_map: dict[str, str] | None,
    likelihood_threshold: float,
) -> list[dict]:
    """Two separate single-animal DLC CSVs, one per mouse."""
    bh, bd, bn = _read_dlc_csv(black_csv)
    wh, wd, wn = _read_dlc_csv(white_csv)
    black_coords = _extract_animal_coords(bh, bd, None, bn)
    white_coords = _extract_animal_coords(wh, wd, None, wn)

    all_parts = list(set(list(black_coords.keys()) + list(white_coords.keys())))
    if keypoint_map is None:
        keypoint_map = _build_default_keypoint_map(all_parts)

    covered = set(keypoint_map.values())
    missing = [c for c in CANONICAL_LABELS if c not in covered]
    if missing:
        raise SystemExit(
            f"Keypoint map missing canonical slots: {missing}\n"
            f"DLC bodyparts found: {all_parts}"
        )

    return _build_records(black_coords, white_coords, keypoint_map, video_path, likelihood_threshold)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert DeepLabCut CSV output to canonical pose JSON"
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--input", type=Path, help="Multi-animal DLC CSV")
    mode_group.add_argument("--input-black", type=Path, help="Single-animal DLC CSV (black mouse)")

    parser.add_argument("--input-white", type=Path, help="Single-animal DLC CSV (white mouse)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--black-id", default="mouse1", help="Individual ID for black mouse (multi-animal DLC)")
    parser.add_argument("--white-id", default="mouse2", help="Individual ID for white mouse (multi-animal DLC)")
    parser.add_argument("--video", default="", help="Source video path (stored in JSON metadata)")
    parser.add_argument("--keypoint-map", type=Path, default=None)
    parser.add_argument("--likelihood-threshold", type=float, default=0.6,
                        help="Minimum DLC likelihood to accept a keypoint (default 0.6)")
    args = parser.parse_args()

    keypoint_map: dict[str, str] | None = None
    if args.keypoint_map is not None:
        keypoint_map = json.loads(args.keypoint_map.read_text(encoding="utf-8"))

    if args.input is not None:
        records = convert_dlc_multianimal(
            csv_path=args.input,
            black_id=args.black_id,
            white_id=args.white_id,
            video_path=args.video,
            keypoint_map=keypoint_map,
            likelihood_threshold=args.likelihood_threshold,
        )
    else:
        if args.input_white is None:
            raise SystemExit("--input-white is required when using --input-black")
        records = convert_dlc_single_animal_pair(
            black_csv=args.input_black,
            white_csv=args.input_white,
            video_path=args.video,
            keypoint_map=keypoint_map,
            likelihood_threshold=args.likelihood_threshold,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} canonical pose records -> {args.output}")


if __name__ == "__main__":
    main()
