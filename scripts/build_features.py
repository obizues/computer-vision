from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def load_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text(encoding="utf-8"))


def flatten_top_view(records: list[dict], keypoints_per_mouse: int) -> pd.DataFrame:
    rows: list[dict] = []

    for frame_index, rec in enumerate(records):
        filename = rec["filename"]
        width = rec["width"]
        height = rec["height"]
        black = rec["coords"]["black"]
        white = rec["coords"]["white"]

        b_nose_x, b_nose_y = black["x"][0], black["y"][0]
        w_nose_x, w_nose_y = white["x"][0], white["y"][0]

        b_neck_x, b_neck_y = black["x"][3], black["y"][3]
        w_neck_x, w_neck_y = white["x"][3], white["y"][3]

        b_tail_x, b_tail_y = black["x"][6], black["y"][6]
        w_tail_x, w_tail_y = white["x"][6], white["y"][6]

        # Prefer source_frame_idx (actual video frame number) over enumerate counter
        frame_idx_val = rec.get("source_frame_idx", frame_index)

        rows.append(
            {
                "frame_idx": frame_idx_val,
                "filename": filename,
                "img_w": width,
                "img_h": height,
                "b_nose_x": b_nose_x,
                "b_nose_y": b_nose_y,
                "w_nose_x": w_nose_x,
                "w_nose_y": w_nose_y,
                "b_neck_x": b_neck_x,
                "b_neck_y": b_neck_y,
                "w_neck_x": w_neck_x,
                "w_neck_y": w_neck_y,
                "b_tail_x": b_tail_x,
                "b_tail_y": b_tail_y,
                "w_tail_x": w_tail_x,
                "w_tail_y": w_tail_y,
                "keypoints_per_mouse": keypoints_per_mouse,
            }
        )

    return pd.DataFrame(rows)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np

    out = df.copy()

    # ── static spatial features ───────────────────────────────────────────────
    out["nose_dist"] = np.hypot(out["b_nose_x"] - out["w_nose_x"], out["b_nose_y"] - out["w_nose_y"])
    out["neck_dist"] = np.hypot(out["b_neck_x"] - out["w_neck_x"], out["b_neck_y"] - out["w_neck_y"])
    out["tail_dist"] = np.hypot(out["b_tail_x"] - out["w_tail_x"], out["b_tail_y"] - out["w_tail_y"])

    out["b_body_len"] = np.hypot(out["b_nose_x"] - out["b_tail_x"], out["b_nose_y"] - out["b_tail_y"])
    out["w_body_len"] = np.hypot(out["w_nose_x"] - out["w_tail_x"], out["w_nose_y"] - out["w_tail_y"])

    # Minimum of all pairwise nose/neck/tail distances — catches close contact
    # regardless of which body part is nearest
    out["min_pairwise_dist"] = out[["nose_dist", "neck_dist", "tail_dist"]].min(axis=1)

    # Body-axis orientation angle for each mouse (direction nose→tail, in radians)
    out["b_orientation"] = np.arctan2(
        out["b_nose_y"] - out["b_tail_y"],
        out["b_nose_x"] - out["b_tail_x"],
    )
    out["w_orientation"] = np.arctan2(
        out["w_nose_y"] - out["w_tail_y"],
        out["w_nose_x"] - out["w_tail_x"],
    )

    # Angle of the vector FROM black centroid TO white centroid
    cx_b = (out["b_nose_x"] + out["b_tail_x"]) / 2
    cy_b = (out["b_nose_y"] + out["b_tail_y"]) / 2
    cx_w = (out["w_nose_x"] + out["w_tail_x"]) / 2
    cy_w = (out["w_nose_y"] + out["w_tail_y"]) / 2
    inter_angle = np.arctan2(cy_w - cy_b, cx_w - cx_b)

    # cosine of angle between each mouse's body-axis and the inter-mouse vector
    # +1 means that mouse is facing directly toward the other; −1 means facing away
    out["b_facing_toward_w"] = np.cos(out["b_orientation"] - inter_angle)
    out["w_facing_toward_b"] = np.cos(out["w_orientation"] - (inter_angle + np.pi))

    # ── temporal features (sort by frame_idx for correct temporal ordering) ───
    out = out.sort_values("frame_idx").reset_index(drop=True)

    out["b_nose_speed"] = np.hypot(out["b_nose_x"].diff().fillna(0), out["b_nose_y"].diff().fillna(0))
    out["w_nose_speed"] = np.hypot(out["w_nose_x"].diff().fillna(0), out["w_nose_y"].diff().fillna(0))

    # Signed: negative → mice getting closer (approaching), positive → diverging
    out["nose_approach_velocity"] = out["nose_dist"].diff().fillna(0)
    # Keep the unsigned version for backward compat
    out["relative_dist_change"] = out["nose_approach_velocity"].abs()

    # Rolling statistics over a 5-frame window
    out["nose_dist_roll5_mean"] = out["nose_dist"].rolling(5, min_periods=1).mean()
    out["nose_dist_roll5_min"] = out["nose_dist"].rolling(5, min_periods=1).min()
    out["nose_dist_roll5_std"] = out["nose_dist"].rolling(5, min_periods=1).std().fillna(0)

    # Was an approach happening over the last 5 frames?
    out["approach_in_window"] = (out["nose_dist_roll5_mean"] < out["nose_dist_roll5_mean"].shift(5).fillna(out["nose_dist_roll5_mean"])).astype(int)

    return out


def add_proxy_label(df: pd.DataFrame, quantile: float) -> tuple[pd.DataFrame, float, str]:
    out = df.copy()
    valid_nose_dist = out.loc[out["nose_dist"].notna() & (out["nose_dist"] > 0), "nose_dist"]

    strategy_used = "quantile_nose_distance"
    if valid_nose_dist.empty:
        out["is_close_interaction"] = (out["frame_idx"] % 2).astype(int)
        return out, 0.0, "fallback_frame_parity_no_valid_distances"

    threshold = float(valid_nose_dist.quantile(quantile))
    out["is_close_interaction"] = ((out["nose_dist"] > 0) & (out["nose_dist"] <= threshold)).astype(int)

    if int(out["is_close_interaction"].nunique()) < 2:
        out["is_close_interaction"] = (out["frame_idx"] % 2).astype(int)
        strategy_used = "fallback_frame_parity"

    return out, threshold, strategy_used


def _normalize_binary_label(value: Any, positive_labels: set[str]) -> int | None:
    if pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        numeric = int(value)
        if numeric in (0, 1):
            return numeric
        return None

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "positive"}:
        return 1
    if text in {"0", "false", "no", "n", "negative"}:
        return 0
    if text in positive_labels:
        return 1
    return 0


def apply_ground_truth_labels(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, str, dict]:
    label_cfg = cfg.get("label_stage", {})
    label_file = label_cfg.get("ground_truth_labels_file")
    if not label_file:
        return df, "proxy_only", {"ground_truth_rows": 0, "ground_truth_matches": 0}

    label_path = Path(label_file)
    if not label_path.exists():
        return df, "proxy_only_missing_label_file", {"ground_truth_rows": 0, "ground_truth_matches": 0}

    merge_key = str(label_cfg.get("merge_key", "frame_idx"))
    label_col = str(label_cfg.get("label_column", "is_close_interaction"))
    positive_labels = {str(item).strip().lower() for item in label_cfg.get("positive_labels", [])}

    labels_df = pd.read_csv(label_path)
    if merge_key not in labels_df.columns:
        return df, "proxy_only_invalid_merge_key", {"ground_truth_rows": int(len(labels_df)), "ground_truth_matches": 0}
    if label_col not in labels_df.columns:
        return df, "proxy_only_missing_label_column", {"ground_truth_rows": int(len(labels_df)), "ground_truth_matches": 0}

    labels_df = labels_df[[merge_key, label_col]].copy()
    labels_df["__gt_label"] = labels_df[label_col].apply(lambda value: _normalize_binary_label(value, positive_labels))
    labels_df = labels_df.dropna(subset=["__gt_label"])
    if labels_df.empty:
        return df, "proxy_only_empty_ground_truth", {"ground_truth_rows": int(len(labels_df)), "ground_truth_matches": 0}

    if merge_key == "frame_idx":
        labels_df[merge_key] = labels_df[merge_key].astype(int)

    merged = df.merge(labels_df[[merge_key, "__gt_label"]], on=merge_key, how="left")
    gt_matches = int(merged["__gt_label"].notna().sum())
    merged["is_close_interaction"] = merged["__gt_label"].fillna(merged["is_close_interaction"]).astype(int)
    merged = merged.drop(columns=["__gt_label"])

    strategy = "ground_truth_override" if gt_matches > 0 else "proxy_only_no_ground_truth_matches"
    stats = {
        "ground_truth_rows": int(len(labels_df)),
        "ground_truth_matches": gt_matches,
    }
    return merged, strategy, stats


def quality_report(df: pd.DataFrame) -> dict:
    x_cols = [c for c in df.columns if c.endswith("_x")]
    y_cols = [c for c in df.columns if c.endswith("_y")]

    bad_x = 0
    bad_y = 0
    for col in x_cols:
        bad_x += int(((df[col] < 0) | (df[col] > df["img_w"])).sum())
    for col in y_cols:
        bad_y += int(((df[col] < 0) | (df[col] > df["img_h"])).sum())

    return {
        "rows": int(len(df)),
        "null_cells": int(df.isna().sum().sum()),
        "out_of_bounds_x": int(bad_x),
        "out_of_bounds_y": int(bad_y),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build model-ready features from MARS pose JSON")
    parser.add_argument("--config", type=Path, default=Path("configs/mvp_config.json"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(".")

    pose_stage_cfg = cfg.get("pose_stage", {})
    pose_output = pose_stage_cfg.get("output_top_keypoints_file")
    if pose_output:
        top_path = data_dir / pose_output
    else:
        top_path = data_dir / cfg["dataset"]["top_keypoints_file"]
    output_dir = data_dir / cfg["outputs"]["artifacts_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    records = json.loads(top_path.read_text(encoding="utf-8"))
    keypoints_per_mouse = int(cfg["feature_build"]["keypoints_per_mouse"])
    quantile = float(cfg["feature_build"]["close_interaction_quantile"])

    base_df = flatten_top_view(records, keypoints_per_mouse=keypoints_per_mouse)
    feat_df = add_derived_features(base_df)
    feat_df, threshold, strategy_used = add_proxy_label(feat_df, quantile=quantile)
    feat_df, gt_strategy, gt_stats = apply_ground_truth_labels(feat_df, cfg)

    features_path = output_dir / "features_top_view.csv"
    feat_df.to_csv(features_path, index=False)

    report = quality_report(feat_df)
    report["close_threshold_px"] = threshold
    report["label_strategy_used"] = strategy_used
    report["ground_truth_strategy_used"] = gt_strategy
    report["ground_truth_rows"] = gt_stats.get("ground_truth_rows", 0)
    report["ground_truth_matches"] = gt_stats.get("ground_truth_matches", 0)
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()

    report_path = output_dir / "feature_quality_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote features: {features_path}")
    print(f"Wrote quality report: {report_path}")


if __name__ == "__main__":
    main()
