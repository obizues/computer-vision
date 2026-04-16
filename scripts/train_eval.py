from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def load_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text(encoding="utf-8"))


# All features produced by build_features.py — listed in priority order so
# missing columns (e.g. from old feature files) degrade gracefully.
ALL_FEATURE_COLS = [
    # Core distances
    "nose_dist",
    "neck_dist",
    "tail_dist",
    "min_pairwise_dist",
    # Body shape
    "b_body_len",
    "w_body_len",
    # Orientation / facing
    "b_facing_toward_w",
    "w_facing_toward_b",
    # Speed / dynamics
    "b_nose_speed",
    "w_nose_speed",
    "nose_approach_velocity",
    "relative_dist_change",
    # Rolling / temporal context
    "nose_dist_roll5_mean",
    "nose_dist_roll5_min",
    "nose_dist_roll5_std",
    "approach_in_window",
]


def temporal_train_test_split(df: pd.DataFrame, test_size: float):
    """Split by time: last `test_size` fraction of frames is the test set.

    This avoids the data-leakage that random splitting causes on video data
    (consecutive frames share nearly identical pose values and labels).
    """
    df = df.sort_values("frame_idx").reset_index(drop=True)
    cutoff = int(len(df) * (1 - test_size))
    return df.iloc[:cutoff], df.iloc[cutoff:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate baseline behavior model")
    parser.add_argument("--config", type=Path, default=Path("configs/mvp_config.json"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["outputs"]["artifacts_dir"])

    feature_path = output_dir / "features_top_view.csv"
    if not feature_path.exists():
        raise SystemExit(f"Missing feature file: {feature_path}. Run scripts/build_features.py first.")

    df = pd.read_csv(feature_path)

    # Use whichever features are actually present in this feature file
    feature_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
    if not feature_cols:
        raise SystemExit("No recognized feature columns found in feature file.")

    target_col = "is_close_interaction"
    if target_col not in df.columns:
        raise SystemExit(f"Target column '{target_col}' not found in feature file.")

    # ── train / test split ────────────────────────────────────────────────────
    split_cfg = cfg["split"]
    split_method = str(split_cfg.get("method", "temporal"))
    test_size = float(split_cfg["test_size"])

    if split_method == "temporal" and "frame_idx" in df.columns:
        train_df, test_df = temporal_train_test_split(df, test_size)
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
    else:
        # Fallback to random split (use when ground-truth labels are sparse / not temporally ordered)
        X = df[feature_cols]
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=int(split_cfg["random_seed"]),
            stratify=y if bool(split_cfg.get("stratify", True)) else None,
        )

    if int(y_train.nunique()) < 2:
        raise SystemExit("Training set contains only one class — check labeling or lower the quantile threshold.")

    # ── model ─────────────────────────────────────────────────────────────────
    model_cfg = cfg["model"]
    model_name = str(model_cfg.get("name", "gradient_boosting"))

    if model_name == "gradient_boosting":
        estimator = GradientBoostingClassifier(
            n_estimators=int(model_cfg.get("n_estimators", 200)),
            max_depth=int(model_cfg.get("max_depth", 4)),
            learning_rate=float(model_cfg.get("learning_rate", 0.05)),
            subsample=float(model_cfg.get("subsample", 0.8)),
            random_state=int(split_cfg["random_seed"]),
        )
        # GBM doesn't require standard scaling but it doesn't hurt
        pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", estimator),
        ])
    else:
        estimator = LogisticRegression(
            max_iter=int(model_cfg.get("max_iter", 1000)),
            random_state=int(split_cfg["random_seed"]),
        )
        pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=True, with_std=bool(model_cfg.get("standardize", True)))),
            ("model", estimator),
        ])

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]

    auc = float(roc_auc_score(y_test, proba))
    cls_report = classification_report(y_test, pred, output_dict=True)
    cm = confusion_matrix(y_test, pred)

    # ── feature importance ────────────────────────────────────────────────────
    inner = pipeline.named_steps["model"]
    if hasattr(inner, "feature_importances_"):
        importances = inner.feature_importances_.tolist()
    elif hasattr(inner, "coef_"):
        importances = np.abs(inner.coef_[0]).tolist()
    else:
        importances = []
    feature_importance = dict(zip(feature_cols, importances)) if importances else {}

    # ── save outputs ──────────────────────────────────────────────────────────
    scored = X_test.copy()
    scored["y_true"] = y_test.values
    scored["y_pred"] = pred
    scored["y_proba_close"] = proba
    if "frame_idx" in df.columns:
        if split_method == "temporal":
            scored["frame_idx"] = test_df["frame_idx"].values
        scored = scored.reset_index(drop=True)
    if "filename" in df.columns:
        if split_method == "temporal":
            scored["filename"] = test_df["filename"].values

    scored_path = output_dir / "baseline_scored_sample.csv"
    scored.head(1000).to_csv(scored_path, index=False)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": model_name,
        "split_method": split_method,
        "features": feature_cols,
        "target": target_col,
        "auc": round(auc, 6),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "confusion_matrix": cm.tolist(),
        "classification_report": cls_report,
        "feature_importance": feature_importance,
    }

    summary_path = output_dir / "baseline_model_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    model_path = output_dir / "baseline_model.joblib"
    joblib.dump(pipeline, model_path)

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {scored_path}")
    print(f"Wrote: {model_path}")
    print(f"Model: {model_name}  |  Split: {split_method}  |  AUC: {auc:.5f}")
    if feature_importance:
        ranked = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        print("Top features:", ", ".join(f"{k}={v:.3f}" for k, v in ranked[:5]))


if __name__ == "__main__":
    main()
