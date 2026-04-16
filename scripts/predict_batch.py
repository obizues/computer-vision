from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

# Import canonical feature list from train_eval so predict always matches train
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from train_eval import ALL_FEATURE_COLS


def load_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch inference on feature table")
    parser.add_argument("--config", type=Path, default=Path("configs/mvp_config.json"))
    parser.add_argument("--input", type=Path, default=None, help="Optional custom feature CSV")
    parser.add_argument("--output", type=Path, default=None, help="Optional prediction CSV output path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["outputs"]["artifacts_dir"])

    model_path = output_dir / "baseline_model.joblib"
    if not model_path.exists():
        raise SystemExit(f"Missing model artifact: {model_path}. Run scripts/train_eval.py first.")

    input_path = args.input if args.input is not None else (output_dir / "features_top_view.csv")
    if not input_path.exists():
        raise SystemExit(f"Missing feature input: {input_path}. Run scripts/build_features.py first.")

    df = pd.read_csv(input_path)
    model = joblib.load(model_path)

    # Prefer feature names baked into the model (sklearn ≥1.0 stores them in the scaler)
    model_features: list[str] | None = None
    scaler = model.named_steps.get("scale")
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        model_features = list(scaler.feature_names_in_)

    feature_cols = model_features if model_features else [c for c in ALL_FEATURE_COLS if c in df.columns]
    if not feature_cols:
        raise SystemExit("No recognised feature columns found in input CSV.")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Input is missing feature columns: {missing}\n"
            "Re-run scripts/build_features.py to regenerate the feature file."
        )

    proba = model.predict_proba(df[feature_cols])[:, 1]
    pred = (proba >= 0.5).astype(int)

    out = df[["frame_idx", "filename"]].copy() if {"frame_idx", "filename"}.issubset(df.columns) else pd.DataFrame(index=df.index)
    out["y_pred"] = pred
    out["y_proba_close"] = proba

    output_path = args.output if args.output is not None else (output_dir / "batch_predictions.csv")
    out.to_csv(output_path, index=False)
    print(f"Wrote {len(out)} scored rows -> {output_path}")


if __name__ == "__main__":
    main()
