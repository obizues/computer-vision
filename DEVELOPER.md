# Developer Guide: Mouse Vision

This guide covers setup, workflow, testing, and extending the Mouse Vision system.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Running the Pipeline](#running-the-pipeline)
3. [Running the Dashboard](#running-the-dashboard)
4. [Project Structure](#project-structure)
5. [Making Changes](#making-changes)
6. [Testing & Validation](#testing--validation)
7. [Debugging](#debugging)
8. [Configuration](#configuration)
9. [Common Tasks](#common-tasks)

---

## Environment Setup

### Prerequisites

- **Python 3.10+** (tested on 3.14.3)
- **Windows, macOS, or Linux**
- **Git** (for version control)
- **~2 GB disk space** minimum (more for raw videos and intermediate artifacts)

### 1. Clone the repository

```bash
git clone <repo-url>
cd "Machine Vision"
```

### 2. Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux (Bash):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify installation

```bash
python -c "import pandas, sklearn, cv2, streamlit; print('✓ All dependencies installed')"
```

---

## Running the Pipeline

### Quick start (recommended for most users)

**Windows (PowerShell):**
```powershell
.\launch_mvp.bat
```

**Command line:**
```bash
python scripts/run_pipeline.py --config configs/mvp_config.json
```

### Command-line options

```bash
# Run with default config (MVP setup)
python scripts/run_pipeline.py --config configs/mvp_config.json

# Run with long-video config (larger dataset)
python scripts/run_pipeline.py --config configs/mvp_config.long_video.json

# Run specific stages (intermediate; not recommended for reproducibility)
python scripts/build_features.py --config configs/mvp_config.json
python scripts/train_eval.py --config configs/mvp_config.json
python scripts/predict_batch.py --config configs/mvp_config.json
```

For `continuous_video_external_inference` mode, set these in config:

- `pose_inference_runtime.video_file`: local target path
- `pose_inference_runtime.video_url`: hosted downloadable URL
- `pose_inference_runtime.video_sha256`: optional integrity check

If `video_file` is missing, runtime automatically downloads from `video_url` before executing the inference command.

### Pipeline stages (order matters)

1. **Pose Ingestion** (`video_to_pose.py`) — Extracts or loads pose keypoints from video/JSON
2. **Feature Engineering** (`build_features.py`) — Computes interaction and movement features
3. **Model Training** (`train_eval.py`) — Trains gradient boosting classifier on features
4. **Batch Inference** (`predict_batch.py`) — Scores all frames with trained model
5. **Registry Update** — Logs run metadata to `data/run_registry.jsonl`

All outputs go to `data/eda_outputs/` by default.

### Monitoring progress

The pipeline logs to console. Watch for:

- `✓ Pose ingestion complete` — Keypoints loaded and normalized
- `Training on X frames, testing on Y frames` — Train/test split confirmed
- `AUC: X.XXX` — Model quality metric
- `Wrote N scored rows` — Inference finished
- `Pipeline completed successfully` — All stages passed

---

## Running the Dashboard

### Start the Streamlit app

**Windows (PowerShell):**
```powershell
& ".\.venv\Scripts\streamlit.exe" run app/scientist_dashboard.py --server.port 8501
```

**macOS/Linux (Bash):**
```bash
streamlit run app/scientist_dashboard.py --server.port 8501
```

### Access the dashboard

Open a browser to: **`http://localhost:8501`**

### Dashboard tabs

1. **Event Replay** — Select a video, run pipeline, review detected interaction segments with pose overlay
2. **Analytics & Details** — Model metrics, data quality, full-run probability timeline, frame-level review
3. **Artifacts** — Links to generated model files, predictions, and quality reports

### Demo mode (for hosted/shared environments)

```bash
# Set environment variable before running
$env:MOUSE_VISION_DEMO_MODE = "1"

# Then start dashboard (will disable in-app pipeline execution)
streamlit run app/scientist_dashboard.py --server.port 8501
```

When demo mode is enabled:
- **"Run Full Pipeline" button is hidden**
- Dashboard acts as a **viewer-only** interface for precomputed artifacts
- Safe for Streamlit Community Cloud or shared hosting

---

## Project Structure

```
Machine Vision/
├── app/
│   └── scientist_dashboard.py    # Streamlit dashboard for segment review & model diagnostics
├── configs/
│   ├── mvp_config.json           # Main configuration (data paths, model params, thresholds)
│   ├── mvp_config.long_video.json # Alternative config for larger datasets
│   └── data_contract.json        # Schema validation for pose records
├── data/
│   ├── raw_videos/               # Input videos (user-provided or downloaded)
│   ├── raw_images_top/           # MARS dataset: raw frame images
│   ├── processed/                # Intermediate outputs (pose JSON, normalized records)
│   ├── eda_outputs/              # Final artifacts (model, predictions, reports)
│   ├── downloads/                # Downloaded datasets (MARS, etc.)
│   ├── MARS_keypoints_top.json   # MARS pose annotations (top view)
│   ├── MARS_keypoints_front.json # MARS pose annotations (front view)
│   └── run_registry.jsonl        # Audit log of all pipeline runs
├── docs/
│   ├── datasets.md               # Guide to available datasets & sources
│   ├── system-architecture.md    # Technical architecture & design decisions
│   └── video-mlops-playbook.md   # MLOps patterns & deployment strategies
├── scripts/
│   ├── run_pipeline.py           # Orchestrator: runs all stages in order
│   ├── video_to_pose.py          # Stage 1: Pose ingestion & normalization
│   ├── build_features.py         # Stage 2: Feature engineering
│   ├── train_eval.py             # Stage 3: Model training & evaluation
│   ├── predict_batch.py          # Stage 4: Batch inference
│   ├── render_pose_overlay.py    # Optional: Generate video with pose overlay
│   ├── external_pose_inference_dlc.py  # DeepLabCut integration
│   ├── download_data.py          # Download MARS dataset
│   └── ... (utility scripts)
├── notes/
│   └── poc-stories.md            # Implementation narratives and walkthrough examples
├── launch_mvp.bat                # Windows menu launcher
├── requirements.txt              # Python dependencies
├── README.md                     # Main project overview
├── DEVELOPER.md                  # This file
└── .gitignore                    # Version control exclusions
```

---

## Making Changes

### Code style

- **Format:** Follow PEP 8 — use 4 spaces for indentation
- **Type hints:** Encouraged for function signatures (Python 3.10+)
- **Documentation:** Add docstrings to new functions
- **Testing:** Add a simple test or manual verification step

Example function:

```python
def my_feature_fn(pose_record: dict, window_size: int = 5) -> float:
    """Compute a feature from pose keypoints.
    
    Args:
        pose_record: Canonical pose record with 'coords' key
        window_size: Temporal window for rolling aggregation
        
    Returns:
        Feature value (float), or NaN if invalid
    """
    # Implementation...
    return value
```

### Adding a new feature

1. **Edit `scripts/build_features.py`**
   - Add computation in `add_derived_features()`
   - Update `ALL_FEATURE_COLS` if needed
   - Document the feature name and meaning

2. **Test on sample data**
   ```bash
   python scripts/build_features.py --config configs/mvp_config.json
   ```
   - Check `data/eda_outputs/features_top_view.csv`
   - Verify no NaN explosion or infinite values

3. **Re-train the model**
   ```bash
   python scripts/train_eval.py --config configs/mvp_config.json
   ```
   - Compare AUC and feature importance
   - Edit `ALL_FEATURE_COLS` if the new feature should be used

### Adding a new data source

1. **Create an adapter in `scripts/pose_adapters.py`**
   - Implement `load_<your_source>()` function
   - Return list of canonical pose records (see `data_contract.json`)

2. **Update config option selection in `scripts/video_to_pose.py`**
   - Add a new mode string (e.g., `"your_source_name"`)
   - Call your adapter in the mode dispatcher

3. **Test end-to-end**
   ```bash
   # Update config to point to your data
   python scripts/run_pipeline.py --config configs/mvp_config.json
   ```

### Modifying the model

Current model: **GradientBoostingClassifier** (scikit-learn)

To change:

1. **Edit `scripts/train_eval.py`**
   - Modify `model_builder()` function
   - Keep the same input (`X`) and output (`y`) shapes
   - Update hyperparameter comments

2. **Re-evaluate**
   ```bash
   python scripts/train_eval.py --config configs/mvp_config.json
   ```

3. **Update `.gitignore`** if new artifact types are generated

---

## Testing & Validation

### Manual smoke test (5 minutes)

```bash
# Ensure config is valid
python -c "import json; json.load(open('configs/mvp_config.json'))"

# Run a quick feature build
python scripts/build_features.py --config configs/mvp_config.json

# Spot-check the output
python -c "import pandas as pd; df = pd.read_csv('data/eda_outputs/features_top_view.csv'); print(f'Rows: {len(df)}'); print(df.dtypes)"
```

### Data validation checklist

After running the pipeline, verify:

- ✅ `data/processed/pose_top_keypoints.json` — Valid JSON, all frames present
- ✅ `data/eda_outputs/features_top_view.csv` — No corrupted rows, feature columns match model input
- ✅ `data/eda_outputs/baseline_model_summary.json` — AUC > 0.5 (at minimum)
- ✅ `data/eda_outputs/batch_predictions.csv` — Correct frame count, probability in [0, 1]
- ✅ `data/eda_outputs/feature_quality_report.json` — Reports any schema violations

### Example validation script

```python
import json
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
OUT = BASE / "data" / "eda_outputs"

# Check pose normalization
pose = json.loads((BASE / "data" / "processed" / "pose_top_keypoints.json").read_text())
assert len(pose) > 0, "No pose records found"
print(f"✓ {len(pose)} pose records")

# Check features
features = pd.read_csv(OUT / "features_top_view.csv")
assert features.shape[0] == len(pose), "Feature count mismatch"
print(f"✓ {features.shape[1]} features computed")

# Check predictions
preds = pd.read_csv(OUT / "batch_predictions.csv")
assert (preds["y_proba_close"] >= 0).all() and (preds["y_proba_close"] <= 1).all()
print(f"✓ {len(preds)} predictions, probabilities in [0, 1]")

# Check model
summary = json.loads((OUT / "baseline_model_summary.json").read_text())
print(f"✓ Model AUC: {summary['auc']:.3f}")

print("\n✅ All validation checks passed!")
```

---

## Debugging

### Pipeline fails at pose ingestion

**Symptom:** `FileNotFoundError: pose source not found`

**Solutions:**
1. Check config `pose_stage.mode` and `pose_stage.input_file` exist
2. For video mode: verify `pose_inference_runtime.video_file` points to a real file
3. For JSON mode: validate JSON is valid (`python -m json.tool <file.json>`)

### Model training produces NaN

**Symptom:** `Warning: invalid value encountered in scalar_power`

**Solutions:**
1. Check features for infinite values: `df[df.isin([np.inf, -np.inf])].any()`
2. Drop rows with all-NaN features: `df.dropna(how='all')`
3. Check feature engineering logic for division-by-zero or log(0)

### Dashboard shows "No segments detected"

**Symptom:** Event Replay tab displays info message, no segments appear

**Solutions:**
1. Verify `batch_predictions.csv` has high-probability frames:
   ```python
   import pandas as pd
   df = pd.read_csv('data/eda_outputs/batch_predictions.csv')
   print(df['y_proba_close'].describe())  # Should have values > 0.85
   ```
2. Check distance gate filter in `app/scientist_dashboard.py` — may be too strict
3. Lower confidence threshold in dashboard UI (slider at top of Event Replay)

### Memory or performance issues

**Symptom:** Dashboard is slow, or Python runs out of memory

**Solutions:**
1. Reduce video resolution in config: `pose_inference_runtime.target_size`
2. Reduce batch size in `scripts/predict_batch.py` if using large feature tables
3. Use a smaller dataset (e.g., sample 500 frames instead of 2400)

---

## Configuration

All runtime behavior is controlled by `configs/mvp_config.json`.

### Key sections

```json
{
  "pose_stage": {
    "mode": "dataset_json_passthrough",    // Data source mode
    "input_file": "data/MARS_keypoints_top.json",
    "external_pose_file": "data/processed/external_pose_predictions.json"
  },
  "feature_target_builder": {
    "close_interaction_quantile": 0.25    // Label: bottom 25% of nose_dist
  },
  "model_config": {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05
  },
  "outputs": {
    "artifacts_dir": "data/eda_outputs",
    "run_registry": "data/run_registry.jsonl"
  }
}
```

### Editing config safely

1. **Never change** `outputs.artifacts_dir` unless you move the entire `data/eda_outputs/` directory
2. **Always validate** JSON syntax after editing:
   ```bash
   python -m json.tool configs/mvp_config.json > /dev/null && echo "✓ Valid JSON"
   ```
3. **Back up** before major changes:
   ```bash
   cp configs/mvp_config.json configs/mvp_config.json.backup
   ```

---

## Common Tasks

### Clean up old artifacts

```bash
# Remove everything except raw downloads and source data
rm -rf data/processed/*
rm -rf data/eda_outputs/*
```

### Download fresh MARS dataset

```bash
python scripts/download_data.py --config configs/mvp_config.json
python scripts/download_data.py --config configs/mvp_config.json --include-raw-images-top
```

### Export model for deployment

```bash
# Model is already at:
# data/eda_outputs/baseline_model.joblib

import joblib
model = joblib.load('data/eda_outputs/baseline_model.joblib')

# Use on new feature data:
y_proba = model.predict_proba(X_new_features)[:, 1]
```

### Review run history

```bash
# View all runs
tail -20 data/run_registry.jsonl | python -m json.tool

# Find runs with AUC > 0.90
python -c "
import json
with open('data/run_registry.jsonl') as f:
    for line in f:
        run = json.loads(line)
        if run.get('metrics', {}).get('auc', 0) > 0.90:
            print(f\"{run['timestamp']}: AUC={run['metrics']['auc']:.3f}\")
"
```

### Create a custom dataset

1. **Prepare pose JSON** in canonical format (see `data_contract.json`)
2. **Save to** `data/processed/your_dataset.json`
3. **Update config** to point to it:
   ```json
   {
     "pose_stage": {
       "mode": "dataset_json_passthrough",
       "input_file": "data/processed/your_dataset.json"
     }
   }
   ```
4. **Run pipeline**:
   ```bash
   python scripts/run_pipeline.py --config configs/mvp_config.json
   ```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- Code review expectations
- Commit message conventions
- Pull request workflow
- Branch naming

---

## Questions?

- **Architecture questions:** See [docs/system-architecture.md](docs/system-architecture.md)
- **Dataset references:** See [docs/datasets.md](docs/datasets.md)
- **PoC walkthroughs:** See [notes/poc-stories.md](notes/poc-stories.md)
- **MLOps philosophy:** See [docs/video-mlops-playbook.md](docs/video-mlops-playbook.md)

Good luck! 🧬👀
