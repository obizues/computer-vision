# Mouse Vision Proof of Concept

This workspace is a proof of concept for using computer vision to identify mouse social interactions from pose signals. It includes a complete ML pipeline for detecting close interactions, a scientist-facing dashboard for results analysis, and reproducible configuration-driven workflows.

## Goal

Build enough hands-on understanding that you can discuss:

- raw video ingestion
- camera and recording constraints
- frame extraction and storage tradeoffs
- tracking and pose estimation
- behavior labeling and classifier training
- model packaging and deployment
- data quality, drift, and annotation ops

## Quick Start (5 minutes)

### One-command local setup (recommended)

If you want the fastest path to run locally with guided options:

1. Open a terminal in the project root
2. Run:

```bash
.\launch_mvp.bat
```

Then choose:

- **1** to download required data
- **2** to run the full pipeline
- **3** to open the dashboard
- **4** to do all steps in sequence

This gives a complete local flow for users who want to reproduce results and run the PoC on their own machine.

### I want to...

| Task | Go to |
|------|-------|
| **Set up my local environment** | [DEVELOPER.md → Environment Setup](DEVELOPER.md#environment-setup) |
| **Run the full pipeline** | [DEVELOPER.md → Running the Pipeline](DEVELOPER.md#running-the-pipeline) |
| **Explore results in the dashboard** | [DEVELOPER.md → Running the Dashboard](DEVELOPER.md#running-the-dashboard) |
| **Understand the system architecture** | [docs/system-architecture.md](docs/system-architecture.md) |
| **Add a new feature to the model** | [DEVELOPER.md → Adding a new feature](DEVELOPER.md#adding-a-new-feature) |
| **Deploy to Streamlit Cloud for PoC viewing** | [PoC hosting](#poc-hosting-streamlit-viewer) section below |
| **Contribute code changes** | [CONTRIBUTING.md](CONTRIBUTING.md) |

## Quick Links

- **For developers:** See [DEVELOPER.md](DEVELOPER.md) for setup, running, and extending the code
- **For architects:** See [docs/system-architecture.md](docs/system-architecture.md) for technical design  
- **For contributors:** See [CONTRIBUTING.md](CONTRIBUTING.md) for collaboration guidelines
- **For PoC viewers:** See "[PoC hosting](#poc-hosting-streamlit-viewer)" section below

**Documentation:**
- [DEVELOPER.md](DEVELOPER.md) — **Start here** for environment setup, running pipeline & dashboard, making changes
- [CONTRIBUTING.md](CONTRIBUTING.md) — Code style, pull request process, testing expectations
- [README.md](README.md) — This file; overview, quick start, configuration reference
- [docs/datasets.md](docs/datasets.md) — Real mouse video datasets and what each is good for
- [docs/system-architecture.md](docs/system-architecture.md) — Technical design, model inventory, orchestrator responsibilities
- [docs/video-mlops-playbook.md](docs/video-mlops-playbook.md) — MLOps patterns and deployment strategies for video behavior systems
- [notes/poc-stories.md](notes/poc-stories.md) — Reusable implementation narratives for presenting this proof of concept

**Code:**
- [scripts/](scripts) — Pipeline stages (pose ingestion, feature building, training, inference)
- [app/scientist_dashboard.py](app/scientist_dashboard.py) — Streamlit UI for segment review and model diagnostics
- [configs/](configs) — Configuration files (mvp_config.json, data_contract.json)

## Best starting datasets

1. CalMS21
2. MARS behavior annotation data
3. MARS pose annotation data

## Suggested first week

### Day 1
- Read [docs/datasets.md](docs/datasets.md)
- Pick one dataset to study first
- Write down the supervision type: tracking, pose, or behavior labels

### Day 2
- Map the pipeline from raw video to model input
- Identify storage formats, annotations, and evaluation outputs

### Day 3
- Use the scripts in [scripts](scripts) on a small local video set
- Practice talking through video QA checks

### Day 4
- Read [docs/video-mlops-playbook.md](docs/video-mlops-playbook.md)
- Prepare answers about data versioning, retraining, and deployment

### Day 5
- Review 3 implementation narratives from [notes/poc-stories.md](notes/poc-stories.md)

## If you want to go deeper

A strong demo project is:

- ingest a small set of mouse videos
- compute clip metadata and sampled frames
- define a labeling schema
- store annotations in a reproducible format
- train or fine-tune a simple behavior classifier
- expose inference as a batch job plus analysis dashboard

## MVP Success Criteria

A run is considered MVP-complete when all of the following are true:

- input data exists at `data/MARS_keypoints_top.json` and `data/MARS_keypoints_front.json`
- one command can execute feature build, train, evaluate, and export artifacts
- outputs include:
	- summary metrics JSON
	- scored sample CSV
	- score distribution plot
	- data quality report with validation counts
- each run writes a run record with:
	- timestamp
	- dataset version
	- config version
	- model version
	- evaluation metrics
- all artifacts are reproducible from a pinned config file in `configs/`

## Contracts and Configuration

- `configs/mvp_config.json`: runtime config for data paths, split settings, model settings, and output locations
- `configs/data_contract.json`: required schema and quality gates for pose records and label values

These two files are the source of truth for pipeline behavior and validation checks.

## Run the MVP pipeline

**→ See [DEVELOPER.md](DEVELOPER.md#running-the-pipeline) for detailed setup and execution instructions.**

Quick start:

```bash
# Windows (PowerShell)
.\launch_mvp.bat

# Or directly:
python scripts/run_pipeline.py --config configs/mvp_config.json
```


The download step is safe to rerun and skips files that already exist.

`--include-raw-images-top` downloads and extracts real MARS top-view mouse frames (large file) so the dashboard can display real raw frames beside keypoint overlays.

`--include-sample-video` downloads the hosted sample video to `pose_inference_runtime.video_file` using `pose_inference_runtime.video_url` from config.

This executes:

1. `scripts/build_features.py`
2. `scripts/train_eval.py`
3. `scripts/predict_batch.py`

And writes:

- model + metrics to `data/eda_outputs/`
- batch predictions to `data/eda_outputs/batch_predictions.csv`
- run lineage entry to `data/run_registry.jsonl`

## Front-to-back mode options

`pose_stage.mode` in `configs/mvp_config.json` controls how the pipeline starts:

- `dataset_json_passthrough`: uses downloaded MARS pose JSON as the pose stage input.
- `video_stub`: runs a lightweight CV stub on local videos in `data/raw_videos/` to create canonical pose records.
- `deeplabcut_canonical_json`: loads canonical pose JSON exported from a DeepLabCut workflow.
- `sleap_canonical_json`: loads canonical pose JSON exported from a SLEAP workflow.
- `continuous_video_external_inference`: runs a configured external inference command on raw video, writes canonical pose JSON, then validates contiguous frame coverage.

For DeepLabCut/SLEAP modes, set `pose_stage.external_pose_file` to your exported canonical pose JSON path.

For `continuous_video_external_inference`, configure `pose_inference_runtime.command` in `configs/mvp_config.json`.
Supported placeholders:

- `{video}` selected input video path
- `{output}` canonical pose JSON output path (`pose_stage.external_pose_file`)
- `{config}` active config path

If `pose_inference_runtime.video_file` is missing locally and `pose_inference_runtime.video_url` is set, runtime auto-downloads the video before inference. If `pose_inference_runtime.video_sha256` is set, the download is verified before running.

DeepLabCut runtime helper:

- `scripts/external_pose_inference_dlc.py` runs DLC on `{video}` and converts the resulting CSV to canonical pose JSON at `{output}`.
- Set `pose_inference_runtime.dlc.project_config` to your DLC project `config.yaml` before running.

Canonical pose format is a JSON list where each item includes:

- `filename`, `width`, `height`, `labels`
- `coords.black.x`, `coords.black.y`
- `coords.white.x`, `coords.white.y`

The pipeline always normalizes to `data/processed/pose_top_keypoints.json`, then runs feature build → train/eval → batch predict.

Note: MARS pose annotation data provides real image frames and pose labels, not continuous source videos.

## Ground-truth behavior labels (high-ROI upgrade)

You can override proxy closeness labels with real labels by creating:

- `data/processed/behavior_labels.csv`

Expected columns (default config):

- `frame_idx`
- `is_close_interaction`

`is_close_interaction` can be `0/1`, boolean-like values, or string labels listed in `label_stage.positive_labels` in `configs/mvp_config.json`.

If the file is missing, the pipeline automatically falls back to proxy labels and records that in `data/eda_outputs/feature_quality_report.json`.

## Scientist-facing dashboard

**→ See [DEVELOPER.md](DEVELOPER.md#running-the-dashboard) for detailed dashboard instructions.**

Quick start:

```bash
streamlit run app/scientist_dashboard.py
```

Open browser to: `http://localhost:8501`

**Tabs:**
- **Event Replay** — Select video, run pipeline, browse detected interaction segments with pose overlay
- **Analytics & Details** — Model metrics, data quality, full-timeline probability chart, frame-level review
- **Artifacts** — Links to model files, predictions, quality reports

## PoC hosting (Streamlit viewer)

Use this when you want a clean hosted experience for stakeholders.

### 1) Local prep before pushing

- Run one successful local pipeline pass so artifacts are up to date:
	- `python scripts/run_pipeline.py --config configs/mvp_config.long_video.json`
- Ensure these outputs exist and look correct:
	- `data/processed/pose_top_keypoints.json`
	- `data/eda_outputs/batch_predictions.csv`
	- `data/eda_outputs/baseline_model_summary.json`
	- `data/eda_outputs/feature_quality_report.json`

### 2) Push repo to GitHub

- `.gitignore` excludes large local-only folders (venvs, raw videos, DLC intermediate outputs).
- Keep only files needed for a reproducible viewer/demo.

### 3) Deploy to Streamlit Community Cloud

- App entrypoint: `app/scientist_dashboard.py`
- Python dependencies: `requirements.txt`
- Add environment variable in app settings:
	- `MOUSE_VISION_DEMO_MODE=1`

When `MOUSE_VISION_DEMO_MODE=1`, the dashboard disables in-app pipeline execution and acts as a stable viewer for precomputed artifacts.

Hosted sample video (for local pipeline runs):

- https://github.com/obizues/computer-vision/releases/download/video-assets-v1/mars_top_dataset_sample.mp4

Notes:

- Streamlit hosted mode is intended for viewing precomputed results, not running full DLC inference in-app.
- Local runtime can still auto-download this sample video using `pose_inference_runtime.video_url` if `video_file` is missing.

### 4) Stakeholder walkthrough (30–60 seconds)

- Open Event Replay tab
- Select a segment and replay Raw vs Overlay
- Scroll to probability trace for the selected segment
- Open Analytics tab for model/data-quality summary

This demonstrates end-to-end PoC capability (pipeline + UX + diagnostics) without requiring heavy inference on the hosted environment.

## Windows launcher

For a menu-driven launcher on Windows, run:

- `launch_mvp.bat`

It provides options to:

- download required data
- run the full pipeline
- launch the scientist dashboard
- run all steps sequentially
- create a desktop shortcut to the launcher
