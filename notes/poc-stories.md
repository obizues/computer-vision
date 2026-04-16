# PoC Implementation Narratives

## Narrative 1: Treating video as a product
A mouse monitoring platform should not start with model selection. It should start with a reliable video pipeline: acquisition, metadata, quality checks, annotation workflows, and reproducible derived artifacts. This lets research teams iterate on models without losing traceability.

## Narrative 2: Why pose is a strategic intermediate
Pose is often the right abstraction between raw video and behavior classification. It compresses data volume, improves interpretability, and lets multiple downstream tasks reuse the same representation.

## Narrative 3: Human-in-the-loop behavior labeling
Use a loop where uncertain clips are prioritized for review, corrected labels are versioned, and retraining jobs are triggered on a schedule. This turns annotation into a measurable operational process rather than ad hoc cleanup.

## Narrative 4: Failure modes in mouse video
Common failure modes include occlusion, bedding changes, reflections, night recording, camera drift, broken timestamps, identity swaps, and inconsistent annotation rules. The PoC should track and report each class of issue.

## Narrative 5: A simple platform architecture
- object storage for raw video
- metadata database for recordings and experiments
- batch pipeline for frame, clip, and pose extraction
- annotation store with versioned labels
- model registry and evaluation reports
- scientist-facing analysis UI

## Strong questions for stakeholders
- What is the current unit of prediction: frame, clip, or event?
- Do you rely more on raw video models or pose-derived models?
- How often do annotations get revised?
- What are the hardest failure modes today?
- Is the bottleneck data quality, model quality, or operational usability?

## 5-minute architecture walkthrough

1. Input layer
- Start with versioned pose inputs from `data/MARS_keypoints_top.json` and `data/MARS_keypoints_front.json`.
- Data expectations are pinned in `configs/data_contract.json`.

2. Feature layer
- `scripts/build_features.py` flattens pose records into model-ready features and writes:
	- `data/eda_outputs/features_top_view.csv`
	- `data/eda_outputs/feature_quality_report.json`

3. Training and evaluation layer
- `scripts/train_eval.py` trains a baseline classifier, evaluates it, and writes:
	- `data/eda_outputs/baseline_model.joblib`
	- `data/eda_outputs/baseline_model_summary.json`
	- `data/eda_outputs/baseline_scored_sample.csv`

4. Inference layer
- `scripts/predict_batch.py` runs batch scoring on feature tables and writes:
	- `data/eda_outputs/batch_predictions.csv`

5. Orchestration and lineage layer
- `scripts/run_pipeline.py` runs all steps end-to-end from one command.
- It appends run metadata to `data/run_registry.jsonl` so every run is traceable by dataset version, config, metrics, and artifact paths.

6. Why this matters in production
- This structure separates concerns cleanly: contracts, transforms, modeling, inference, and lineage.
- It makes retraining and debugging operational instead of ad hoc.

## Demo dry-run script (3 minutes)

Run command:
- `python scripts/run_pipeline.py --config configs/mvp_config.json`

Say while it runs:
- This command executes feature engineering, model training/evaluation, and batch inference.
- All behavior is controlled by `configs/mvp_config.json`, and inputs are validated by `configs/data_contract.json`.

Show artifacts:
- `data/eda_outputs/feature_quality_report.json` for data QC
- `data/eda_outputs/baseline_model_summary.json` for model metrics
- `data/eda_outputs/batch_predictions.csv` for inference output
- `data/run_registry.jsonl` for lineage and reproducibility

Close with impact statement:
- This is an MVP production loop: reproducible config, automated checks, model outputs, and run tracking.
- Next step is replacing proxy labels with true behavior annotations and adding periodic retraining triggers.
