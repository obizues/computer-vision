# Video + MLOps Playbook for Mouse Monitoring

## End-to-end system view

### 1. Video acquisition
Questions to ask:
- How many cameras per enclosure?
- Frame rate and resolution requirements?
- IR or low-light recording?
- How is time synchronization handled?
- Are videos continuous or event-triggered?

Risks:
- dropped frames
- unsynced cameras
- codec corruption
- lighting changes
- occlusion and cage artifacts

### 2. Data contracts
Define a contract for every recording unit:
- `video_id`
- `camera_id`
- `timestamp_start`
- `fps`
- `resolution`
- `duration_sec`
- `experiment_id`
- `mouse_ids` if known
- annotation status
- QC status

### 3. Labeling layers
Typical layers:
- detection boxes
- identity tracking
- pose keypoints
- frame-level behaviors
- clip-level summaries

Key idea:
Behavior labels are downstream of pose and context, so label provenance matters.

### 4. Training stages
A realistic stack is:
1. detection or segmentation
2. tracking and identity assignment
3. pose estimation
4. behavior classification from pose, image crops, or both
5. post-processing and smoothing

### 5. Evaluation
Metrics by stage:
- detection: precision and recall
- tracking: IDF1, switches
- pose: keypoint error metrics
- behavior: frame F1, bout-level F1, confusion by behavior

Always break results down by:
- cage setup
- lighting condition
- mouse strain or phenotype
- camera angle
- crowded vs simple scenes

### 6. Deployment patterns
Common options:
- offline batch processing after experiments
- near-real-time scoring in the lab
- human review queue for uncertain events
- scheduled retraining from newly labeled clips

### 7. MLOps controls
You should be able to discuss:
- versioned datasets and annotation schemas
- reproducible feature extraction
- lineage from video to prediction
- model registry and rollback
- drift monitoring
- active learning for hard clips
- review tooling for scientists

## MVP architecture for this repo

### Components
- `configs/mvp_config.json`: run-time settings and versioned decisions
- `configs/data_contract.json`: schema + quality gates for input records and labels
- feature build step: converts raw pose records to model-ready table
- train/eval step: fits model and writes metrics + confusion outputs
- batch inference step: scores new pose/video batches
- run registry step: appends one run manifest per execution

### Required run artifacts
- `baseline_model_summary.json`
- `baseline_scored_sample.csv`
- `baseline_score_distribution.png`
- `dataset_summary.json`
- run manifest row in `data/run_registry.jsonl`

### Acceptance checks
- all required artifacts are present
- data quality gate failures are zero or explicitly reported
- model metrics are written with timestamp and config reference
- rerunning with same config reproduces equivalent outputs

## Good interview framing

### If they ask about building the platform
Say:
- separate raw data from curated training data
- treat annotations as first-class versioned assets
- keep pose extraction as a reusable intermediate layer
- support both research workflows and production batch inference

### If they ask about video specifically
Say:
- video is expensive, so sampling, caching, and derived artifacts matter
- failure analysis should link metrics back to clips and frames
- temporal context matters, so clip boundaries and smoothing policies must be explicit

### If they ask about scientists using the system
Say:
- optimize for easy relabeling and rapid review
- expose uncertain predictions for correction
- make experiment metadata searchable
- make every model result traceable to the exact input video and code version

## Concrete 30-60-90 day plan

### First 30 days
- inventory cameras, formats, storage, and labels
- map the current pipeline
- identify one high-value benchmark task
- establish dataset and model versioning conventions

### 60 days
- build reproducible preprocessing jobs
- define evaluation dashboards by cohort and condition
- standardize inference outputs and review workflow

### 90 days
- automate retraining triggers
- add drift and data-quality checks
- operationalize a human-in-the-loop correction loop
