# Real Mouse Video Datasets

## 1. CalMS21
- URL: https://data.caltech.edu/records/s0vdx-0k302
- Includes: videos, tracked poses, frame-level behavior labels, unlabeled pose data, MARS-derived features
- Best for: social interaction modeling, pose-to-behavior pipelines, benchmark discussion
- PoC value: high
- Download friction: medium to high
- Starter fit: good if you work on a subset

## 2. MARS behavior annotation data
- URL: https://data.caltech.edu/records/7mh1s-yph35
- Includes: top-view videos, pose JSON, extracted features, frame-level behavior annotations
- Best for: end-to-end pipeline discussion from video to behavior classification
- PoC value: very high
- Download friction: high
- Starter fit: excellent if scoped to a few videos

## 3. MARS pose annotation data
- URL: https://data.caltech.edu/records/j1ww1-mdc55
- Includes: annotated images, keypoints, worker labels, processed pose data
- Best for: pose estimation, annotation quality, consensus labeling
- PoC value: high
- Download friction: medium
- Starter fit: best clean starting point

## 4. CRIM13
- URL: https://data.caltech.edu/records/1892
- Includes: synchronized top and side-view videos, action annotations, track files
- Best for: classic social behavior recognition and multi-view discussion
- PoC value: high
- Download friction: high
- Starter fit: good for talking points, less convenient for first build

## 5. MARS multi-worker behavior annotations
- URL: https://data.caltech.edu/records/1zp0n-nfs07
- Includes: resident-intruder videos and multi-annotator frame labels
- Best for: label quality, inter-rater agreement, annotation ops
- PoC value: very high for MLOps and data quality
- Download friction: medium
- Starter fit: very good

## 6. MABe22
- URL: https://data.caltech.edu/records/rdsa8-rde65
- Includes: mouse clips, keypoints, labels, train/test splits, benchmark tasks
- Best for: representation learning and benchmark discussion
- PoC value: high
- Download friction: medium
- Starter fit: more advanced

## 7. DANNCE markerless mouse demos
- URLs:
  - https://github.com/spoonsso/dannce
  - https://www.dropbox.com/sh/wn1x8erb5k3n9vr/AADE_Ca-2farKhd38ZvsNi84a?dl=0
  - https://www.dropbox.com/sh/tspmwo36gbj6b4x/AAA_sWJA6K1ksX8f6hBoZf7Ia?dl=0
- Includes: multi-view mouse videos and 3D pose demo assets
- Best for: multi-camera synchronization and 3D pose
- PoC value: high if the team uses multiple cameras
- Download friction: medium to high
- Starter fit: good for advanced prep

## Recommended shortlist

### Best first hands-on dataset
- MARS pose annotation data

### Best end-to-end pipeline dataset
- MARS behavior annotation data

### Best benchmark for comparative evaluation
- CalMS21

## Canonical tools to know
- DeepLabCut: popular pose estimation in animal behavior labs
- SLEAP: multi-animal pose estimation and tracking
- MARS: mouse action recognition pipeline
- DANNCE: 3D pose from multi-view video

## How to frame this PoC

A strong answer is:

> I would treat mouse monitoring as a video data product, not just a model problem. I would define the pipeline from acquisition, synchronization, and storage all the way to pose extraction, behavior labels, evaluation, serving, and continuous error analysis. Datasets like CalMS21 and MARS make it easier to reason concretely about that stack.
