# TrafficFlow AI | Occlusion-Aware Vehicle Counting & Traffic Analytics

**Portfolio focus:** Computer Vision / ML Engineering / Data Science with time-series events. Original project: https://github.com/srinivasarao120005/realtime-traffic-monitoring.

## Goal and experimental innovation
A YOLO-only detector sees vehicles in each frame but cannot tell whether frame-level counts refer to *new* vehicles. This portfolio refactor adds a Hungarian assignment tracker with simple velocity prediction, short-occlusion retention, unique stable line-crossing counts, vehicle occupancy *proxy*, time-series plotting and offline detection/identity evaluation. These components are **established techniques**; the contribution to be tested is the specific fused decision policy and its ablations, not invention of tracking itself.

## Dataset and data-quality workflow
**UA-DETRAC** is the intended data source. Obtain videos/frame sequences and matching XML annotations from the official benchmark or authorized mirrors. See [UA-DETRAC data instructions](https://github.com/auag92/eva/blob/master/data/ua_detrac/README.md). Images/videos and trained weights are not redistributed.

1. Place raw sequence frames/videos under `data/raw/` and annotation XML under `data/raw/annotations/`.
2. `python src/prepare_detrac.py --xml-dir data/raw/annotations --out results` parses every frame, flags invalid boxes, rejects duplicate (sequence, frame, target ID), audits class/sequence counts and saves `ground_truth.csv` / `class_distribution.png`.
3. Split by **whole video sequence** for any detector training and evaluation; do not randomly mix adjacent frames. Review ignored regions and annotation policy before reporting full detection metrics.
4. `pip install ultralytics` (optional, in addition to `requirements.txt`), then run `python src/run_video.py --video data/raw/MVI_20011.mp4 --weights yolov8n.pt --out results/MVI_20011` (requires network to download weights unless cached). The first run may need several minutes and CPU/GPU depending on environment.
5. Use per-frame YOLO detections and tracked identity outputs against protocol-matched GT. For saved `tracked_boxes.csv` (or other tracking predictions) containing columns `frame,id,x1,y1,x2,y2`, run `python src/evaluate.py --gt gt_one_video.csv --pred pred_track_one_video.csv --out results/tracking_metrics.json`.
6. To run tracking on existing model detections without downloading YOLO weights: `python src/track_from_csv.py --csv detections.csv --width 960 --height 540 --out results/from_csv`; this creates tracked identity CSV and plots. Do **not** treat ground-truth-as-detections as measured detector performance.
7. `pytest -q` for geometry/crossing/identity behavior unit tests.

## Outputs
`annotated.mp4`, `frame_metrics.csv`, `crossing_events.csv`, `detections.csv`, `tracked_boxes.csv`, `traffic_dashboard.png`, `metrics.json`. `crossing_events.csv` counts unique *track crossing events*, not raw bounding boxes. `bbox_area_occupancy_proxy` sums observed box areas over the full frame (overlap can double-count); this is **not** road occupancy percentage nor speed. Without calibrated camera geometry and lanes, don't present speed in km/h.

## Tests and experimental comparisons needed
- Detector-only count per frame vs tracked unique crossings; line-crossing MAE against manually labeled crossing events.
- IoU-only vs motion+IoU matching with same detections and threshold; IDF1, identity switches, count error and processing FPS on held-out sequences.
- Clear weather/daytime/nighttime stratification and occlusion severity, if annotation is available.
- Compare total pipeline FPS on named CPU/GPU and model configuration; prerecorded-video throughput ≠ real-time CCTV claim.

## Limitations
`src/evaluate.py` computes simplified frame-level IoU and globally-assigned IDF1, **not an official UA-DETRAC leaderboard score** (ignore regions and distractors are not fully implemented). The tracker is a transparent portfolio baseline, not a proven new MOT model. Tracking ground truth alone is an oracle diagnostic and must never be presented as end-to-end YOLO performance. The scripts require real videos + annotations for benchmark results.

## Development-only preview
**Synthetic sequence, not a real UA-DETRAC result.**

![Synthetic traffic trace](examples/synthetic_tracking_dashboard.png)

See [results status](docs/RESULTS_STATUS.md).
