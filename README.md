# Real-Time Traffic Monitoring

YOLO object detection with OpenCV for recorded traffic videos and local webcam/IP-camera streams. Detects supported road-user classes and counts detections per frame.

## Data and model
- Public traffic-video benchmark: [UA-DETRAC](https://detrac-db.rit.albany.edu/) (check access and research-use terms on the official site).
- Optional sample videos: use traffic videos you own or have permission to use. Place them under `data/videos/`.
- Detector weights: Ultralytics COCO-pretrained `yolo11n.pt` are fetched by Ultralytics on first run. Review Ultralytics licensing before commercial use.

## Install and run
```bash
python -m venv .venv
# activate environment
pip install -r requirements.txt
python detect.py --source data/videos/traffic.mp4
python detect.py --source 0
```
For an IP camera, pass its authorized stream URL as `--source`. Keep camera credentials out of source control.

## Streamlit interface
```bash
streamlit run app.py
```
Upload a recorded clip for annotated output. Live camera operation is available through the command line (`--source 0` or your authorized RTSP URL).

## Limitations
Counts are frame-level detections, not unique vehicle tracks. For reliable unique-vehicle counts and speed estimation, add tracking (e.g. ByteTrack), camera calibration and a defined counting line. Performance depends on camera angle, weather, lighting and compute.
