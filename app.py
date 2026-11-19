from pathlib import Path
import tempfile
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Traffic Monitoring",page_icon="🚦")
st.title("Traffic Monitoring — YOLO + OpenCV")
st.caption("Upload a recorded traffic video for vehicle detection and frame-level counts.")
weights=st.selectbox("YOLO weights",["yolo11n.pt"])
uploaded=st.file_uploader("Traffic video",type=["mp4","avi","mov","mkv"])
if uploaded and st.button("Process video"):
    with tempfile.NamedTemporaryFile(delete=False,suffix=Path(uploaded.name).suffix) as f:
        f.write(uploaded.getvalue()); source=f.name
    model=YOLO(weights)
    with st.spinner("Processing video..."):
        result=model.predict(source=source,conf=.30,save=True,
            project="runs/streamlit",name="traffic",exist_ok=True,verbose=False)
    st.success(f"Processed {len(result)} frames. Annotated output is saved in runs/streamlit/traffic.")
    st.info("For live local webcam or authorized IP-camera streams, run detect.py on a machine with camera/network access.")
