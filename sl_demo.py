import streamlit as st
import streamlit_antd_components as sac
from streamlit_drawable_canvas import st_canvas

import os
import cv2
import numpy as np
import tempfile
import pandas as pd
import imutils
from PIL import Image

import supervision as sv
from ultralytics import YOLO

# Streamlit
st.set_page_config(layout="wide")

model_path = "y8s800full_v27.pt"
video_path = None
imgsz = 640 if "640" in model_path else 800
urls = {"Gümüşsuyu": "https://hls.ibb.gov.tr/tkm1/hls/193.stream/chunklist.m3u8",
         "Sirkeci": "https://hls.ibb.gov.tr/tkm1/hls/796.stream/chunklist.m3u8",
         "Beşiktaş": "https://hls.ibb.gov.tr/tkm4/hls/43.stream/chunklist.m3u8",
         "Üsküdar": "https://hls.ibb.gov.tr/tkm2/hls/405.stream/chunklist.m3u8",
         "Kapalıçarşı": "https://hls.ibb.gov.tr/ls/cam_turistik/cam_trsk_kapali_carsi.stream/chunklist_w2139077786.m3u8",
         "Eyüpsultan": "https://hls.ibb.gov.tr/ls/cam_turistik/cam_trsk_eyup.stream/playlist.m3u8",
         "Akçay 1": "https://content.tvkur.com/l/cj69rf5052aveu3s8ls0/master.m3u8", 
         "Akçay 2": "https://content.tvkur.com/l/cj69rut052aveu3s8ltg/master.m3u8", 
         "Altınoluk 1": "https://content.tvkur.com/l/cooanmhpkpqs73c1ui8g/master.m3u8",
         "Altınoluk 2": "https://content.tvkur.com/l/cj69tk5052aveu3s8m3g/master.m3u8",
         "Konya 1": "https://content.tvkur.com/l/c77ibcnbb2nj4i0fr8cg/master.m3u8",
         "Konya 2": "https://content.tvkur.com/l/cei9a1fmm25sdv36p5d0/master.m3u8",
         "Konya 5": "https://content.tvkur.com/l/c77ia4vbb2nj4i0fr85g/master.m3u8",
         "Konya 6": "https://content.tvkur.com/l/c77i7dr84cnrb6mlji80/master.m3u8",
         "Konya 7": "https://content.tvkur.com/l/cei9bavbb2nv2u3dv3k0/master.m3u8",
         "Brüksel Meydan": "https://livecam.brucity.be/LiveBrusselsWebcams/streams/IIbCJLnggPzOAFSL1701416341591.m3u8",
         "Düsseldorf Havalimanı": "https://stream-5gw.1000eyes.de/cupertino/dusfcae.stream/chunklist_w628542230.m3u8",
         }

# Model yükleme işlemi, model zaten yüklenmişse yüklemeyi atla
if 'model' not in st.session_state:
    st.session_state.model = YOLO(model_path, task="detect")

model = st.session_state.model  # Modeli yerel değişkene ata

# Streamlit arayüzü ----------------------------------------------------------
# Sidebar
with st.sidebar:
    infer = st.toggle("Oynat Uğurcum", value=False)
    sac.divider(label='Seç bakalım', icon=sac.BsIcon('camera-video', size=20), align="center", color="red")
    streams = [sac.SegmentedItem(label=key) for key in urls.keys()]
    vid_choice = sac.segmented(items=streams, use_container_width=True, direction="vertical", divider=False)
    video_path = urls[vid_choice]

# Video işleme
cap = cv2.VideoCapture(video_path)
video_info = sv.VideoInfo.from_video_path(video_path=video_path)

tab_main, tab_draw, tab_trace = st.tabs(["Pedestrian Tracking", "Perspective", "Traces"])

with tab_main:
    main_col1, main_col2 = st.columns([0.75, 0.25])

    with main_col1:
        stframe = st.empty()

    with main_col2:
        col1, col2, col3 = st.columns(3)
        with col1:
            metric1_col = st.empty()
        with col2:
            metric2_col = st.empty()
        with col3:
            metric3_col = st.empty()

        tab1, tab2 = st.tabs(["Pedestrian", "FPS"])
        with tab1:
            ped_col = st.container()
        with tab2:
            fps_col = st.container()

    btm_col1, btm_col2, btm_col3, btm_col4 = st.columns(4, gap="large")

    with btm_col1:
        skip_frames = st.slider("Skipped Frames", 1, 4, 2, step=1)
        conf = st.slider("Confidence", 0.2, 1.0, 0.6, step=0.05)
        iou = st.slider("IoU", 0.2, 1.0, 0.7, step=0.1)
    with btm_col2:
        track_activation_threshold = st.slider("Track Activation", 0.0, 1.0, 0.3, step=0.1)
        lost_track_buffer = st.slider("Lost Track Buffer", 30, 240, 120, step=30)
        minimum_matching_threshold = st.slider("Minimum Matching", 0.4, 1.0, 0.8, step=0.1)
    with btm_col3:
        stream = sac.switch(label="Stream", size="lg", value=True)
        stream_buffer = sac.switch(label="Stream Buffer", size="lg", value=True)
        half = sac.switch(label="Half Precision", size="lg", value=True)
        verbose = sac.switch(label="Verbose", size="lg")

with tab_draw:
    #cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        first_frame = imutils.resize(frame, width=1280)
        height, width = first_frame.shape[:2]
    #cap.release()

    if first_frame is not None:
        #st.image(first_frame, channels="BGR")
        canvas_result = st_canvas(
            drawing_mode="polygon",
            point_display_radius=5,            
            stroke_width=2,
            stroke_color="rgba(0, 255, 0, 1)",
            fill_color="rgba(0, 255, 0, 0.2)",
            background_image=Image.fromarray(first_frame),
            height=height,
            width=width,
            update_streamlit=False,
            display_toolbar=True,
            key="test"
            )

    #if canvas_result.json_data is not None:
        #st.text(canvas_result.json_data["objects"][0]["path"])

ped_chart = ped_col.line_chart()
fps_chart = fps_col.line_chart()


tracker = sv.ByteTrack(track_activation_threshold=track_activation_threshold, 
                       lost_track_buffer=lost_track_buffer,
                       minimum_matching_threshold=minimum_matching_threshold,
                       minimum_consecutive_frames=int(5/skip_frames), 
                       frame_rate=video_info.fps)
smoother = sv.DetectionsSmoother(length=int(10/skip_frames))

fps_monitor = sv.FPSMonitor(sample_size=10)
vid_fps = video_info.fps / skip_frames

box_annotator = sv.BoxCornerAnnotator(thickness=2, corner_length=10, color_lookup=sv.annotators.utils.ColorLookup.TRACK)
color_annotator = sv.ColorAnnotator(opacity=0.2, color_lookup=sv.annotators.utils.ColorLookup.TRACK)
label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=3, border_radius=5, color_lookup=sv.annotators.utils.ColorLookup.TRACK, text_position=sv.geometry.core.Position.TOP_CENTER)
trace_annotator = sv.TraceAnnotator(trace_length=180, color_lookup=sv.annotators.utils.ColorLookup.TRACK, position=sv.geometry.core.Position.CENTER, thickness=2)

def callback(frame: np.ndarray, vid_fps: float) -> np.ndarray:
    results = model(frame, imgsz=imgsz, conf=conf, iou=iou, stream=stream, stream_buffer=stream_buffer, verbose=verbose, half=half, device="mps")
    results = list(results)[0]

    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)
    detections = smoother.update_with_detections(detections)

    labels = [f"{tracker_id:.2f}" for tracker_id in detections.confidence]
    ped_count = len(detections)
    fps = fps_monitor.fps
    inference_time = results.speed['inference']

    annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
    annotated_frame = color_annotator.annotate(annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
    annotated_frame = trace_annotator.annotate(annotated_frame, detections=detections)
    
    metric1_col.metric(label="Video", value=f"{video_info.height}p", delta=f"{video_info.fps}fps", delta_color="off")
    metric2_col.metric(label=f"SF: {skip_frames}", value=f"{fps:.1f}", delta=f"/{vid_fps:.1f}fps", delta_color="off")
    metric3_col.metric(label="Inference", value=f"{inference_time:.1f}", delta=f"ms", delta_color="off")

    ped_chart.add_rows([ped_count])  # Line chart'a yaya sayısı ekleniyor
    fps_chart.add_rows({'FPS': [fps], 'Video FPS': [vid_fps]})  # Line chart'a fps ekleniyor

    fps_monitor.tick()

    return annotated_frame, fps

while cap.isOpened() and infer:
    ret, frame = cap.read()
    if not ret:
        break
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % skip_frames == 0 or int(cap.get(cv2.CAP_PROP_POS_FRAMES)) == 1:
        annotated_frame, _ = callback(frame, vid_fps)
        stframe.image(annotated_frame, channels="BGR")

cap.release()
