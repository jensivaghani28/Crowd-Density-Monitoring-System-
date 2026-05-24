import streamlit as st
import pandas as pd
import pickle
import cv2
import numpy as np
from ultralytics import YOLO

from yolo.image_detection import process_image
from yolo.video_detection import process_frame

st.set_page_config(layout="wide")

st.title("🚦 SMART CROWD MANAGEMENT SYSTEM")
st.markdown("### 📊 Real-Time AI Monitoring Dashboard")
st.markdown("---")

# ==========================
# LOAD DATA + MODEL
# ==========================

df = pd.read_csv("crowd_data.csv")
st.dataframe(df.head(10))

model_ml = pickle.load(open("model/hybrid_model.pkl", "rb"))
columns = pickle.load(open("model/columns.pkl", "rb"))

model_yolo = YOLO("yolov8n.pt")


# ==========================
# SIDEBAR CONTROLS
# ==========================

st.sidebar.title("⚙️ Control Center")

run = st.sidebar.toggle("▶ Start Detection")
threshold = st.sidebar.slider("🚨 Alert Threshold", 10, 200, 50)
area_live = st.sidebar.slider("📐 Area Size (Live)", 40, 100, 60)

st.sidebar.markdown("---")
st.sidebar.success("System Running with YOLOv8 + ML")

# ==========================
# METRICS
# ==========================

st.markdown("## 📊 System Overview")

col1, col2, col3 = st.columns(3)

col1.metric("👥 Max People", int(df['people_count'].max()))
col2.metric("📊 Avg Density", round(df['density'].mean(), 2))
col3.metric("🗂 Total Records", len(df))

# ==========================
# CHARTS
# ==========================
st.markdown("## 📊 Analytics Dashboard")

tab1, tab2, tab3 = st.tabs(["📈 Trend", "📍 Location", "📊 Density"])

with tab1:
    st.line_chart(df['people_count'])

with tab2:
    st.bar_chart(df.groupby('location')['people_count'].mean())

with tab3:
    st.line_chart(df['density'])

# ==========================
# 🔥 HEATMAP FUNCTION
# ==========================

def generate_heatmap(frame, boxes):
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

    for box in boxes:
        x1, y1, x2, y2 = box
        heatmap[y1:y2, x1:x2] += 1

    heatmap = cv2.GaussianBlur(heatmap, (25,25), 0)

    if np.max(heatmap) != 0:
        heatmap = (heatmap / np.max(heatmap)) * 255

    heatmap = heatmap.astype("uint8")
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

# ==========================
# 🔮 MANUAL PREDICTION UI (IMPROVED)
# ==========================

st.markdown("## 🔮 Crowd Prediction Panel")

col1, col2 = st.columns(2)

with col1:
    people = st.slider("👥 People Count", 1, 1000)

with col2:
    area = st.slider("📐 Area Size", 10, 100)

if st.button("🚀 Predict Now"):
    input_df = pd.DataFrame(columns=columns)
    input_df.loc[0] = 0

    if 'people_count' in input_df.columns:
        input_df['people_count'] = people

    if 'area_size' in input_df.columns:
        input_df['area_size'] = area

    pred_density = model_ml.predict(input_df)[0]

    density_ratio = people / area

    if density_ratio > 12:
        status = "Overcrowded"
    elif density_ratio > 7:
        status = "High Density"
    else:
        status = "Normal"

    # DEBUG (IMPORTANT)
    st.write("Debug Ratio:", round(density_ratio,2))

    # OUTPUT
    st.metric("📊 Density", round(pred_density,2))


    if status == "Overcrowded":
        st.error("🚨 OVERCROWD!")
    elif status == "High Density":
        st.warning("⚠️ HIGH DENSITY AREA")
    else:
        st.success("✅ NORMAL CROWD")

# ==========================
# 🔴 LIVE DETECTION UI
# ==========================

st.markdown("## 🎥 Live Crowd Detection")

status_placeholder = st.empty()

if run:
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not working")
            break

        results = model_yolo(frame)

        people_count = 0
        boxes_list = []

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:
                    people_count += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes_list.append([x1, y1, x2, y2])

                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        # Heatmap
        frame = generate_heatmap(frame, boxes_list)

        # ==========================
        # ML PREDICTION
        # ==========================

        input_df = pd.DataFrame(columns=columns)
        input_df.loc[0] = 0

        if 'people_count' in input_df.columns:
            input_df['people_count'] = people_count

        if 'area_size' in input_df.columns:
            input_df['area_size'] = area_live

        pred_density = model_ml.predict(input_df)[0]

        # ==========================
        # CROWD STATUS
        # ==========================

        if people_count > threshold:
            crowd_status = "Overcrowded"
            color = (0,0,255)

        elif pred_density > 6:
            crowd_status = "High Density"
            color = (0,165,255)

        else:
            crowd_status = "Normal"
            color = (0,255,0)

        # Display on frame
        cv2.putText(frame, f"People: {people_count}", (50,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.putText(frame, f"Density: {pred_density:.2f}", (50,150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        cv2.putText(frame, f"Status: {crowd_status}", (50,200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame)

        # ==========================
        # STREAMLIT LIVE STATUS PANEL
        # ==========================

        with status_placeholder.container():
            st.markdown("### 🚦 Live Status")

            col1, col2, col3 = st.columns(3)
            col1.metric("👥 People", people_count)
            col2.metric("📊 Density", round(pred_density,2))
            col3.metric("🚦 Status", crowd_status)

            if crowd_status == "Overcrowded":
                st.error("🚨 OVERCROWD ALERT!")
            elif crowd_status == "High Density":
                st.warning("⚠️ HIGH DENSITY AREA")
            else:
                st.success("✅ NORMAL CROWD")

    cap.release()

# ==========================
# 🔴 IMAGE DETECTION UI
# ==========================
st.markdown("## 🖼️ Image Crowd Detection")

uploaded_image = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_image is not None:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # 🔥 CALL FUNCTION
    image, people_count, pred_density, status = process_image(image, model_ml, columns)

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    st.write(f"👥 People: {people_count}")
    st.write(f"📊 Density: {round(pred_density,2)}")

    if status == "Overcrowded":
        st.error("🚨 OVERCROWD!")
    elif status == "High Density":
        st.warning("⚠️ HIGH DENSITY CROWD")
    else:
        st.success("✅ NORMAL CROWD")

# ==========================
# 🔴 Video DETECTION UI
# ==========================
st.markdown("## 🎥 Video Crowd Detection")

uploaded_video = st.file_uploader("Upload Video", type=["mp4","avi"])

if uploaded_video is not None:
    tfile = open("temp.mp4", "wb")
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture("temp.mp4")
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 🔥 CALL FUNCTION
        frame, people_count, pred_density, status = process_frame(frame, model_ml, columns)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame)

        st.write(f"👥 People: {people_count}")
        st.write(f"📊 Density: {round(pred_density,2)}")

        if status == "Overcrowded":
            st.error("🚨 OVERCROWD")
        elif status == "High Density":
            st.warning("⚠️ HIGH DENSITY")
        else:
            st.success("✅ NORMAL")

    cap.release()
# ==========================
# FOOTER
# ==========================

st.markdown("---")
st.caption("© AI Crowd Management System | YOLOv8 + Machine Learning")