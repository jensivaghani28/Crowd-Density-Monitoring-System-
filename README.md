# 🚦 Smart Crowd Management System

An AI-powered Smart Crowd Management System using YOLOv8, OpenCV, Machine Learning, and Streamlit for real-time crowd detection, density monitoring, heatmap visualization, and overcrowding alerts through live webcam, image, and video analysis.

---

# 📌 Features

- 👥 Real-time Crowd Detection
- 🎥 Live Webcam Monitoring
- 🖼️ Image Crowd Detection
- 🎬 Video Crowd Detection
- 🔥 Heatmap Visualization
- 📊 Crowd Density Prediction
- 🚨 Overcrowding Alerts
- 📈 Interactive Streamlit Dashboard

---

# 🛠️ Technologies Used

- Python
- YOLOv8
- OpenCV
- Streamlit
- Machine Learning
- Pandas
- NumPy

---

# 📂 Project Structure

```bash
Smart-Crowd-Management-System/
│
├── yolov8n.pt
├── crowd_data.csv
├── dashb.py
├── generate_dataset.py
├── requirements.txt
│
├── detection/
│   ├── image_detection.py
│   ├── live_detection.py
│   └── video_detection.py
│
├── model/
│   ├── columns.pkl
│   └── hybrid_model.pkl
│
├── notebooks/
│   └── crowd.ipynb
│
├── utils/
│   ├── heatmap.py
│   └── __init__.py
```

---

# 🚀 Installation

## Clone Repository

```bash
git clone YOUR_GITHUB_REPO_LINK
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Project

```bash
streamlit run dashb.py
```

---

# 📊 Modules

## 1. Live Detection
Detects people from webcam in real-time.

## 2. Image Detection
Detects crowd from uploaded images.

## 3. Video Detection
Processes uploaded videos frame-by-frame.

## 4. Heatmap Generation
Displays crowd density visually using heatmaps.

## 5. Density Prediction
Predicts crowd density using Machine Learning.

---

# 🔥 Future Enhancements

- Face Recognition
- Multi-camera Support
- Cloud Deployment
- Mobile Application
- Emergency Alert Integration

---

# 👨‍💻 Developed By

Jensi Vaghani
---

# 📄 License

This project is licensed under the MIT License.
