from ultralytics import YOLO
import cv2
import pandas as pd
from utils.heatmap import generate_heatmap


model = YOLO("yolov8n.pt")

def process_frame(frame, model_ml, columns):

    results = model(frame)

    people_count = 0
    boxes_list = []

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:
                people_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes_list.append([x1,y1,x2,y2])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    # 🔥 Heatmap
    frame = generate_heatmap(frame, boxes_list)

    # 🔮 Density
    input_df = pd.DataFrame(columns=columns)
    input_df.loc[0] = 0
    input_df['people_count'] = people_count
    input_df['area_size'] = 60

    pred_density = model_ml.predict(input_df)[0]

    # 🚦 Status
    ratio = people_count / 60

    if ratio > 12:
        status = "Overcrowded"
    elif ratio > 7:
        status = "High Density"
    else:
        status = "Normal"

    return frame, people_count, pred_density, status