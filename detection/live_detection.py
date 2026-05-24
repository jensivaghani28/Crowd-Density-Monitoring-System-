import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ultralytics import YOLO
import cv2
from utils.heatmap import generate_heatmap

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

THRESHOLD = 50

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    people_count = 0
    boxes_list = []

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:
                people_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes_list.append([x1, y1, x2, y2])

                # bounding box
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    # 🔥 Heatmap
    output_frame = generate_heatmap(frame, boxes_list)

    # Alert
    if people_count > THRESHOLD:
        cv2.putText(output_frame, "⚠️ OVERCROWD ALERT", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    # Info
    cv2.putText(output_frame, f"People: {people_count}", (50,100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Smart Crowd Detection", output_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()