import cv2
import numpy as np

def generate_heatmap(frame, boxes):
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

    for box in boxes:
        x1, y1, x2, y2 = box
        heatmap[y1:y2, x1:x2] += 1

    heatmap = cv2.GaussianBlur(heatmap, (25,25), 0)

    if np.max(heatmap) != 0:
        heatmap = (heatmap / np.max(heatmap)) * 255

    heatmap = heatmap.astype("uint8")
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)