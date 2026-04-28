import cv2
import numpy as np
import time
from collections import deque
from ultralytics import YOLO

print("🚀 FINAL DLCI SYSTEM (CLEAN LANES) RUNNING...")

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("inference/test_2.mp4")

# -----------------------------
# MEMORY
# -----------------------------
track_history = {}
speed_history = {}

NUM_LANES = 3
lane_histories = [deque(maxlen=20) for _ in range(NUM_LANES)]

prev_time = 0

# -----------------------------
# CLASS + WEIGHTS
# -----------------------------
names = {2: "Car", 3: "Bike", 5: "Bus", 7: "Truck"}

weights = {
    "Car": 1.0,
    "Bike": 0.5,
    "Bus": 2.5,
    "Truck": 2.0
}

# -----------------------------
def classify(dlci):
    if dlci < 0.3:
        return "LOW"
    elif dlci < 0.6:
        return "MEDIUM"
    else:
        return "HIGH"

def get_color(dlci):
    if dlci < 0.3:
        return (0, 255, 0)
    elif dlci < 0.6:
        return (0, 255, 255)
    else:
        return (0, 0, 255)

def draw_text(img, text, pos, color):
    x, y = pos
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    # 🔥 Improved background
    cv2.rectangle(img, (x, y - h - 10), (x + w + 10, y + 5), (30, 30, 30), -1)
    cv2.putText(img, text, (x + 5, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# -----------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    results = model.track(frame, persist=True, conf=0.4, verbose=False)[0]

    lane_density = [0] * NUM_LANES
    lane_speeds = [[] for _ in range(NUM_LANES)]

    ids = results.boxes.id.int().cpu().tolist() if results.boxes.id is not None else []

    for box, track_id in zip(results.boxes, ids):

        conf = float(box.conf[0])
        if conf < 0.4:
            continue

        cls = int(box.cls[0])
        if cls not in names:
            continue

        label = names[cls]
        weight = weights.get(label, 1)

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # SPEED
        speed = 0
        if track_history.get(track_id):
            px, py = track_history[track_id]
            dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)

            if dist > 50:
                dist = 0

            speed = dist

        track_history[track_id] = (cx, cy)

        speed = min(speed / 25, 1)
        if speed < 0.02:
            speed = 0

        if track_id not in speed_history:
            speed_history[track_id] = deque(maxlen=5)

        speed_history[track_id].append(speed)
        speed = sum(speed_history[track_id]) / len(speed_history[track_id])

        # LANES
        if cx < w // 3:
            lane_id = 0
        elif cx < 2 * w // 3:
            lane_id = 1
        else:
            lane_id = 2

        lane_density[lane_id] += weight
        lane_speeds[lane_id].append(speed)

        color = [(0,255,0), (0,200,255), (255,0,0)][lane_id]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame,
                    f"{label} ID:{track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

    # ---------------- DLCI ----------------
    lane_dlci = []

    for i in range(NUM_LANES):
        density = min(lane_density[i] / 6, 1)

        speed_avg = np.mean(lane_speeds[i]) if lane_speeds[i] else 0
        congestion = 1 - speed_avg

        if speed_avg < 0.1:
            congestion *= 1.2

        dlci = (0.75 * density) + (0.25 * congestion)
        dlci = min(dlci, 1)

        if len(lane_histories[i]) == 0:
            smooth = dlci
        else:
            smooth = 0.85 * lane_histories[i][-1] + 0.15 * dlci

        lane_histories[i].append(smooth)
        lane_dlci.append(smooth)

    # ---------------- COLOR LANES ----------------
    lane1_end = w // 3
    lane2_end = 2 * w // 3

    colors = [get_color(lane_dlci[i]) for i in range(3)]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (lane1_end, h), colors[0], -1)
    cv2.rectangle(overlay, (lane1_end, 0), (lane2_end, h), colors[1], -1)
    cv2.rectangle(overlay, (lane2_end, 0), (w, h), colors[2], -1)

    frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)

    # 🔥 Lane divider lines
    cv2.line(frame, (lane1_end, 0), (lane1_end, h), (255,255,255), 2)
    cv2.line(frame, (lane2_end, 0), (lane2_end, h), (255,255,255), 2)

    # ---------------- LABELS ----------------
    positions = [
        (lane1_end // 2, 60),
        ((lane1_end + lane2_end) // 2, 60),
        ((lane2_end + w) // 2, 60)
    ]

    for i in range(NUM_LANES):
        status = classify(lane_dlci[i])
        lane_color = get_color(lane_dlci[i])

        # 🔥 Added DLCI value
        draw_text(frame,
                  f"Lane {i+1}: {status} ({lane_dlci[i]:.2f})",
                  (positions[i][0] - 110, positions[i][1]),
                  lane_color)

    # ---------------- FPS ----------------
    curr = time.time()
    fps = 1 / (curr - prev_time) if prev_time else 0
    prev_time = curr

    draw_text(frame, f"FPS: {int(fps)}", (w - 150, 50), (0,255,255))

    cv2.imshow("FINAL DLCI SYSTEM", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()