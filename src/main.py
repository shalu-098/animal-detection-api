import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision.models import efficientnet_b4
import torchvision.transforms as transforms
import time
import winsound
import os
import csv
import uuid
import json
from alert import trigger_alert

BASE_URL = "https://api.pahadix.in"

# ===== SETUP =====
device = torch.device("cpu")

os.makedirs("detections", exist_ok=True)

LOG_FILE = "detections/log.csv"

# create CSV file if not exists
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "id",
            "timestamp",
            "animal",
            "confidence",
            "danger",
            "image",
            "box",
            "detection_count",
            "location"
        ])

# ===== LOAD MODELS =====
yolo_model = YOLO("model/best.pt")

classifier = efficientnet_b4(weights=None)
classifier.classifier[1] = nn.Linear(classifier.classifier[1].in_features, 17) # type: ignore

classifier.load_state_dict(
    torch.load("model/animal_classifier_efficientnet.pth", map_location=device)
)

classifier = classifier.to(device)
classifier.eval()

# ===== CLASS NAMES =====
class_names = [
    "Bear","Bull","Camel","Cat","Cattle","Coyote","Deer",
    "Dogs","Elephant","Fox","Goat","Horse","Hyena",
    "Jaguar","Leopard","Monkey","Pig"
]

DANGEROUS_ANIMALS = ["Leopard", "Bear", "Hyena", "Coyote"]

# ===== CONFIG =====
CONF_THRESHOLD = 0.6
CLS_THRESHOLD = 0.7
PADDING = 10
COOLDOWN = 5

# ===== ALERT STATE =====
detection_counter = 0
last_alert_time = 0
last_confidence = 0
last_box = None

# ===== TRANSFORM =====
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def log_json(data):
    with open("detections/log.json", "a") as f:
        f.write(json.dumps(data) + "\n")

# ===== LOG FUNCTION =====
def log_detection(label, confidence, image_path, box, danger, count):
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            str(uuid.uuid4()),  # unique id
            time.strftime("%Y-%m-%d %H:%M:%S"),
            label,
            f"{confidence:.2f}",
            danger,
            image_path,
            box,
            count,
            "unknown"  # placeholder for GPS later
        ])

# ===== CAMERA =====
cap = cv2.VideoCapture(0)

# ===== FPS TRACKING =====
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detected_this_frame = False
    detected_label = None

    results = yolo_model(frame)[0]

    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()

        for box, score in zip(boxes, scores):

            if score < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box)

            x1 = max(0, x1 - PADDING)
            y1 = max(0, y1 - PADDING)
            x2 = min(frame.shape[1], x2 + PADDING)
            y2 = min(frame.shape[0], y2 + PADDING)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            input_tensor = transform(crop_rgb).unsqueeze(0).to(device) # type: ignore

            with torch.no_grad():
                outputs = classifier(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, dim=1)

            conf = conf.item()
            pred = pred.item()

            label_name = class_names[pred] # type: ignore

            if conf < CLS_THRESHOLD:
                label = "Unknown"
            else:
                label = f"{label_name} {conf:.2f}"

                if label_name in DANGEROUS_ANIMALS:
                    detected_this_frame = True
                    detected_label = label_name
                    last_confidence = conf
                    last_box = [x1, y1, x2, y2]

            # Draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # ===== ALERT LOGIC =====
    if detected_this_frame:
        detection_counter += 1
    else:
        detection_counter = 0

    if detection_counter >= 5 and time.time() - last_alert_time > COOLDOWN:
        trigger_alert(detected_label, last_confidence)

        filename = f"detections/{detected_label}_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)

        image_url = f"{BASE_URL}/{filename}"

        log_detection(
            detected_label,
            last_confidence,
            image_url,   # ✅ now accessible in UI
            last_box,
            True,
            detection_counter
        )
        log_json({
    "animal": detected_label,
    "confidence": last_confidence,
    "box": last_box,
    "time": time.time()
})
        last_alert_time = time.time()
        detection_counter = 0

    # ===== FPS DISPLAY =====
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("Animal Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()