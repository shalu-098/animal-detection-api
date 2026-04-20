import os
import csv
import time
import uuid
from fastapi.staticfiles import StaticFiles
import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from torchvision.models import efficientnet_b4
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from alert import trigger_alert

# ===== INIT APP =====
app = FastAPI()

BASE_URL = "https://api.pahadix.in"

os.makedirs("detections", exist_ok=True)

LOG_FILE = "detections/log.csv"

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

device = torch.device("cpu")

app.mount("/detections", StaticFiles(directory="detections"), name="detections")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for testing (later restrict this)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== LOAD MODELS =====
yolo_model = YOLO("model/best.pt")

classifier = efficientnet_b4(weights=None)
classifier.classifier[1] = nn.Linear(classifier.classifier[1].in_features, 17)

classifier.load_state_dict(
    torch.load("model/animal_classifier_efficientnet.pth", map_location=device)
)

classifier.eval()

def log_detection(label, confidence, image_url, box, danger, count):
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            str(uuid.uuid4()),
            time.strftime("%Y-%m-%d %H:%M:%S"),
            label,
            f"{confidence:.2f}",
            danger,
            image_url,
            box,
            count,
            "api"
        ])

# ===== LABELS =====
class_names = [
    "Bear","Bull","Camel","Cat","Cattle","Coyote","Deer",
    "Dogs","Elephant","Fox","Goat","Horse","Hyena",
    "Jaguar","Leopard","Monkey","Pig"
]

DANGEROUS_ANIMALS = ["Leopard", "Bear", "Hyena", "Coyote"]

# ===== TRANSFORM =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ===== API ENDPOINT =====
@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    # read image
    image = Image.open(file.file).convert("RGB")
    img = np.array(image)

    results = yolo_model(img)[0]

    detections = []

    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        detections = []
        saved_image_url = None

        for box, score in zip(boxes, scores):

            if score < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box)
            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            input_tensor = transform(Image.fromarray(crop)).unsqueeze(0)

            with torch.no_grad():
                outputs = classifier(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, dim=1)

            label = class_names[pred.item()]
            confidence = conf.item()
            if confidence < 0.70:
                continue
            danger = label in DANGEROUS_ANIMALS

            if danger:
                trigger_alert(label, confidence)

            if saved_image_url is None:
                filename = f"detections/{int(time.time())}.jpg"
                cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                saved_image_url = f"{BASE_URL}/{filename}"

            image_url = saved_image_url

            # log it
            log_detection(
                label,
                confidence,
                image_url,
                [x1, y1, x2, y2],
                danger,
                1
            )

            detections.append({
                "label": label,
                "confidence": float(confidence),
                "danger": danger,
                "box": [x1, y1, x2, y2],
                "image": image_url
            })

    return {
        "detections": detections,
        "count": len(detections)
    }

@app.get("/history")
def get_history():
    logs = []

    try:
        with open(LOG_FILE, newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                logs.append(row)
    except:
        return {"history": []}

    return {
        "history": logs[::-1][:20]
    }