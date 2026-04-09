import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from torchvision.models import efficientnet_b4
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile, File
from PIL import Image

# ===== INIT APP =====
app = FastAPI()

device = torch.device("cpu")

# ===== LOAD MODELS =====
yolo_model = YOLO("model/best.pt")

classifier = efficientnet_b4(weights=None)
classifier.classifier[1] = nn.Linear(classifier.classifier[1].in_features, 17)

classifier.load_state_dict(
    torch.load("model/animal_classifier_efficientnet.pth", map_location=device)
)

classifier.eval()

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

        for box, score in zip(boxes, scores):

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

            detections.append({
                "label": label,
                "confidence": float(confidence),
                "danger": label in DANGEROUS_ANIMALS,
                "box": [x1, y1, x2, y2]
            })

    return {
        "detections": detections,
        "count": len(detections)
    }