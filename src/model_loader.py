import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision.models import efficientnet_b4

# ===== DEVICE =====
device = torch.device("cpu")

# ===== GLOBAL VARIABLES (singleton pattern) =====
yolo_model = None
classifier = None


def load_models():
    global yolo_model, classifier

    # Prevent reloading
    if yolo_model is not None and classifier is not None:
        return yolo_model, classifier

    print("🔄 Loading models...")

    # ===== YOLO =====
    yolo_model = YOLO("model/best.pt")

    # ===== CLASSIFIER =====
    classifier = efficientnet_b4(weights=None)

    # Safe replacement of final layer
    if isinstance(classifier.classifier[1], nn.Linear):
        in_features = classifier.classifier[1].in_features
        classifier.classifier[1] = nn.Linear(in_features, 17) # type: ignore

    classifier.load_state_dict(
        torch.load("model/animal_classifier_efficientnet.pth", map_location=device)
    )

    classifier = classifier.to(device)
    classifier.eval()

    print("✅ Models loaded successfully")

    return yolo_model, classifier