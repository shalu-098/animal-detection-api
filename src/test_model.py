import torch
from torchvision import models

# number of classes (CHANGE THIS)
NUM_CLASSES = 17  # e.g. leopard, tiger, other

# recreate EfficientNetB4
model = models.efficientnet_b4()

# modify classifier
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

# load weights
model.load_state_dict(torch.load('model/animal_classifier_efficientnet.pth', map_location='cpu'))

model.eval()

print("✅ Model loaded successfully")

# test input (EfficientNet uses 380x380)
dummy = torch.randn(1, 3, 380, 380)

output = model(dummy)

print("Output shape:", output.shape)