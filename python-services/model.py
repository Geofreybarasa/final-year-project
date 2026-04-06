import torch
import torch.nn as nn
from torchvision import models
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pth")

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Base architecture (must match training)
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)
    )

    # 🔴 STRICT: fail if missing
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")

    # Load weights safely
    state_dict = torch.load(MODEL_PATH, map_location=device)

    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print("⚠️ Strict load failed, trying relaxed mode...")
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    print("✅ Model loaded successfully")
    return model, device