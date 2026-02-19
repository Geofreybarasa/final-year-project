import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import base64
from io import BytesIO
from model import load_model
from utils.preprocess import preprocess_image
import matplotlib.pyplot as plt
import tempfile
import os

app = Flask(__name__)
CORS(app)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = load_model(device)

model, device = load_model()  # device is set inside load_model()

# Class names: 0 = fake, 1 = real
CLASS_NAMES = ['fake', 'real']


def generate_saliency_map(model, input_tensor):
    """Generate a saliency map for the given input tensor."""
    input_copy = input_tensor.clone().detach().requires_grad_(True)

    output = model(input_copy)
    score = output.max()

    model.zero_grad()
    score.backward()

    saliency = input_copy.grad.data.abs()
    saliency, _ = torch.max(saliency, dim=1)
    saliency = saliency.squeeze().cpu().numpy()

    # Normalize to [0, 1]
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    return saliency


def saliency_to_base64(saliency: np.ndarray, original_image_path: str) -> str:
    """Overlay saliency map on the original image and return as base64 string."""
    orig = cv2.imread(original_image_path)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    # Resize saliency to match original image
    saliency_resized = cv2.resize(saliency, (orig.shape[1], orig.shape[0]))

    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * saliency_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend with original
    overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

    # Convert to base64
    pil_img = Image.fromarray(overlay)
    buffer = BytesIO()
    pil_img.save(buffer, format='JPEG')
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return f"data:image/jpeg;base64,{encoded}"


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    temp_path = os.path.join(tempfile.gettempdir(), 'temp_image.jpg')
    file.save(temp_path)

    # Preprocess
    input_tensor = preprocess_image(temp_path).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)

    pred_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_idx].item() * 100

    # Generate saliency map (requires grad, so done outside no_grad block)
    saliency = generate_saliency_map(model, input_tensor)
    heatmap_b64 = saliency_to_base64(saliency, temp_path)

    result = {
        'prediction': CLASS_NAMES[pred_idx],
        'confidence': round(confidence, 2),
        'heatmap': heatmap_b64
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)