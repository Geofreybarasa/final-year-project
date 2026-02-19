import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
import base64
from io import BytesIO
from model import load_model
from utils.preprocess import preprocess_image
import tempfile
import os

app = Flask(__name__)
CORS(app)

model, device = load_model()
CLASS_NAMES = ['fake', 'real']

model.eval()
for name, param in model.named_parameters():
    param.requires_grad = 'layer4' in name

# ─── OpenCV Cascades (safe loading) ───────────────────────────
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
_eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

_nose_cascade_path  = cv2.data.haarcascades + 'haarcascade_mcs_nose.xml'
_mouth_cascade_path = cv2.data.haarcascades + 'haarcascade_mcs_mouth.xml'
_nose_cascade  = cv2.CascadeClassifier(_nose_cascade_path)  if os.path.exists(_nose_cascade_path)  else None
_mouth_cascade = cv2.CascadeClassifier(_mouth_cascade_path) if os.path.exists(_mouth_cascade_path) else None


# ─── Grad-CAM ─────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.gradients    = None
        self.activations  = None
        self.hook_handles = []
        self.hook_handles.append(target_layer.register_forward_hook(self._save_activation))
        self.hook_handles.append(target_layer.register_full_backward_hook(self._save_gradient))

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[0, class_idx].backward()
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam     = cam.squeeze().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


gradcam = GradCAM(model, target_layer=model.layer4[-1])


# ══════════════════════════════════════════════════════════════
# ANALYSIS 1 — GRAD-CAM
# ══════════════════════════════════════════════════════════════
def run_gradcam(cam: np.ndarray, orig_rgb: np.ndarray) -> np.ndarray:
    h, w        = orig_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_smooth  = cv2.GaussianBlur(cam_resized, (11, 11), 0)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_smooth), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    alpha   = cam_smooth[:, :, np.newaxis]
    blended = (orig_rgb * (1 - alpha * 0.7) + heatmap * alpha * 0.7).astype(np.uint8)

    for threshold in [0.4, 0.6, 0.8]:
        binary      = (cam_smooth >= threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color       = (255, 255, 255) if threshold == 0.8 else (200, 200, 200)
        cv2.drawContours(blended, contours, -1, color, 1)

    return blended


# ══════════════════════════════════════════════════════════════
# ANALYSIS 2 — FFT FREQUENCY DOMAIN
# ══════════════════════════════════════════════════════════════
def run_fft(orig_rgb: np.ndarray) -> np.ndarray:
    h, w = orig_rgb.shape[:2]

    gray      = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    fft       = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shift))

    mag_norm  = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    mag_uint8 = np.uint8(255 * mag_norm)

    fft_color = cv2.applyColorMap(mag_uint8, cv2.COLORMAP_INFERNO)
    fft_color = cv2.cvtColor(fft_color, cv2.COLOR_BGR2RGB)
    fft_color = cv2.resize(fft_color, (w, h))

    cx, cy = w // 2, h // 2
    cv2.line(fft_color, (cx - 20, cy), (cx + 20, cy), (255, 255, 255), 1)
    cv2.line(fft_color, (cx, cy - 20), (cx, cy + 20), (255, 255, 255), 1)

    for r in [int(min(h, w) * f) for f in [0.1, 0.2, 0.35, 0.5]]:
        cv2.circle(fft_color, (cx, cy), r, (80, 80, 80), 1)

    anomaly_mask = np.zeros_like(mag_norm)
    anomaly_mask[:h//6, :w//6]   = mag_norm[:h//6, :w//6]
    anomaly_mask[:h//6, -w//6:]  = mag_norm[:h//6, -w//6:]
    anomaly_mask[-h//6:, :w//6]  = mag_norm[-h//6:, :w//6]
    anomaly_mask[-h//6:, -w//6:] = mag_norm[-h//6:, -w//6:]

    anomaly_norm  = (anomaly_mask - anomaly_mask.min()) / (anomaly_mask.max() - anomaly_mask.min() + 1e-8)
    anomaly_color = cv2.applyColorMap(np.uint8(255 * anomaly_norm), cv2.COLORMAP_HOT)
    anomaly_color = cv2.cvtColor(anomaly_color, cv2.COLOR_BGR2RGB)
    anomaly_color = cv2.resize(anomaly_color, (w, h))

    result = np.clip(
        fft_color.astype(np.float32) + anomaly_color.astype(np.float32) * 0.6,
        0, 255
    ).astype(np.uint8)

    return result


# ══════════════════════════════════════════════════════════════
# ANALYSIS 3 — ERROR LEVEL ANALYSIS (ELA)
# ══════════════════════════════════════════════════════════════
def run_ela(image_path: str, orig_rgb: np.ndarray, quality: int = 90) -> np.ndarray:
    h, w = orig_rgb.shape[:2]

    pil_orig = Image.fromarray(orig_rgb)
    buf      = BytesIO()
    pil_orig.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    compressed = np.array(Image.open(buf).convert('RGB'))

    ela        = np.abs(orig_rgb.astype(np.float32) - compressed.astype(np.float32))
    ela_scaled = np.clip(ela * 15, 0, 255).astype(np.uint8)

    ela_gray  = cv2.cvtColor(ela_scaled, cv2.COLOR_RGB2GRAY)
    ela_color = cv2.applyColorMap(ela_gray, cv2.COLORMAP_JET)
    ela_color = cv2.cvtColor(ela_color, cv2.COLOR_BGR2RGB)

    ela_norm     = ela_gray.astype(np.float32) / 255.0
    threshold    = np.percentile(ela_norm, 85)
    hotspot_mask = (ela_norm >= threshold).astype(np.uint8) * 255
    hotspot_mask = cv2.dilate(hotspot_mask, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(hotspot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant = [c for c in contours if cv2.contourArea(c) > (h * w * 0.001)]
    cv2.drawContours(ela_color, significant, -1, (255, 255, 255), 2)

    return ela_color


# ══════════════════════════════════════════════════════════════
# ANALYSIS 4 — FACIAL STRUCTURE (OpenCV only, no MediaPipe)
# ══════════════════════════════════════════════════════════════
def run_face_mesh(orig_rgb: np.ndarray) -> np.ndarray:
    h, w   = orig_rgb.shape[:2]
    canvas = orig_rgb.copy()
    gray   = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY)

    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (8, 8, 18), -1)
        canvas = cv2.addWeighted(canvas, 0.3, overlay, 0.7, 0)
        cv2.putText(canvas, 'NO FACE DETECTED', (w//2 - 110, h//2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(canvas, 'geometric analysis unavailable', (w//2 - 120, h//2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)
        return canvas

    for (fx, fy, fw, fh) in faces:
        # Face bounding box
        cv2.rectangle(canvas, (fx, fy), (fx+fw, fy+fh), (0, 255, 150), 2)

        # Mesh grid inside face
        for i in range(1, 8):
            x = fx + int(fw * i / 8)
            cv2.line(canvas, (x, fy), (x, fy+fh), (0, 160, 160), 1)
        for i in range(1, 8):
            y = fy + int(fh * i / 8)
            cv2.line(canvas, (fx, y), (fx+fw, y), (0, 160, 160), 1)

        # Facial thirds
        for t in [1, 2]:
            y = fy + (fh // 3) * t
            cv2.line(canvas, (fx, y), (fx+fw, y), (255, 200, 0), 1)

        # Symmetry axis
        mid_x = fx + fw // 2
        cv2.line(canvas, (mid_x, fy), (mid_x, fy+fh), (100, 100, 255), 1)

        # Eye detection (restrict to top half of face ROI)
        eye_roi     = gray[fy: fy + fh//2, fx: fx+fw]
        eyes        = _eye_cascade.detectMultiScale(eye_roi, scaleFactor=1.1, minNeighbors=5)
        eye_centers = []
        for (ex, ey, ew, eh) in eyes[:2]:
            cx = fx + ex + ew//2
            cy = fy + ey + eh//2
            eye_centers.append((cx, cy))
            cv2.ellipse(canvas, (cx, cy), (ew//2, eh//2), 0, 0, 360, (255, 100, 100), 2)
            cv2.circle(canvas, (cx, cy), 2, (255, 255, 255), -1)
            cv2.rectangle(canvas, (fx+ex, fy+ey), (fx+ex+ew, fy+ey+eh), (255, 80, 80), 1)

        # Eye alignment + tilt
        if len(eye_centers) == 2:
            cv2.line(canvas, eye_centers[0], eye_centers[1], (255, 200, 0), 1)
            dx    = eye_centers[1][0] - eye_centers[0][0]
            dy    = eye_centers[1][1] - eye_centers[0][1]
            angle = abs(np.degrees(np.arctan2(dy, max(abs(dx), 1))))
            a_col = (0, 255, 100) if angle < 10 else (255, 80, 80)
            cv2.putText(canvas, f'Eye tilt: {angle:.1f}deg',
                        (fx, max(fy - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, a_col, 1)

        # Nose detection (only if cascade loaded successfully)
        if _nose_cascade is not None and not _nose_cascade.empty():
            nose_roi = gray[fy + fh//4: fy + 3*fh//4, fx: fx+fw]
            if nose_roi.size > 0:
                noses = _nose_cascade.detectMultiScale(
                    nose_roi, scaleFactor=1.1, minNeighbors=4
                )
                for (nx, ny, nw, nh) in noses[:1]:
                    abs_ny = fy + fh//4 + ny
                    cv2.rectangle(canvas,
                                  (fx+nx, abs_ny), (fx+nx+nw, abs_ny+nh),
                                  (0, 200, 255), 1)

        # Mouth detection (only if cascade loaded successfully)
        if _mouth_cascade is not None and not _mouth_cascade.empty():
            mouth_roi = gray[fy + fh//2: fy+fh, fx: fx+fw]
            if mouth_roi.size > 0:
                mouths = _mouth_cascade.detectMultiScale(
                    mouth_roi, scaleFactor=1.1, minNeighbors=6
                )
                for (mx, my, mw, mh) in mouths[:1]:
                    abs_my = fy + fh//2 + my
                    cv2.rectangle(canvas,
                                  (fx+mx, abs_my), (fx+mx+mw, abs_my+mh),
                                  (255, 200, 0), 1)

        # Symmetry score
        img_mid   = w // 2
        face_mid  = fx + fw // 2
        offset    = abs(face_mid - img_mid)
        sym_score = max(0.0, 100.0 - (offset / max(w / 2, 1)) * 100.0)
        sym_color = (0, 255, 100) if sym_score > 70 else (255, 80, 80)
        cv2.putText(canvas, f'Symmetry: {sym_score:.1f}%',
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.52, sym_color, 2)

    return canvas


# ══════════════════════════════════════════════════════════════
# COMPOSE FORENSIC PANEL
# ══════════════════════════════════════════════════════════════
def compose_forensic_panel(gradcam_img, fft_img, ela_img, mesh_img,
                            orig_rgb, prediction, confidence) -> str:
    TH, TW = 300, 300

    def resize_tile(img):
        return cv2.resize(img, (TW, TH))

    def add_label(img, text, sub=''):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (TW, 28), (10, 10, 20), -1)
        cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 229, 255), 1)
        if sub:
            cv2.rectangle(out, (0, TH - 22), (TW, TH), (10, 10, 20), -1)
            cv2.putText(out, sub, (6, TH - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 160), 1)
        return out

    t_orig    = add_label(resize_tile(orig_rgb),    'ORIGINAL IMAGE',       'input')
    t_gradcam = add_label(resize_tile(gradcam_img), 'GRAD-CAM',             'model attention')
    t_fft     = add_label(resize_tile(fft_img),     'FFT FREQUENCY DOMAIN', 'GAN artifact detection')
    t_ela     = add_label(resize_tile(ela_img),     'ERROR LEVEL ANALYSIS', 'compression forensics')
    t_mesh    = add_label(resize_tile(mesh_img),    'FACIAL STRUCTURE',     'geometric analysis')

    # Verdict tile
    is_fake     = prediction == 'fake'
    v_color     = (255, 80, 80)  if is_fake else (0, 220, 120)
    verdict_txt = 'DEEPFAKE'     if is_fake else 'AUTHENTIC'
    risk_txt    = 'HIGH RISK'    if confidence > 80 else ('MEDIUM RISK' if confidence > 60 else 'LOW RISK')

    vt      = np.zeros((TH, TW, 3), dtype=np.uint8)
    vt[:]   = (12, 12, 24)

    cv2.putText(vt, 'VERDICT',
                (TW//2 - 38, 55),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120,120,140), 1)
    cv2.putText(vt, verdict_txt,
                (TW//2 - 60, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.85, v_color,       2)
    cv2.putText(vt, f'{confidence:.1f}%',
                (TW//2 - 35, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
    cv2.putText(vt, 'CONFIDENCE',
                (TW//2 - 50, 182), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120,120,140), 1)
    cv2.putText(vt, risk_txt,
                (TW//2 - 48, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.52, v_color,       1)

    # Confidence bar
    bx, by, bw, bh = 20, 245, TW - 40, 10
    cv2.rectangle(vt, (bx, by), (bx + bw, by + bh), (35, 35, 50), -1)
    fill = int(bw * min(confidence, 100) / 100)
    cv2.rectangle(vt, (bx, by), (bx + fill, by + bh), v_color, -1)
    cv2.line(vt, (20, 270), (TW - 20, 270), (40, 40, 60), 1)

    for i, tag in enumerate(['GRAD-CAM', 'FFT', 'ELA', 'FACIAL']):
        cv2.putText(vt, tag, (20 + i * 65, 290),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 100), 1)

    # 2×3 grid with gaps
    gap   = 3
    gap_v = np.zeros((TH, gap, 3), dtype=np.uint8)
    gap_h = np.zeros((gap, TW * 3 + gap * 2, 3), dtype=np.uint8)

    row1 = np.concatenate([t_orig,  gap_v, t_gradcam, gap_v, t_fft], axis=1)
    row2 = np.concatenate([t_ela,   gap_v, t_mesh,    gap_v, vt],    axis=1)
    grid = np.concatenate([row1, gap_h, row2], axis=0)

    # Header bar
    header      = np.zeros((44, grid.shape[1], 3), dtype=np.uint8)
    header[:]   = (6, 8, 16)
    cv2.putText(header, 'VERITAS  //  FORENSIC DEEPFAKE ANALYSIS PANEL',
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 229, 255), 1)
    cv2.putText(header, '4-CHANNEL ANALYSIS  |  MODEL: ResNet18',
                (grid.shape[1] - 295, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 120), 1)

    final = np.concatenate([header, grid], axis=0)

    pil_img = Image.fromarray(final)
    buf     = BytesIO()
    pil_img.save(buf, format='JPEG', quality=95)
    encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded}"


# ─── Predict route ─────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file      = request.files['image']
    temp_path = os.path.join(tempfile.gettempdir(), 'temp_image.jpg')
    file.save(temp_path)

    orig     = cv2.imread(temp_path)
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    # Inference
    input_tensor = preprocess_image(temp_path).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs   = torch.softmax(outputs, dim=1)

    pred_idx   = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_idx].item() * 100

    # Grad-CAM — separate forward pass with layer4 grad enabled
    input_grad = preprocess_image(temp_path).to(device)
    cam        = gradcam.generate(input_grad, class_idx=pred_idx)

    # Four analyses
    gradcam_img = run_gradcam(cam, orig_rgb)
    fft_img     = run_fft(orig_rgb)
    ela_img     = run_ela(temp_path, orig_rgb)
    mesh_img    = run_face_mesh(orig_rgb)

    panel_b64 = compose_forensic_panel(
        gradcam_img, fft_img, ela_img, mesh_img,
        orig_rgb, CLASS_NAMES[pred_idx], confidence
    )

    return jsonify({
        'prediction': CLASS_NAMES[pred_idx],
        'confidence': round(confidence, 2),
        'heatmap':    panel_b64
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)