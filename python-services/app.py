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
import tempfile
import os
import torchvision.transforms as transforms
import traceback

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ─── LOAD MODEL ───────────────────────────────────────────────
try:
    model, device = load_model()
    CLASS_NAMES = ['fake', 'real']
    model.eval()
    print(f"✅ Model loaded successfully on {device}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    device = 'cpu'

# ─── OPENCV CASCADES (Full set from code b) ───────────────────
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
_eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

_nose_cascade_path = cv2.data.haarcascades + 'haarcascade_mcs_nose.xml'
_mouth_cascade_path = cv2.data.haarcascades + 'haarcascade_mcs_mouth.xml'
_nose_cascade = cv2.CascadeClassifier(_nose_cascade_path) if os.path.exists(_nose_cascade_path) else None
_mouth_cascade = cv2.CascadeClassifier(_mouth_cascade_path) if os.path.exists(_mouth_cascade_path) else None

print("✅ OpenCV cascades loaded successfully")

# ─── PREPROCESS (Improved from code a) ────────────────────────
def preprocess_image(image_path):
    """Preprocess image for model inference"""
    try:
        image = Image.open(image_path).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        return transform(image).unsqueeze(0)
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        raise

# ─── GRAD-CAM (Enhanced from code b) ──────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
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
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam

# Initialize GradCAM safely
gradcam = None
if model:
    try:
        gradcam = GradCAM(model, target_layer=model.layer4[-1])
        print("✅ GradCAM initialized successfully")
    except Exception as e:
        print(f"⚠️ GradCAM init failed: {e}")

# ─── ANALYSIS FUNCTIONS (Enhanced from code b) ─────────────────
def run_gradcam(cam, orig_rgb):
    """Generate Grad-CAM heatmap overlay"""
    try:
        h, w = orig_rgb.shape[:2]
        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
        cam_smooth = cv2.GaussianBlur(cam_resized, (11, 11), 0)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam_smooth), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        alpha = cam_smooth[:, :, np.newaxis]
        blended = (orig_rgb * (1 - alpha * 0.7) + heatmap * alpha * 0.7).astype(np.uint8)

        # Draw contour lines at different thresholds
        for threshold in [0.4, 0.6, 0.8]:
            binary = (cam_smooth >= threshold).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = (255, 255, 255) if threshold == 0.8 else (200, 200, 200)
            cv2.drawContours(blended, contours, -1, color, 1)

        return blended
    except Exception as e:
        print(f"❌ Error in run_gradcam: {e}")
        return orig_rgb

def run_fft(orig_rgb):
    """Generate FFT frequency domain analysis"""
    try:
        h, w = orig_rgb.shape[:2]
        gray = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log1p(np.abs(fft_shift))

        mag_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        mag_uint8 = np.uint8(255 * mag_norm)

        fft_color = cv2.applyColorMap(mag_uint8, cv2.COLORMAP_INFERNO)
        fft_color = cv2.cvtColor(fft_color, cv2.COLOR_BGR2RGB)
        fft_color = cv2.resize(fft_color, (w, h))

        # Add crosshair and circles
        cx, cy = w // 2, h // 2
        cv2.line(fft_color, (cx - 20, cy), (cx + 20, cy), (255, 255, 255), 1)
        cv2.line(fft_color, (cx, cy - 20), (cx, cy + 20), (255, 255, 255), 1)

        for r in [int(min(h, w) * f) for f in [0.1, 0.2, 0.35, 0.5]]:
            cv2.circle(fft_color, (cx, cy), r, (80, 80, 80), 1)

        return fft_color
    except Exception as e:
        print(f"❌ Error in run_fft: {e}")
        return orig_rgb

def run_ela(image_path, orig_rgb, quality=90):
    """Error Level Analysis for compression artifacts"""
    try:
        pil_orig = Image.fromarray(orig_rgb)
        buf = BytesIO()
        pil_orig.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        compressed = np.array(Image.open(buf).convert('RGB'))

        ela = np.abs(orig_rgb.astype(np.float32) - compressed.astype(np.float32))
        ela_scaled = np.clip(ela * 15, 0, 255).astype(np.uint8)

        ela_gray = cv2.cvtColor(ela_scaled, cv2.COLOR_RGB2GRAY)
        ela_color = cv2.applyColorMap(ela_gray, cv2.COLORMAP_JET)
        ela_color = cv2.cvtColor(ela_color, cv2.COLOR_BGR2RGB)

        # Highlight high-error regions
        ela_norm = ela_gray.astype(np.float32) / 255.0
        threshold = np.percentile(ela_norm, 85)
        hotspot_mask = (ela_norm >= threshold).astype(np.uint8) * 255
        hotspot_mask = cv2.dilate(hotspot_mask, np.ones((5, 5), np.uint8), iterations=1)

        contours, _ = cv2.findContours(hotspot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = orig_rgb.shape[:2]
        significant = [c for c in contours if cv2.contourArea(c) > (h * w * 0.001)]
        cv2.drawContours(ela_color, significant, -1, (255, 255, 255), 2)

        return ela_color
    except Exception as e:
        print(f"❌ Error in run_ela: {e}")
        return orig_rgb

def run_face_mesh(orig_rgb):
    """Advanced facial structure analysis"""
    try:
        h, w = orig_rgb.shape[:2]
        canvas = orig_rgb.copy()
        gray = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY)

        faces = _face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if len(faces) == 0:
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (8, 8, 18), -1)
            canvas = cv2.addWeighted(canvas, 0.3, overlay, 0.7, 0)
            cv2.putText(canvas, '⚠ NO FACE DETECTED', (w//2 - 110, h//2 - 10),
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

            # Eye detection
            eye_roi = gray[fy: fy + fh//2, fx: fx+fw]
            eyes = _eye_cascade.detectMultiScale(eye_roi, scaleFactor=1.1, minNeighbors=5)
            eye_centers = []
            for (ex, ey, ew, eh) in eyes[:2]:
                cx = fx + ex + ew//2
                cy = fy + ey + eh//2
                eye_centers.append((cx, cy))
                cv2.ellipse(canvas, (cx, cy), (ew//2, eh//2), 0, 0, 360, (255, 100, 100), 2)
                cv2.circle(canvas, (cx, cy), 2, (255, 255, 255), -1)

            # Eye alignment
            if len(eye_centers) == 2:
                cv2.line(canvas, eye_centers[0], eye_centers[1], (255, 200, 0), 1)
                dx = eye_centers[1][0] - eye_centers[0][0]
                dy = eye_centers[1][1] - eye_centers[0][1]
                angle = abs(np.degrees(np.arctan2(dy, max(abs(dx), 1))))
                a_col = (0, 255, 100) if angle < 10 else (255, 80, 80)
                cv2.putText(canvas, f'Eye tilt: {angle:.1f}°',
                            (fx, max(fy - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, a_col, 1)

            # Nose detection
            if _nose_cascade is not None and not _nose_cascade.empty():
                nose_roi = gray[fy + fh//4: fy + 3*fh//4, fx: fx+fw]
                if nose_roi.size > 0:
                    noses = _nose_cascade.detectMultiScale(nose_roi, scaleFactor=1.1, minNeighbors=4)
                    for (nx, ny, nw, nh) in noses[:1]:
                        abs_ny = fy + fh//4 + ny
                        cv2.rectangle(canvas, (fx+nx, abs_ny), (fx+nx+nw, abs_ny+nh), (0, 200, 255), 1)

            # Mouth detection
            if _mouth_cascade is not None and not _mouth_cascade.empty():
                mouth_roi = gray[fy + fh//2: fy+fh, fx: fx+fw]
                if mouth_roi.size > 0:
                    mouths = _mouth_cascade.detectMultiScale(mouth_roi, scaleFactor=1.1, minNeighbors=6)
                    for (mx, my, mw, mh) in mouths[:1]:
                        abs_my = fy + fh//2 + my
                        cv2.rectangle(canvas, (fx+mx, abs_my), (fx+mx+mw, abs_my+mh), (255, 200, 0), 1)

            # Symmetry score
            img_mid = w // 2
            face_mid = fx + fw // 2
            offset = abs(face_mid - img_mid)
            sym_score = max(0.0, 100.0 - (offset / max(w / 2, 1)) * 100.0)
            sym_color = (0, 255, 100) if sym_score > 70 else (255, 80, 80)
            cv2.putText(canvas, f'Symmetry: {sym_score:.1f}%',
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.52, sym_color, 2)

        return canvas
    except Exception as e:
        print(f"❌ Error in run_face_mesh: {e}")
        return orig_rgb

# ─── FORENSIC PANEL COMPOSITION (Enhanced from code b) ─────────
def compose_forensic_panel(gradcam_img, fft_img, ela_img, mesh_img, orig_rgb, prediction, confidence):
    """Create 2x3 grid forensic analysis panel"""
    try:
        TH, TW = 300, 300

        def resize_tile(img):
            return cv2.resize(img, (TW, TH))

        def add_label(img, text, sub=''):
            out = img.copy()
            # Top label bar
            cv2.rectangle(out, (0, 0), (TW, 28), (10, 10, 20), -1)
            cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 229, 255), 1)
            # Bottom sublabel (if provided)
            if sub:
                cv2.rectangle(out, (0, TH - 22), (TW, TH), (10, 10, 20), -1)
                cv2.putText(out, sub, (6, TH - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 160), 1)
            return out

        # Prepare all tiles
        t_orig = add_label(resize_tile(orig_rgb), 'ORIGINAL IMAGE', 'input')
        t_gradcam = add_label(resize_tile(gradcam_img), 'GRAD-CAM', 'model attention')
        t_fft = add_label(resize_tile(fft_img), 'FFT FREQUENCY DOMAIN', 'GAN artifact detection')
        t_ela = add_label(resize_tile(ela_img), 'ERROR LEVEL ANALYSIS', 'compression forensics')
        t_mesh = add_label(resize_tile(mesh_img), 'FACIAL STRUCTURE', 'geometric analysis')

        # Verdict tile
        is_fake = prediction == 'fake'
        v_color = (255, 80, 80) if is_fake else (0, 220, 120)
        verdict_txt = 'DEEPFAKE' if is_fake else 'AUTHENTIC'
        risk_txt = '🔴 HIGH RISK' if confidence > 80 else ('🟡 MEDIUM RISK' if confidence > 60 else '🟢 LOW RISK')

        vt = np.zeros((TH, TW, 3), dtype=np.uint8)
        vt[:] = (12, 12, 24)

        # Verdict text
        cv2.putText(vt, 'VERDICT', (TW//2 - 38, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120,120,140), 1)
        cv2.putText(vt, verdict_txt, (TW//2 - 60, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.85, v_color, 2)
        cv2.putText(vt, f'{confidence:.1f}%', (TW//2 - 35, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
        cv2.putText(vt, 'CONFIDENCE', (TW//2 - 50, 182), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120,120,140), 1)
        cv2.putText(vt, risk_txt, (TW//2 - 48, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.52, v_color, 1)

        # Confidence bar
        bx, by, bw, bh = 20, 245, TW - 40, 10
        cv2.rectangle(vt, (bx, by), (bx + bw, by + bh), (35, 35, 50), -1)
        fill = int(bw * min(confidence, 100) / 100)
        cv2.rectangle(vt, (bx, by), (bx + fill, by + bh), v_color, -1)
        cv2.line(vt, (20, 270), (TW - 20, 270), (40, 40, 60), 1)

        # Analysis labels at bottom
        for i, tag in enumerate(['GRAD-CAM', 'FFT', 'ELA', 'FACIAL']):
            cv2.putText(vt, tag, (20 + i * 65, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 100), 1)

        # Assemble grid
        gap = 3
        gap_v = np.zeros((TH, gap, 3), dtype=np.uint8)
        gap_h = np.zeros((gap, TW * 3 + gap * 2, 3), dtype=np.uint8)

        row1 = np.concatenate([t_orig, gap_v, t_gradcam, gap_v, t_fft], axis=1)
        row2 = np.concatenate([t_ela, gap_v, t_mesh, gap_v, vt], axis=1)
        grid = np.concatenate([row1, gap_h, row2], axis=0)

        # Header bar
        header = np.zeros((44, grid.shape[1], 3), dtype=np.uint8)
        header[:] = (6, 8, 16)
        cv2.putText(header, 'VERITAS  //  FORENSIC DEEPFAKE ANALYSIS PANEL',
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 229, 255), 1)
        cv2.putText(header, '4-CHANNEL ANALYSIS  |  MODEL: ResNet18',
                    (grid.shape[1] - 295, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 120), 1)

        final = np.concatenate([header, grid], axis=0)

        # Convert to base64
        pil_img = Image.fromarray(final)
        buf = BytesIO()
        pil_img.save(buf, format='JPEG', quality=95)
        encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded}"
        
    except Exception as e:
        print(f"❌ Error in compose_forensic_panel: {e}")
        traceback.print_exc()
        return None

# ─── PREDICT ROUTE (Enhanced from code b) ──────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    print("📥 Received prediction request")
    
    if 'image' not in request.files:
        print("❌ No image in request")
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    temp_path = None
    
    try:
        # Save temporary file
        temp_path = os.path.join(tempfile.gettempdir(), f'temp_image_{os.urandom(8).hex()}.jpg')
        file.save(temp_path)
        print(f"💾 Image saved to {temp_path}")
        
        # Read image
        orig = cv2.imread(temp_path)
        if orig is None:
            print("❌ Failed to read image")
            return jsonify({'error': 'Invalid image file'}), 400
            
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        print(f"📷 Image loaded: {orig_rgb.shape}")
        
        # Check if model is loaded
        if model is None:
            print("❌ Model not loaded")
            return jsonify({'error': 'Model not loaded. Check best_model.pth file.'}), 500
        
        # Inference
        print("🧠 Running inference...")
        input_tensor = preprocess_image(temp_path).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)

        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item() * 100
        prediction = CLASS_NAMES[pred_idx]
        
        print(f"🎯 Prediction: {prediction}, Confidence: {confidence:.2f}%")
        
        # Generate Grad-CAM
        gradcam_img = orig_rgb
        if gradcam:
            print("🎨 Generating Grad-CAM...")
            input_grad = preprocess_image(temp_path).to(device)
            cam = gradcam.generate(input_grad, class_idx=pred_idx)
            gradcam_img = run_gradcam(cam, orig_rgb)
        
        # Generate all analyses
        print("🔬 Generating analysis panels...")
        fft_img = run_fft(orig_rgb)
        ela_img = run_ela(temp_path, orig_rgb)
        mesh_img = run_face_mesh(orig_rgb)
        
        # Compose final panel
        print("📊 Composing final panel...")
        panel_b64 = compose_forensic_panel(
            gradcam_img, fft_img, ela_img, mesh_img,
            orig_rgb, prediction, confidence
        )
        
        if panel_b64 is None:
            print("❌ Failed to compose panel")
            return jsonify({'error': 'Failed to generate analysis panel'}), 500
        
        result = {
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'heatmap': panel_b64
        }
        
        print(f"✅ Success! Returning result: {result['prediction']}, {result['confidence']}%")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error in prediction: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
        
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"🗑️ Cleaned up {temp_path}")

# ─── HEALTH CHECK ROUTE ───────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device),
        'cascades_loaded': {
            'face': _face_cascade is not None,
            'eye': _eye_cascade is not None,
            'nose': _nose_cascade is not None,
            'mouth': _mouth_cascade is not None
        }
    })

# ─── RUN SERVER ───────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting VERITAS Forensic Analysis Server")
    print("=" * 60)
    print(f"📍 Model status: {'✅ Loaded' if model else '❌ Not loaded'}")
    print(f"💻 Device: {device}")
    print(f"🌐 Server running on http://0.0.0.0:5000")
    print(f"🔍 Health check: http://localhost:5000/health")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)