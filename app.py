import os, uuid
import cv2
import numpy as np
import torch
import joblib
from PIL import Image
from flask import Flask, render_template, request
from transformers import AutoImageProcessor, AutoModel

from skin_detect import detect_skin

app = Flask(__name__)
UPLOAD_DIR = os.path.join("static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

bundle = joblib.load("dino_lr_v4.joblib")
clf, MU, SD, CLASSES = bundle["clf"], bundle["mu"], bundle["sd"], bundle["classes"]

proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dino = AutoModel.from_pretrained("facebook/dinov2-base").eval()

LOW_CONFIDENCE = 0.60
NORMAL_CONFIDENCE = 0.60

def features(rgb_array):
    im = Image.fromarray(rgb_array)
    x = proc(images=im, return_tensors="pt")["pixel_values"]
    with torch.no_grad():
        o = dino(pixel_values=x)
    f = torch.cat([o.last_hidden_state[:, 0], o.last_hidden_state[:, 1:].mean(1)], dim=1)
    return f.numpy()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        f = request.files.get("image")
        if f and f.filename:
            data = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                result = {"error": "Could not read this image. Please upload a JPG or PNG."}
            else:
                crop, frac, found = detect_skin(img)
                name = uuid.uuid4().hex[:8]
                cv2.imwrite(os.path.join(UPLOAD_DIR, name + "_orig.jpg"), img)
                cv2.imwrite(os.path.join(UPLOAD_DIR, name + "_crop.jpg"), crop)

                if not found:
                    result = {
                        "not_skin": True,
                        "orig": "uploads/" + name + "_orig.jpg",
                        "crop": "uploads/" + name + "_crop.jpg",
                        "skin_pct": frac * 100,
                    }
                    return render_template("index.html", result=result)

                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                probs = clf.predict_proba((features(rgb) - MU) / SD)[0]
                order = np.argsort(probs)[::-1]
                top = [(CLASSES[i], float(probs[i]) * 100) for i in order[:3]]
                top_name = CLASSES[order[0]]
                top_p = float(probs[order[0]])
                result = {
                    "label": top[0][0],
                    "confidence": top[0][1],
                    "top3": top,
                    "low": top_p < LOW_CONFIDENCE,
                    "pigmented": top_name == "Pigmented lesion",
                    "normal": top_name == "Normal" and top_p >= NORMAL_CONFIDENCE,
                    "skin_found": found,
                    "skin_pct": frac * 100,
                    "orig": "uploads/" + name + "_orig.jpg",
                    "crop": "uploads/" + name + "_crop.jpg",
                }
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=False, port=5000)