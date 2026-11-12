import os, csv
import cv2
import numpy as np
from model_def import load_medai
from skin_detect import detect_skin

FOLDER = "test_photos"
LABELS = {
    "Acne and Rosacea Photos": "Acne / Rosacea",
    "Eczema Photos": "Eczema",
    "Psoriasis pictures Lichen Planus and related diseases": "Psoriasis",
    "Tinea Ringworm Candidiasis and other Fungal Infections": "Fungal Infection",
    "Melanoma Skin Cancer Nevi and Moles": "Melanoma",
}
KEYWORDS = {
    "acne": "Acne / Rosacea", "rosacea": "Acne / Rosacea",
    "eczema": "Eczema", "eczma": "Eczema",
    "psoriasis": "Psoriasis", "posaris": "Psoriasis",
    "ringworm": "Fungal Infection", "fungal": "Fungal Infection", "tinea": "Fungal Infection",
    "melanoma": "Melanoma", "mole": "Melanoma",
}
TONES = ["light", "medium", "indian", "dark", "brown"]

model, class_names = load_medai()
rows = []

for fn in sorted(os.listdir(FOLDER)):
    if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
        continue
    img = cv2.imdecode(np.fromfile(os.path.join(FOLDER, fn), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("cannot read:", fn)
        continue
    crop, frac, found = detect_skin(img)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224)).astype("float32")
    probs = model.predict(rgb[None, ...], verbose=0)[0]
    i = int(np.argmax(probs))
    pred = LABELS.get(class_names[i], class_names[i])
    conf = float(probs[i]) * 100

    low = fn.lower()
    truth = next((v for k, v in KEYWORDS.items() if k in low), "?")
    tone = next((t for t in TONES if t in low), "?")
    ok = "YES" if truth == pred else ("NO" if truth != "?" else "?")
    rows.append([fn[:24], truth, tone, pred, round(conf, 1), "yes" if found else "NO", ok])

print()
print("%-25s %-17s %-7s %-17s %5s %-5s %s" % ("file", "true", "tone", "predicted", "conf", "skin", "correct"))
for r in rows:
    print("%-25s %-17s %-7s %-17s %5s %-5s %s" % tuple(r))

done = [r for r in rows if r[6] in ("YES", "NO")]
if done:
    good = sum(1 for r in done if r[6] == "YES")
    print("\nOVERALL: %d / %d correct (%.0f%%)" % (good, len(done), 100 * good / len(done)))
    for t in TONES:
        sub = [r for r in done if r[2] == t]
        if sub:
            g = sum(1 for r in sub if r[6] == "YES")
            print("  %-7s %d / %d correct" % (t, g, len(sub)))

with open("results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "true", "tone", "predicted", "confidence", "skin_found", "correct"])
    w.writerows(rows)
print("\nSaved results.csv")