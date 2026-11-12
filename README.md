
# MedAI - Skin Condition Screening (Early Prototype)

A Flask web app where users upload a skin photo and receive a screening suggestion with a confidence score.

This is a student prototype for learning and research. It is NOT a medical diagnosis tool.

## How It Works

1. OpenCV checks whether the photo contains skin and crops the skin area.
2. DINOv2 extracts image features.
3. A Logistic Regression classifier predicts one of the following:
   - Acne/Rosacea
   - Eczema
   - Fungal
   - Psoriasis
   - Pigmented lesion
   - Normal/Not a disease
4. Low-confidence predictions show a warning to consult a dermatologist.

## Datasets Used

- DermNet
- SCIN (Google)
- Fitzpatrick17k
- PAD-UFES-20 (moles only)
- pacificrm Unknown_Normal (non-skin images)

## Current Results

Tested on 664 held-out SCIN phone photos.

- Accuracy: Approximately 69%
- Macro F1-score: Approximately 0.57
- Better performance: Acne/Rosacea and Eczema
- Weak performance: Psoriasis (approximately 21% recall) and Fungal (approximately 42% recall).

The model cannot distinguish melanoma from harmless moles. Such photos are labelled as Pigmented lesion with a recommendation to consult a doctor.

Dark-skin results are based on small samples and are not conclusive.

## Limitations

- This project is not a medical diagnosis tool.
- Predictions may be incorrect.
- Psoriasis and fungal conditions need improvement.
- Users should consult a qualified dermatologist for medical advice.

## Installation and Setup

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install flask opencv-python pillow numpy torch transformers scikit-learn joblib
```

Run the application:

```bash
python app.py
```

Open in your browser:

http://127.0.0.1:5000

Note: The first run downloads the pretrained DINOv2 model (approximately 350 MB).

## Main Project Files

- app.py
- skin_detect.py
- templates/index.html
- dino_lr_v4.joblib

## Disclaimer

MedAI is an experimental academic project intended for educational and research purposes only. It must not be used for diagnosis or treatment decisions.
