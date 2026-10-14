# Breast Cytology AI

A Streamlit research demonstration for a saved MobileNetV2 Keras model that classifies breast FNA cytology photomicrographs as Benign or Malignant.

## Deploy

1. Upload `app.py`, `requirements.txt`, and the trained model `best_cytology_baseline.keras` to the root of your GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
3. Create an app from `Arnab-coding/breast-cytology-ai`, branch `main`, main file `app.py`.
4. In Advanced settings, select Python 3.12 if offered, then deploy.

## Important model assumptions to verify

- Input shape: 224 × 224 × 3 RGB.
- Label convention supplied by the project owner: 0 = Benign; 1 = Malignant.
- The single sigmoid output is interpreted as the score for class 1 (Malignant).
- Preprocessing divides image pixels by 255 before the model's saved Lambda layer scales values with `x * 2 - 1`, yielding values in [-1, 1].

These assumptions must be compared with the original training pipeline before results are relied upon. Scores are not necessarily calibrated probabilities.

## Safety and privacy

This is an educational/research prototype, not a medical device. It must not be used for clinical diagnosis, screening, triage or treatment decisions. Grad-CAM is only a coarse model explanation, not proof of biological reasoning. Upload only images you are authorised to process, and do not upload directly identifiable patient data.

The `.keras` model contains a serialized Lambda layer and the app loads it with Keras `safe_mode=False`. Use only the trusted model file created by your own training notebook. Never use this setting with a model file from an unknown or untrusted source.
