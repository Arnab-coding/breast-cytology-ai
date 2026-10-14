import hashlib
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

# ------------------------------------------------------------
# Page setup and visual theme
# ------------------------------------------------------------
st.set_page_config(
    page_title="Breast Cytology AI | Research Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    :root { --bg:#09080f; --panel:#15111e; --pink:#f29acb; --violet:#b9a0ff;
            --white:#f8f5fc; --muted:#bcb3c8; --line:rgba(213,190,255,.17); }
    [data-testid="stAppViewContainer"] {
      background: radial-gradient(circle at 10% 0%, #2a1736 0, transparent 30%),
                  radial-gradient(circle at 94% 3%, #211b43 0, transparent 27%), var(--bg);
      color:var(--white); font-family:'DM Sans',sans-serif;
    }
    [data-testid="stHeader"] { background:transparent; }
    .block-container { max-width:1220px; padding-top:2rem; padding-bottom:3rem; }
    h1,h2,h3 { font-family:'Space Grotesk',sans-serif; letter-spacing:-.025em; }
    .eyebrow { color:var(--pink); font-weight:700; font-size:.76rem; letter-spacing:.16em; text-transform:uppercase; }
    .hero-title { font-family:'Space Grotesk',sans-serif; font-size:clamp(2.4rem,5.4vw,4.3rem); line-height:1.04; font-weight:700; margin:.55rem 0; }
    .gradient-text { background:linear-gradient(90deg,var(--pink),var(--violet)); -webkit-background-clip:text; background-clip:text; color:transparent; }
    .hero-copy { max-width:760px; color:var(--muted); font-size:1.04rem; line-height:1.7; }
    .pill { display:inline-block; border:1px solid var(--line); background:#191422; color:#e9e0f5; border-radius:999px; padding:.36rem .72rem; margin:.15rem .22rem .15rem 0; font-size:.78rem; }
    .panel { background:linear-gradient(145deg,rgba(30,23,42,.96),rgba(17,14,24,.97)); border:1px solid var(--line); border-radius:20px; padding:1.2rem; }
    .section-label { font-size:.75rem; color:var(--violet); font-weight:700; letter-spacing:.14em; text-transform:uppercase; }
    .muted { color:var(--muted); }
    .score-card { background:#191422; border:1px solid var(--line); border-radius:16px; padding:1rem; }
    .score-num { font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:700; }
    .small { color:var(--muted); font-size:.86rem; line-height:1.65; }
    .notice { background:#1b1424; border-left:3px solid var(--pink); border-radius:0 12px 12px 0; padding:.9rem 1rem; line-height:1.65; color:#e8deef; }
    div.stButton > button { min-height:3rem; border:1px solid #d6baff; border-radius:12px; background:linear-gradient(90deg,#ef9bc9,#b7a0ff); color:#100b17; font-weight:700; }
    div.stButton > button:hover { border-color:white; filter:brightness(1.05); color:#100b17; }
    [data-testid="stFileUploader"] { border:1px dashed #9f82d1; border-radius:16px; background:rgba(30,22,43,.45); padding:.45rem; }
    [data-testid="stMetric"] { background:#191422; border:1px solid var(--line); border-radius:14px; padding:1rem; }
    hr { border-color:var(--line); }
    footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="padding:.8rem 0 1rem">
      <div class="eyebrow">Computational Cytopathology · Research Demonstration</div>
      <div class="hero-title">Breast Cytology <span class="gradient-text">AI</span></div>
      <div class="hero-copy">An experimental deep-learning interface for exploring binary classification of breast fine-needle aspiration (FNA) cytology photomicrographs.</div>
      <div style="margin-top:.8rem">
        <span class="pill">MobileNetV2</span><span class="pill">Benign vs Malignant</span>
        <span class="pill">Grad-CAM explainability</span><span class="pill">Research use only</span>
      </div>
    </div>
    <div class="notice"><b>Research-use notice:</b> This prototype is not a medical device and must not be used to diagnose, exclude, screen, or manage breast disease. The displayed values are model scores, not necessarily calibrated clinical probabilities. Final interpretation requires a qualified pathologist and appropriate clinical correlation.</div>
    """,
    unsafe_allow_html=True,
)
st.write("")

MODEL_PATH = os.environ.get("MODEL_PATH", "best_cytology_baseline.keras")

@st.cache_resource(show_spinner="Loading the trained Keras model…")
def load_model(model_path: str):
    # The project's trusted .keras archive contains a serialized Lambda layer.
    # safe_mode=False is used only for this trusted, user-created artifact.
    return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)

@st.cache_resource
def build_gradcam_model(_loaded_model):
    """Expose MobileNetV2's final feature map for Grad-CAM."""
    try:
        preprocessing_layer = _loaded_model.get_layer("lambda_2")
        backbone = _loaded_model.get_layer("mobilenetv2_1.00_224")
        conv_layer = backbone.get_layer("out_relu")
        feature_extractor = tf.keras.Model(backbone.input, conv_layer.output)
        feature_map = feature_extractor(preprocessing_layer.output)
        return tf.keras.Model(_loaded_model.inputs, [feature_map, _loaded_model.output])
    except Exception:
        # Some Keras versions name internal MobileNetV2 layers differently.
        return None

def prepare_image(image: Image.Image) -> np.ndarray:
    """Resize to training input dimensions and scale to [0,1].

    The saved Lambda layer multiplies by 2 and subtracts 1, so [0,1]
    input values are expected to become [-1,1] within the model.
    This assumption must be compared with the original training pipeline.
    """
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((224, 224), Image.Resampling.BILINEAR)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(pixels, axis=0)

def make_gradcam(batch: np.ndarray, loaded_model):
    grad_model = build_gradcam_model(loaded_model)
    if grad_model is None:
        return None
    try:
        x = tf.convert_to_tensor(batch)
        with tf.GradientTape() as tape:
            feature_maps, scores = grad_model(x, training=False)
            target_score = scores[:, 0]  # sigmoid score for class label 1
        gradients = tape.gradient(target_score, feature_maps)
        if gradients is None:
            return None
        weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
        cam = tf.reduce_sum(weights * feature_maps, axis=-1)
        cam = tf.nn.relu(cam)[0]
        cam = cam / (tf.reduce_max(cam) + tf.keras.backend.epsilon())
        cam = tf.image.resize(cam[..., tf.newaxis], (224, 224)).numpy().squeeze()
        return cam
    except Exception:
        return None

def make_overlay(image: Image.Image, cam: np.ndarray) -> Image.Image:
    source = ImageOps.exif_transpose(image).convert("RGB").resize((224, 224))
    base = np.asarray(source, dtype=np.float32) / 255.0
    heat = cm.get_cmap("magma")(cam)[..., :3]
    overlay = np.clip(0.58 * base + 0.42 * heat, 0.0, 1.0)
    return Image.fromarray((overlay * 255).astype(np.uint8))

# Persistent result state: results remain visible when the app reruns.
if "result_for_hash" not in st.session_state:
    st.session_state.result_for_hash = None
if "prediction_payload" not in st.session_state:
    st.session_state.prediction_payload = None

left, right = st.columns([1.08, .92], gap="large")

with left:
    st.markdown('<div class="section-label">01 / Upload a photomicrograph</div>', unsafe_allow_html=True)
    st.write("")
    uploaded = st.file_uploader(
        "Choose an image (PNG, JPG, JPEG or WEBP)",
        type=["png", "jpg", "jpeg", "webp"],
        help="Upload only an image you are authorised to process. Avoid patient identifiers.",
    )
    image_bytes = uploaded.getvalue() if uploaded is not None else None
    image_hash = hashlib.sha256(image_bytes).hexdigest() if image_bytes else None

    if image_hash != st.session_state.result_for_hash:
        st.session_state.prediction_payload = None

    original_image = None
    if image_bytes:
        try:
            original_image = Image.open(io.BytesIO(image_bytes))
            original_image.load()
            st.image(original_image, caption=f"Uploaded image · {original_image.width} × {original_image.height}px", width="stretch")
            st.caption("The model input is resized to 224 × 224 pixels. Resizing may discard fine cellular detail.")
        except (UnidentifiedImageError, OSError, ValueError):
            st.error("This image could not be opened. Please try a PNG or JPEG image.")
            original_image = None
            image_hash = None
    else:
        st.markdown(
            """<div class="panel" style="min-height:260px;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center">
            <div style="font-size:3rem;color:#b9a0ff">＋</div><h3 style="margin:.4rem 0">Your image appears here</h3>
            <div class="small">Choose a breast FNA cytology photomicrograph above to preview it before running the model.</div></div>""",
            unsafe_allow_html=True,
        )
    st.write("")
    analyse = st.button("Analyse image", type="primary", width="stretch", disabled=(original_image is None))

    if analyse and original_image is not None and image_hash is not None:
        try:
            batch = prepare_image(original_image)
            with st.spinner("Running model inference…"):
                loaded_model = load_model(MODEL_PATH)
                raw_score = float(np.asarray(loaded_model.predict(batch, verbose=0)).reshape(-1)[0])
            if not np.isfinite(raw_score) or not 0.0 <= raw_score <= 1.0:
                st.error("The model returned an invalid sigmoid score. No interpretation is shown.")
                st.session_state.prediction_payload = None
            else:
                heatmap = make_gradcam(batch, loaded_model)
                st.session_state.prediction_payload = {
                    "raw_score": raw_score,
                    "image_hash": image_hash,
                    "gradcam": heatmap,
                }
                st.session_state.result_for_hash = image_hash
        except FileNotFoundError:
            st.error("Model file not found. Add best_cytology_baseline.keras to the repository root, or configure MODEL_PATH.")
        except Exception as exc:
            st.error("The image could not be analysed. Check that the model file and dependencies are compatible.")
            st.caption(f"Technical detail: {type(exc).__name__}: {exc}")

with right:
    st.markdown('<div class="section-label">02 / Model interpretation</div>', unsafe_allow_html=True)
    st.write("")
    payload = st.session_state.prediction_payload
    if payload and payload.get("image_hash") == image_hash:
        score = payload["raw_score"]
        malignant = score * 100.0
        benign = (1.0 - score) * 100.0
        predicted = "Malignant" if score >= 0.5 else "Benign"
        accent = "#f29acb" if predicted == "Malignant" else "#b9a0ff"
        st.markdown(
            f"""<div class="panel"><div class="small">MODEL PREDICTION · EXPERIMENTAL</div>
            <div style="font-size:2.2rem;font-weight:700;color:{accent};margin:.3rem 0">{predicted}</div>
            <div class="small">Demonstration threshold: 0.50. This is not a clinically validated decision threshold.</div></div>""",
            unsafe_allow_html=True,
        )
        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="score-card"><div class="small">Benign score</div><div class="score-num" style="color:#b9a0ff">{benign:.1f}%</div></div>', unsafe_allow_html=True)
            st.progress(int(round(benign)), text="Benign")
        with c2:
            st.markdown(f'<div class="score-card"><div class="small">Malignant score</div><div class="score-num" style="color:#f29acb">{malignant:.1f}%</div></div>', unsafe_allow_html=True)
            st.progress(int(round(malignant)), text="Malignant")
        with st.expander("How should I interpret these scores?"):
            st.write("Using the label convention supplied for this project (0 = Benign; 1 = Malignant), the sigmoid output is displayed as the malignant-class score and its complement as the benign-class score.")
            st.warning("These are raw model scores and may not be calibrated probabilities. A score is not the same as the chance that a patient has cancer.")
        st.markdown('<div class="section-label" style="margin-top:1.2rem">03 / Explainability · Grad-CAM</div>', unsafe_allow_html=True)
        heatmap = payload.get("gradcam")
        if heatmap is not None and original_image is not None:
            g1, g2 = st.columns(2)
            with g1:
                st.image(original_image.convert("RGB").resize((224, 224)), caption="Model input (resized)", width="stretch")
            with g2:
                st.image(make_overlay(original_image, heatmap), caption="Grad-CAM overlay", width="stretch")
            st.markdown('<div class="small">Warm regions indicate areas contributing more to the model’s malignant-class score. Grad-CAM is a coarse visual explanation, not a validated map of malignant cells and not proof of biological reasoning.</div>', unsafe_allow_html=True)
        else:
            st.info("Grad-CAM was not available for this runtime/model configuration. The classification score is shown without a heatmap.")
    else:
        st.markdown(
            """<div class="panel" style="min-height:260px"><div class="small">RESULTS WILL APPEAR HERE</div>
            <h3 style="margin-top:1rem">Benign / Malignant scores</h3>
            <p class="muted">Upload an image and select <b>Analyse image</b>. The saved model's output will appear here.</p>
            <hr><div class="small">No result is displayed until the image has been submitted for inference.</div></div>""",
            unsafe_allow_html=True,
        )

st.write("")
st.markdown('<div class="section-label">04 / Model limitations</div>', unsafe_allow_html=True)
a, b = st.columns(2, gap="large")
with a:
    st.markdown("""<div class="panel"><h3>What this prototype can do</h3><div class="small">Apply a previously trained binary image classifier to an uploaded image and display the model output. Where technically supported, Grad-CAM gives a rough visual indication of regions influencing that output.</div></div>""", unsafe_allow_html=True)
with b:
    st.markdown("""<div class="panel"><h3>What it cannot establish</h3><div class="small">It cannot independently diagnose cancer, replace cytopathological review, establish invasion, determine tumour subtype, or guarantee performance on images from different microscopes, stains, laboratories, or patient populations.</div></div>""", unsafe_allow_html=True)

with st.expander("Validation and responsible use"):
    st.markdown("""
    - This is an educational and research prototype, not a clinically validated diagnostic system.
    - The 0.50 threshold is a demonstration setting, not a validated clinical cut-off.
    - Dataset-level performance does not guarantee correct classification of an individual image.
    - Evaluation should use an independent test set with patient/case-level separation where identifiers permit, and report sensitivity, specificity, ROC-AUC, precision, calibration, and confidence intervals.
    - Randomly splitting images from the same patient or slide across training and test sets can inflate apparent performance through data leakage.
    - Upload only images you are authorised to process. Do not upload directly identifiable patient data.
    """)

st.markdown("""<hr><div class="small" style="text-align:center">Breast Cytology AI · Capstone research demonstration<br>Not for clinical diagnosis, screening, triage, or treatment decisions.</div>""", unsafe_allow_html=True)
