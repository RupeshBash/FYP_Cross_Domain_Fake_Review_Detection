import streamlit as st
from pathlib import Path
import numpy as np
from typing import Dict
import random

from src.model import load_model, get_bert_embeddings
from src.preprocess import clean_text
from src.utils import MODELS_DIR, read_metadata

# -------------------------
# Page config & styling
# -------------------------
st.set_page_config(page_title="Fake Review Detector", page_icon="🕵️", layout="centered")
st.markdown(
    """
    <style>
      .stApp { font-family: 'Segoe UI', sans-serif; }
      h1 { text-align: center; }
      textarea { border-radius: 8px; }
      .stButton>button { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🕵️ Cross-Domain Fake Review Detection")
st.write("Detect whether a review is **Fake** or **Genuine** using a BERT + Ensemble model.")

# -------------------------
# Discover models & metadata
# -------------------------
meta = read_metadata()
available = sorted([p.name for p in Path(MODELS_DIR).glob("*.pkl")])

models_list = [m for m in meta.get("models", []) if (Path(MODELS_DIR) / m).exists()] or available
if not models_list:
    st.error(" No model .pkl files found in models/. Please copy your model files and reload.")
    st.stop()

default_model = meta.get("best_by_auc")
if default_model not in models_list:
    default_model = models_list[0]

# -------------------------
# Corrected label mapping
# -------------------------
label_map_meta = meta.get("label_map", None)
num_to_label: Dict[int, str] = {}

if isinstance(label_map_meta, dict):
    try:
        num_to_label = {int(v): k.lower() for k, v in label_map_meta.items()}
    except Exception:
        num_to_label = {}

if not num_to_label:
    try:
        tmp_model = load_model(str(Path(MODELS_DIR) / default_model))
        classes = list(map(int, getattr(tmp_model, "classes_", [0, 1])))
    except Exception:
        classes = [0, 1]

    if set(classes) == {0, 1}:
        #  Corrected: your model uses 0 = genuine, 1 = fake
        num_to_label = {0: "genuine", 1: "fake"}
    else:
        num_to_label = {classes[0]: "genuine", classes[-1]: "fake"}

st.info(f" Using label map: {num_to_label}")

# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    pick = st.selectbox("Select Model File", options=models_list, index=models_list.index(default_model))
    max_len = st.slider("BERT max tokens", 64, 256, 128, 32)
    default_thr = float(meta.get("thresholds", {}).get(pick, 0.5))
    thr = st.slider("Decision threshold (Fake if P(fake) ≥ thr)", 0.01, 0.99, default_thr, 0.01)

# -------------------------
# Load model (cached)
# -------------------------
@st.cache_resource(show_spinner=False)
def load_model_cached(path: str):
    return load_model(path)

model_path = str(Path(MODELS_DIR) / pick)
model = load_model_cached(model_path)

# -------------------------
# Input area
# -------------------------
review_input = st.text_area(" Enter a review", height=160)
domain = st.selectbox("Select domain (optional)", ["app", "hotel", "yelp"])

col1, col2, col3 = st.columns(3)
with col1:
    predict_btn = st.button("Predict")
with col2:
    clear_btn = st.button("Clear")
with col3:
    sample_btn = st.button("Use sample")

SAMPLES = {
    "app": [
        "Absolutely love this product! Works as advertised and arrived quickly.",
        "Received compensation for this review. Best item ever!! 10/10 would buy again!!",
    ],
    "yelp": [
        "Service was slow and the food was undercooked. Not coming back.",
        "Great ambiance. Staff repeatedly asked for 5-star review which felt pushy.",
    ],
    "hotel": [
        "Room was clean, staff were friendly, and check-in was smooth.",
        "Review seems templated: 'The best stay of my life' repeated across listings.",
    ],
}

def safe_rerun():
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except Exception:
            pass

if sample_btn:
    st.session_state["review_input"] = random.choice(SAMPLES[domain])
    safe_rerun()

if clear_btn:
    st.session_state.pop("review_input", None)
    safe_rerun()

# -------------------------
# Compute probability of 'fake'
# -------------------------
def compute_p_fake(probs: np.ndarray, classes: np.ndarray, num2label: Dict[int, str]) -> float:
    class_prob = {int(c): float(p) for c, p in zip(classes, probs)}
    fake_num = next((int(num) for num, name in num2label.items() if name == "fake"), None)
    if fake_num is None:
        fake_num = 1  # by default
    return class_prob.get(fake_num, 0.0)

# -------------------------
# Prediction flow
# -------------------------
def predict_and_explain(text: str, threshold: float):
    cleaned = clean_text(text)
    with st.spinner(" Computing BERT embeddings..."):
        vec = get_bert_embeddings([cleaned], max_len=max_len)
    if len(vec.shape) == 1:
        vec = vec.reshape(1, -1)

    probs = model.predict_proba(vec)[0]
    p_fake = compute_p_fake(probs, model.classes_, num_to_label)
    is_fake = float(p_fake) >= float(threshold)

    class_probs_map = {int(c): float(p) for c, p in zip(model.classes_, probs)}
    pred_num = int(max(class_probs_map, key=class_probs_map.get))
    pred_name = num_to_label.get(pred_num, str(pred_num))

    per_est = []
    if hasattr(model, "estimators_"):
        for name, est in getattr(model, "named_estimators_", {}).items():
            try:
                p = est.predict_proba(vec)[0]
                est_map = {int(c): float(pp) for c, pp in zip(est.classes_, p)}
                per_est.append((name, est_map))
            except Exception:
                per_est.append((name, None))

    return {
        "cleaned": cleaned,
        "p_fake": p_fake,
        "pred_is_fake": bool(is_fake),
        "pred_label": pred_name,
        "predicted_numeric": pred_num,
        "probs": class_probs_map,
        "per_estimator": per_est,
    }

# -------------------------
# Predict button logic
# -------------------------
if predict_btn:
    if not review_input or not review_input.strip():
        st.warning("⚠️ Please enter a review to predict.")
    else:
        res = predict_and_explain(review_input, thr)
        label_display = "Fake" if res["pred_is_fake"] else "Genuine"
        conf_pct = res["p_fake"] * 100 if res["pred_is_fake"] else (100 - res["p_fake"] * 100)

        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", label_display)
        c2.metric("Confidence", f"{conf_pct:.1f}%")
        c3.metric("Model file", pick)

        with st.expander("🧹 Cleaned text used by model"):
            st.write(res["cleaned"])

        st.write("**Model class probabilities (numeric_label → prob):**")
        st.json(res["probs"])

        st.write(f"**Probability assigned to `fake`: {res['p_fake'] * 100:.2f}%**")

        if res["per_estimator"]:
            st.write("**Per-estimator probabilities (if available):**")
            for name, est_map in res["per_estimator"]:
                if est_map is None:
                    st.write(f"- {name}: (no probabilities)")
                else:
                    st.write(f"- {name}: {est_map}")

st.markdown("---")
st.caption("Tip: Adjust threshold in sidebar. If results seem off, verify `num_to_label` mapping above.")
