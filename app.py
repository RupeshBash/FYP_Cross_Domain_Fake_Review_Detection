import streamlit as st
from src.model import load_model, get_bert_embeddings
from src.preprocess import clean_text

# --- Load model once to avoid reloading every time ---
@st.cache_resource
def load_prediction_model():
    return load_model("models/fake_review_model.pkl")

model = load_prediction_model()

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🕵️",
    layout="centered"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    /* Background and layout */
    .stApp {
        background: radial-gradient(circle at top left, #0f2027, #203a43, #2c5364);
        color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Title */
    h1 {
        color: #f8f9fa;
        text-align: center;
        font-size: 2.2rem !important;
        margin-bottom: 0.5rem;
    }

    /* Subtext */
    .stMarkdown {
        text-align: center;
        font-size: 1rem;
        color: #bdbdbd;
        margin-bottom: 2rem;
    }

    /* Text area */
    textarea {
        border-radius: 10px !important;
        border: 1px solid #6c757d !important;
        background-color: #1c1f26 !important;
        color: #fff !important;
    }

    /* Dropdown */
    div[data-baseweb="select"] {
        border-radius: 10px !important;
    }

    /* Buttons */
    div.stButton > button {
        width: 100%;
        height: 2.6rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 10px;
        border: none;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        transform: scale(1.05);
    }

    /* Predict button */
    div[data-testid="column"]:first-child button {
        background-color: #28a745 !important;
        color: white !important;
    }

    /* Clear button */
    div[data-testid="column"]:last-child button {
        background-color: #6c757d !important;
        color: white !important;
    }

    /* Success + Info boxes */
    .stSuccess {
        background-color: rgba(40, 167, 69, 0.2) !important;
        border: 1px solid #28a745 !important;
        border-radius: 10px;
    }

    .stInfo {
        background-color: rgba(23, 162, 184, 0.2) !important;
        border: 1px solid #17a2b8 !important;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("Cross-Domain Fake Review Detection")
st.write("Detect whether a review is **Fake** or **Genuine** using BERT + Ensemble model.")

# --- Input Section ---
review_input = st.text_area("Enter a review", height=150, key="review_input")
domain = st.selectbox("Select domain", ["amazon", "hotel", "yelp"], key="domain")

# --- Buttons ---
col1, col2 = st.columns(2)
with col1:
    predict_button = st.button("Predict")
with col2:
    clear_button = st.button("Clear")

# --- Prediction Logic ---
if predict_button and review_input.strip():
    clean = clean_text(review_input)
    vec = get_bert_embeddings([clean])
    pred = model.predict(vec)[0]
    label = "Genuine ✅" if pred == 1 else "Fake ❌"

    st.success(f"**Prediction:** {label}")
    st.info(f"**Domain:** {domain.title()}")

elif predict_button and not review_input.strip():
    st.warning("Please enter a review before predicting.")

# --- Clear button logic ---
if clear_button:
    for key in ["review_input", "domain"]:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state["review_input"] = ""  # explicitly clear input box
    st.session_state["domain"] = "amazon"  # reset dropdown
    st.rerun()


