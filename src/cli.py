# src/cli.py
from src.model import load_model, BertEmbedder
from src.preprocess import clean_text

# --------------------------------------------------
# Load model once globally
# --------------------------------------------------
MODEL_PATH = "models/bert_fake_review_model.pkl"
print(f" Loading model from {MODEL_PATH} ...")
model = load_model(MODEL_PATH)

# Initialize BERT embedder (same settings as training)
embedder = BertEmbedder(max_len=128)


def predict_review(review: str, domain: str = "unknown") -> str:
    """
    Predict whether a single review is fake or genuine.
    Uses the same BERT embedder and trained ensemble as training.
    """
    if not review.strip():
        return " Empty input."

    # 1. Clean the input text (consistent preprocessing)
    cleaned = clean_text(review)

    # 2. Extract embedding (1 × 768)
    vec = embedder.encode([cleaned])
    if vec.shape[1] != model.estimators_[0].n_features_in_:
        raise ValueError(
            f"Feature mismatch: got {vec.shape[1]} features, "
            f"expected {model.estimators_[0].n_features_in_}"
        )

    # 3. Predict probability & label
    prob_fake = float(model.predict_proba(vec)[:, 1])
    label = "Fake" if prob_fake >= 0.5 else "Genuine"
    conf = prob_fake if label == "Fake" else 1 - prob_fake

    return f"Domain: {domain.title()} | Prediction: {label} ({conf*100:.1f}% confidence)"


def run_cli():
    """Interactive command-line interface for fake review detection."""
    print("\n🕵️ Cross-Domain Fake Review Detection CLI")
    print("Type 'quit' anytime to exit.\n")

    while True:
        review = input("Enter a review: ").strip()
        if review.lower() == "quit":
            print("Exiting CLI. Goodbye!")
            break

        domain = input("Enter domain (app/hotel/yelp): ").strip().lower()
        if domain not in ["app", "hotel", "yelp"]:
            print(" Invalid domain. Please choose: app / hotel / yelp.\n")
            continue

        try:
            print(predict_review(review, domain), "\n")
        except Exception as e:
            print(f" Error: {e}\n")    
