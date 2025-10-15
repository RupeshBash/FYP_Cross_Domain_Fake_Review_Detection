from src.model import load_model, get_bert_embeddings
from src.preprocess import clean_text
import numpy as np

#  Load the model once (not every time predict_review is called)
model = load_model("models/fake_review_model.pkl")


def predict_review(review, domain="unknown"):
    """Predict whether a single review is fake or genuine using BERT + Ensemble."""
    # 1. Clean the input text
    clean = clean_text(review)

    # 2. Extract BERT embeddings (should match training shape, typically 768)
    vec = get_bert_embeddings([clean])  # shape: (1, 768)

    # 3. Ensure shape is correct
    if vec.shape[1] != model.estimators_[0].n_features_in_:
        raise ValueError(
            f" Feature mismatch: embedding shape {vec.shape[1]} "
            f"!= expected {model.estimators_[0].n_features_in_}"
        )

    # 4. Make prediction
    pred = model.predict(vec)[0]
    label = "Genuine" if pred == 1 else "Fake"

    return f"Domain: {domain.title()} | Prediction: {label}"


def run_cli():
    """Interactive CLI for fake review detection."""
    print("Cross-Domain Fake Review Detection CLI")
    print("Type 'quit' anytime to exit.\n")

    while True:
        review = input("Enter a review: ").strip()
        if review.lower() == "quit":
            print("Exiting CLI. Goodbye!")
            break

        domain = input("Enter domain (amazon/hotel/yelp): ").strip().lower()
        if domain not in ["amazon", "hotel", "yelp"]:
            print("Invalid domain. Please enter amazon/hotel/yelp.\n")
            continue

        try:
            print(predict_review(review, domain), "\n")
        except Exception as e:
            print(f"⚠️ Error: {e}\n")
