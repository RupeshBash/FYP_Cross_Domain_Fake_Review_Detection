from src.model import load_model
from src.preprocess import clean_text

def predict_review(review, domain="unknown"):
    """Predict fake/genuine for a single review."""
    model, vectorizer = load_model()
    clean = clean_text(review)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    label = "Genuine " if pred == 1 else "Fake ❌"
    return f"Domain: {domain.title()} | Prediction: {label}"

def run_cli():
    print(" Cross-Domain Fake Review Detection CLI")
    print("Type 'quit' anytime to exit.\n")
    while True:
        review = input(" Enter a review: ")
        if review.lower() == "quit":
            break
        domain = input("Enter domain (amazon/hotel/yelp): ")
        print(predict_review(review, domain), "\n")
