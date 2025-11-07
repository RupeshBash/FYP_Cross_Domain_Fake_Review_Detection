import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# ------------------------------------------------------------------
# BERT Setup
# ------------------------------------------------------------------
MODEL_NAME = "bert-base-uncased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
bert_model = BertModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

# ------------------------------------------------------------------
# Feature Extraction (Batched BERT embeddings)
# ------------------------------------------------------------------
@torch.inference_mode()
def get_bert_embeddings(texts, batch_size: int = 64, max_len: int = 128, pool: str = "mean"):
    """
    Batched BERT embeddings (GPU-accelerated if available).
    pool: 'mean' averages token embeddings, 'cls' uses [CLS].
    Returns: np.ndarray (n_samples, 768)
    """
    texts = list(texts)
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)

    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="BERT embeddings"):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        ).to(DEVICE)

        out = bert_model(**enc)

        if pool == "cls":
            vec = out.last_hidden_state[:, 0, :]
        else:
            mask = enc["attention_mask"].unsqueeze(-1)
            summed = (out.last_hidden_state * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1)
            vec = summed / lengths

        embeddings.append(vec.detach().cpu().numpy().astype(np.float32))

    return np.vstack(embeddings)


# ------------------------------------------------------------------
# Model Training: Voting Ensemble (Fast + Probabilities)
# ------------------------------------------------------------------
def train_model(train_df, test_df, text_col="clean", label_col="Label"):
    """
    Train a soft Voting Ensemble (Logistic Regression + Calibrated LinearSVC + Random Forest)
    using BERT embeddings from text data.
    """
    print("\nGenerating BERT embeddings (train + test)...")
    X_train = get_bert_embeddings(train_df[text_col])
    X_test  = get_bert_embeddings(test_df[text_col])
    y_train, y_test = train_df[label_col], test_df[label_col]

    # Base models (fast + with probability via calibration)
    lr  = LogisticRegression(max_iter=200, random_state=42)
    svm = CalibratedClassifierCV(LinearSVC(random_state=42), method="sigmoid", cv=3)
    rf  = RandomForestClassifier(n_estimators=100, random_state=42)

    # Ensemble
    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("svm", svm), ("rf", rf)],
        voting="soft"
    )

    print("Training ensemble model...")
    ensemble.fit(X_train, y_train)

    # Evaluation
    y_pred = ensemble.predict(X_test)
    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1":        f1_score(y_test, y_pred, average="weighted", zero_division=0)
    }

    print("Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"  {k:<10}: {v:.4f}")

    return ensemble, metrics


# ------------------------------------------------------------------
# Save / Load utilities
# ------------------------------------------------------------------
def save_model(model, model_path="models/bert_fake_review_model.pkl"):
    """Save trained ensemble model."""
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")


def load_model(model_path="models/bert_fake_review_model.pkl"):
    """Load trained ensemble model."""
    model = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")
    return model
