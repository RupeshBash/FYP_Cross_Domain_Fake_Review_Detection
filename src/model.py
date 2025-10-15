import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# --------------------------------------------
#  BERT Setup
# -------------------------------------------
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)
model.eval()


# -----------------------------------------------------------
#  Feature Extraction: BERT Embeddings
# -----------------------------------------------------------
def get_bert_embeddings(texts):
    """
    Convert list of texts into BERT [CLS] embeddings.
    Returns a NumPy array of shape (n_samples, 768).
    """
    all_embeddings = []
    for text in tqdm(texts, desc="Extracting BERT embeddings"):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        all_embeddings.append(cls_embedding)
    return np.vstack(all_embeddings)


# -----------------------------------------------------------
#  Training: Single Voting Ensemble
# -----------------------------------------------------------
def train_model(train_df, test_df):
    """
    Train a soft Voting Ensemble (Logistic Regression + SVM + Random Forest)
    on BERT embeddings and evaluate.
    """
    # Step 1: Extract BERT features
    X_train = get_bert_embeddings(train_df['clean_text'])
    X_test = get_bert_embeddings(test_df['clean_text'])
    y_train, y_test = train_df['label'], test_df['label']

    # Step 2: Define base models
    lr = LogisticRegression(max_iter=200, random_state=42)
    svm = SVC(probability=True, kernel='linear', random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Step 3: Define ensemble
    ensemble = VotingClassifier(
        estimators=[('lr', lr), ('svm', svm), ('rf', rf)],
        voting='soft'
    )

    # Step 4: Train
    print("\n Training ensemble model...")
    ensemble.fit(X_train, y_train)

    # Step 5: Evaluate
    y_pred = ensemble.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }

    print("\n Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")

    return ensemble, metrics


# ------------------------------------------------------------
#  Save & Load Model
# ------------------------------------------------------------
def save_model(model, model_path='models/fake_review_model.pkl'):
    """Save trained ensemble model."""
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


def load_model(model_path='models/fake_review_model.pkl'):
    """Load trained ensemble model."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model



