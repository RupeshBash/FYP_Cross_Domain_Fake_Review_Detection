import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model(train_df, test_df):
    """Train Logistic Regression on TF-IDF and evaluate."""
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df['clean_text'])
    X_test = vectorizer.transform(test_df['clean_text'])

    y_train, y_test = train_df['label'], test_df['label']

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }

    return model, vectorizer, metrics

def save_model(model, vectorizer, model_path='models/fake_review_model.pkl', vec_path='models/vectorizer.pkl'):
    """Save trained model and vectorizer."""
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(vec_path, 'wb') as f:
        pickle.dump(vectorizer, f)

def load_model(model_path='models/fake_review_model.pkl', vec_path='models/vectorizer.pkl'):
    """Load model and vectorizer from disk."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vec_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer
