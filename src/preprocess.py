import re
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """Lowercase, remove punctuation, stopwords, and tokenize."""
    text = str(text).lower()
    text = re.sub(f"[{string.punctuation}]", " ", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(tokens)

def preprocess_dataframe(df: pd.DataFrame, text_col='review') -> pd.DataFrame:
    """Apply text cleaning to a DataFrame."""
    df['clean_text'] = df[text_col].apply(clean_text)
    return df
