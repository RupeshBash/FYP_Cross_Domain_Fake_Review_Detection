import re
import string
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

#  Download required NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

#  Load spaCy model (English)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

stop_words = set(stopwords.words('english'))

#  URL pattern
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')

#  Emoji pattern (basic)
EMOJI_PATTERN = re.compile(
    "["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags
    "]+", flags=re.UNICODE
)

def clean_text(text: str) -> str:
    """Lowercase, remove URLs, emojis, punctuation, stopwords, lemmatize."""
    text = str(text).lower()
    text = URL_PATTERN.sub(" ", text)  # remove URLs
    text = EMOJI_PATTERN.sub(" ", text)  # remove emojis
    text = re.sub(f"[{string.punctuation}]", " ", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # extra spaces

    # Tokenize
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words]

    # Lemmatize using spaCy
    doc = nlp(" ".join(tokens))
    lemmas = [token.lemma_ for token in doc if token.lemma_ not in stop_words]

    return " ".join(lemmas)

def preprocess_dataframe(df: pd.DataFrame, text_col='review') -> pd.DataFrame:
    """Apply text cleaning to a DataFrame column."""
    df['clean_text'] = df[text_col].apply(clean_text)
    return df


