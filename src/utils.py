import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------
# Load and prepare datasets
# -----------------------------------------------------
def load_datasets():
    """Load and merge Amazon, Hotel, and Yelp datasets."""

    # --- Amazon Fine Food Reviews ---
    amazon_df = pd.read_csv("data/Reviews.csv").sample(1000, random_state=42)  #replace this last ma hai, its only using 1000 rows for faster testing
    if 'Text' in amazon_df.columns and 'Score' in amazon_df.columns:
        amazon_df['label'] = amazon_df['Score'].apply(lambda x: 1 if x >= 4 else 0)
        amazon_df.rename(columns={'Text': 'review'}, inplace=True)
    else:
        raise ValueError(" Reviews.csv must have 'Text' and 'Score' columns.")
    amazon_df['domain'] = 'amazon'

    # --- Deceptive Opinion Spam Corpus ---
    hotel_df = pd.read_csv("data/deceptive-opinion.csv").sample(1000, random_state=42)#replace this last ma hai, its only using 1000 rows for faster testing


    # find the review column automatically
    possible_review_cols = ['review', 'text', 'Review', 'Text', 'sentence']
    review_col = next((c for c in possible_review_cols if c in hotel_df.columns), None)
    if review_col is None:
        raise ValueError(f" No valid review column found in deceptive-opinion.csv. Columns are: {list(hotel_df.columns)}")
    
    # find label column automatically
    possible_label_cols = ['deceptive', 'label', 'Label', 'truth', 'is_deceptive']
    label_col = next((c for c in possible_label_cols if c in hotel_df.columns), None)
    if label_col is None:
        raise ValueError(f" No valid label column found in deceptive-opinion.csv. Columns are: {list(hotel_df.columns)}")

    # rename to standard names
    hotel_df = hotel_df.rename(columns={review_col: 'review', label_col: 'label'})
    # normalize label values (if deceptive=1, genuine=0)
    hotel_df['label'] = hotel_df['label'].apply(lambda x: 0 if str(x).lower() in ['deceptive', '1', '-1'] else 1)
    hotel_df['domain'] = 'hotel'

    # --- Yelp Labelled Dataset ---
    yelp_df = pd.read_csv("data/Labelled Yelp Dataset.csv").sample(1000, random_state=42)#replace this last ma hai, its only using 1000 rows for faster testing
    if {'Review', 'Label'}.issubset(yelp_df.columns):
        yelp_df = yelp_df[['Review', 'Label']].rename(columns={'Review': 'review', 'Label': 'label'})
    else:
        raise ValueError(f" Expected 'Review' and 'Label' columns in Yelp dataset, got {list(yelp_df.columns)}")
    yelp_df['domain'] = 'yelp'

    # Normalize labels to binary (no -1 values)
    amazon_df['label'] = amazon_df['label'].replace(-1, 0)
    hotel_df['label'] = hotel_df['label'].replace(-1, 0)
    yelp_df['label'] = yelp_df['label'].replace(-1, 0)

    # Combine all datasets
    df = pd.concat([
        amazon_df[['review', 'label', 'domain']],
        hotel_df[['review', 'label', 'domain']],
        yelp_df[['review', 'label', 'domain']]
    ], ignore_index=True)

    print(" Loaded datasets successfully!")
    print(f"  Amazon: {amazon_df.shape}, Hotel: {hotel_df.shape}, Yelp: {yelp_df.shape}")
    print(f"  Combined shape: {df.shape}")

    return shuffle(df, random_state=42).reset_index(drop=True)


# -----------------------------------------------------
# Leave-One-Domain-Out Split
# -----------------------------------------------------
def leave_one_domain_out(df, test_domain):
    """Split into train and test sets by excluding one domain."""
    train_df = df[df['domain'] != test_domain]
    test_df = df[df['domain'] == test_domain]
    return train_df, test_df


# -------------------------------------------------------
# Evaluation Helpers
# -------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    """Plot confusion matrix with seaborn heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels or ['Fake', 'Genuine'],
                yticklabels=labels or ['Fake', 'Genuine'])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def cross_validate_model(model, X, y, cv_splits=5):
    """Perform Stratified K-Fold cross-validation."""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    f1 = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')

    print(f" Cross-Validation Results ({cv_splits}-fold):")
    print(f"  Accuracy: {acc.mean():.4f} ± {acc.std():.4f}")
    print(f"  F1 Score: {f1.mean():.4f} ± {f1.std():.4f}")
    return {'cv_accuracy': acc.mean(), 'cv_f1': f1.mean()}


# ------------------------------------------------------
# Cross-Domain Evaluation Helpers
# ------------------------------------------------------
def save_eval_results(df, path="results/cross_domain_eval.csv"):
    """Save evaluation results to CSV."""
    df.to_csv(path, index=False)
    print(f" Evaluation results saved to: {path}")

def print_table_summary(df):
    """Pretty print summary of cross-domain evaluation results."""
    print("\n Leave-One-Domain-Out Summary:")
    print(df.to_string(index=False))
    print("\nAverage Scores:")
    print(df.mean(numeric_only=True).round(3))