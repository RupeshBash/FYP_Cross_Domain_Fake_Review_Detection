import os
import pandas as pd
from src.utils import load_datasets, leave_one_domain_out
from src.preprocess import preprocess_dataframe
from src.model import train_model, save_model
from src.cli import run_cli

def main():
    # Step 1: Load Data
    print("📦Loading datasets...")
    df = load_datasets()
    df = preprocess_dataframe(df)
    domains = df['domain'].unique()

    # Step 2: Cross-domain training & evaluation
    results = []
    for domain in domains:
        print(f"\n Testing on excluded domain: {domain.upper()}")
        train_df, test_df = leave_one_domain_out(df, domain)
        model, vectorizer, metrics = train_model(train_df, test_df)
        print(f"Accuracy={metrics['accuracy']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
        results.append({'Test Domain': domain, **metrics})
    
    results_df = pd.DataFrame(results)
    print("\n Cross-domain results summary:\n", results_df)

    # Step 3: Train final model on all data
    print("\n Training final model on all domains...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label']

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    save_model(model, vectorizer)
    print(" Final model saved in /models")

    # Step 4: Run CLI Interface
    run_cli()

if __name__ == "__main__":
    main()
