import os
import argparse
import pandas as pd
from src.utils import load_datasets, leave_one_domain_out, save_eval_results
from src.preprocess import preprocess_dataframe
from src.model import train_model, save_model
from src.cli import run_cli


def main():
    parser = argparse.ArgumentParser(description="Fake Review Detection (Phase 2)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run cross-domain Leave-One-Domain-Out evaluation"
    )
    args = parser.parse_args()

    # Step 1: Load and preprocess data
    print(" Loading datasets...")
    df = load_datasets()
    df = preprocess_dataframe(df)

    #  TEMPORARY SAMPLING FOR FAST TESTING (REMOVE LATER FOR FULL TRAINING)
    # df = df.groupby('domain').apply(lambda x: x.sample(n=min(100, len(x)), random_state=42)).reset_index(drop=True)

    domains = df['domain'].unique()

    # Step 2: Cross-domain evaluation
    if args.eval:
        print("\n Running Leave-One-Domain-Out evaluation...")
        results = []

        for domain in domains:
            print(f"\n Testing on excluded domain: {domain.upper()}")
            train_df, test_df = leave_one_domain_out(df, domain)
            model, metrics = train_model(train_df, test_df)

            print(f"  {domain.title()} → Acc: {metrics['accuracy']:.3f} | "
                  f"Prec: {metrics['precision']:.3f} | Rec: {metrics['recall']:.3f} | F1: {metrics['f1']:.3f}")

            results.append({'Test Domain': domain, **metrics})

        results_df = pd.DataFrame(results)
        print("\n Cross-Domain Results Summary:\n", results_df)

        os.makedirs("results", exist_ok=True)
        csv_path = "results/cross_domain_eval.csv"
        save_eval_results(results_df, csv_path)
        print(f" Results saved to {csv_path}")

    # Step 3: Train final model on all data (BERT embeddings)
    print("\n Training final model on all domains...")
    model, _ = train_model(df, df)   #  Removed final_train=True

    os.makedirs("models", exist_ok=True)
    save_model(model, "models/fake_review_model.pkl")
    print(" Final model saved to /models folder")

    # Step 4: Launch CLI interface
    run_cli()


if __name__ == "__main__":
    main()
