# main.py
import os
import argparse
import pandas as pd
from datetime import date

from src.utils import (
    load_datasets,
    leave_one_domain_out,
    save_eval_results,
    MODELS_DIR,
    META_PATH,
    write_metadata,
)
from src.model import train_model, save_model
from src.preprocess import preprocess_dataframe  # keeps your old name; returns Review, Label, y, clean


def main():
    parser = argparse.ArgumentParser(description="Fake Review Detection (Phase 2)")
    parser.add_argument("--eval", action="store_true",
                        help="Run cross-domain Leave-One-Domain-Out evaluation")
    parser.add_argument("--model", default="voting",
                        choices=["voting", "svm_cal", "lr", "rf"],
                        help="Which classifier to train/use")
    parser.add_argument("--max_len", type=int, default=128,
                        help="BERT max sequence length")
    args = parser.parse_args()

    # ── Step 1: Load & preprocess (enforces Review/Label → y, adds clean)
    print(" Loading datasets…")
    df = load_datasets()               # concat of 3 CSVs + domain column
    df = preprocess_dataframe(df)      # -> ['Review','Label','y','clean', 'domain']

    domains = df["domain"].unique()
    os.makedirs("results", exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Step 2: Leave-One-Domain-Out evaluation (optional)
    thresholds_used = {}
    results_rows = []

    if args.eval:
        print("\n Running Leave-One-Domain-Out evaluation…")
        for domain in domains:
            print(f"\n  Holding out domain: {domain.upper()}")
            train_df, test_df = leave_one_domain_out(df, domain)

            model, metrics = train_model(
                train_df, test_df,
                text_col="clean", label_col="y",
                model_key=args.model, max_len=args.max_len
            )

            print(
                f"   {domain.title()} → Acc: {metrics['accuracy']:.3f} | "
                f"F1_macro: {metrics['f1_macro']:.3f} | AUC: {metrics['auc']:.3f}"
            )

            row = {"Test Domain": domain, **{k: v for k, v in metrics.items() if k != "report"}}
            results_rows.append(row)

        results_df = pd.DataFrame(results_rows)
        print("\n Cross-Domain Results Summary:\n", results_df)
        save_eval_results(results_df, "results/cross_domain_eval.csv")
        print(" Results saved to results/cross_domain_eval.csv")

    # ── Step 3: Train final model on ALL data ( can also do train/test split if future)
    print("\n Training final model on all domains…")
    final_model, final_metrics = train_model(
        df, df,
        text_col="clean", label_col="y",
        model_key=args.model, max_len=args.max_len
    )

    # Versioned filename + stable alias
    stamp = date.today().isoformat()
    model_fname = f"{args.model}_{stamp}.pkl"
    save_model(final_model, str(MODELS_DIR / model_fname))
    print(f" Final model saved to models/{model_fname}")

    # ── Step 4: Write/merge metadata for Streamlit
    write_metadata(
        best_model=model_fname,
        add_model=model_fname,
        thresholds={},  
    )
    print(f" Metadata updated at {META_PATH}")

    # (Optional) if you still want to run a CLI after training:
    # from src.cli import run_cli
    # run_cli()


if __name__ == "__main__":
    main()
