import pandas as pd
from sklearn.utils import shuffle

def load_datasets():
    """Load and merge Amazon, Hotel, and Yelp datasets."""

    # --- Amazon Fine Food Reviews ---
    amazon_df = pd.read_csv("data/Reviews.csv").sample(5000, random_state=42)
    if 'Text' in amazon_df.columns and 'Score' in amazon_df.columns:
        amazon_df['label'] = amazon_df['Score'].apply(lambda x: 1 if x >= 4 else 0)
        amazon_df.rename(columns={'Text': 'review'}, inplace=True)
    else:
        raise ValueError(" Reviews.csv must have 'Text' and 'Score' columns.")
    amazon_df['domain'] = 'amazon'

    # --- Deceptive Opinion Spam Corpus ---
    hotel_df = pd.read_csv("data/deceptive-opinion.csv")

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
    yelp_df = pd.read_csv("data/Labelled Yelp Dataset.csv")
    if {'Review', 'Label'}.issubset(yelp_df.columns):
        yelp_df = yelp_df[['Review', 'Label']].rename(columns={'Review': 'review', 'Label': 'label'})
    else:
        raise ValueError(f" Expected 'Review' and 'Label' columns in Yelp dataset, got {list(yelp_df.columns)}")
    yelp_df['domain'] = 'yelp'

    #  Normalize labels to binary (no -1 values)
    amazon_df['label'] = amazon_df['label'].replace(-1, 0)
    hotel_df['label'] = hotel_df['label'].replace(-1, 0)
    yelp_df['label'] = yelp_df['label'].replace(-1, 0)


    # --- Combine all ---
    df = pd.concat([
        amazon_df[['review', 'label', 'domain']],
        hotel_df[['review', 'label', 'domain']],
        yelp_df[['review', 'label', 'domain']]
    ], ignore_index=True)

    print(" Loaded datasets successfully!")
    print(f"  Amazon: {amazon_df.shape}, Hotel: {hotel_df.shape}, Yelp: {yelp_df.shape}")
    print(f"  Combined shape: {df.shape}")

    return shuffle(df, random_state=42).reset_index(drop=True)


def leave_one_domain_out(df, test_domain):
    """Split into train and test sets by excluding one domain."""
    train_df = df[df['domain'] != test_domain]
    test_df = df[df['domain'] == test_domain]
    return train_df, test_df
