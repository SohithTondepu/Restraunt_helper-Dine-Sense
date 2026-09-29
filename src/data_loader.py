import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit, train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'Restaurant reviews.csv')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')


def rating_to_sentiment(rating: float) -> str:
    """
    Standard 3-class sentiment mapping:
    - Negative: rating <= 2.0
    - Neutral: 2.5 <= rating <= 3.5
    - Positive: rating >= 4.0
    """
    if rating <= 2.0:
        return 'Negative'
    elif rating <= 3.5:
        return 'Neutral'
    else:
        return 'Positive'


def sentiment_to_label(sentiment: str) -> int:
    mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    return mapping.get(sentiment, 1)


def label_to_sentiment(label: int) -> str:
    mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return mapping.get(label, 'Neutral')


def load_and_clean_data(raw_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Loads raw CSV, strips corrupted columns (like '7514'), handles NaNs,
    deduplicates reviews, and builds clean sentiment labels.
    """
    if not os.path.exists(raw_path):
        # Fallback to data/Restaurant reviews.csv if raw hasn't copied
        raw_path = os.path.join(PROJECT_ROOT, 'data', 'Restaurant reviews.csv')
        
    df = pd.read_csv(raw_path)
    
    # 1. Drop corrupted column if present
    if '7514' in df.columns:
        df.drop(columns=['7514'], inplace=True)
        
    # 2. Filter invalid ratings
    df = df[df['Rating'] != 'Like'].copy()
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    
    # 3. Clean textual review column
    df['Review'] = df['Review'].astype(str).str.strip()
    df = df[df['Review'].str.len() > 3].copy()
    
    # 4. Drop rows with null ratings, reviews, or restaurants
    df.dropna(subset=['Rating', 'Review', 'Restaurant'], inplace=True)
    
    # Fill missing Reviewer with 'Anonymous'
    if 'Reviewer' in df.columns:
        df['Reviewer'] = df['Reviewer'].fillna('Anonymous').astype(str).str.strip()
        df.loc[df['Reviewer'] == '', 'Reviewer'] = 'Anonymous'
    else:
        df['Reviewer'] = 'Anonymous'
        
    # 5. Remove exact duplicates (same restaurant & review)
    initial_len = len(df)
    df.drop_duplicates(subset=['Restaurant', 'Review'], inplace=True)
    dedup_dropped = initial_len - len(df)
    
    # 6. Map ratings to 3-class sentiment
    df['Sentiment'] = df['Rating'].apply(rating_to_sentiment)
    df['Label'] = df['Sentiment'].apply(sentiment_to_label)
    df['Review_Length'] = df['Review'].apply(len)
    df['Word_Count'] = df['Review'].apply(lambda x: len(x.split()))
    
    df.reset_index(drop=True, inplace=True)
    print(f"Data Loaded: {len(df)} clean reviews (dropped {dedup_dropped} duplicates).")
    return df


def create_splits(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15, random_state: int = 42):
    """
    Creates stratified splits grouped by Reviewer to prevent reviewer-style data leakage.
    Reviews by 'Anonymous' or single-review users are stratified directly.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # For reviewers with multiple reviews, keep them in the same split
    # For robust grouping across the dataset, use StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    # First split into train_val (80%) and test (20%), then train (70%) and val (10% or 15%)
    # Let's do a clean 70/15/15 split using GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=random_state)
    train_idx, temp_idx = next(gss.split(df, groups=df['Reviewer']))
    
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    temp_df = df.iloc[temp_idx].copy().reset_index(drop=True)
    
    # Split temp into val and test (50/50 of temp = 15%/15% of total)
    gss_val = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=random_state)
    val_idx, test_idx = next(gss_val.split(temp_df, groups=temp_df['Reviewer']))
    
    val_df = temp_df.iloc[val_idx].copy().reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].copy().reset_index(drop=True)
    
    # Save splits
    train_df.to_csv(os.path.join(PROCESSED_DIR, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, 'test.csv'), index=False)
    
    df.to_csv(os.path.join(PROCESSED_DIR, 'cleaned_reviews.csv'), index=False)
    try:
        df.to_parquet(os.path.join(PROCESSED_DIR, 'cleaned_reviews.parquet'), index=False)
    except Exception as e:
        print(f"Parquet save notice: {e}")
        
    print(f"Splits Created: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df


if __name__ == '__main__':
    df = load_and_clean_data()
    train_df, val_df, test_df = create_splits(df)
    print("\nClass distribution in Test Set:")
    print(test_df['Sentiment'].value_counts(normalize=True))
