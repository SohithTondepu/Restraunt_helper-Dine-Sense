import os
import sys
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = r"d:\Aspect-Based-Sentimental-Analysis-on-Food-Reviews"
PROCESSED_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_reviews.csv")
PILOT_MANIFEST = os.path.join(PROJECT_ROOT, "annotations", "pilot_sample_manifest.csv")
OUTPUT_MANIFEST_1900 = os.path.join(PROJECT_ROOT, "annotations", "batch_1900", "sample_manifest_1900.csv")

def generate_1900_sample():
    print(f"Loading cleaned reviews from: {PROCESSED_CSV}")
    df = pd.read_csv(PROCESSED_CSV, keep_default_na=False)
    
    print(f"Loading approved pilot manifest to prevent overlap: {PILOT_MANIFEST}")
    pilot = pd.read_csv(PILOT_MANIFEST, keep_default_na=False)
    pilot_indices = set(pilot['original_row_id'])
    print(f"Pilot contains {len(pilot_indices)} reviews.")
    
    remaining_df = df[~df.index.isin(pilot_indices)].copy()
    remaining_df['original_row_id'] = remaining_df.index
    remaining_df['review_id'] = remaining_df.index.map(lambda i: f"REV_{i+1:05d}")
    remaining_df['parsed_time'] = pd.to_datetime(remaining_df['Time'])
    remaining_df['year'] = remaining_df['parsed_time'].dt.year

    def get_rating_bin(r):
        if r <= 2.5: return 'Low (1.0-2.5)'
        elif r <= 3.5: return 'Mid (3.0-3.5)'
        else: return 'High (4.0-5.0)'

    def get_length_bin(l):
        if l < 150: return 'Short (<150)'
        elif l <= 350: return 'Medium (150-350)'
        else: return 'Long (>350)'

    remaining_df['rating_bin'] = remaining_df['Rating'].apply(get_rating_bin)
    remaining_df['length_bin'] = remaining_df['Review_Length'].apply(get_length_bin)

    print(f"Remaining available reviews: {len(remaining_df)} across {remaining_df['Restaurant'].nunique()} establishments.")
    
    # Stratified sampling: exactly 19 reviews per establishment across all 100 establishments = 1,900 reviews
    sampled_1900 = (
        remaining_df.groupby('Restaurant', group_keys=False)
        .apply(lambda g: g.sample(n=19, random_state=42), include_groups=True)
        .sort_values('original_row_id')
        .reset_index(drop=True)
    )

    assert len(sampled_1900) == 1900, f"Expected 1900 reviews, got {len(sampled_1900)}"
    assert sampled_1900['Restaurant'].nunique() == 100, f"Expected 100 restaurants, got {sampled_1900['Restaurant'].nunique()}"
    assert len(set(sampled_1900['original_row_id']).intersection(pilot_indices)) == 0, "Overlap found with pilot!"

    manifest_cols = [
        'review_id',
        'original_row_id',
        'Restaurant',
        'Rating',
        'rating_bin',
        'Review_Length',
        'Word_Count',
        'length_bin',
        'Time',
        'year',
        'Reviewer',
        'Metadata',
        'Pictures',
        'Sentiment',
        'Review'
    ]
    manifest_rename = {
        'Restaurant': 'establishment_id',
        'Rating': 'star_rating',
        'Review_Length': 'review_char_length',
        'Word_Count': 'review_word_count',
        'Time': 'review_timestamp',
        'Reviewer': 'reviewer_name',
        'Metadata': 'reviewer_metadata',
        'Pictures': 'picture_count',
        'Sentiment': 'legacy_document_sentiment',
        'Review': 'review_text'
    }

    manifest_df = sampled_1900[manifest_cols].rename(columns=manifest_rename)
    manifest_df.to_csv(OUTPUT_MANIFEST_1900, index=False, encoding='utf-8')
    print(f"Saved 1900-review manifest to: {OUTPUT_MANIFEST_1900}")

    print("\n--- Manifest Summary ---")
    print(f"Total reviews: {len(manifest_df)}")
    print(f"Establishments: {manifest_df['establishment_id'].nunique()} (19 reviews each)")
    print("\nRating Distribution:")
    print(manifest_df['rating_bin'].value_counts())
    print("\nLength Distribution:")
    print(manifest_df['length_bin'].value_counts())
    print("\nYear Distribution:")
    print(manifest_df['year'].value_counts().sort_index())

if __name__ == '__main__':
    generate_1900_sample()
