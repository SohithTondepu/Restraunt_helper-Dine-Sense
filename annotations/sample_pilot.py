import os
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint
from scipy.sparse import dok_matrix

PROJECT_ROOT = r"d:\Aspect-Based-Sentimental-Analysis-on-Food-Reviews"
PROCESSED_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_reviews.csv")
OUTPUT_MANIFEST = os.path.join(PROJECT_ROOT, "annotations", "pilot_sample_manifest.csv")

def generate_pilot_sample():
    print(f"Reading cleaned dataset from: {PROCESSED_CSV}")
    # Read preserving exact strings
    df = pd.read_csv(PROCESSED_CSV, keep_default_na=False)
    df['original_row_id'] = df.index
    df['review_id'] = df.index.map(lambda i: f"REV_{i+1:05d}")
    df['parsed_time'] = pd.to_datetime(df['Time'])
    df['year'] = df['parsed_time'].dt.year

    def get_rating_bin(r):
        if r <= 2.5: return 'Low (1.0-2.5)'
        elif r <= 3.5: return 'Mid (3.0-3.5)'
        else: return 'High (4.0-5.0)'

    def get_length_bin(l):
        if l < 150: return 'Short (<150)'
        elif l <= 350: return 'Medium (150-350)'
        else: return 'Long (>350)'

    df['rating_bin'] = df['Rating'].apply(get_rating_bin)
    df['length_bin'] = df['Review_Length'].apply(get_length_bin)

    rest_list = sorted(df['Restaurant'].unique())
    rest_to_idx = {r: i for i, r in enumerate(rest_list)}
    N = len(df)

    print(f"Total establishments: {len(rest_list)}")
    print(f"Total reviews in source: {N}")

    # Matrix formulation
    # 100 establishment constraints (sum = 1)
    # 3 rating constraints (25 Low, 25 Mid, 50 High)
    # 3 length constraints (30 Short, 40 Med, 30 Long)
    # Year constraints: 2 in 2016, 6 in 2017, 40-50 in 2018
    num_constraints = 100 + 3 + 3 + 3
    A = dok_matrix((num_constraints, N))
    
    b_l = [1.0] * 100 + [25.0, 25.0, 50.0] + [30.0, 40.0, 30.0] + [2.0, 6.0, 40.0]
    b_u = [1.0] * 100 + [25.0, 25.0, 50.0] + [30.0, 40.0, 30.0] + [2.0, 8.0, 50.0]

    for i, row in df.iterrows():
        # Establishment constraint
        A[rest_to_idx[row['Restaurant']], i] = 1.0
        
        # Rating constraint
        if row['rating_bin'] == 'Low (1.0-2.5)': A[100, i] = 1.0
        elif row['rating_bin'] == 'Mid (3.0-3.5)': A[101, i] = 1.0
        elif row['rating_bin'] == 'High (4.0-5.0)': A[102, i] = 1.0

        # Length constraint
        if row['length_bin'] == 'Short (<150)': A[103, i] = 1.0
        elif row['length_bin'] == 'Medium (150-350)': A[104, i] = 1.0
        elif row['length_bin'] == 'Long (>350)': A[105, i] = 1.0

        # Year constraint
        if row['year'] == 2016: A[106, i] = 1.0
        elif row['year'] == 2017: A[107, i] = 1.0
        elif row['year'] == 2018: A[108, i] = 1.0

    constraints = LinearConstraint(A.tocsc(), b_l, b_u)
    
    # Deterministic objective
    np.random.seed(42)
    c = np.random.uniform(0.1, 1.0, N)
    integrality = np.ones(N)

    res = milp(c=c, integrality=integrality, constraints=constraints)
    if not res.success:
        raise RuntimeError(f"MILP failed with status {res.status}")

    selected_indices = np.where(res.x > 0.5)[0]
    sample_df = df.iloc[selected_indices].copy()
    
    # Sort deterministically by review_id
    sample_df = sample_df.sort_values('original_row_id').reset_index(drop=True)

    # Prepare manifest columns
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
    
    manifest_df = sample_df[manifest_cols].rename(columns=manifest_rename)
    
    os.makedirs(os.path.dirname(OUTPUT_MANIFEST), exist_ok=True)
    manifest_df.to_csv(OUTPUT_MANIFEST, index=False, encoding='utf-8')
    print(f"Saved pilot manifest ({len(manifest_df)} rows) to: {OUTPUT_MANIFEST}")
    
    # Summary report
    print("\n--- Final Pilot Sample Summary ---")
    print(f"Total reviews: {len(manifest_df)}")
    print(f"Total unique establishments: {manifest_df['establishment_id'].nunique()}")
    print("\nRating Distribution (Target: 25 Low, 25 Mid, 50 High):")
    print(manifest_df['rating_bin'].value_counts())
    print("\nLength Distribution (Target: 30 Short, 40 Med, 30 Long):")
    print(manifest_df['length_bin'].value_counts())
    print("\nYear Distribution (Target: Temporal coverage across 2016-2019):")
    print(manifest_df['year'].value_counts().sort_index())

if __name__ == '__main__':
    generate_pilot_sample()
