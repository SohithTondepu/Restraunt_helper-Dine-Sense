import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.aspect_engine import extract_aspects_from_review
from src.root_cause import cluster_restaurant_complaints
from src.health_index import compute_restaurant_scorecard, validate_health_index, ASPECTS

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'cleaned_reviews.csv')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')


def run_pipeline():
    print("=== STARTING FULL OFFLINE PRECOMPUTE PIPELINE ===")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    df = pd.read_csv(DATA_PATH)
    restaurants = df['Restaurant'].unique().tolist()
    print(f"Total reviews: {len(df)} across {len(restaurants)} restaurants.")
    
    # 1. Precompute Aspect Sentiments per review
    print("\n[Step 1/3] Extracting aspect sentiments across reviews...")
    # Group reviews by restaurant
    scorecards = []
    all_complaint_clusters = {}
    
    summary_rows = []
    
    for rest in tqdm(restaurants, desc="Processing Restaurants"):
        rest_df = df[df['Restaurant'] == rest]
        
        # Extract aspect mentions for all reviews in this restaurant
        all_aspect_findings = []
        for _, row in rest_df.iterrows():
            findings = extract_aspects_from_review(str(row['Review']))
            all_aspect_findings.extend(findings)
            
        # Compute scorecard
        card = compute_restaurant_scorecard(rest, rest_df, all_aspect_findings)
        scorecards.append(card)
        
        # Flatten for tabular Parquet/CSV storage
        row_dict = {
            'Restaurant': rest,
            'Overall_Health': card['overall_health_index'],
            'Average_Stars': card['average_stars'],
            'Total_Reviews': card['total_reviews'],
            'Food_Score': card['aspects']['Food']['score'],
            'Food_Status': card['aspects']['Food']['status'],
            'Food_Mentions': card['aspects']['Food']['total_mentions'],
            'Service_Score': card['aspects']['Service']['score'],
            'Service_Status': card['aspects']['Service']['status'],
            'Service_Mentions': card['aspects']['Service']['total_mentions'],
            'Price_Score': card['aspects']['Price']['score'],
            'Price_Status': card['aspects']['Price']['status'],
            'Price_Mentions': card['aspects']['Price']['total_mentions'],
            'Ambience_Score': card['aspects']['Ambience']['score'],
            'Ambience_Status': card['aspects']['Ambience']['status'],
            'Ambience_Mentions': card['aspects']['Ambience']['total_mentions']
        }
        summary_rows.append(row_dict)
        
        # Extract complaint clusters
        clusters = cluster_restaurant_complaints(rest, rest_df, top_k=5)
        all_complaint_clusters[rest] = clusters

    # 2. Save Tabular Summaries
    print("\n[Step 2/3] Exporting processed summary tables...")
    summary_df = pd.DataFrame(summary_rows)
    summary_df.sort_values(by='Overall_Health', ascending=False, inplace=True)
    summary_df.reset_index(drop=True, inplace=True)
    
    parquet_path = os.path.join(PROCESSED_DIR, 'restaurant_health_summary.parquet')
    csv_path = os.path.join(PROCESSED_DIR, 'restaurant_health_summary.csv')
    try:
        summary_df.to_parquet(parquet_path, index=False)
        print(f"Saved Parquet summary to {parquet_path}")
    except Exception as e:
        print(f"Parquet notice: {e}")
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved CSV summary to {csv_path}")
    
    # Save complaint clusters JSON
    clusters_path = os.path.join(PROCESSED_DIR, 'complaint_clusters.json')
    with open(clusters_path, 'w') as f:
        json.dump(all_complaint_clusters, f, indent=2)
    print(f"Saved complaint clusters to {clusters_path}")
    
    # 3. Statistical Validation of Health Index
    print("\n[Step 3/3] Validating Health Index vs. Customer Star Ratings...")
    val_stats = validate_health_index(scorecards)
    print(f"Spearman Rank Correlation: {val_stats['spearman_correlation']} (p-value: {val_stats['p_value']:.4e})")
    
    val_file = os.path.join(RESULTS_DIR, 'health_index_validation.json')
    with open(val_file, 'w') as f:
        json.dump(val_stats, f, indent=2)
    print(f"Validation statistics written to {val_file}")
    
    print("\n=== OFFLINE PIPELINE EXECUTION COMPLETE ===")
    return summary_df


if __name__ == '__main__':
    run_pipeline()
