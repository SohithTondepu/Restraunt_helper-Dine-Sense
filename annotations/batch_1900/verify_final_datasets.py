import pandas as pd
import numpy as np

def verify_dataset(name, path):
    print(f"\n==========================================")
    print(f"VERIFYING DATASET: {name}")
    print(f"Path: {path}")
    print(f"==========================================")
    df = pd.read_csv(path)
    
    n_reviews = df['review_id'].nunique()
    n_clauses = df['clause_id'].nunique()
    n_assertions = len(df)
    
    print(f"Total Reviews: {n_reviews:,}")
    print(f"Total Clauses: {n_clauses:,}")
    print(f"Total Assertions: {n_assertions:,}")
    
    print("\nAnnotation Status:")
    print(df['annotation_status'].value_counts(dropna=False).to_string())
    
    print("\nAspect Breakdown:")
    print(df['aspect'].value_counts(dropna=False).to_string())
    
    print("\nSentiment Breakdown:")
    print(df['sentiment'].value_counts(dropna=False).to_string())
    
    # Character Offset Verifications
    offset_errors = 0
    checked_spans = 0
    for idx, r in df.iterrows():
        rev = str(r['review_text'])
        # 1. Clause offset
        c_start = int(r['clause_start_char'])
        c_end = int(r['clause_end_char'])
        c_text = str(r['clause_text'])
        if rev[c_start:c_end] != c_text:
            offset_errors += 1
            if offset_errors <= 3:
                print(f"Clause offset error at row {idx}")
        
        # 2. Opinion offset
        if pd.notna(r['opinion_span']) and pd.notna(r['opinion_start_char']) and pd.notna(r['opinion_end_char']):
            op_start = int(r['opinion_start_char'])
            op_end = int(r['opinion_end_char'])
            op_text = str(r['opinion_span'])
            if rev[op_start:op_end] != op_text:
                offset_errors += 1
                if offset_errors <= 3:
                    print(f"Opinion offset error at row {idx}: {repr(rev[op_start:op_end])} vs {repr(op_text)}")
            checked_spans += 1
            
        # 3. Target offset
        if pd.notna(r['aspect_target_span']) and pd.notna(r['target_start_char']) and pd.notna(r['target_end_char']):
            tg_start = int(r['target_start_char'])
            tg_end = int(r['target_end_char'])
            tg_text = str(r['aspect_target_span'])
            if rev[tg_start:tg_end] != tg_text:
                offset_errors += 1
                if offset_errors <= 3:
                    print(f"Target offset error at row {idx}: {repr(rev[tg_start:tg_end])} vs {repr(tg_text)}")
            checked_spans += 1

    print(f"\nOffset Verification Result: {offset_errors} errors across {checked_spans:,} opinion/target spans and {n_clauses:,} clauses!")
    assert offset_errors == 0, "Character offset mismatch detected!"
    return df

df_1900 = verify_dataset("1,900 Batch", r'D:\Aspect-Based-Sentimental-Analysis-on-Food-Reviews\annotations\batch_1900\annotations_1900.csv')
df_2000 = verify_dataset("2,000 Combined Final", r'D:\Aspect-Based-Sentimental-Analysis-on-Food-Reviews\annotations\combined_2000\annotations_2000_final.csv')
